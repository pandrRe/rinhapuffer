//! Linux-only async HTTP/1.1 server using io_uring. Replaces the epoll loop
//! in `http_async.zig`. All allocation is at startup; the hot path is one
//! `io_uring_enter` syscall per loop tick, amortized across the batch of CQEs
//! drained that tick.
//!
//! Milestones (incremental, see plan in `~/.claude/plans/`):
//!   M1 — skeleton: multishot accept, close on accept.
//!   M2 — one-shot recv + writev + close (functional correctness).
//!   M3 — registered files via accept_multishot_direct.
//!   M4 — BUF_RING for recv (kernel-provided buffers).
//!   M5 — multishot recv.
//!   M6 (this commit) — generation-tagged user_data for stale-CQE drop.
//!
//! Deferred from M6:
//!   - TCP_NODELAY on accepted sockets — registered files require io_uring
//!     SETSOCKOPT (kernel 6.7+, past our 5.19 floor). Prod is UDS anyway,
//!     where TCP_NODELAY is meaningless.
//!   - Helper dedup with http_async.zig — both paths still co-exist;
//!     consolidate when the epoll path is removed.
//!
//! Kernel floor is 5.19 (multishot accept). Multishot recv (6.0) lands in M5.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const http = @import("http.zig");
const instrument = @import("instrument.zig");

comptime {
    if (builtin.os.tag != .linux) {
        @compileError("http_uring is Linux-only — main.zig must comptime-branch");
    }
}

const linux = std.os.linux;
const posix = std.posix;

pub const MAX_CONNS: u32 = 256;
pub const READ_BUF_SIZE: usize = 8192;
pub const HEAD_BUF_SIZE: usize = 256;

// SQ depth holds at most: 1 accept-multishot + (1 recv + 1 writev + 1 close)
// per conn ≈ 3·MAX_CONNS = 768 in the worst burst. Round up to the next
// power of two with headroom.
const SQ_DEPTH: u16 = 1024;
const CQES_PER_BATCH: usize = 256;

// Provided-buffer ring: one 8 KiB buffer per slot. `accept_multishot_direct`
// caps live conns at MAX_CONNS, so PBUF_COUNT == MAX_CONNS guarantees the
// kernel never runs out of buffers under normal load. Power-of-two enforced
// by `setup_buf_ring`.
const PBUF_COUNT: u16 = MAX_CONNS;
const PBUF_SIZE: usize = READ_BUF_SIZE;
const PBUF_GROUP: u16 = 0;
const PBUF_MASK: u16 = PBUF_COUNT - 1;

// user_data tag layout (u64):
//   bits 0..15   slot index (0..MAX_CONNS-1)
//   bits 16..23  op tag      (ACCEPT, RECV, WRITEV, CLOSE)
//   bits 24..31  generation  (bumped on every accept-init; CQE generation
//                             must match the slot's current generation,
//                             else the CQE is from a previous tenant and
//                             gets dropped)
//   bits 32..63  reserved
const OP_ACCEPT: u8 = 0;
const OP_RECV: u8 = 1;
const OP_WRITEV: u8 = 2;
const OP_CLOSE: u8 = 3;

inline fn make_user_data(slot: u16, op: u8, gen: u8) u64 {
    return (@as(u64, gen) << 24) | (@as(u64, op) << 16) | @as(u64, slot);
}

inline fn ud_op(ud: u64) u8 {
    return @intCast((ud >> 16) & 0xff);
}

inline fn ud_slot(ud: u64) u16 {
    return @intCast(ud & 0xffff);
}

inline fn ud_gen(ud: u64) u8 {
    return @intCast((ud >> 24) & 0xff);
}

const ConnState = enum(u8) { normal, closing };

const Conn = struct {
    fd: i32,
    state: ConnState,
    in_use: bool,
    // True while a multishot recv SQE is live on this slot. The kernel keeps
    // re-firing recv CQEs (one per data arrival) until F_MORE is clear or an
    // error terminates the multishot — only then do we re-arm.
    recv_armed: bool,
    write_in_flight: bool,
    // Bumped on every accept-init for this slot. The current value is encoded
    // into every SQE's user_data; mismatched CQEs are dropped (stale from a
    // prior tenant of the same slot, e.g. recv-multishot in flight when the
    // slot got reused after close_direct).
    generation: u8,

    read_buf: [READ_BUF_SIZE]u8,
    have: u16,

    head_buf: [HEAD_BUF_SIZE]u8,
    // Two iovecs (head + body). After a partial WRITEV we mutate `base`/`len`
    // in place and bump `write_iov_idx` once a vec drains. `write_iov_idx`
    // == 2 means fully drained.
    write_iovs: [2]posix.iovec_const,
    write_iov_idx: u8,
    write_keep_alive: bool,
    // The iovec slice the kernel reads from while the WRITEV is in flight.
    // Must outlive the SQE submission — io_uring stores the pointer in the
    // SQE and the kernel dereferences it asynchronously. Stack-local would
    // race under concurrent load.
    submit_iovs: [2]posix.iovec_const,
    // Per-request stage timings. `t_total_ns` is set when the request body
    // has fully arrived (start of synchronous CPU work); `t_write_ns` is
    // set just before `submit_writev`. Both are observed in `handle_writev`
    // when the WRITEV CQE reports the response fully drained, so the
    // hist_total / hist_write percentiles cover the full kernel send time
    // (not just the SQE-submit time).
    t_total_ns: u64,
    t_write_ns: u64,
};

// Conn slot is identified directly by the kernel-allocated registered-file
// index; `accept_multishot_direct` picks a free slot in [0, MAX_CONNS) and
// returns it in `cqe.res`. No user-space free list — the kernel manages
// slot allocation against the sparse table registered at startup.
var conns: [MAX_CONNS]Conn = undefined;

// Provided-buffer ring backing storage. Page-aligned so the addresses the
// kernel sees (via `buf_ring_add`) are stable across reuse. `pbuf_ring` is
// the mmap'd shared header from `setup_buf_ring` — kernel and user share
// it lock-free (kernel reads tail, user writes tail).
var pbuf_storage: [PBUF_COUNT][PBUF_SIZE]u8 align(std.heap.page_size_min) = undefined;
var pbuf_ring: *align(std.heap.page_size_min) linux.io_uring_buf_ring = undefined;

pub const RunError = error{
    InitFailed,
    SubmitFailed,
    AcceptMultishotLost,
};

pub fn run(listen_fd: i32, comptime dispatch: http.Handler) RunError!noreturn {
    // `dispatch` is a comptime parameter so the compiler specializes every
    // helper below for this exact handler — the call from `parse_loop`
    // becomes a direct call, allowing inlining across parse → dispatch →
    // vectorize → search → format. Removes the indirect-call cost (~1 ns
    // + branch-predictor pollution) on every request and lets cross-fn
    // optimizations fire.

    // Tail-latency tuning. SINGLE_ISSUER tells the kernel exactly one task
    // ever submits SQEs (true here — `run` is the only submitter), skipping
    // multi-issuer serialization. DEFER_TASKRUN coalesces task-work onto the
    // next io_uring_enter instead of running it inline at completion time,
    // which cuts wakeup churn on multishot CQEs. COOP_TASKRUN avoids inter-
    // processor interrupts when the kernel posts CQEs from another core.
    //
    // SQPOLL alternative (build with `-Dsqpoll=true`): kernel spawns a poller
    // thread that drains the SQ ring without `io_uring_enter`, eliminating
    // the per-loop-tick syscall entirely. Steady-state cost is ~one core
    // burned. Mutually exclusive with DEFER_TASKRUN (DEFER_TASKRUN requires
    // the issuer to drain task-work via enter; SQPOLL bypasses that). We
    // also push the idle window to 2 ms so brief gaps don't park the poller.
    var params: linux.io_uring_params = std.mem.zeroes(linux.io_uring_params);
    if (comptime build_options.sqpoll) {
        params.flags = linux.IORING_SETUP_SQPOLL |
            linux.IORING_SETUP_SINGLE_ISSUER |
            linux.IORING_SETUP_COOP_TASKRUN;
        params.sq_thread_idle = 2_000;
    } else {
        params.flags = linux.IORING_SETUP_SINGLE_ISSUER |
            linux.IORING_SETUP_DEFER_TASKRUN |
            linux.IORING_SETUP_COOP_TASKRUN;
    }
    var ring = linux.IoUring.init_params(SQ_DEPTH, &params) catch blk: {
        // Older kernel rejected the flag combo. Retry with no flags.
        params = std.mem.zeroes(linux.io_uring_params);
        break :blk linux.IoUring.init_params(SQ_DEPTH, &params) catch return error.InitFailed;
    };
    defer ring.deinit();

    // Sparse registered-file table: MAX_CONNS slots reserved for
    // accept_multishot_direct's auto-allocation. Listen fd stays raw — we
    // never IOSQE_FIXED_FILE on the accept SQE's input fd.
    ring.register_files_sparse(MAX_CONNS) catch return error.InitFailed;

    // Provided-buffer ring: register, then publish all PBUF_COUNT buffers
    // in one batched advance. After this, recv SQEs can use BUFFER_SELECT
    // and the kernel picks a buf for each completion.
    pbuf_ring = linux.IoUring.setup_buf_ring(ring.fd, PBUF_COUNT, PBUF_GROUP, .{ .inc = false }) catch return error.InitFailed;
    linux.IoUring.buf_ring_init(pbuf_ring);
    for (0..PBUF_COUNT) |i| {
        linux.IoUring.buf_ring_add(
            pbuf_ring,
            pbuf_storage[i][0..],
            @intCast(i),
            PBUF_MASK,
            @intCast(i),
        );
    }
    linux.IoUring.buf_ring_advance(pbuf_ring, PBUF_COUNT);

    for (0..MAX_CONNS) |i| {
        conns[i].in_use = false;
        conns[i].generation = 0;
    }
    instrument.init();

    arm_accept(&ring, listen_fd) catch return error.SubmitFailed;

    var cqes: [CQES_PER_BATCH]linux.io_uring_cqe = undefined;
    while (true) {
        // submit_and_wait flushes pending SQEs and blocks for ≥1 CQE in one
        // io_uring_enter — THE syscall budget per loop tick.
        _ = ring.submit_and_wait(1) catch return error.SubmitFailed;

        while (true) {
            const n = ring.copy_cqes(&cqes, 0) catch return error.SubmitFailed;
            if (n == 0) break;
            for (cqes[0..n]) |cqe| dispatch_cqe(&ring, listen_fd, cqe, dispatch) catch |err| return err;
        }
    }
}

fn dispatch_cqe(ring: *linux.IoUring, listen_fd: i32, cqe: linux.io_uring_cqe, comptime dispatch: http.Handler) RunError!void {
    switch (ud_op(cqe.user_data)) {
        OP_ACCEPT => try handle_accept(ring, listen_fd, cqe),
        OP_RECV => try handle_recv(ring, cqe, dispatch),
        OP_WRITEV => try handle_writev(ring, cqe, dispatch),
        OP_CLOSE => handle_close(cqe),
        else => {},
    }
}

// ─── accept ─────────────────────────────────────────────────────────────────

fn arm_accept(ring: *linux.IoUring, listen_fd: i32) !void {
    // Direct multishot accept: kernel-allocates a registered-file slot per
    // accepted connection and reports that slot index in cqe.res. Saves a
    // per-accept io_uring_register(REGISTER_FILES_UPDATE) syscall vs the
    // raw-fd accept_multishot variant.
    _ = try ring.accept_multishot_direct(make_user_data(0, OP_ACCEPT, 0), listen_fd, null, null, 0);
}

fn handle_accept(ring: *linux.IoUring, listen_fd: i32, cqe: linux.io_uring_cqe) RunError!void {
    if (cqe.res >= 0) {
        const slot: u16 = @intCast(cqe.res);
        std.debug.assert(slot < MAX_CONNS);
        instrument.inc(&instrument.accepts, 1);

        const conn = &conns[slot];
        // Wraparound is fine: u8 cycles every 256 reuses. Stale CQEs are
        // serviced microseconds after the kernel posts them; even at insane
        // reuse rates the wrap never collides with an actually-stale CQE.
        const next_gen: u8 = conn.generation +% 1;
        conn.* = .{
            // fd is left zeroed — all subsequent ops use the registered-file
            // index (`slot`) with IOSQE_FIXED_FILE, never the raw fd.
            .fd = 0,
            .state = .normal,
            .in_use = true,
            .recv_armed = false,
            .write_in_flight = false,
            .generation = next_gen,
            .read_buf = undefined,
            .have = 0,
            .head_buf = undefined,
            .write_iovs = .{
                .{ .base = undefined, .len = 0 },
                .{ .base = undefined, .len = 0 },
            },
            .write_iov_idx = 2,
            .write_keep_alive = true,
            .submit_iovs = .{
                .{ .base = undefined, .len = 0 },
                .{ .base = undefined, .len = 0 },
            },
            .t_total_ns = 0,
            .t_write_ns = 0,
        };

        submit_recv(ring, slot) catch return error.SubmitFailed;
    }
    // res < 0 covers per-accept errors (incl. -ENFILE when the registered
    // table is full → connection stays in the listen queue, kernel keeps
    // the multishot armed via F_MORE). Drop silently.

    if (cqe.flags & linux.IORING_CQE_F_MORE == 0) {
        arm_accept(ring, listen_fd) catch return error.AcceptMultishotLost;
    }
}

// ─── recv ──────────────────────────────────────────────────────────────────

fn submit_recv(ring: *linux.IoUring, slot: u16) !void {
    const conn = &conns[slot];
    if (conn.have >= READ_BUF_SIZE) return; // scratch full; can't accept more
    if (conn.recv_armed) return;
    const ud = make_user_data(slot, OP_RECV, conn.generation);
    // BUFFER_SELECT + RECV_MULTISHOT: kernel keeps firing CQEs for every
    // recv on this fd, picking a free buf from BUF_RING per completion,
    // until F_MORE is clear (typically -ENOBUFS or peer-close). One SQE
    // per conn lifetime, not per request — the steady-state syscall budget
    // collapses to one io_uring_enter per loop tick.
    const sqe = try ring.recv(ud, @intCast(slot), .{ .buffer_selection = .{
        .group_id = PBUF_GROUP,
        .len = PBUF_SIZE,
    } }, 0);
    sqe.flags |= linux.IOSQE_FIXED_FILE;
    sqe.ioprio |= linux.IORING_RECV_MULTISHOT;
    conn.recv_armed = true;
}

fn handle_recv(ring: *linux.IoUring, cqe: linux.io_uring_cqe, comptime dispatch: http.Handler) RunError!void {
    const slot = ud_slot(cqe.user_data);
    if (slot >= MAX_CONNS) return;
    const conn = &conns[slot];
    if (!conn.in_use) return;
    if (ud_gen(cqe.user_data) != conn.generation) return; // stale CQE — slot was reused

    // Multishot recv: F_MORE clear means the kernel has retired this SQE.
    // Common causes: -ENOBUFS (ring empty), -ECONNRESET, peer half-close
    // with res==0, or some error paths. Once retired we must explicitly
    // re-arm to keep receiving.
    if (cqe.flags & linux.IORING_CQE_F_MORE == 0) {
        conn.recv_armed = false;
    }

    if (cqe.res < 0) {
        // -ENOBUFS is recoverable — we always release a buf back on every
        // successful recv, so it's effectively only the burst-startup case.
        // Re-arm and continue. Other negatives are I/O errors → close.
        if (cqe.res == -@as(i32, @intFromEnum(linux.E.NOBUFS))) {
            if (conn.state == .normal and !conn.recv_armed and conn.have < READ_BUF_SIZE) {
                submit_recv(ring, slot) catch return error.SubmitFailed;
            }
            return;
        }
        start_close(ring, slot) catch return error.SubmitFailed;
        return;
    }
    if (cqe.res == 0) {
        // peer half-closed. (BUFFER_SELECT recvs with res==0 don't have
        // F_BUFFER set — no buffer to release.)
        start_close(ring, slot) catch return error.SubmitFailed;
        return;
    }

    // Decode the kernel-picked buffer ID from cqe.flags, copy into the
    // parser-stable scratch, then re-publish the buf back to the ring so
    // it's immediately available for the next recv.
    const buf_id = cqe.buffer_id() catch {
        // CQE has no F_BUFFER bit despite res > 0 — unexpected; treat as
        // recv error and close.
        start_close(ring, slot) catch return error.SubmitFailed;
        return;
    };
    const n: usize = @intCast(cqe.res);
    const room = READ_BUF_SIZE - @as(usize, conn.have);
    const take = @min(n, room);
    @memcpy(conn.read_buf[conn.have .. @as(usize, conn.have) + take], pbuf_storage[buf_id][0..take]);
    conn.have += @intCast(take);
    linux.IoUring.buf_ring_add(pbuf_ring, pbuf_storage[buf_id][0..], buf_id, PBUF_MASK, 0);
    linux.IoUring.buf_ring_advance(pbuf_ring, 1);

    parse_loop(ring, slot, dispatch) catch return error.SubmitFailed;

    // If the multishot terminated (F_MORE was clear) and we're still alive
    // with room, re-arm. Otherwise the kernel keeps streaming — no SQE.
    if (conn.state == .normal and !conn.recv_armed and conn.have < READ_BUF_SIZE) {
        submit_recv(ring, slot) catch return error.SubmitFailed;
    }
    // Buffer is full but no parseable request — header oversize.
    if (conn.have == READ_BUF_SIZE and !conn.write_in_flight and conn.state == .normal) {
        send_status_close(ring, slot, 413) catch return error.SubmitFailed;
    }
}

// ─── parse + dispatch ──────────────────────────────────────────────────────

fn parse_loop(ring: *linux.IoUring, slot: u16, comptime dispatch: http.Handler) !void {
    const conn = &conns[slot];
    while (conn.state == .normal and !conn.write_in_flight) {
        // Single header walk: returns null when \r\n\r\n isn't here yet,
        // otherwise yields method/path/keep_alive + the offsets we need
        // to size the body. Replaces the old `indexOf+sniff+parse` triple.
        const t_parse = instrument.now_ns();
        const hp = http.parse_headers(conn.read_buf[0..conn.have]) catch |err| {
            instrument.inc(&instrument.req_parse_err, 1);
            try send_status_close(ring, slot, map_parse_status(err));
            return;
        } orelse return;
        const total = hp.headers_end + hp.content_length;
        if (total > READ_BUF_SIZE) {
            try send_status_close(ring, slot, 413);
            return;
        }
        if (total > conn.have) return; // need more bytes
        instrument.observe_since(&instrument.hist_parse, t_parse);
        // Mark the start of the full request lifecycle. Observed in
        // handle_writev when WRITEV fully drains.
        conn.t_total_ns = t_parse;

        const body: []const u8 = if (hp.method == .post)
            conn.read_buf[hp.headers_end..total]
        else
            "";
        const req: http.Request = .{
            .method = hp.method,
            .path = hp.path,
            .body = body,
            .keep_alive = hp.keep_alive,
        };

        const resp = dispatch(req);

        const head_slice: []const u8 = if (http.fast_head(resp)) |fh| blk: {
            instrument.inc(&instrument.head_fast, 1);
            break :blk fh;
        } else blk: {
            instrument.inc(&instrument.head_slow, 1);
            break :blk http.format_head(&conn.head_buf, resp) catch {
                start_close(ring, slot) catch {};
                return;
            };
        };

        conn.write_iovs[0] = .{ .base = head_slice.ptr, .len = head_slice.len };
        conn.write_iovs[1] = .{ .base = resp.body.ptr, .len = resp.body.len };
        conn.write_iov_idx = 0;
        conn.write_keep_alive = resp.keep_alive;
        conn.t_write_ns = instrument.now_ns();
        try submit_writev(ring, slot);
        advance_read(conn, total);
        // Loop to look for the next pipelined request — but only if writev
        // didn't end up needing the buffer (it shouldn't, we already sliced).
    }
}

fn advance_read(conn: *Conn, consumed: usize) void {
    if (conn.have > consumed) {
        std.mem.copyForwards(
            u8,
            conn.read_buf[0 .. @as(usize, conn.have) - consumed],
            conn.read_buf[consumed..conn.have],
        );
    }
    conn.have -= @intCast(consumed);
}

// ─── write ──────────────────────────────────────────────────────────────────

fn submit_writev(ring: *linux.IoUring, slot: u16) !void {
    const conn = &conns[slot];
    // The iovec backing storage MUST outlive the SQE — io_uring captures the
    // pointer and dereferences it asynchronously. Use the per-conn
    // `submit_iovs` field, not a stack-local: under concurrent load the
    // stack frame would be reused before the kernel reads it, corrupting
    // the iovec and emitting a garbled head/body on the wire.
    var iov_count: usize = 0;
    if (conn.write_iov_idx == 0) {
        conn.submit_iovs[0] = conn.write_iovs[0];
        iov_count = 1;
        if (conn.write_iovs[1].len > 0) {
            conn.submit_iovs[1] = conn.write_iovs[1];
            iov_count = 2;
        }
    } else if (conn.write_iov_idx == 1) {
        conn.submit_iovs[0] = conn.write_iovs[1];
        iov_count = 1;
    } else {
        return; // already drained
    }
    const ud = make_user_data(slot, OP_WRITEV, conn.generation);
    const sqe = try ring.writev(ud, @intCast(slot), conn.submit_iovs[0..iov_count], 0);
    sqe.flags |= linux.IOSQE_FIXED_FILE;
    conn.write_in_flight = true;
}

fn handle_writev(ring: *linux.IoUring, cqe: linux.io_uring_cqe, comptime dispatch: http.Handler) RunError!void {
    const slot = ud_slot(cqe.user_data);
    if (slot >= MAX_CONNS) return;
    const conn = &conns[slot];
    if (!conn.in_use) return;
    if (ud_gen(cqe.user_data) != conn.generation) return; // stale CQE — slot was reused
    conn.write_in_flight = false;

    if (cqe.res < 0) {
        // Write error → close.
        start_close(ring, slot) catch return error.SubmitFailed;
        return;
    }

    advance_write(conn, @intCast(cqe.res));

    if (conn.write_iov_idx != 2) {
        // Partial — resubmit the tail.
        submit_writev(ring, slot) catch return error.SubmitFailed;
        return;
    }

    // Fully drained — observe write + total stage timings here so they
    // include kernel send completion, not just the SQE submit.
    instrument.observe_since(&instrument.hist_write, conn.t_write_ns);
    instrument.observe_since(&instrument.hist_total, conn.t_total_ns);

    if (!conn.write_keep_alive) {
        start_close(ring, slot) catch return error.SubmitFailed;
        return;
    }
    // Try to dispatch the next pipelined request.
    parse_loop(ring, slot, dispatch) catch return error.SubmitFailed;
    // Re-arm recv if room and no recv pending.
    if (conn.state == .normal and !conn.recv_armed and conn.have < READ_BUF_SIZE) {
        submit_recv(ring, slot) catch return error.SubmitFailed;
    }
}

fn advance_write(conn: *Conn, n: usize) void {
    var rem: usize = n;
    if (conn.write_iov_idx == 0) {
        const head_len = conn.write_iovs[0].len;
        if (rem >= head_len) {
            rem -= head_len;
            conn.write_iovs[0].len = 0;
            conn.write_iov_idx = if (conn.write_iovs[1].len == 0) 2 else 1;
        } else {
            conn.write_iovs[0].base = conn.write_iovs[0].base + rem;
            conn.write_iovs[0].len -= rem;
            return;
        }
    }
    if (conn.write_iov_idx == 1) {
        const body_len = conn.write_iovs[1].len;
        if (rem >= body_len) {
            conn.write_iovs[1].len = 0;
            conn.write_iov_idx = 2;
        } else {
            conn.write_iovs[1].base = conn.write_iovs[1].base + rem;
            conn.write_iovs[1].len -= rem;
        }
    }
}

// ─── close ──────────────────────────────────────────────────────────────────

fn start_close(ring: *linux.IoUring, slot: u16) !void {
    const conn = &conns[slot];
    if (conn.state == .closing) return;
    conn.state = .closing;
    if (conn.write_in_flight) {
        // Wait for the pending WRITEV CQE; it'll re-enter start_close from
        // its own error/keep-alive=false path. Don't double-submit close.
        return;
    }
    // close_direct closes the underlying fd AND releases the registered-file
    // slot back to the kernel's auto-alloc pool — next accept_multishot_direct
    // can pick this slot again.
    const ud = make_user_data(slot, OP_CLOSE, conn.generation);
    _ = try ring.close_direct(ud, slot);
}

fn handle_close(cqe: linux.io_uring_cqe) void {
    const slot = ud_slot(cqe.user_data);
    if (slot >= MAX_CONNS) return;
    // CLOSE generation is the generation we owned at the time we submitted.
    // If `conns[slot].generation` already moved past us, a previous close
    // CQE already cleared in_use and the slot is the new tenant — drop.
    if (ud_gen(cqe.user_data) != conns[slot].generation) return;
    conns[slot].in_use = false;
    instrument.inc(&instrument.conn_closes, 1);
}

// ─── error responses ───────────────────────────────────────────────────────

fn send_status_close(ring: *linux.IoUring, slot: u16, status: u16) !void {
    const conn = &conns[slot];
    const resp: http.Response = .{
        .status = status,
        .body = "",
        .content_type = "text/plain",
        .keep_alive = false,
    };
    const head = http.format_head(&conn.head_buf, resp) catch {
        try start_close(ring, slot);
        return;
    };
    conn.write_iovs[0] = .{ .base = head.ptr, .len = head.len };
    conn.write_iovs[1] = .{ .base = head.ptr, .len = 0 };
    conn.write_iov_idx = 0;
    conn.write_keep_alive = false;
    try submit_writev(ring, slot);
}

// ─── parse helpers ──────────────────────────────────────────────────────────

fn map_parse_status(err: http.ParseError) u16 {
    return switch (err) {
        error.UnsupportedMethod, error.UnsupportedVersion, error.Malformed, error.MissingContentLength => 400,
        error.BodyTooLarge, error.HeadersOversize => 413,
    };
}

// ─── tests ─────────────────────────────────────────────────────────────────

test "user_data tag pack/unpack round-trip" {
    const ud = make_user_data(42, OP_RECV, 7);
    try std.testing.expectEqual(@as(u16, 42), ud_slot(ud));
    try std.testing.expectEqual(OP_RECV, ud_op(ud));
}

test "advance_write: head only, partial then full" {
    var conn: Conn = undefined;
    conn.write_iovs = .{
        .{ .base = "abcdefghij".ptr, .len = 10 },
        .{ .base = "xyz".ptr, .len = 3 },
    };
    conn.write_iov_idx = 0;
    advance_write(&conn, 4);
    try std.testing.expectEqual(@as(u8, 0), conn.write_iov_idx);
    try std.testing.expectEqual(@as(usize, 6), conn.write_iovs[0].len);

    advance_write(&conn, 6);
    try std.testing.expectEqual(@as(u8, 1), conn.write_iov_idx);

    advance_write(&conn, 3);
    try std.testing.expectEqual(@as(u8, 2), conn.write_iov_idx);
}

test "advance_write: spans head and body in one call" {
    var conn: Conn = undefined;
    conn.write_iovs = .{
        .{ .base = "abc".ptr, .len = 3 },
        .{ .base = "xyzpqr".ptr, .len = 6 },
    };
    conn.write_iov_idx = 0;
    advance_write(&conn, 5); // 3 head + 2 body
    try std.testing.expectEqual(@as(u8, 1), conn.write_iov_idx);
    try std.testing.expectEqual(@as(usize, 4), conn.write_iovs[1].len);
}

test "map_parse_status — 400 vs 413" {
    try std.testing.expectEqual(@as(u16, 400), map_parse_status(error.UnsupportedMethod));
    try std.testing.expectEqual(@as(u16, 413), map_parse_status(error.BodyTooLarge));
}
