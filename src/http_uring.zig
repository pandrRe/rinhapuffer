//! Linux-only async HTTP/1.1 server using io_uring with **fd-passing accept**.
//!
//! lb is no longer in the data path. Instead, lb (rinhalb) accepts the
//! public TCP connection and hands the fd to this process via SCM_RIGHTS
//! over a persistent Unix STREAM control connection. This module:
//!
//!   1. Listens on the api's UDS (bound by main.zig).
//!   2. Accepts the single control connection from lb (one-shot).
//!   3. Loops on `recvmsg` against that control connection, extracting one
//!      client TCP fd per record from the SCM_RIGHTS cmsg.
//!   4. For each recovered fd, spins up a Conn slot and runs the existing
//!      multishot-recv → parse → writev → keep-alive loop directly on the
//!      TCP fd. The kernel handles cross-net-namespace usage transparently
//!      (the socket's `sk_net` was set when lb accepted; our process ns
//!      doesn't matter for I/O on the fd).
//!
//! Per-request lb work: zero. Per-connection lb work: one sendmsg(SCM_RIGHTS).
//! lb scheduling contention with this process under cgroup CFS is no longer
//! the dominant tail source.
//!
//! Trade vs the old accept-multishot-direct path: client fds are raw, not
//! registered. recv/writev/close SQEs carry the raw fd. Per-SQE cost is a
//! few hundred ns higher (refcount bump per SQE) but we save the per-conn
//! IORING_REGISTER_FILES_UPDATE call. Net impact: negligible at our load.

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

// SQ depth: 1 control-recvmsg + per-conn (recv + writev + close) ≈ 3·MAX_CONNS
// = 768. Round up to 1024 with headroom.
const SQ_DEPTH: u16 = 1024;
const CQES_PER_BATCH: usize = 256;

// Provided-buffer ring: one 8 KiB buffer per slot. PBUF_COUNT == MAX_CONNS
// guarantees the kernel never runs out of buffers under normal load. Power
// of two enforced by setup_buf_ring.
const PBUF_COUNT: u16 = MAX_CONNS;
const PBUF_SIZE: usize = READ_BUF_SIZE;
const PBUF_GROUP: u16 = 0;
const PBUF_MASK: u16 = PBUF_COUNT - 1;

// user_data tag layout (u64):
//   bits 0..15   slot index (0..MAX_CONNS-1) for client conns
//                — sentinel SLOT_LISTEN / SLOT_CONTROL for the control path
//   bits 16..23  op tag (ACCEPT, RECVMSG, RECV, WRITEV, CLOSE)
//   bits 24..31  generation (bumped on every Conn slot init; CQE generation
//                must match the slot's current generation, else the CQE is
//                from a previous tenant and is dropped)
const OP_ACCEPT: u8 = 0;       // one-shot: control conn from lb
const OP_RECVMSG: u8 = 1;      // one-shot: pull client fd from control conn
const OP_RECV: u8 = 2;         // multishot: data on a client TCP fd
const OP_WRITEV: u8 = 3;       // one-shot: drain response to client TCP fd
const OP_CLOSE: u8 = 4;        // one-shot: close client TCP fd

// Sentinel slot values for non-Conn-array CQEs (control-path ops). Any value
// >= MAX_CONNS is treated as a control-path slot, never indexes `conns[]`.
const SLOT_LISTEN: u16 = MAX_CONNS;
const SLOT_CONTROL: u16 = MAX_CONNS + 1;

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
    // Raw TCP fd received via SCM_RIGHTS. Used directly in recv/writev/close
    // SQEs — no registered-file slot, no IOSQE_FIXED_FILE.
    fd: i32,
    state: ConnState,
    in_use: bool,
    recv_armed: bool,
    write_in_flight: bool,
    // Bumped on every slot init. Encoded into every SQE's user_data; CQEs
    // with mismatched generation are dropped (stale tenant after close).
    generation: u8,

    read_buf: [READ_BUF_SIZE]u8,
    have: u16,

    head_buf: [HEAD_BUF_SIZE]u8,
    write_iovs: [2]posix.iovec_const,
    write_iov_idx: u8,
    write_keep_alive: bool,
    submit_iovs: [2]posix.iovec_const,
    t_total_ns: u64,
    t_write_ns: u64,
};

var conns: [MAX_CONNS]Conn = undefined;

// Free-slot stack for Conn array. Replaces accept_multishot_direct's
// kernel-managed slot allocation. A single u16 array indexed by `free_top`;
// pop returns the highest free slot; push puts it back.
var free_slots: [MAX_CONNS]u16 = undefined;
var free_top: usize = 0;

fn init_free_slots() void {
    free_top = MAX_CONNS;
    for (0..MAX_CONNS) |i| free_slots[i] = @intCast(MAX_CONNS - 1 - i);
}

fn pop_slot() ?u16 {
    if (free_top == 0) return null;
    free_top -= 1;
    return free_slots[free_top];
}

fn push_slot(slot: u16) void {
    free_slots[free_top] = slot;
    free_top += 1;
}

// Provided-buffer ring backing storage. Page-aligned so the addresses the
// kernel sees (via buf_ring_add) are stable across reuse.
var pbuf_storage: [PBUF_COUNT][PBUF_SIZE]u8 align(std.heap.page_size_min) = undefined;
var pbuf_ring: *align(std.heap.page_size_min) linux.io_uring_buf_ring = undefined;

// ─── control-channel state ──────────────────────────────────────────────────
//
// Single persistent connection from lb. recvmsg one-shot pulls one fd per
// completion via SCM_RIGHTS cmsg.

var control_fd: i32 = -1;

// msghdr/iovec/cmsg storage for recvmsg. Static so the pointers remain valid
// across SQE submission. One set is enough because recvmsg is one-shot — the
// kernel reads the layout when the SQE is processed and the ABI populates
// it during the operation; we read the result on CQE and immediately re-arm.
var recvmsg_iov_data: [16]u8 = undefined; // 1-byte payload "F" arrives here
var recvmsg_iov: posix.iovec = undefined;
var recvmsg_cmsg_buf: [256]u8 align(@alignOf(linux.cmsghdr)) = undefined;
var recvmsg_hdr: linux.msghdr = undefined;

fn init_recvmsg_hdr() void {
    recvmsg_iov.base = &recvmsg_iov_data;
    recvmsg_iov.len = recvmsg_iov_data.len;
    recvmsg_hdr = .{
        .name = null,
        .namelen = 0,
        .iov = @ptrCast(&recvmsg_iov),
        .iovlen = 1,
        .control = &recvmsg_cmsg_buf,
        .controllen = recvmsg_cmsg_buf.len,
        .flags = 0,
    };
}

// Walk the cmsg buffer after a recvmsg completion and extract the first
// SCM_RIGHTS fd. Returns -1 if no fd was found (shouldn't happen if lb is
// behaving). Linux cmsg layout: each cmsghdr is followed by data, padded
// to alignment; a chain of cmsghdrs ends when one runs past controllen.
fn extract_fd_from_cmsg() i32 {
    // Scan the cmsg buffer for the first SCM_RIGHTS record.
    const got_len: usize = @intCast(recvmsg_hdr.controllen);
    if (got_len < @sizeOf(linux.cmsghdr)) return -1;
    var off: usize = 0;
    while (off + @sizeOf(linux.cmsghdr) <= got_len) {
        const cmsg: *const linux.cmsghdr = @ptrCast(@alignCast(&recvmsg_cmsg_buf[off]));
        const cmsg_len: usize = @intCast(cmsg.len);
        if (cmsg_len < @sizeOf(linux.cmsghdr)) return -1;
        if (cmsg.level == linux.SOL.SOCKET and cmsg.type == 0x01) {
            // SCM_RIGHTS == 0x01 on Linux. Extract the first int payload.
            const data_off = off + cmsg_align(@sizeOf(linux.cmsghdr));
            if (data_off + @sizeOf(i32) > got_len) return -1;
            const fd_ptr: *const i32 = @ptrCast(@alignCast(&recvmsg_cmsg_buf[data_off]));
            return fd_ptr.*;
        }
        off += cmsg_align(cmsg_len);
    }
    return -1;
}

// CMSG_ALIGN — round up to sizeof(size_t). Linux uses sizeof(long) which on
// x86_64-musl is 8.
inline fn cmsg_align(len: usize) usize {
    const a = @sizeOf(usize);
    return (len + a - 1) & ~(@as(usize, a - 1));
}

pub const RunError = error{
    InitFailed,
    SubmitFailed,
    AcceptFailed,
};

pub fn run(listen_fd: i32, comptime dispatch: http.Handler) RunError!noreturn {
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
        params = std.mem.zeroes(linux.io_uring_params);
        break :blk linux.IoUring.init_params(SQ_DEPTH, &params) catch return error.InitFailed;
    };
    defer ring.deinit();

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

    init_free_slots();
    init_recvmsg_hdr();
    for (0..MAX_CONNS) |i| {
        conns[i].in_use = false;
        conns[i].generation = 0;
        conns[i].fd = -1;
    }
    instrument.init();
    listen_fd_g = listen_fd;

    arm_accept_control(&ring, listen_fd) catch return error.SubmitFailed;

    var cqes: [CQES_PER_BATCH]linux.io_uring_cqe = undefined;
    while (true) {
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
        OP_ACCEPT => try handle_accept_control(ring, listen_fd, cqe),
        OP_RECVMSG => try handle_recvmsg(ring, cqe),
        OP_RECV => try handle_recv(ring, cqe, dispatch),
        OP_WRITEV => try handle_writev(ring, cqe, dispatch),
        OP_CLOSE => handle_close(cqe),
        else => {},
    }
}

// ─── control-conn accept ────────────────────────────────────────────────────
//
// One-shot accept on the listen UDS. Fires once per lb session (typically
// once for the lifetime of this process). On a CQE we stash control_fd and
// arm the recvmsg loop. If lb disconnects later, we re-arm accept to wait
// for a new lb instance.

fn arm_accept_control(ring: *linux.IoUring, listen_fd: i32) !void {
    const ud = make_user_data(SLOT_LISTEN, OP_ACCEPT, 0);
    _ = try ring.accept(ud, listen_fd, null, null, 0);
}

fn handle_accept_control(ring: *linux.IoUring, listen_fd: i32, cqe: linux.io_uring_cqe) RunError!void {
    if (cqe.res < 0) {
        // Accept failed — wait a tick and retry.
        arm_accept_control(ring, listen_fd) catch return error.AcceptFailed;
        return;
    }
    control_fd = cqe.res;
    instrument.inc(&instrument.accepts, 1);
    arm_recvmsg(ring) catch return error.SubmitFailed;
}

// ─── recvmsg from lb (fd handoff) ───────────────────────────────────────────

fn arm_recvmsg(ring: *linux.IoUring) !void {
    if (control_fd < 0) return;
    // Reset cmsg controllen for the next receive — the kernel writes the
    // actual size back, so we have to restore the buffer cap each time.
    recvmsg_hdr.controllen = recvmsg_cmsg_buf.len;
    recvmsg_hdr.flags = 0;
    const ud = make_user_data(SLOT_CONTROL, OP_RECVMSG, 0);
    _ = try ring.recvmsg(ud, control_fd, &recvmsg_hdr, 0);
}

fn handle_recvmsg(ring: *linux.IoUring, cqe: linux.io_uring_cqe) RunError!void {
    if (cqe.res <= 0) {
        // lb disconnected (peer close or error). Drop control_fd and re-arm
        // accept so a new lb can attach.
        if (control_fd >= 0) {
            _ = linux.close(control_fd);
            control_fd = -1;
        }
        // Caller's loop will re-arm accept on next tick — but we'd lose that
        // signal here. Instead: caller should detect control_fd==-1 and
        // arm accept. Simpler: arm here, but we need listen_fd which we
        // don't have. Stash it at startup as a global to keep this tidy.
        listen_fd_g_arm_accept(ring) catch {};
        return;
    }
    const client_fd = extract_fd_from_cmsg();
    if (client_fd < 0) {
        // Malformed cmsg — re-arm and continue.
        arm_recvmsg(ring) catch return error.SubmitFailed;
        return;
    }
    const slot = pop_slot() orelse {
        // Conn table full — close the fd to avoid leaking.
        _ = linux.close(client_fd);
        arm_recvmsg(ring) catch return error.SubmitFailed;
        return;
    };
    init_conn(slot, client_fd);
    submit_recv(ring, slot) catch return error.SubmitFailed;
    arm_recvmsg(ring) catch return error.SubmitFailed;
}

// listen_fd is needed by handle_recvmsg's lb-disconnect path. Stashed at
// run() entry into a module-level global so the dispatch handlers don't
// need it threaded through.
var listen_fd_g: i32 = -1;
fn listen_fd_g_arm_accept(ring: *linux.IoUring) !void {
    if (listen_fd_g >= 0) try arm_accept_control(ring, listen_fd_g);
}

fn init_conn(slot: u16, client_fd: i32) void {
    const conn = &conns[slot];
    const next_gen: u8 = conn.generation +% 1;
    conn.* = .{
        .fd = client_fd,
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
}

// ─── recv ──────────────────────────────────────────────────────────────────

fn submit_recv(ring: *linux.IoUring, slot: u16) !void {
    const conn = &conns[slot];
    if (conn.have >= READ_BUF_SIZE) return; // scratch full; can't accept more
    if (conn.recv_armed) return;
    if (conn.fd < 0) return;
    const ud = make_user_data(slot, OP_RECV, conn.generation);
    const sqe = try ring.recv(ud, conn.fd, .{ .buffer_selection = .{
        .group_id = PBUF_GROUP,
        .len = PBUF_SIZE,
    } }, 0);
    sqe.ioprio |= linux.IORING_RECV_MULTISHOT;
    conn.recv_armed = true;
}

fn handle_recv(ring: *linux.IoUring, cqe: linux.io_uring_cqe, comptime dispatch: http.Handler) RunError!void {
    const slot = ud_slot(cqe.user_data);
    if (slot >= MAX_CONNS) return;
    const conn = &conns[slot];
    if (!conn.in_use) return;
    if (ud_gen(cqe.user_data) != conn.generation) return;

    if (cqe.flags & linux.IORING_CQE_F_MORE == 0) {
        conn.recv_armed = false;
    }

    if (cqe.res < 0) {
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
        start_close(ring, slot) catch return error.SubmitFailed;
        return;
    }

    const buf_id = cqe.buffer_id() catch {
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

    if (conn.state == .normal and !conn.recv_armed and conn.have < READ_BUF_SIZE) {
        submit_recv(ring, slot) catch return error.SubmitFailed;
    }
    if (conn.have == READ_BUF_SIZE and !conn.write_in_flight and conn.state == .normal) {
        send_status_close(ring, slot, 413) catch return error.SubmitFailed;
    }
}

// ─── parse + dispatch ──────────────────────────────────────────────────────

fn parse_loop(ring: *linux.IoUring, slot: u16, comptime dispatch: http.Handler) !void {
    const conn = &conns[slot];
    while (conn.state == .normal and !conn.write_in_flight) {
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
        if (total > conn.have) return;
        instrument.observe_since(&instrument.hist_parse, t_parse);
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
        return;
    }
    const ud = make_user_data(slot, OP_WRITEV, conn.generation);
    _ = try ring.writev(ud, conn.fd, conn.submit_iovs[0..iov_count], 0);
    conn.write_in_flight = true;
}

fn handle_writev(ring: *linux.IoUring, cqe: linux.io_uring_cqe, comptime dispatch: http.Handler) RunError!void {
    const slot = ud_slot(cqe.user_data);
    if (slot >= MAX_CONNS) return;
    const conn = &conns[slot];
    if (!conn.in_use) return;
    if (ud_gen(cqe.user_data) != conn.generation) return;
    conn.write_in_flight = false;

    if (cqe.res < 0) {
        start_close(ring, slot) catch return error.SubmitFailed;
        return;
    }

    advance_write(conn, @intCast(cqe.res));

    if (conn.write_iov_idx != 2) {
        submit_writev(ring, slot) catch return error.SubmitFailed;
        return;
    }

    instrument.observe_since(&instrument.hist_write, conn.t_write_ns);
    instrument.observe_since(&instrument.hist_total, conn.t_total_ns);

    if (!conn.write_keep_alive) {
        start_close(ring, slot) catch return error.SubmitFailed;
        return;
    }
    parse_loop(ring, slot, dispatch) catch return error.SubmitFailed;
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
    if (conn.write_in_flight) return;
    if (conn.fd < 0) {
        // Already closed somehow — just free the slot.
        conn.in_use = false;
        push_slot(slot);
        return;
    }
    const ud = make_user_data(slot, OP_CLOSE, conn.generation);
    _ = try ring.close(ud, conn.fd);
}

fn handle_close(cqe: linux.io_uring_cqe) void {
    const slot = ud_slot(cqe.user_data);
    if (slot >= MAX_CONNS) return;
    if (ud_gen(cqe.user_data) != conns[slot].generation) return;
    conns[slot].in_use = false;
    conns[slot].fd = -1;
    push_slot(slot);
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
    try std.testing.expectEqual(@as(u8, 7), ud_gen(ud));
}

test "free-slot stack pop/push round-trip" {
    init_free_slots();
    try std.testing.expectEqual(@as(usize, MAX_CONNS), free_top);
    const a = pop_slot().?;
    const b = pop_slot().?;
    try std.testing.expect(a != b);
    push_slot(a);
    push_slot(b);
    try std.testing.expectEqual(@as(usize, MAX_CONNS), free_top);
}

test "free-slot stack exhaustion returns null" {
    init_free_slots();
    var i: usize = 0;
    while (i < MAX_CONNS) : (i += 1) _ = pop_slot().?;
    try std.testing.expect(pop_slot() == null);
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
    advance_write(&conn, 5);
    try std.testing.expectEqual(@as(u8, 1), conn.write_iov_idx);
    try std.testing.expectEqual(@as(usize, 4), conn.write_iovs[1].len);
}

test "map_parse_status — 400 vs 413" {
    try std.testing.expectEqual(@as(u16, 400), map_parse_status(error.UnsupportedMethod));
    try std.testing.expectEqual(@as(u16, 413), map_parse_status(error.BodyTooLarge));
}

test "extract_fd_from_cmsg: well-formed SCM_RIGHTS frame" {
    init_recvmsg_hdr();
    // Build a cmsg with SCM_RIGHTS containing fd=42.
    @memset(&recvmsg_cmsg_buf, 0);
    const cmsg: *linux.cmsghdr = @ptrCast(@alignCast(&recvmsg_cmsg_buf));
    cmsg.len = @sizeOf(linux.cmsghdr) + @sizeOf(i32);
    cmsg.level = linux.SOL.SOCKET;
    cmsg.type = 0x01; // SCM_RIGHTS
    const data_off = cmsg_align(@sizeOf(linux.cmsghdr));
    const fd_target_ptr: *i32 = @ptrCast(@alignCast(&recvmsg_cmsg_buf[data_off]));
    fd_target_ptr.* = 42;
    recvmsg_hdr.controllen = cmsg_align(cmsg.len);

    const got = extract_fd_from_cmsg();
    try std.testing.expectEqual(@as(i32, 42), got);
}
