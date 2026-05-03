//! Linux-only async HTTP/1.1 server using io_uring. Replaces the epoll loop
//! in `http_async.zig`. All allocation is at startup; the hot path is one
//! `io_uring_enter` syscall per loop tick, amortized across the batch of CQEs
//! drained that tick.
//!
//! Milestones (incremental, see plan in `~/.claude/plans/`):
//!   M1 — skeleton: multishot accept, close on accept.
//!   M2 (this commit) — one-shot recv + writev + close (functional correctness).
//!   M3 — registered files (fixed-file table).
//!   M4 — BUF_RING for recv.
//!   M5 — multishot recv.
//!   M6 — TCP_NODELAY, generation tag, dedup helpers, polish.
//!
//! Kernel floor is 5.19 (multishot accept). Multishot recv (6.0) lands in M5.

const std = @import("std");
const builtin = @import("builtin");
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

// user_data tag layout (u64):
//   bits 0..15   slot index (0..MAX_CONNS-1, or 0xffff for orphan close)
//   bits 16..23  op tag      (ACCEPT, RECV, WRITEV, CLOSE)
//   bits 24..31  generation  (M6 — bumped on slot reuse for stale-CQE drop)
//   bits 32..63  reserved
const OP_ACCEPT: u8 = 0;
const OP_RECV: u8 = 1;
const OP_WRITEV: u8 = 2;
const OP_CLOSE: u8 = 3;

// Sentinel slot for ops that aren't tied to a pool entry (e.g. async-close
// of an accept-orphan when the pool is exhausted).
const ORPHAN_SLOT: u16 = 0xffff;

inline fn make_user_data(slot: u16, op: u8, gen: u8) u64 {
    return (@as(u64, gen) << 24) | (@as(u64, op) << 16) | @as(u64, slot);
}

inline fn ud_op(ud: u64) u8 {
    return @intCast((ud >> 16) & 0xff);
}

inline fn ud_slot(ud: u64) u16 {
    return @intCast(ud & 0xffff);
}

const ConnState = enum(u8) { normal, closing };

const Conn = struct {
    fd: i32,
    state: ConnState,
    in_use: bool,
    recv_in_flight: bool,
    write_in_flight: bool,

    read_buf: [READ_BUF_SIZE]u8,
    have: u16,

    head_buf: [HEAD_BUF_SIZE]u8,
    // Two iovecs (head + body). After a partial WRITEV we mutate `base`/`len`
    // in place and bump `write_iov_idx` once a vec drains. `write_iov_idx`
    // == 2 means fully drained.
    write_iovs: [2]posix.iovec_const,
    write_iov_idx: u8,
    write_keep_alive: bool,
};

var conn_pool: [MAX_CONNS]Conn = undefined;
var free_list: [MAX_CONNS]u16 = undefined;
var free_top: u16 = 0;

// Set once by `run()`; read by callbacks. Single process per backend → safe.
var dispatch_fn: http.Handler = undefined;

pub const RunError = error{
    InitFailed,
    SubmitFailed,
    AcceptMultishotLost,
};

pub fn run(listen_fd: i32, dispatch: http.Handler) RunError!noreturn {
    dispatch_fn = dispatch;

    // Plain io_uring init for now; SETUP optimization flags layered later.
    var params: linux.io_uring_params = std.mem.zeroes(linux.io_uring_params);
    var ring = linux.IoUring.init_params(SQ_DEPTH, &params) catch return error.InitFailed;
    defer ring.deinit();

    pool_reset();
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
            for (cqes[0..n]) |cqe| dispatch_cqe(&ring, listen_fd, cqe) catch |err| return err;
        }
    }
}

fn dispatch_cqe(ring: *linux.IoUring, listen_fd: i32, cqe: linux.io_uring_cqe) RunError!void {
    switch (ud_op(cqe.user_data)) {
        OP_ACCEPT => try handle_accept(ring, listen_fd, cqe),
        OP_RECV => try handle_recv(ring, cqe),
        OP_WRITEV => try handle_writev(ring, cqe),
        OP_CLOSE => handle_close(cqe),
        else => {},
    }
}

// ─── connection pool ───────────────────────────────────────────────────────

fn pool_reset() void {
    for (0..MAX_CONNS) |i| {
        conn_pool[i].in_use = false;
        // LIFO: pop returns smaller indices first (cache-friendly).
        free_list[i] = @intCast(MAX_CONNS - 1 - i);
    }
    free_top = MAX_CONNS;
}

fn pool_alloc() ?u16 {
    if (free_top == 0) return null;
    free_top -= 1;
    return free_list[free_top];
}

fn pool_free(idx: u16) void {
    conn_pool[idx].in_use = false;
    free_list[free_top] = idx;
    free_top += 1;
}

// ─── accept ─────────────────────────────────────────────────────────────────

fn arm_accept(ring: *linux.IoUring, listen_fd: i32) !void {
    _ = try ring.accept_multishot(make_user_data(0, OP_ACCEPT, 0), listen_fd, null, null, 0);
}

fn handle_accept(ring: *linux.IoUring, listen_fd: i32, cqe: linux.io_uring_cqe) RunError!void {
    if (cqe.res >= 0) {
        const fd: i32 = cqe.res;
        instrument.inc(&instrument.accepts, 1);

        const slot = pool_alloc() orelse {
            // Pool exhausted — close the orphan asynchronously so haproxy can
            // rebalance. No slot to free; user_data carries ORPHAN_SLOT.
            _ = ring.close(make_user_data(ORPHAN_SLOT, OP_CLOSE, 0), fd) catch {
                // Ring full of SQEs — fall back to sync close so we don't leak fds.
                _ = linux.close(fd);
            };
            return;
        };

        const conn = &conn_pool[slot];
        conn.* = .{
            .fd = fd,
            .state = .normal,
            .in_use = true,
            .recv_in_flight = false,
            .write_in_flight = false,
            .read_buf = undefined,
            .have = 0,
            .head_buf = undefined,
            .write_iovs = .{
                .{ .base = undefined, .len = 0 },
                .{ .base = undefined, .len = 0 },
            },
            .write_iov_idx = 2,
            .write_keep_alive = true,
        };

        submit_recv(ring, slot) catch return error.SubmitFailed;
    }
    // Per-accept error path: multishot stays armed via F_MORE — only re-arm
    // when the kernel says it's done with this multishot SQE.

    if (cqe.flags & linux.IORING_CQE_F_MORE == 0) {
        arm_accept(ring, listen_fd) catch return error.AcceptMultishotLost;
    }
}

// ─── recv ──────────────────────────────────────────────────────────────────

fn submit_recv(ring: *linux.IoUring, slot: u16) !void {
    const conn = &conn_pool[slot];
    if (conn.have >= READ_BUF_SIZE) return; // buffer full; can't accept more
    if (conn.recv_in_flight) return;
    const ud = make_user_data(slot, OP_RECV, 0);
    const buf = conn.read_buf[conn.have..];
    _ = try ring.recv(ud, conn.fd, .{ .buffer = buf }, 0);
    conn.recv_in_flight = true;
}

fn handle_recv(ring: *linux.IoUring, cqe: linux.io_uring_cqe) RunError!void {
    const slot = ud_slot(cqe.user_data);
    if (slot >= MAX_CONNS) return;
    const conn = &conn_pool[slot];
    if (!conn.in_use) return;
    conn.recv_in_flight = false;

    if (cqe.res <= 0) {
        // res < 0: I/O error. res == 0: peer half-closed. Either way, close.
        start_close(ring, slot) catch return error.SubmitFailed;
        return;
    }
    conn.have += @intCast(cqe.res);

    parse_loop(ring, slot) catch return error.SubmitFailed;

    // Re-arm recv unless we're closing or the buffer is full.
    if (conn.state == .normal and !conn.recv_in_flight and conn.have < READ_BUF_SIZE) {
        submit_recv(ring, slot) catch return error.SubmitFailed;
    }
    // Buffer is full but no parseable request — header oversize.
    if (conn.have == READ_BUF_SIZE and !conn.write_in_flight and conn.state == .normal) {
        send_status_close(ring, slot, 413) catch return error.SubmitFailed;
    }
}

// ─── parse + dispatch ──────────────────────────────────────────────────────

fn parse_loop(ring: *linux.IoUring, slot: u16) !void {
    const conn = &conn_pool[slot];
    while (conn.state == .normal and !conn.write_in_flight) {
        const head_end_opt = std.mem.indexOf(u8, conn.read_buf[0..conn.have], "\r\n\r\n");
        const head_end = head_end_opt orelse return;
        const headers_end = head_end + 4;

        const cl = sniff_content_length(conn.read_buf[0..headers_end]);
        const total = headers_end + cl;
        if (total > READ_BUF_SIZE) {
            try send_status_close(ring, slot, 413);
            return;
        }
        if (total > conn.have) return; // need more bytes

        const req = http.parse(conn.read_buf[0..total]) catch |err| {
            instrument.inc(&instrument.req_parse_err, 1);
            try send_status_close(ring, slot, map_parse_status(err));
            return;
        };

        const resp = dispatch_fn(req);

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
    const conn = &conn_pool[slot];
    var iov_buf: [2]posix.iovec_const = undefined;
    var iov_count: usize = 0;
    if (conn.write_iov_idx == 0) {
        iov_buf[0] = conn.write_iovs[0];
        iov_count = 1;
        if (conn.write_iovs[1].len > 0) {
            iov_buf[1] = conn.write_iovs[1];
            iov_count = 2;
        }
    } else if (conn.write_iov_idx == 1) {
        iov_buf[0] = conn.write_iovs[1];
        iov_count = 1;
    } else {
        return; // already drained
    }
    const ud = make_user_data(slot, OP_WRITEV, 0);
    _ = try ring.writev(ud, conn.fd, iov_buf[0..iov_count], 0);
    conn.write_in_flight = true;
}

fn handle_writev(ring: *linux.IoUring, cqe: linux.io_uring_cqe) RunError!void {
    const slot = ud_slot(cqe.user_data);
    if (slot >= MAX_CONNS) return;
    const conn = &conn_pool[slot];
    if (!conn.in_use) return;
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

    // Drained.
    if (!conn.write_keep_alive) {
        start_close(ring, slot) catch return error.SubmitFailed;
        return;
    }
    // Try to dispatch the next pipelined request.
    parse_loop(ring, slot) catch return error.SubmitFailed;
    // Re-arm recv if room and no recv pending.
    if (conn.state == .normal and !conn.recv_in_flight and conn.have < READ_BUF_SIZE) {
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
    const conn = &conn_pool[slot];
    if (conn.state == .closing) return;
    conn.state = .closing;
    if (conn.write_in_flight) {
        // Wait for the pending WRITEV CQE; it'll re-enter start_close from
        // its own error/keep-alive=false path. Don't double-submit close.
        return;
    }
    const ud = make_user_data(slot, OP_CLOSE, 0);
    _ = try ring.close(ud, conn.fd);
}

fn handle_close(cqe: linux.io_uring_cqe) void {
    const slot = ud_slot(cqe.user_data);
    if (slot == ORPHAN_SLOT) return; // pool-exhausted accept-orphan close
    if (slot >= MAX_CONNS) return;
    pool_free(slot);
    instrument.inc(&instrument.conn_closes, 1);
}

// ─── error responses ───────────────────────────────────────────────────────

fn send_status_close(ring: *linux.IoUring, slot: u16, status: u16) !void {
    const conn = &conn_pool[slot];
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
// TODO(M6): dedup with `http_async.zig:410-447` — both backends use the same
// helpers verbatim. Move into `http.zig` once the io_uring path is the proven
// default.

fn sniff_content_length(headers: []const u8) usize {
    var p: usize = std.mem.indexOf(u8, headers, "\r\n").? + 2;
    while (p + 2 < headers.len) {
        const line_end = std.mem.indexOfPos(u8, headers, p, "\r\n").?;
        if (line_end == p) break;
        const colon = std.mem.indexOfScalarPos(u8, headers, p, ':') orelse {
            p = line_end + 2;
            continue;
        };
        if (colon < line_end) {
            const name = headers[p..colon];
            if (name_eql_ci(name, "content-length")) {
                var vstart = colon + 1;
                while (vstart < line_end and (headers[vstart] == ' ' or headers[vstart] == '\t')) vstart += 1;
                return std.fmt.parseInt(usize, headers[vstart..line_end], 10) catch 0;
            }
        }
        p = line_end + 2;
    }
    return 0;
}

fn name_eql_ci(name: []const u8, target_lower: []const u8) bool {
    if (name.len != target_lower.len) return false;
    for (name, target_lower) |a, b| {
        const al = if (a >= 'A' and a <= 'Z') a + 32 else a;
        if (al != b) return false;
    }
    return true;
}

fn map_parse_status(err: http.ParseError) u16 {
    return switch (err) {
        error.UnsupportedMethod, error.UnsupportedVersion, error.Malformed, error.MissingContentLength => 400,
        error.BodyTooLarge, error.HeadersOversize => 413,
    };
}

// ─── tests ─────────────────────────────────────────────────────────────────

test "pool: alloc/free LIFO" {
    pool_reset();
    const a = pool_alloc().?;
    const b = pool_alloc().?;
    try std.testing.expect(a != b);
    pool_free(a);
    pool_free(b);
    const c = pool_alloc().?;
    const d = pool_alloc().?;
    try std.testing.expectEqual(b, c);
    try std.testing.expectEqual(a, d);
}

test "pool: exhaustion returns null" {
    pool_reset();
    var allocated: [MAX_CONNS]u16 = undefined;
    for (0..MAX_CONNS) |i| allocated[i] = pool_alloc().?;
    try std.testing.expectEqual(@as(?u16, null), pool_alloc());
    for (allocated) |idx| pool_free(idx);
    try std.testing.expect(pool_alloc() != null);
}

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

test "sniff_content_length — present" {
    const headers = "POST /x HTTP/1.1\r\nContent-Length: 42\r\nHost: y\r\n\r\n";
    try std.testing.expectEqual(@as(usize, 42), sniff_content_length(headers));
}

test "sniff_content_length — case-insensitive" {
    const headers = "POST /x HTTP/1.1\r\ncontent-length:  17\r\n\r\n";
    try std.testing.expectEqual(@as(usize, 17), sniff_content_length(headers));
}

test "sniff_content_length — absent" {
    const headers = "GET /x HTTP/1.1\r\nHost: y\r\n\r\n";
    try std.testing.expectEqual(@as(usize, 0), sniff_content_length(headers));
}

test "map_parse_status — 400 vs 413" {
    try std.testing.expectEqual(@as(u16, 400), map_parse_status(error.UnsupportedMethod));
    try std.testing.expectEqual(@as(u16, 413), map_parse_status(error.BodyTooLarge));
}
