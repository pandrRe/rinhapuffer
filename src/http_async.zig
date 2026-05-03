//! Linux-only async HTTP/1.1 server: epoll loop, edge-triggered, per-connection
//! buffers. Designed for HAProxy `mode tcp` + `option splice-auto`: each backend
//! conn is a long-lived persistent HTTP/1.1 conversation, multiplexed across
//! ~256 conns on a single thread.
//!
//! Same handler model as `http.zig::serve()`: dispatch is sync and CPU-bound.
//! We do not yield while it runs — with sub-ms handlers and one process per
//! backend that's a non-issue.

const std = @import("std");
const builtin = @import("builtin");
const http = @import("http.zig");
const instrument = @import("instrument.zig");

comptime {
    if (builtin.os.tag != .linux) {
        @compileError("http_async is Linux-only — main.zig must comptime-branch");
    }
}

const linux = std.os.linux;
const posix = std.posix;

pub const MAX_CONNS: u32 = 256;
pub const READ_BUF_SIZE: usize = 8192;
pub const HEAD_BUF_SIZE: usize = 256;
const EVENTS_PER_WAIT: usize = 64;

// Reserved data.u64 value to identify the listen fd in epoll events.
// All real conn slots use 0..MAX_CONNS-1, so any value outside that range
// works; pick the max to make accidental collisions impossible.
const LISTEN_TAG: u64 = std.math.maxInt(u64);

const ConnState = enum(u8) { idle, writing };

const Conn = struct {
    fd: i32,
    state: ConnState,
    in_use: bool,

    read_buf: [READ_BUF_SIZE]u8,
    have: usize, // bytes in read_buf[0..have]

    // Backing buffer for the slow `format_head` path. The hot path
    // (200/JSON/len ∈ {35,36}) skips formatting entirely and `write_head`
    // just aliases a static `http.HEAD_200_JSON_*` constant.
    head_buf: [HEAD_BUF_SIZE]u8,
    write_head: []const u8,

    // Pending write description. write_off counts bytes written across the
    // head+body pair. write_body aliases a static response string (no copy).
    write_body: []const u8,
    write_off: usize,
    write_keep_alive: bool,
};

var conn_pool: [MAX_CONNS]Conn = undefined;
var free_list: [MAX_CONNS]u16 = undefined;
var free_top: u16 = 0;

// Set once by `run()`; read by callbacks. Single process per backend → safe.
var dispatch_fn: http.Handler = undefined;

pub const RunError = error{
    EpollCreateFailed,
    EpollCtlFailed,
    EpollWaitFailed,
    SetNonblockFailed,
};

pub fn run(listen_fd: i32, dispatch: http.Handler) RunError!noreturn {
    dispatch_fn = dispatch;
    try set_nonblocking(listen_fd);

    const ep_rc = linux.epoll_create1(linux.EPOLL.CLOEXEC);
    if (@as(isize, @bitCast(ep_rc)) < 0) return error.EpollCreateFailed;
    const epfd: i32 = @intCast(ep_rc);
    defer _ = linux.close(epfd);

    var listen_ev: linux.epoll_event = .{
        .events = linux.EPOLL.IN | linux.EPOLL.ET,
        .data = .{ .u64 = LISTEN_TAG },
    };
    if (@as(isize, @bitCast(linux.epoll_ctl(epfd, linux.EPOLL.CTL_ADD, listen_fd, &listen_ev))) != 0) {
        return error.EpollCtlFailed;
    }

    pool_reset();
    instrument.init();

    var events: [EVENTS_PER_WAIT]linux.epoll_event = undefined;
    while (true) {
        const wait_rc = linux.epoll_wait(epfd, &events, EVENTS_PER_WAIT, -1);
        const wait_signed: isize = @bitCast(wait_rc);
        if (wait_signed < 0) {
            const errno_raw: u32 = @intCast(-wait_signed);
            if (errno_raw == @intFromEnum(linux.E.INTR)) continue;
            return error.EpollWaitFailed;
        }
        const n: usize = @intCast(wait_signed);
        instrument.inc(&instrument.epoll_wakeups, 1);
        instrument.inc(&instrument.epoll_events, @intCast(n));

        for (events[0..n]) |ev| {
            if (ev.data.u64 == LISTEN_TAG) {
                accept_loop(epfd, listen_fd);
                continue;
            }
            const idx: u16 = @intCast(ev.data.u64);
            const conn = &conn_pool[idx];
            if (!conn.in_use) continue;

            const flags = ev.events;
            // ERR/HUP/RDHUP after we've still got data to read is OK — we'll
            // notice EOF on the next read(). But if there's no IN bit set
            // either, the conn is genuinely dead.
            if (flags & (linux.EPOLL.ERR | linux.EPOLL.HUP) != 0) {
                close_conn(epfd, idx);
                continue;
            }
            if (flags & linux.EPOLL.IN != 0 and conn.state == .idle) {
                try_read_and_dispatch(epfd, idx);
                if (!conn.in_use) continue;
            }
            if (flags & linux.EPOLL.OUT != 0 and conn.state == .writing) {
                try_drain_write(epfd, idx);
            }
        }
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

// ─── socket plumbing ───────────────────────────────────────────────────────

fn set_nonblocking(fd: i32) RunError!void {
    const get_rc = linux.fcntl(fd, linux.F.GETFL, 0);
    const get_signed: isize = @bitCast(get_rc);
    if (get_signed < 0) return error.SetNonblockFailed;
    const flags: u32 = @intCast(get_signed);
    const set_rc = linux.fcntl(fd, linux.F.SETFL, flags | @as(u32, posix.SOCK.NONBLOCK));
    if (@as(isize, @bitCast(set_rc)) < 0) return error.SetNonblockFailed;
}

fn accept_loop(epfd: i32, listen_fd: i32) void {
    while (true) {
        const fd_rc = linux.accept4(listen_fd, null, null, posix.SOCK.NONBLOCK | posix.SOCK.CLOEXEC);
        const fd_signed: isize = @bitCast(fd_rc);
        if (fd_signed < 0) return; // EAGAIN drained; any other error → also stop
        const fd: i32 = @intCast(fd_signed);

        const idx = pool_alloc() orelse {
            // No slots — close the fresh fd so HAProxy can rebalance.
            _ = linux.close(fd);
            return;
        };
        const conn = &conn_pool[idx];
        conn.* = .{
            .fd = fd,
            .state = .idle,
            .in_use = true,
            .read_buf = undefined,
            .have = 0,
            .head_buf = undefined,
            .write_head = "",
            .write_body = "",
            .write_off = 0,
            .write_keep_alive = true,
        };
        instrument.inc(&instrument.accepts, 1);

        // Always-armed: register IN | OUT | ET once. ET only delivers
        // events on transitions, so an idle conn with nothing to write
        // never fires OUT spuriously. Removes the per-state-transition
        // CTL_MOD that the previous code did when entering/leaving the
        // .writing state (~2 syscalls per request saved).
        var ev: linux.epoll_event = .{
            .events = linux.EPOLL.IN | linux.EPOLL.OUT | linux.EPOLL.ET,
            .data = .{ .u64 = idx },
        };
        if (@as(isize, @bitCast(linux.epoll_ctl(epfd, linux.EPOLL.CTL_ADD, fd, &ev))) != 0) {
            _ = linux.close(fd);
            pool_free(idx);
            continue;
        }
    }
}

fn close_conn(epfd: i32, idx: u16) void {
    const conn = &conn_pool[idx];
    if (!conn.in_use) return;
    _ = linux.epoll_ctl(epfd, linux.EPOLL.CTL_DEL, conn.fd, null);
    _ = linux.close(conn.fd);
    pool_free(idx);
    instrument.inc(&instrument.conn_closes, 1);
}

// Always-armed EPOLLIN|EPOLLOUT|ET (set once at accept). State transitions
// don't issue CTL_MOD anymore — the kernel only re-fires events on edge,
// and the conn state machine ignores any flag it doesn't care about right
// now. `rearm` kept as a no-op for callsite clarity and easy revert.
inline fn rearm(epfd: i32, fd: i32, idx: u16, want_out: bool) void {
    _ = epfd;
    _ = fd;
    _ = idx;
    _ = want_out;
}

// ─── read + dispatch ────────────────────────────────────────────────────────

fn try_read_and_dispatch(epfd: i32, idx: u16) void {
    const conn = &conn_pool[idx];

    // Outer loop: drain reads (edge-triggered → must read until EAGAIN).
    while (true) {
        // Inner loop: process all complete requests already buffered.
        while (true) {
            const head_end_opt = std.mem.indexOf(u8, conn.read_buf[0..conn.have], "\r\n\r\n");
            const head_end = head_end_opt orelse break;
            const headers_end = head_end + 4;

            const cl = sniff_content_length(conn.read_buf[0..headers_end]);
            const total = headers_end + cl;
            if (total > READ_BUF_SIZE) {
                send_status_close(epfd, idx, 413);
                return;
            }
            if (total > conn.have) break; // wait for more bytes

            const t_total = instrument.now_ns();
            const t_parse = instrument.now_ns();
            const req = http.parse(conn.read_buf[0..total]) catch |err| {
                instrument.inc(&instrument.req_parse_err, 1);
                send_status_close(epfd, idx, map_parse_status(err));
                return;
            };
            instrument.observe_since(&instrument.hist_parse, t_parse);

            const resp = dispatch_fn(req);

            // Hot path: fraud-score 200/json responses use a precomputed
            // head constant; everything else goes through `format_head`'s
            // bufPrint-based slow path.
            if (http.fast_head(resp)) |fh| {
                conn.write_head = fh;
                instrument.inc(&instrument.head_fast, 1);
            } else {
                const head = http.format_head(&conn.head_buf, resp) catch {
                    close_conn(epfd, idx);
                    return;
                };
                conn.write_head = head;
                instrument.inc(&instrument.head_slow, 1);
            }
            conn.write_body = resp.body;
            conn.write_off = 0;
            conn.write_keep_alive = resp.keep_alive;

            const t_write = instrument.now_ns();
            if (!try_drain_write_inner(conn)) {
                // Partial write → wait for OUT. Advance read buffer first so
                // any subsequent buffered request waits behind this write.
                advance_read(conn, total);
                conn.state = .writing;
                rearm(epfd, conn.fd, idx, true);
                instrument.inc(&instrument.partial_writes, 1);
                return;
            }
            instrument.observe_since(&instrument.hist_write, t_write);
            instrument.observe_since(&instrument.hist_total, t_total);
            if (!resp.keep_alive) {
                close_conn(epfd, idx);
                return;
            }
            advance_read(conn, total);
            // Loop to next buffered request.
        }

        // No complete request buffered; if buffer is full it can never become
        // complete (header oversize) — bail.
        if (conn.have == READ_BUF_SIZE) {
            send_status_close(epfd, idx, 413);
            return;
        }

        // Read more.
        const rc = linux.read(conn.fd, conn.read_buf[conn.have..].ptr, READ_BUF_SIZE - conn.have);
        const signed: isize = @bitCast(rc);
        if (signed < 0) {
            const errno_raw: u32 = @intCast(-signed);
            if (errno_raw == @intFromEnum(linux.E.AGAIN)) {
                instrument.inc(&instrument.read_eagain, 1);
                return; // drained — wait for next IN edge
            }
            close_conn(epfd, idx);
            return;
        }
        if (signed == 0) {
            // peer closed cleanly
            close_conn(epfd, idx);
            return;
        }
        conn.have += @intCast(signed);
        // Loop back to try parsing.
    }
}

fn advance_read(conn: *Conn, consumed: usize) void {
    if (conn.have > consumed) {
        std.mem.copyForwards(u8, conn.read_buf[0 .. conn.have - consumed], conn.read_buf[consumed..conn.have]);
    }
    conn.have -= consumed;
}

// ─── write drain ────────────────────────────────────────────────────────────

// Returns true if all pending bytes were written. Returns false on EAGAIN
// (caller should leave conn in .writing state and wait for OUT) OR on real
// I/O error (caller should close — distinguishable: writev_error_close is
// signaled by setting conn.in_use = false via close_conn from caller).
//
// Internal-only: this function does NOT close the conn on error. It just
// reports "couldn't drain". Caller decides.
fn try_drain_write_inner(conn: *Conn) bool {
    while (true) {
        const head = conn.write_head;
        const total_len = head.len + conn.write_body.len;
        if (conn.write_off >= total_len) return true;

        var iovs: [2]posix.iovec_const = undefined;
        var iov_count: usize = 0;
        if (conn.write_off < head.len) {
            iovs[0] = .{ .base = head.ptr + conn.write_off, .len = head.len - conn.write_off };
            iov_count = 1;
            if (conn.write_body.len > 0) {
                iovs[1] = .{ .base = conn.write_body.ptr, .len = conn.write_body.len };
                iov_count = 2;
            }
        } else {
            const off_in_body = conn.write_off - head.len;
            iovs[0] = .{ .base = conn.write_body.ptr + off_in_body, .len = conn.write_body.len - off_in_body };
            iov_count = 1;
        }

        const rc = linux.writev(conn.fd, &iovs, iov_count);
        const signed: isize = @bitCast(rc);
        if (signed < 0) {
            const errno_raw: u32 = @intCast(-signed);
            if (errno_raw == @intFromEnum(linux.E.AGAIN)) {
                instrument.inc(&instrument.write_eagain, 1);
                return false;
            }
            // Real error — we report not-drained; caller's outer logic will
            // decide. We mark write_keep_alive=false so caller closes after.
            conn.write_keep_alive = false;
            return false;
        }
        if (signed == 0) {
            conn.write_keep_alive = false;
            return false;
        }
        conn.write_off += @intCast(signed);
    }
}

fn try_drain_write(epfd: i32, idx: u16) void {
    const conn = &conn_pool[idx];
    const drained = try_drain_write_inner(conn);
    if (!drained) {
        if (!conn.write_keep_alive) {
            // Real I/O error path — close. (Distinguishing from EAGAIN by the
            // fact that write_keep_alive is false: see try_drain_write_inner.)
            close_conn(epfd, idx);
        }
        return;
    }
    if (!conn.write_keep_alive) {
        close_conn(epfd, idx);
        return;
    }
    conn.state = .idle;
    rearm(epfd, conn.fd, idx, false);
    // We may already have buffered the next request — process it.
    try_read_and_dispatch(epfd, idx);
}

// ─── parse helpers ──────────────────────────────────────────────────────────

fn sniff_content_length(headers: []const u8) usize {
    // headers includes the trailing "\r\n\r\n".
    var p: usize = std.mem.indexOf(u8, headers, "\r\n").? + 2;
    while (p + 2 < headers.len) {
        const line_end = std.mem.indexOfPos(u8, headers, p, "\r\n").?;
        if (line_end == p) break; // empty line = end of headers
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

fn send_status_close(epfd: i32, idx: u16, status: u16) void {
    const conn = &conn_pool[idx];
    const resp: http.Response = .{ .status = status, .body = "", .content_type = "text/plain", .keep_alive = false };
    const head = http.format_head(&conn.head_buf, resp) catch {
        close_conn(epfd, idx);
        return;
    };
    conn.write_head = head;
    conn.write_body = "";
    conn.write_off = 0;
    conn.write_keep_alive = false;
    _ = try_drain_write_inner(conn);
    close_conn(epfd, idx);
}

// ─── tests ─────────────────────────────────────────────────────────────────

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

test "name_eql_ci" {
    try std.testing.expect(name_eql_ci("Content-Length", "content-length"));
    try std.testing.expect(name_eql_ci("CONTENT-LENGTH", "content-length"));
    try std.testing.expect(!name_eql_ci("content-typex", "content-length"));
    try std.testing.expect(!name_eql_ci("content-typ", "content-length"));
}

test "pool: alloc/free LIFO" {
    pool_reset();
    const a = pool_alloc().?;
    const b = pool_alloc().?;
    try std.testing.expect(a != b);
    pool_free(a);
    pool_free(b);
    const c = pool_alloc().?;
    const d = pool_alloc().?;
    // LIFO: last freed (b) returned first.
    try std.testing.expectEqual(b, c);
    try std.testing.expectEqual(a, d);
}

test "pool: exhaustion returns null" {
    pool_reset();
    var allocated: [MAX_CONNS]u16 = undefined;
    for (0..MAX_CONNS) |i| {
        allocated[i] = pool_alloc().?;
    }
    try std.testing.expectEqual(@as(?u16, null), pool_alloc());
    for (allocated) |idx| pool_free(idx);
    try std.testing.expect(pool_alloc() != null);
}

test "map_parse_status — 400 vs 413" {
    try std.testing.expectEqual(@as(u16, 400), map_parse_status(error.UnsupportedMethod));
    try std.testing.expectEqual(@as(u16, 400), map_parse_status(error.MissingContentLength));
    try std.testing.expectEqual(@as(u16, 413), map_parse_status(error.BodyTooLarge));
    try std.testing.expectEqual(@as(u16, 413), map_parse_status(error.HeadersOversize));
}
