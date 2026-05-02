//! Hand-rolled HTTP/1.1 server: blocking accept loop, libc syscalls direct,
//! every buffer allocated at program load (BSS) — zero per-request, per-
//! connection, or per-accept allocation.
//!
//! Two routes only (`GET /ready`, `POST /fraud-score`); anything else gets
//! 400 or 404. Method/version/header parsing is schema-rigid: we reject
//! permissively rather than degrade to a permissive HTTP parser.
//!
//! Single-threaded server → at most one in-flight connection → the static
//! `read_buf` and `head_buf` are safe to share across the program lifetime.
//! Phase 9 would have to revisit this if we ever introduce worker threads.

const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;
const libc = std.c;

pub const REQ_BUF_SIZE: usize = 8192;
pub const HEAD_BUF_SIZE: usize = 256;
pub const LISTEN_BACKLOG: u32 = 128;

// Module-level static buffers. Allocated once at program load; never grow.
// Single-threaded accept loop guarantees at most one in-flight connection,
// so sharing across requests is safe.
var read_buf: [REQ_BUF_SIZE]u8 = undefined;
var head_buf: [HEAD_BUF_SIZE]u8 = undefined;

pub const Method = enum { get, post };

pub const Request = struct {
    method: Method,
    path: []const u8,
    body: []const u8,
    keep_alive: bool,
};

pub const Response = struct {
    status: u16,
    body: []const u8,
    content_type: []const u8 = "application/json",
    keep_alive: bool = true,
};

pub const Handler = *const fn (req: Request) Response;

pub const ParseError = error{
    Malformed,
    UnsupportedMethod,
    UnsupportedVersion,
    MissingContentLength,
    BodyTooLarge,
    HeadersOversize,
};

pub const ListenError = error{
    SocketFailed,
    SetsockoptFailed,
    BindFailed,
    ListenFailed,
};

pub const AcceptError = error{ AcceptFailed };

// ─── parser ────────────────────────────────────────────────────────────────

inline fn ascii_lower(b: u8) u8 {
    return if (b >= 'A' and b <= 'Z') b + 32 else b;
}

fn header_name_eql(name: []const u8, target_lower: []const u8) bool {
    if (name.len != target_lower.len) return false;
    for (name, target_lower) |a, b| {
        if (ascii_lower(a) != b) return false;
    }
    return true;
}

fn header_value_eql_ci(value: []const u8, target_lower: []const u8) bool {
    if (value.len != target_lower.len) return false;
    for (value, target_lower) |a, b| {
        if (ascii_lower(a) != b) return false;
    }
    return true;
}

/// Parse a fully-buffered HTTP/1.1 request. All slices in the returned
/// `Request` alias `bytes`; caller must keep `bytes` alive while using the
/// request.
pub fn parse(bytes: []const u8) ParseError!Request {
    const first_crlf = std.mem.indexOf(u8, bytes, "\r\n") orelse return error.HeadersOversize;
    const request_line = bytes[0..first_crlf];

    const sp1 = std.mem.indexOfScalar(u8, request_line, ' ') orelse return error.Malformed;
    const method_str = request_line[0..sp1];
    const method: Method = blk: {
        if (std.mem.eql(u8, method_str, "GET")) break :blk .get;
        if (std.mem.eql(u8, method_str, "POST")) break :blk .post;
        return error.UnsupportedMethod;
    };

    const after_method = request_line[sp1 + 1 ..];
    const sp2 = std.mem.indexOfScalar(u8, after_method, ' ') orelse return error.Malformed;
    const path = after_method[0..sp2];
    const version = after_method[sp2 + 1 ..];
    if (!std.mem.eql(u8, version, "HTTP/1.1")) return error.UnsupportedVersion;

    var p: usize = first_crlf + 2;
    var content_length: ?usize = null;
    var keep_alive = true;
    var headers_end: usize = 0;
    while (p <= bytes.len) {
        const line_end = std.mem.indexOfPos(u8, bytes, p, "\r\n") orelse return error.HeadersOversize;
        if (line_end == p) {
            headers_end = line_end + 2;
            break;
        }
        const colon = std.mem.indexOfScalarPos(u8, bytes, p, ':') orelse return error.Malformed;
        if (colon >= line_end) return error.Malformed;
        const name = bytes[p..colon];
        var vstart: usize = colon + 1;
        while (vstart < line_end and (bytes[vstart] == ' ' or bytes[vstart] == '\t')) vstart += 1;
        const value = bytes[vstart..line_end];

        if (header_name_eql(name, "content-length")) {
            content_length = std.fmt.parseInt(usize, value, 10) catch return error.Malformed;
        } else if (header_name_eql(name, "connection")) {
            if (header_value_eql_ci(value, "close")) keep_alive = false;
        }
        p = line_end + 2;
    }
    if (headers_end == 0) return error.HeadersOversize;

    var body: []const u8 = "";
    if (method == .post) {
        const len = content_length orelse return error.MissingContentLength;
        if (headers_end + len > bytes.len) return error.BodyTooLarge;
        body = bytes[headers_end .. headers_end + len];
    }

    return .{ .method = method, .path = path, .body = body, .keep_alive = keep_alive };
}

// ─── response formatting ───────────────────────────────────────────────────

fn reason(status: u16) []const u8 {
    return switch (status) {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        else => "Unknown",
    };
}

/// Format the response head into `buf`. Returns the slice that was written.
/// Pure byte function — no I/O — for unit testability.
pub fn format_head(buf: []u8, resp: Response) error{NoSpaceLeft}![]u8 {
    const r = reason(resp.status);
    if (resp.body.len == 0) {
        if (resp.keep_alive) {
            return std.fmt.bufPrint(
                buf,
                "HTTP/1.1 {d} {s}\r\nContent-Length: 0\r\n\r\n",
                .{ resp.status, r },
            );
        }
        return std.fmt.bufPrint(
            buf,
            "HTTP/1.1 {d} {s}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            .{ resp.status, r },
        );
    }
    if (resp.keep_alive) {
        return std.fmt.bufPrint(
            buf,
            "HTTP/1.1 {d} {s}\r\nContent-Length: {d}\r\nContent-Type: {s}\r\n\r\n",
            .{ resp.status, r, resp.body.len, resp.content_type },
        );
    }
    return std.fmt.bufPrint(
        buf,
        "HTTP/1.1 {d} {s}\r\nContent-Length: {d}\r\nContent-Type: {s}\r\nConnection: close\r\n\r\n",
        .{ resp.status, r, resp.body.len, resp.content_type },
    );
}

// ─── socket helpers ────────────────────────────────────────────────────────

pub const IoError = error{IoError};

/// Open + bind + listen on `port`. SO_REUSEADDR set; SO_NOSIGPIPE set on
/// macOS (best-effort). Returns the listening fd. On port 0, kernel picks
/// the port — recover it with `local_port`.
pub fn bind_listen(port: u16) ListenError!libc.fd_t {
    const fd = libc.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (fd < 0) return error.SocketFailed;
    errdefer _ = libc.close(fd);

    const reuse: c_int = 1;
    if (libc.setsockopt(
        fd,
        posix.SOL.SOCKET,
        posix.SO.REUSEADDR,
        @ptrCast(&reuse),
        @sizeOf(c_int),
    ) != 0) return error.SetsockoptFailed;

    if (builtin.os.tag.isDarwin()) {
        const nosigpipe: c_int = 1;
        _ = libc.setsockopt(
            fd,
            posix.SOL.SOCKET,
            posix.SO.NOSIGPIPE,
            @ptrCast(&nosigpipe),
            @sizeOf(c_int),
        );
    }

    var sin: libc.sockaddr.in = .{
        .port = std.mem.nativeToBig(u16, port),
        .addr = 0, // INADDR_ANY
    };
    if (libc.bind(fd, @ptrCast(&sin), @sizeOf(@TypeOf(sin))) != 0) return error.BindFailed;
    if (libc.listen(fd, LISTEN_BACKLOG) != 0) return error.ListenFailed;
    return fd;
}

/// Read back the kernel-assigned port from a bound listening fd. Useful
/// when `bind_listen(0)` lets the kernel pick.
pub fn local_port(fd: libc.fd_t) error{GetsocknameFailed}!u16 {
    var sin: libc.sockaddr.in = undefined;
    var len: libc.socklen_t = @sizeOf(@TypeOf(sin));
    if (libc.getsockname(fd, @ptrCast(&sin), &len) != 0) return error.GetsocknameFailed;
    return std.mem.bigToNative(u16, sin.port);
}

/// Loop `accept` retrying on EINTR. On real failure returns AcceptFailed.
pub fn accept_one(listen_fd: libc.fd_t) AcceptError!libc.fd_t {
    while (true) {
        const fd = libc.accept(listen_fd, null, null);
        if (fd >= 0) return fd;
        const e = libc.errno(fd);
        if (e == .INTR) continue;
        return error.AcceptFailed;
    }
}

fn write_all(fd: libc.fd_t, bytes: []const u8) IoError!void {
    var off: usize = 0;
    while (off < bytes.len) {
        const rc = libc.write(fd, bytes.ptr + off, bytes.len - off);
        if (rc < 0) {
            const e = libc.errno(rc);
            if (e == .INTR) continue;
            return error.IoError;
        }
        if (rc == 0) return error.IoError;
        off += @intCast(rc);
    }
}

fn writev_all(fd: libc.fd_t, head: []const u8, body: []const u8) IoError!void {
    // Two-iov path (head + body). Loop on partial writev by advancing the
    // current iov; when the head iov is fully drained, continue with a
    // single-iov body via write_all.
    var head_off: usize = 0;
    while (head_off < head.len) {
        var iovs: [2]posix.iovec_const = .{
            .{ .base = head.ptr + head_off, .len = head.len - head_off },
            .{ .base = body.ptr, .len = body.len },
        };
        const rc = libc.writev(fd, &iovs, 2);
        if (rc < 0) {
            const e = libc.errno(rc);
            if (e == .INTR) continue;
            return error.IoError;
        }
        if (rc == 0) return error.IoError;
        var n: usize = @intCast(rc);
        if (n >= head.len - head_off) {
            n -= head.len - head_off;
            head_off = head.len;
            // Tail of body remaining.
            if (n < body.len) return write_all(fd, body[n..]);
            return;
        }
        head_off += n;
    }
}

pub fn write_response(fd: libc.fd_t, resp: Response) IoError!void {
    const head = format_head(&head_buf, resp) catch return error.IoError;
    if (resp.body.len == 0) return write_all(fd, head);
    return writev_all(fd, head, resp.body);
}

const ReadOutcome = union(enum) {
    ok: usize, // total bytes consumed (head + body) from start of read_buf
    closed,
    err,
};

fn find_double_crlf(haystack: []const u8) ?usize {
    return std.mem.indexOf(u8, haystack, "\r\n\r\n");
}

/// Block reading into `read_buf[have..]` until a complete request (head +
/// body) is buffered. Returns the byte-count consumed by the request.
fn read_until_request(client_fd: libc.fd_t, have_in_out: *usize) ReadOutcome {
    var have = have_in_out.*;
    var head_end: ?usize = find_double_crlf(read_buf[0..have]);

    while (head_end == null) {
        if (have == REQ_BUF_SIZE) return .err; // headers wouldn't fit
        const rc = libc.read(client_fd, read_buf[have..].ptr, REQ_BUF_SIZE - have);
        if (rc < 0) {
            const e = libc.errno(rc);
            if (e == .INTR) continue;
            return .err;
        }
        if (rc == 0) {
            have_in_out.* = have;
            return if (have == 0) .closed else .err;
        }
        have += @intCast(rc);
        head_end = find_double_crlf(read_buf[0..have]);
    }
    const headers_end = head_end.? + 4;

    // Sniff Content-Length quickly so we know how much body to wait for.
    // (Re-parsing later is cheap; doing it here lets us reject overflow
    // before reading further bytes.)
    var content_length: usize = 0;
    var p: usize = std.mem.indexOf(u8, read_buf[0..headers_end], "\r\n").? + 2;
    while (p < headers_end - 2) {
        const line_end = std.mem.indexOfPos(u8, read_buf[0..headers_end], p, "\r\n").?;
        const colon = std.mem.indexOfScalarPos(u8, read_buf[0..headers_end], p, ':') orelse {
            p = line_end + 2;
            continue;
        };
        if (colon < line_end) {
            const name = read_buf[p..colon];
            if (header_name_eql(name, "content-length")) {
                var vstart = colon + 1;
                while (vstart < line_end and (read_buf[vstart] == ' ' or read_buf[vstart] == '\t')) vstart += 1;
                content_length = std.fmt.parseInt(usize, read_buf[vstart..line_end], 10) catch 0;
                break;
            }
        }
        p = line_end + 2;
    }

    const total = headers_end + content_length;
    if (total > REQ_BUF_SIZE) {
        have_in_out.* = have;
        return .err;
    }

    while (have < total) {
        const rc = libc.read(client_fd, read_buf[have..].ptr, REQ_BUF_SIZE - have);
        if (rc < 0) {
            const e = libc.errno(rc);
            if (e == .INTR) continue;
            return .err;
        }
        if (rc == 0) {
            have_in_out.* = have;
            return .err;
        }
        have += @intCast(rc);
    }
    have_in_out.* = have;
    return .{ .ok = total };
}

fn map_parse_status(err: ParseError) u16 {
    return switch (err) {
        error.UnsupportedMethod, error.UnsupportedVersion, error.Malformed, error.MissingContentLength => 400,
        error.BodyTooLarge, error.HeadersOversize => 413,
    };
}

/// Per-connection serve loop. Reads requests, dispatches via `handler`,
/// writes responses. Exits on peer close, parse error, write error, or
/// `Connection: close`. Caller owns `client_fd` (does not close).
pub fn serve(client_fd: libc.fd_t, handler: Handler) void {
    var have: usize = 0;
    while (true) {
        const outcome = read_until_request(client_fd, &have);
        switch (outcome) {
            .closed => return,
            .err => return,
            .ok => |consumed| {
                const req = parse(read_buf[0..consumed]) catch |err| {
                    const status = map_parse_status(err);
                    write_response(client_fd, .{ .status = status, .body = "", .content_type = "text/plain", .keep_alive = false }) catch {};
                    return;
                };
                const resp = handler(req);
                write_response(client_fd, resp) catch return;
                if (!resp.keep_alive) return;

                if (have > consumed) {
                    std.mem.copyForwards(u8, read_buf[0 .. have - consumed], read_buf[consumed..have]);
                }
                have -= consumed;
            },
        }
    }
}

/// Test-only: accept exactly one connection on `listen_fd` and serve it.
/// Returns when the peer closes or a parse/write error occurs.
pub fn serve_one(listen_fd: libc.fd_t, handler: Handler) void {
    const client_fd = accept_one(listen_fd) catch return;
    defer _ = libc.close(client_fd);
    serve(client_fd, handler);
}

// ─── tests ─────────────────────────────────────────────────────────────────

test "parse — well-formed GET" {
    const req = try parse("GET /ready HTTP/1.1\r\nHost: x\r\n\r\n");
    try std.testing.expectEqual(Method.get, req.method);
    try std.testing.expectEqualStrings("/ready", req.path);
    try std.testing.expectEqualStrings("", req.body);
    try std.testing.expect(req.keep_alive);
}

test "parse — well-formed POST with body" {
    const req = try parse("POST /fraud-score HTTP/1.1\r\nContent-Length: 4\r\n\r\nbody");
    try std.testing.expectEqual(Method.post, req.method);
    try std.testing.expectEqualStrings("/fraud-score", req.path);
    try std.testing.expectEqualStrings("body", req.body);
    try std.testing.expect(req.keep_alive);
}

test "parse — Connection: close" {
    const req = try parse("GET /ready HTTP/1.1\r\nConnection: close\r\n\r\n");
    try std.testing.expect(!req.keep_alive);
}

test "parse — Connection: Close (mixed case)" {
    const req = try parse("GET /ready HTTP/1.1\r\nConnection: Close\r\n\r\n");
    try std.testing.expect(!req.keep_alive);
}

test "parse — case-insensitive header name" {
    const req = try parse("POST /x HTTP/1.1\r\ncontent-length: 0\r\nCONNECTION: close\r\n\r\n");
    try std.testing.expectEqualStrings("", req.body);
    try std.testing.expect(!req.keep_alive);
}

test "parse — PUT rejected" {
    try std.testing.expectError(error.UnsupportedMethod, parse("PUT /ready HTTP/1.1\r\n\r\n"));
}

test "parse — HTTP/2.0 rejected" {
    try std.testing.expectError(error.UnsupportedVersion, parse("GET /ready HTTP/2.0\r\n\r\n"));
}

test "parse — POST without Content-Length" {
    try std.testing.expectError(error.MissingContentLength, parse("POST /fraud-score HTTP/1.1\r\nHost: x\r\n\r\n"));
}

test "parse — header section without terminator" {
    var buf: [REQ_BUF_SIZE]u8 = undefined;
    const start = "GET /ready HTTP/1.1\r\nX: ";
    @memcpy(buf[0..start.len], start);
    @memset(buf[start.len..], 'a');
    try std.testing.expectError(error.HeadersOversize, parse(buf[0..]));
}

test "parse — Content-Length larger than buffer" {
    try std.testing.expectError(error.BodyTooLarge, parse("POST /x HTTP/1.1\r\nContent-Length: 99999\r\n\r\nshort"));
}

test "format_head — 200 with body keep-alive" {
    var buf: [HEAD_BUF_SIZE]u8 = undefined;
    const head = try format_head(&buf, .{ .status = 200, .body = "abcde", .keep_alive = true });
    try std.testing.expectEqualStrings(
        "HTTP/1.1 200 OK\r\nContent-Length: 5\r\nContent-Type: application/json\r\n\r\n",
        head,
    );
}

test "format_head — 200 empty body keep-alive" {
    var buf: [HEAD_BUF_SIZE]u8 = undefined;
    const head = try format_head(&buf, .{ .status = 200, .body = "", .keep_alive = true });
    try std.testing.expectEqualStrings(
        "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n",
        head,
    );
}

test "format_head — 400 close" {
    var buf: [HEAD_BUF_SIZE]u8 = undefined;
    const head = try format_head(&buf, .{ .status = 400, .body = "", .keep_alive = false });
    try std.testing.expectEqualStrings(
        "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        head,
    );
}

test "format_head — 200 body close" {
    var buf: [HEAD_BUF_SIZE]u8 = undefined;
    const head = try format_head(&buf, .{ .status = 200, .body = "x", .content_type = "text/plain", .keep_alive = false });
    try std.testing.expectEqualStrings(
        "HTTP/1.1 200 OK\r\nContent-Length: 1\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n",
        head,
    );
}

// ─── integration ───────────────────────────────────────────────────────────

const FRAUD_STUB: []const u8 = "{\"approved\":true,\"fraud_score\":0.0}";

fn test_dispatch(req: Request) Response {
    if (req.method == .get and std.mem.eql(u8, req.path, "/ready")) {
        return .{ .status = 200, .body = "", .content_type = "text/plain", .keep_alive = req.keep_alive };
    }
    if (req.method == .post and std.mem.eql(u8, req.path, "/fraud-score")) {
        return .{ .status = 200, .body = FRAUD_STUB, .keep_alive = req.keep_alive };
    }
    return .{ .status = 404, .body = "", .content_type = "text/plain", .keep_alive = req.keep_alive };
}

fn test_client_connect(port: u16) !libc.fd_t {
    const fd = libc.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (fd < 0) return error.SocketFailed;
    var sin: libc.sockaddr.in = .{
        .port = std.mem.nativeToBig(u16, port),
        .addr = std.mem.nativeToBig(u32, 0x7f000001), // 127.0.0.1
    };
    if (libc.connect(fd, @ptrCast(&sin), @sizeOf(@TypeOf(sin))) != 0) {
        _ = libc.close(fd);
        return error.ConnectFailed;
    }
    return fd;
}

fn test_send_all(fd: libc.fd_t, data: []const u8) !void {
    var off: usize = 0;
    while (off < data.len) {
        const rc = libc.write(fd, data.ptr + off, data.len - off);
        if (rc <= 0) return error.WriteFailed;
        off += @intCast(rc);
    }
}

fn test_recv_until_close(fd: libc.fd_t, buf: []u8) !usize {
    var have: usize = 0;
    while (have < buf.len) {
        const rc = libc.read(fd, buf[have..].ptr, buf.len - have);
        if (rc < 0) return error.ReadFailed;
        if (rc == 0) return have;
        have += @intCast(rc);
    }
    return have;
}

test "serve_one end-to-end via raw libc client" {
    const listen_fd = try bind_listen(0);
    defer _ = libc.close(listen_fd);
    const port = try local_port(listen_fd);

    const Server = struct {
        fn run(fd: libc.fd_t) void {
            serve_one(fd, &test_dispatch);
        }
    };
    const thread = try std.Thread.spawn(.{}, Server.run, .{listen_fd});

    const client_fd = try test_client_connect(port);
    defer _ = libc.close(client_fd);

    try test_send_all(client_fd, "GET /ready HTTP/1.1\r\nHost: x\r\n\r\n");
    try test_send_all(client_fd, "POST /fraud-score HTTP/1.1\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}");

    var buf: [4096]u8 = undefined;
    const n = try test_recv_until_close(client_fd, &buf);
    const got = buf[0..n];

    try std.testing.expect(std.mem.startsWith(u8, got, "HTTP/1.1 200 OK\r\n"));
    try std.testing.expect(std.mem.indexOf(u8, got, "Content-Length: 0\r\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, got, FRAUD_STUB) != null);
    try std.testing.expect(std.mem.indexOf(u8, got, "Content-Length: 35\r\n") != null);

    thread.join();
}
