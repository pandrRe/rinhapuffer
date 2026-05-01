const std = @import("std");
const posix = std.posix;
const libc = std.c;

pub const ParseError = error{
    UnexpectedByte,
    UnexpectedEof,
    InvalidFloat,
};

/// Count occurrences of '}' in `bytes`.
///
/// In the reference dataset format every record has exactly one '}' (the one
/// that closes the per-record object), and '}' cannot occur anywhere else:
/// vectors only contain numerics, and labels are restricted to "legit"/"fraud".
/// So this count equals the record count for any well-formed reference file.
///
/// Uses @Vector(16, u8) lane comparisons; lowers to NEON on aarch64 and SSE2 on x86.
pub fn count_closing_braces(bytes: []const u8) usize {
    var total: usize = 0;
    var i: usize = 0;
    const target: @Vector(16, u8) = @splat('}');
    const ones: @Vector(16, u8) = @splat(1);
    const zeros: @Vector(16, u8) = @splat(0);
    while (i + 16 <= bytes.len) : (i += 16) {
        const v: @Vector(16, u8) = bytes[i..][0..16].*;
        const matches = v == target;
        const counts = @select(u8, matches, ones, zeros);
        total += @as(usize, @reduce(.Add, counts));
    }
    while (i < bytes.len) : (i += 1) {
        if (bytes[i] == '}') total += 1;
    }
    return total;
}

pub const ParsedF32 = struct { value: f32, next: usize };

/// Parse a single f32 in the format `-?\d+(\.\d+)?` (no exponent, no leading sign other than `-`).
///
/// Loses no more than 1 ULP for values in the dataset's range (|x| <= 1, <= 5 fractional digits).
pub fn parse_f32_simple(bytes: []const u8, start: usize) ParseError!ParsedF32 {
    var p = start;
    if (p >= bytes.len) return error.UnexpectedEof;

    const neg = bytes[p] == '-';
    p += @intFromBool(neg);

    if (p >= bytes.len or !is_digit(bytes[p])) return error.InvalidFloat;

    var int_part: u32 = 0;
    while (p < bytes.len and is_digit(bytes[p])) : (p += 1) {
        int_part = int_part * 10 + (bytes[p] - '0');
    }

    if (p >= bytes.len or bytes[p] != '.') {
        const v: f32 = @floatFromInt(int_part);
        return .{ .value = if (neg) -v else v, .next = p };
    }

    p += 1;
    var frac: u32 = 0;
    var scale: u32 = 1;
    while (p < bytes.len and is_digit(bytes[p])) : (p += 1) {
        frac = frac * 10 + (bytes[p] - '0');
        scale *= 10;
    }

    const int_f: f32 = @floatFromInt(int_part);
    const frac_f: f32 = @floatFromInt(frac);
    const scale_f: f32 = @floatFromInt(scale);
    const v = int_f + frac_f / scale_f;
    return .{ .value = if (neg) -v else v, .next = p };
}

inline fn is_digit(b: u8) bool {
    return b >= '0' and b <= '9';
}

/// Skip ASCII whitespace (' ', '\t', '\n', '\r').
pub fn skip_ws(bytes: []const u8, start: usize) usize {
    var p = start;
    while (p < bytes.len) : (p += 1) {
        const b = bytes[p];
        if (b != ' ' and b != '\t' and b != '\n' and b != '\r') return p;
    }
    return p;
}

/// Assert `bytes[p] == b`, return `p + 1`. Returns `error.UnexpectedByte` on mismatch
/// or `error.UnexpectedEof` on end-of-input.
pub fn expect_byte(bytes: []const u8, p: usize, b: u8) ParseError!usize {
    if (p >= bytes.len) return error.UnexpectedEof;
    if (bytes[p] != b) return error.UnexpectedByte;
    return p + 1;
}

/// A read-only mmap of a file. Caller must call `deinit` to release.
pub const Mapped = struct {
    bytes: []align(std.heap.page_size_min) const u8,
    fd: posix.fd_t,

    pub fn deinit(self: Mapped) void {
        posix.munmap(self.bytes);
        _ = libc.close(self.fd);
    }
};

pub const MmapError = posix.OpenError || posix.MMapError || error{StatFailed};

/// mmap `path` for reading. Maps the entire file as PRIVATE+READ. The caller
/// owns the result and must call `Mapped.deinit`.
///
/// Issues a `madvise(SEQUENTIAL)` hint on success; ignores madvise errors.
pub fn mmap_file(path: []const u8) MmapError!Mapped {
    const fd = try posix.openat(posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY }, 0);
    errdefer _ = libc.close(fd);

    var st: libc.Stat = undefined;
    if (libc.fstat(fd, &st) != 0) return error.StatFailed;
    const len: usize = @intCast(st.size);

    if (len == 0) {
        return .{ .bytes = &.{}, .fd = fd };
    }

    const bytes = try posix.mmap(
        null,
        len,
        .{ .READ = true },
        .{ .TYPE = .PRIVATE },
        fd,
        0,
    );
    posix.madvise(bytes.ptr, bytes.len, posix.MADV.SEQUENTIAL) catch {};
    return .{ .bytes = bytes, .fd = fd };
}

// ─── tests ──────────────────────────────────────────────────────────────────

test "count_closing_braces empty" {
    try std.testing.expectEqual(@as(usize, 0), count_closing_braces(""));
}

test "count_closing_braces no matches" {
    try std.testing.expectEqual(@as(usize, 0), count_closing_braces("hello world, no braces here at all"));
}

test "count_closing_braces only matches" {
    try std.testing.expectEqual(@as(usize, 32), count_closing_braces("}" ** 32));
}

test "count_closing_braces mixed" {
    const s = "{ \"a\": 1 }, { \"b\": 2 }, { \"c\": 3 }";
    try std.testing.expectEqual(@as(usize, 3), count_closing_braces(s));
}

test "count_closing_braces under 16 bytes" {
    try std.testing.expectEqual(@as(usize, 2), count_closing_braces("a}b}c"));
}

test "count_closing_braces straddling 16-byte boundary" {
    // 20 bytes: forces SIMD pass + scalar tail; '}' at indices 0, 15, 19.
    const s = "}aaaaaaaaaaaaaa}aaa}";
    try std.testing.expectEqual(@as(usize, 3), count_closing_braces(s));
}

test "parse_f32_simple table" {
    const cases = [_]struct { s: []const u8, v: f32 }{
        .{ .s = "0", .v = 0.0 },
        .{ .s = "1", .v = 1.0 },
        .{ .s = "-1", .v = -1.0 },
        .{ .s = "0.0", .v = 0.0 },
        .{ .s = "0.5", .v = 0.5 },
        .{ .s = "0.01", .v = 0.01 },
        .{ .s = "0.0833", .v = 0.0833 },
        .{ .s = "0.8261", .v = 0.8261 },
        .{ .s = "0.1667", .v = 0.1667 },
        .{ .s = "0.0432", .v = 0.0432 },
        .{ .s = "0.6657", .v = 0.6657 },
        .{ .s = "0.9708", .v = 0.9708 },
        .{ .s = "12.345", .v = 12.345 },
        .{ .s = "-0.0001", .v = -0.0001 },
    };
    for (cases) |c| {
        const got = try parse_f32_simple(c.s, 0);
        try std.testing.expectEqual(c.s.len, got.next);
        // Compare against std.fmt.parseFloat — both should round to the same f32.
        const reference = try std.fmt.parseFloat(f32, c.s);
        try std.testing.expect(@abs(got.value - reference) <= 1.0e-6);
        try std.testing.expect(@abs(got.value - c.v) <= 1.0e-6);
    }
}

test "parse_f32_simple advances past number, leaves trailing chars" {
    const s = "0.25,foo";
    const got = try parse_f32_simple(s, 0);
    try std.testing.expectEqual(@as(usize, 4), got.next);
    try std.testing.expect(@abs(got.value - 0.25) <= 1.0e-6);
}

test "parse_f32_simple errors on no digits" {
    try std.testing.expectError(error.InvalidFloat, parse_f32_simple("abc", 0));
    try std.testing.expectError(error.InvalidFloat, parse_f32_simple("-x", 0));
}

test "skip_ws" {
    try std.testing.expectEqual(@as(usize, 0), skip_ws("abc", 0));
    try std.testing.expectEqual(@as(usize, 4), skip_ws("    abc", 0));
    try std.testing.expectEqual(@as(usize, 5), skip_ws(" \t\n\r abc", 0));
    try std.testing.expectEqual(@as(usize, 3), skip_ws("   ", 0));
    try std.testing.expectEqual(@as(usize, 0), skip_ws("", 0));
}

test "expect_byte" {
    try std.testing.expectEqual(@as(usize, 1), try expect_byte("[abc", 0, '['));
    try std.testing.expectError(error.UnexpectedByte, expect_byte("abc", 0, '['));
    try std.testing.expectError(error.UnexpectedEof, expect_byte("", 0, '['));
}

test "mmap_file matches example-references.json + brace count" {
    var mapped = mmap_file("./resources/example-references.json") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer mapped.deinit();
    try std.testing.expect(mapped.bytes.len > 0);
    try std.testing.expectEqual(@as(usize, 100), count_closing_braces(mapped.bytes));
}
