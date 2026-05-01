const std = @import("std");
const posix = std.posix;
const libc = std.c;

pub const ParseError = error{
    UnexpectedByte,
    UnexpectedEof,
    UnexpectedKey,
    InvalidFloat,
    InvalidBoolean,
    InvalidDate,
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

pub inline fn is_digit(b: u8) bool {
    return b >= '0' and b <= '9';
}

pub const ParsedF64 = struct { value: f64, next: usize };

/// Parse a single f64 in the format `-?\d+(\.\d+)?` (no exponent).
///
/// u64 accumulators handle up to ~18 digits each side without overflow — enough
/// for any realistic JSON-emitted decimal, including the 10-digit fractions in
/// the payload examples.
pub fn parse_f64_simple(bytes: []const u8, start: usize) ParseError!ParsedF64 {
    var p = start;
    if (p >= bytes.len) return error.UnexpectedEof;

    const neg = bytes[p] == '-';
    p += @intFromBool(neg);

    if (p >= bytes.len or !is_digit(bytes[p])) return error.InvalidFloat;

    var int_part: u64 = 0;
    while (p < bytes.len and is_digit(bytes[p])) : (p += 1) {
        int_part = int_part * 10 + (bytes[p] - '0');
    }

    if (p >= bytes.len or bytes[p] != '.') {
        const v: f64 = @floatFromInt(int_part);
        return .{ .value = if (neg) -v else v, .next = p };
    }

    p += 1;
    var frac: u64 = 0;
    var scale: u64 = 1;
    while (p < bytes.len and is_digit(bytes[p])) : (p += 1) {
        frac = frac * 10 + (bytes[p] - '0');
        scale *= 10;
    }

    const int_f: f64 = @floatFromInt(int_part);
    const frac_f: f64 = @floatFromInt(frac);
    const scale_f: f64 = @floatFromInt(scale);
    const v = int_f + frac_f / scale_f;
    return .{ .value = if (neg) -v else v, .next = p };
}

pub const ParsedU32 = struct { value: u32, next: usize };

/// Parse a non-negative integer; up to 10 digits (u32 range).
pub fn parse_u32_simple(bytes: []const u8, start: usize) ParseError!ParsedU32 {
    var p = start;
    if (p >= bytes.len or !is_digit(bytes[p])) return error.InvalidFloat;
    var v: u64 = 0;
    while (p < bytes.len and is_digit(bytes[p])) : (p += 1) {
        v = v * 10 + (bytes[p] - '0');
    }
    if (v > std.math.maxInt(u32)) return error.InvalidFloat;
    return .{ .value = @intCast(v), .next = p };
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

// ─── compound helpers — `skip_ws + structural byte` ────────────────────────

pub inline fn open_obj(bytes: []const u8, p: usize) ParseError!usize {
    return expect_byte(bytes, skip_ws(bytes, p), '{');
}

pub inline fn close_obj(bytes: []const u8, p: usize) ParseError!usize {
    return expect_byte(bytes, skip_ws(bytes, p), '}');
}

pub inline fn comma(bytes: []const u8, p: usize) ParseError!usize {
    return expect_byte(bytes, skip_ws(bytes, p), ',');
}

// ─── string + key helpers ──────────────────────────────────────────────────

pub fn skip_string(bytes: []const u8, start: usize) ParseError!usize {
    var p = try expect_byte(bytes, start, '"');
    while (p < bytes.len) : (p += 1) {
        if (bytes[p] == '"') return p + 1;
    }
    return error.UnexpectedEof;
}

pub const TakenString = struct { value: []const u8, next: usize };

pub fn take_string(bytes: []const u8, start: usize) ParseError!TakenString {
    var p = try expect_byte(bytes, start, '"');
    const begin = p;
    while (p < bytes.len) : (p += 1) {
        if (bytes[p] == '"') return .{ .value = bytes[begin..p], .next = p + 1 };
    }
    return error.UnexpectedEof;
}

/// Match `"key"` literal, return position past closing quote.
pub fn expect_key(bytes: []const u8, start: usize, key: []const u8) ParseError!usize {
    var p = try expect_byte(bytes, start, '"');
    if (p + key.len > bytes.len) return error.UnexpectedEof;
    if (!std.mem.eql(u8, bytes[p..][0..key.len], key)) return error.UnexpectedKey;
    p += key.len;
    return try expect_byte(bytes, p, '"');
}

/// Match `"key" :` (with surrounding ws) and return the position of the value.
pub inline fn enter_key(bytes: []const u8, p: usize, key: []const u8) ParseError!usize {
    var q = try expect_key(bytes, skip_ws(bytes, p), key);
    q = try expect_byte(bytes, skip_ws(bytes, q), ':');
    return skip_ws(bytes, q);
}

// ─── value parsers ─────────────────────────────────────────────────────────

pub const ParsedBool = struct { value: bool, next: usize };

pub fn parse_bool(bytes: []const u8, start: usize) ParseError!ParsedBool {
    if (start + 4 > bytes.len) return error.InvalidBoolean;
    if (bytes[start] == 't') {
        if (!std.mem.eql(u8, bytes[start..][0..4], "true")) return error.InvalidBoolean;
        return .{ .value = true, .next = start + 4 };
    }
    if (bytes[start] == 'f') {
        if (start + 5 > bytes.len) return error.InvalidBoolean;
        if (!std.mem.eql(u8, bytes[start..][0..5], "false")) return error.InvalidBoolean;
        return .{ .value = false, .next = start + 5 };
    }
    return error.InvalidBoolean;
}

/// If `bytes[p..]` starts with `null`, return position past it; else null.
pub inline fn take_null(bytes: []const u8, p: usize) ParseError!?usize {
    if (p < bytes.len and bytes[p] == 'n') {
        if (p + 4 > bytes.len or !std.mem.eql(u8, bytes[p..][0..4], "null")) return error.UnexpectedByte;
        return p + 4;
    }
    return null;
}

/// Skip a JSON value enclosed by `open` / `close`, accounting for nested pairs
/// and quoted strings (no escape handling).
pub fn skip_to_matching_close(bytes: []const u8, start: usize, open: u8, close: u8) ParseError!usize {
    var depth: usize = 1;
    var p = start;
    var in_string = false;
    while (p < bytes.len) : (p += 1) {
        const c = bytes[p];
        if (in_string) {
            if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == open) depth += 1;
        if (c == close) {
            depth -= 1;
            if (depth == 0) return p + 1;
        }
    }
    return error.UnexpectedEof;
}

/// Linear scan over `arr_bytes` (the contents between `[` and `]` of a JSON
/// string array) testing membership of `needle` against each `"..."` element.
pub fn contains_string(arr_bytes: []const u8, needle: []const u8) bool {
    var p: usize = 0;
    while (p < arr_bytes.len) {
        while (p < arr_bytes.len and arr_bytes[p] != '"') : (p += 1) {}
        if (p >= arr_bytes.len) return false;
        p += 1;
        const begin = p;
        while (p < arr_bytes.len and arr_bytes[p] != '"') : (p += 1) {}
        if (p > arr_bytes.len) return false;
        const elem = arr_bytes[begin..p];
        if (std.mem.eql(u8, elem, needle)) return true;
        p += 1;
    }
    return false;
}

// ─── "enter key + parse value" combinators ─────────────────────────────────

pub inline fn read_f64(bytes: []const u8, p: usize, key: []const u8) ParseError!ParsedF64 {
    return parse_f64_simple(bytes, try enter_key(bytes, p, key));
}

pub inline fn read_u32(bytes: []const u8, p: usize, key: []const u8) ParseError!ParsedU32 {
    return parse_u32_simple(bytes, try enter_key(bytes, p, key));
}

pub inline fn read_string(bytes: []const u8, p: usize, key: []const u8) ParseError!TakenString {
    return take_string(bytes, try enter_key(bytes, p, key));
}

pub inline fn read_bool(bytes: []const u8, p: usize, key: []const u8) ParseError!ParsedBool {
    return parse_bool(bytes, try enter_key(bytes, p, key));
}

pub const ReadIsoDate = struct { value: []const u8, next: usize };

/// Read a 20-byte ISO-8601 timestamp `"YYYY-MM-DDTHH:MM:SSZ"`.
/// Returns the 20 content bytes (no quotes) and the position past the closing quote.
pub inline fn read_iso_date(bytes: []const u8, p: usize, key: []const u8) ParseError!ReadIsoDate {
    var q = try enter_key(bytes, p, key);
    q = try expect_byte(bytes, q, '"');
    if (q + 20 > bytes.len) return error.InvalidDate;
    const value = bytes[q .. q + 20];
    const next = try expect_byte(bytes, q + 20, '"');
    return .{ .value = value, .next = next };
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

test "parse_f64_simple long fraction" {
    // Payload examples carry up to 10 fractional digits — must parse without overflow.
    const cases = [_]struct { s: []const u8, v: f64 }{
        .{ .s = "881.6139684714", .v = 881.6139684714 },
        .{ .s = "29.2331036248", .v = 29.2331036248 },
        .{ .s = "13.7090520965", .v = 13.7090520965 },
        .{ .s = "0", .v = 0.0 },
        .{ .s = "-1", .v = -1.0 },
        .{ .s = "12345.678901234", .v = 12345.678901234 },
    };
    for (cases) |c| {
        const got = try parse_f64_simple(c.s, 0);
        try std.testing.expectEqual(c.s.len, got.next);
        try std.testing.expect(@abs(got.value - c.v) <= 1.0e-12);
    }
}

test "parse_u32_simple" {
    {
        const r = try parse_u32_simple("0", 0);
        try std.testing.expectEqual(@as(u32, 0), r.value);
        try std.testing.expectEqual(@as(usize, 1), r.next);
    }
    {
        const r = try parse_u32_simple("12,foo", 0);
        try std.testing.expectEqual(@as(u32, 12), r.value);
        try std.testing.expectEqual(@as(usize, 2), r.next);
    }
    {
        const r = try parse_u32_simple("4294967295", 0);
        try std.testing.expectEqual(@as(u32, 4294967295), r.value);
    }
    try std.testing.expectError(error.InvalidFloat, parse_u32_simple("abc", 0));
    try std.testing.expectError(error.InvalidFloat, parse_u32_simple("4294967296", 0));
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

test "parse_bool" {
    {
        const r = try parse_bool("true,", 0);
        try std.testing.expect(r.value);
        try std.testing.expectEqual(@as(usize, 4), r.next);
    }
    {
        const r = try parse_bool("false,", 0);
        try std.testing.expect(!r.value);
        try std.testing.expectEqual(@as(usize, 5), r.next);
    }
    try std.testing.expectError(error.InvalidBoolean, parse_bool("xyz", 0));
}

test "contains_string" {
    const arr = "\"MERC-003\", \"MERC-016\"";
    try std.testing.expect(contains_string(arr, "MERC-016"));
    try std.testing.expect(contains_string(arr, "MERC-003"));
    try std.testing.expect(!contains_string(arr, "MERC-001"));
    try std.testing.expect(!contains_string("", "MERC-001"));
}

test "expect_key + enter_key" {
    const s = "\"foo\" : 42,";
    try std.testing.expectEqual(@as(usize, 5), try expect_key(s, 0, "foo"));
    try std.testing.expectError(error.UnexpectedKey, expect_key(s, 0, "bar"));
    const after = try enter_key(s, 0, "foo");
    try std.testing.expectEqual(@as(usize, 8), after);
}

test "take_null" {
    try std.testing.expectEqual(@as(?usize, 4), try take_null("null,", 0));
    try std.testing.expectEqual(@as(?usize, null), try take_null("true", 0));
    try std.testing.expectError(error.UnexpectedByte, take_null("nope", 0));
}

test "read_iso_date" {
    const s = "\"timestamp\": \"2026-03-11T18:45:53Z\",";
    const got = try read_iso_date(s, 0, "timestamp");
    try std.testing.expectEqualStrings("2026-03-11T18:45:53Z", got.value);
    try std.testing.expectEqual(@as(usize, 35), got.next);
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
