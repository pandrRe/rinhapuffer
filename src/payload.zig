//! Schema-rigid payload vectorizer for fraud-detection requests.
//!
//! Each request is a single JSON object shaped like the examples in
//! resources/example-payloads.json. We assume:
//!   - Top-level keys appear in this exact order:
//!       id, transaction, customer, merchant, terminal, last_transaction
//!   - Inner keys appear in their documented order too.
//!   - Strings contain no JSON escapes (true for the spec's IDs, MCCs, dates, labels).
//!
//! Any deviation returns an error — the caller maps that to HTTP 400.
//!
//! Per request:
//!   - zero heap allocations
//!   - zero syscalls
//!   - output is written into a caller-supplied `*[14]f32`
//!
//! Lookup tables are built at comptime; nothing loads at startup.

const std = @import("std");
const fj = @import("fast_json.zig");

pub const N_FEATURES: usize = 14;

pub const Error = fj.ParseError || error{InvalidMcc};

// ─── normalization constants (mirror of resources/normalization.json) ──────

const norm = struct {
    pub const max_amount: f32 = 10000;
    pub const max_installments: f32 = 12;
    pub const amount_vs_avg_ratio: f32 = 10;
    pub const max_minutes: f32 = 1440;
    pub const max_km: f32 = 1000;
    pub const max_tx_count_24h: f32 = 20;
    pub const max_merchant_avg_amount: f32 = 10000;
};

// ─── mcc risk lookup (mirror of resources/mcc_risk.json) ───────────────────

pub const MCC_DEFAULT: f32 = 0.5;

/// Comptime-baked 40 KB flat table indexed by mcc (0..9999). Listed values
/// come from `resources/mcc_risk.json`; everything else gets `MCC_DEFAULT`.
const mcc_risk: [10000]f32 = blk: {
    @setEvalBranchQuota(40_000);
    var t: [10000]f32 = undefined;
    var i: usize = 0;
    while (i < 10000) : (i += 1) t[i] = MCC_DEFAULT;
    t[5411] = 0.15;
    t[5812] = 0.30;
    t[5912] = 0.20;
    t[5944] = 0.45;
    t[7801] = 0.80;
    t[7802] = 0.75;
    t[7995] = 0.85;
    t[4511] = 0.35;
    t[5311] = 0.25;
    t[5999] = 0.50;
    break :blk t;
};

// ─── vectorize ──────────────────────────────────────────────────────────────

/// Parse one payload into a 14-feature vector.
///
/// Schema-rigid: keys must appear in the order documented at the top of
/// this file. Deviations return `error.UnexpectedKey` / `UnexpectedByte`.
pub fn vectorize(bytes: []const u8, out: *[N_FEATURES]f32) Error!void {
    var p = try fj.open_obj(bytes, 0);

    // top-level "id" — not vectorized
    p = try fj.enter_key(bytes, p, "id");
    p = try fj.skip_string(bytes, p);
    p = try fj.comma(bytes, p);

    // transaction → fills [0] amount, [1] installments, [3] hour_of_day, [4] day_of_week
    p = try fj.enter_key(bytes, p, "transaction");
    p = try fj.open_obj(bytes, p);
    const amount = try fill_amount(bytes, p, &out[0]);
    p = try fj.comma(bytes, amount.next);
    p = try fill_installments(bytes, p, &out[1]);
    p = try fj.comma(bytes, p);
    const req_at = try fill_requested_at(bytes, p, &out[3], &out[4]);
    p = try fj.close_obj(bytes, req_at.next);
    p = try fj.comma(bytes, p);

    // customer → fills [2] amount_vs_avg, [8] tx_count_24h; captures known_merchants range
    p = try fj.enter_key(bytes, p, "customer");
    p = try fj.open_obj(bytes, p);
    p = try fill_amount_vs_avg(bytes, p, amount.value, &out[2]);
    p = try fj.comma(bytes, p);
    p = try fill_tx_count_24h(bytes, p, &out[8]);
    p = try fj.comma(bytes, p);
    const known = try take_known_merchants(bytes, p);
    p = try fj.close_obj(bytes, known.next);
    p = try fj.comma(bytes, p);

    // merchant → fills [12] mcc_risk, [13] merchant_avg_amount; captures merchant.id
    p = try fj.enter_key(bytes, p, "merchant");
    p = try fj.open_obj(bytes, p);
    const merchant_id = try take_merchant_id(bytes, p);
    p = try fj.comma(bytes, merchant_id.next);
    p = try fill_mcc_risk(bytes, p, &out[12]);
    p = try fj.comma(bytes, p);
    p = try fill_merchant_avg_amount(bytes, p, &out[13]);
    p = try fj.close_obj(bytes, p);
    p = try fj.comma(bytes, p);

    // resolved from earlier captures → fills [11] unknown_merchant
    fill_unknown_merchant(known.range, merchant_id.value, &out[11]);

    // terminal → fills [9] is_online, [10] card_present, [7] km_from_home
    p = try fj.enter_key(bytes, p, "terminal");
    p = try fj.open_obj(bytes, p);
    p = try fill_is_online(bytes, p, &out[9]);
    p = try fj.comma(bytes, p);
    p = try fill_card_present(bytes, p, &out[10]);
    p = try fj.comma(bytes, p);
    p = try fill_km_from_home(bytes, p, &out[7]);
    p = try fj.close_obj(bytes, p);
    p = try fj.comma(bytes, p);

    // last_transaction → fills [5] minutes_since_last_tx, [6] km_from_last_tx (-1 if null)
    p = try fj.enter_key(bytes, p, "last_transaction");
    p = try fill_last_transaction(bytes, p, req_at.date, &out[5], &out[6]);

    _ = try fj.close_obj(bytes, p);
}

// ─── per-field fill helpers ────────────────────────────────────────────────
//
// Each takes the precise `*f32` slot(s) it writes; the call site in `vectorize`
// makes the data-flow explicit (`fill_amount(..., &out[0])` → fills index 0).

const Amount = struct { value: f64, next: usize };

/// fills out_amount (feature 0) and forwards the parsed value for feature 2's ratio.
inline fn fill_amount(bytes: []const u8, p: usize, out_amount: *f32) Error!Amount {
    const r = try fj.read_f64(bytes, p, "amount");
    out_amount.* = clamp01(@as(f32, @floatCast(r.value)) / norm.max_amount);
    return .{ .value = r.value, .next = r.next };
}

/// fills out_installments (feature 1).
inline fn fill_installments(bytes: []const u8, p: usize, out_installments: *f32) Error!usize {
    const r = try fj.read_u32(bytes, p, "installments");
    out_installments.* = clamp01(@as(f32, @floatFromInt(r.value)) / norm.max_installments);
    return r.next;
}

const RequestedAt = struct { date: []const u8, next: usize };

/// fills out_hour_of_day (feature 3) and out_day_of_week (feature 4); forwards the
/// raw 20-byte ISO date for use by `fill_last_transaction`.
inline fn fill_requested_at(
    bytes: []const u8,
    p: usize,
    out_hour_of_day: *f32,
    out_day_of_week: *f32,
) Error!RequestedAt {
    const r = try fj.read_iso_date(bytes, p, "requested_at");
    const hour = try parse_2digit(r.value[11..13]);
    if (hour > 23) return error.InvalidDate;
    const year = try parse_4digit(r.value[0..4]);
    const month = try parse_2digit(r.value[5..7]);
    const day = try parse_2digit(r.value[8..10]);
    if (month < 1 or month > 12 or day < 1 or day > 31) return error.InvalidDate;
    out_hour_of_day.* = @as(f32, @floatFromInt(hour)) / 23.0;
    out_day_of_week.* = @as(f32, @floatFromInt(day_of_week_mon0(year, month, day))) / 6.0;
    return .{ .date = r.value, .next = r.next };
}

/// reads `customer.avg_amount`, fills out_amount_vs_avg (feature 2).
inline fn fill_amount_vs_avg(
    bytes: []const u8,
    p: usize,
    amount: f64,
    out_amount_vs_avg: *f32,
) Error!usize {
    const r = try fj.read_f64(bytes, p, "avg_amount");
    const ratio: f64 = if (r.value == 0) 0 else amount / r.value;
    out_amount_vs_avg.* = clamp01(@as(f32, @floatCast(ratio)) / norm.amount_vs_avg_ratio);
    return r.next;
}

/// fills out_tx_count_24h (feature 8).
inline fn fill_tx_count_24h(bytes: []const u8, p: usize, out_tx_count_24h: *f32) Error!usize {
    const r = try fj.read_u32(bytes, p, "tx_count_24h");
    out_tx_count_24h.* = clamp01(@as(f32, @floatFromInt(r.value)) / norm.max_tx_count_24h);
    return r.next;
}

const KnownMerchants = struct { range: []const u8, next: usize };

/// captures the byte range between `[` and `]` of `customer.known_merchants`
/// for later membership testing. Fills no output directly.
inline fn take_known_merchants(bytes: []const u8, p: usize) Error!KnownMerchants {
    var q = try fj.enter_key(bytes, p, "known_merchants");
    q = try fj.expect_byte(bytes, q, '[');
    const start = q;
    q = try fj.skip_to_matching_close(bytes, q, '[', ']');
    return .{ .range = bytes[start .. q - 1], .next = q };
}

const MerchantId = struct { value: []const u8, next: usize };

/// captures `merchant.id` for later membership testing. Fills no output directly.
inline fn take_merchant_id(bytes: []const u8, p: usize) Error!MerchantId {
    const r = try fj.read_string(bytes, p, "id");
    return .{ .value = r.value, .next = r.next };
}

/// reads `merchant.mcc`, fills out_mcc_risk (feature 12).
inline fn fill_mcc_risk(bytes: []const u8, p: usize, out_mcc_risk: *f32) Error!usize {
    const r = try fj.read_string(bytes, p, "mcc");
    out_mcc_risk.* = mcc_risk[try parse_mcc_str(r.value)];
    return r.next;
}

/// reads `merchant.avg_amount`, fills out_merchant_avg_amount (feature 13).
inline fn fill_merchant_avg_amount(bytes: []const u8, p: usize, out_merchant_avg_amount: *f32) Error!usize {
    const r = try fj.read_f64(bytes, p, "avg_amount");
    out_merchant_avg_amount.* = clamp01(@as(f32, @floatCast(r.value)) / norm.max_merchant_avg_amount);
    return r.next;
}

/// fills out_unknown_merchant (feature 11) — 1 if `merchant.id` is NOT in the
/// `customer.known_merchants` byte range.
inline fn fill_unknown_merchant(
    known_merchants_range: []const u8,
    merchant_id: []const u8,
    out_unknown_merchant: *f32,
) void {
    out_unknown_merchant.* = if (fj.contains_string(known_merchants_range, merchant_id)) 0.0 else 1.0;
}

/// fills out_is_online (feature 9).
inline fn fill_is_online(bytes: []const u8, p: usize, out_is_online: *f32) Error!usize {
    const r = try fj.read_bool(bytes, p, "is_online");
    out_is_online.* = if (r.value) 1.0 else 0.0;
    return r.next;
}

/// fills out_card_present (feature 10).
inline fn fill_card_present(bytes: []const u8, p: usize, out_card_present: *f32) Error!usize {
    const r = try fj.read_bool(bytes, p, "card_present");
    out_card_present.* = if (r.value) 1.0 else 0.0;
    return r.next;
}

/// fills out_km_from_home (feature 7).
inline fn fill_km_from_home(bytes: []const u8, p: usize, out_km_from_home: *f32) Error!usize {
    const r = try fj.read_f64(bytes, p, "km_from_home");
    out_km_from_home.* = clamp01(@as(f32, @floatCast(r.value)) / norm.max_km);
    return r.next;
}

/// fills out_minutes_since_last_tx (feature 5) and out_km_from_last_tx (feature 6).
/// If `last_transaction` is `null`, both outputs become `-1` (sentinel).
inline fn fill_last_transaction(
    bytes: []const u8,
    p: usize,
    requested_at: []const u8,
    out_minutes_since_last_tx: *f32,
    out_km_from_last_tx: *f32,
) Error!usize {
    if (try fj.take_null(bytes, p)) |after_null| {
        out_minutes_since_last_tx.* = -1;
        out_km_from_last_tx.* = -1;
        return after_null;
    }

    var q = try fj.expect_byte(bytes, p, '{');

    const last_at = try fj.read_iso_date(bytes, q, "timestamp");
    const minutes = try minutes_between(last_at.value, requested_at);
    out_minutes_since_last_tx.* = clamp01(@as(f32, @floatFromInt(minutes)) / norm.max_minutes);
    q = try fj.comma(bytes, last_at.next);

    const km_last = try fj.read_f64(bytes, q, "km_from_current");
    out_km_from_last_tx.* = clamp01(@as(f32, @floatCast(km_last.value)) / norm.max_km);
    return try fj.close_obj(bytes, km_last.next);
}

// ─── domain-specific helpers ───────────────────────────────────────────────

inline fn clamp01(x: f32) f32 {
    return @min(@as(f32, 1.0), @max(@as(f32, 0.0), x));
}

inline fn parse_mcc_str(s: []const u8) Error!u16 {
    if (s.len != 4) return error.InvalidMcc;
    var v: u16 = 0;
    for (s) |c| {
        if (!fj.is_digit(c)) return error.InvalidMcc;
        v = v * 10 + (c - '0');
    }
    return v;
}

inline fn parse_2digit(s: []const u8) Error!u8 {
    if (s.len < 2) return error.InvalidDate;
    if (!fj.is_digit(s[0]) or !fj.is_digit(s[1])) return error.InvalidDate;
    return @intCast((s[0] - '0') * 10 + (s[1] - '0'));
}

inline fn parse_4digit(s: []const u8) Error!u16 {
    if (s.len < 4) return error.InvalidDate;
    inline for (0..4) |i| {
        if (!fj.is_digit(s[i])) return error.InvalidDate;
    }
    const v: u32 =
        @as(u32, s[0] - '0') * 1000 +
        @as(u32, s[1] - '0') * 100 +
        @as(u32, s[2] - '0') * 10 +
        @as(u32, s[3] - '0');
    return @intCast(v);
}

/// Sakamoto: 0=Sunday..6=Saturday → shift to 0=Monday..6=Sunday (as spec requires).
/// Correct for Gregorian dates from year 1583 onward.
fn day_of_week_mon0(year: u32, month: u32, day: u32) u32 {
    const t = [_]u32{ 0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4 };
    const y: u32 = if (month < 3) year - 1 else year;
    const sun0 = (y + y / 4 - y / 100 + y / 400 + t[month - 1] + day) % 7;
    return (sun0 + 6) % 7;
}

/// Days from Gregorian epoch (year 0, March 1) — used to compute minutes
/// between two timestamps without an external calendar library.
fn days_from_civil(year: i64, month: u32, day: u32) i64 {
    const y: i64 = if (month <= 2) year - 1 else year;
    const era: i64 = @divFloor(y, 400);
    const yoe: u64 = @intCast(y - era * 400);
    const m: u64 = @intCast(month);
    const d: u64 = @intCast(day);
    const m_adj: u64 = if (m > 2) m - 3 else m + 9;
    const doy: u64 = (153 * m_adj + 2) / 5 + d - 1;
    const doe: u64 = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + @as(i64, @intCast(doe)) - 719468;
}

/// Compute |minutes(later) - minutes(earlier)|. Both inputs are 20-byte
/// "YYYY-MM-DDTHH:MM:SSZ" slices. `current` is `requested_at`.
fn minutes_between(last: []const u8, current: []const u8) Error!u32 {
    if (last.len < 20 or current.len < 20) return error.InvalidDate;
    const last_year: u32 = try parse_4digit(last[0..4]);
    const last_month: u32 = try parse_2digit(last[5..7]);
    const last_day: u32 = try parse_2digit(last[8..10]);
    const last_h: u32 = try parse_2digit(last[11..13]);
    const last_m: u32 = try parse_2digit(last[14..16]);
    const last_s: u32 = try parse_2digit(last[17..19]);

    const cur_year: u32 = try parse_4digit(current[0..4]);
    const cur_month: u32 = try parse_2digit(current[5..7]);
    const cur_day: u32 = try parse_2digit(current[8..10]);
    const cur_h: u32 = try parse_2digit(current[11..13]);
    const cur_m: u32 = try parse_2digit(current[14..16]);
    const cur_s: u32 = try parse_2digit(current[17..19]);

    const last_days = days_from_civil(last_year, last_month, last_day);
    const cur_days = days_from_civil(cur_year, cur_month, cur_day);
    const last_total: i64 = last_days * 86400 +
        @as(i64, last_h) * 3600 + @as(i64, last_m) * 60 + @as(i64, last_s);
    const cur_total: i64 = cur_days * 86400 +
        @as(i64, cur_h) * 3600 + @as(i64, cur_m) * 60 + @as(i64, cur_s);
    const delta_seconds: i64 = if (cur_total >= last_total)
        cur_total - last_total
    else
        last_total - cur_total;
    return @intCast(@divTrunc(delta_seconds, 60));
}

// ─── tests ──────────────────────────────────────────────────────────────────

test "mcc_risk comptime table" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.15), mcc_risk[5411], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.30), mcc_risk[5812], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.80), mcc_risk[7801], 1e-6);
    try std.testing.expectApproxEqAbs(MCC_DEFAULT, mcc_risk[1234], 1e-6);
    try std.testing.expectApproxEqAbs(MCC_DEFAULT, mcc_risk[0], 1e-6);
    try std.testing.expectApproxEqAbs(MCC_DEFAULT, mcc_risk[9999], 1e-6);
}

test "parse_mcc_str" {
    try std.testing.expectEqual(@as(u16, 5411), try parse_mcc_str("5411"));
    try std.testing.expectEqual(@as(u16, 0), try parse_mcc_str("0000"));
    try std.testing.expectEqual(@as(u16, 9999), try parse_mcc_str("9999"));
    try std.testing.expectError(error.InvalidMcc, parse_mcc_str("541"));
    try std.testing.expectError(error.InvalidMcc, parse_mcc_str("54a1"));
}

test "day_of_week_mon0" {
    try std.testing.expectEqual(@as(u32, 2), day_of_week_mon0(2026, 3, 11)); // Wed
    try std.testing.expectEqual(@as(u32, 0), day_of_week_mon0(2026, 3, 9)); // Mon
    try std.testing.expectEqual(@as(u32, 6), day_of_week_mon0(2026, 3, 15)); // Sun
}

test "minutes_between same day" {
    try std.testing.expectEqual(
        @as(u32, 325),
        try minutes_between("2026-03-11T14:58:35Z", "2026-03-11T20:23:35Z"),
    );
}

test "minutes_between across days" {
    try std.testing.expectEqual(
        @as(u32, 120),
        try minutes_between("2026-03-11T23:00:00Z", "2026-03-12T01:00:00Z"),
    );
}

test "vectorize first example payload (last_transaction null)" {
    var mapped = try fj.mmap_file("./resources/example-payloads.json");
    defer mapped.deinit();
    const first = try slice_nth_object(mapped.bytes, 0);

    var out: [N_FEATURES]f32 = undefined;
    try vectorize(first, &out);

    const expected = [_]f32{
        0.0041, 0.1667, 0.05, 0.7826, 0.3333,
        -1,     -1,     0.0292, 0.15, 0,
        1,      0,      0.15,  0.006,
    };
    for (expected, out) |e, got| {
        try std.testing.expectApproxEqAbs(e, got, 1.0e-3);
    }
}

test "vectorize second example payload (with last_transaction)" {
    var mapped = try fj.mmap_file("./resources/example-payloads.json");
    defer mapped.deinit();
    const obj = try slice_nth_object(mapped.bytes, 1);

    var out: [N_FEATURES]f32 = undefined;
    try vectorize(obj, &out);

    try std.testing.expectApproxEqAbs(@as(f32, 384.88 / 10000.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0 / 12.0), out[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.05), out[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0 / 23.0), out[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 6.0), out[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 325.0 / 1440.0), out[5], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 18.8626479774 / 1000.0), out[6], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 13.7090520965 / 1000.0), out[7], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0 / 20.0), out[8], 1e-5);
    try std.testing.expectEqual(@as(f32, 0), out[9]);
    try std.testing.expectEqual(@as(f32, 1), out[10]);
    try std.testing.expectEqual(@as(f32, 0), out[11]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.20), out[12], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 298.95 / 10000.0), out[13], 1e-5);
}

test "vectorize all example payloads round-trip" {
    var mapped = try fj.mmap_file("./resources/example-payloads.json");
    defer mapped.deinit();

    var count: usize = 0;
    var p: usize = 0;
    p = fj.skip_ws(mapped.bytes, p);
    p = try fj.expect_byte(mapped.bytes, p, '[');
    var first = true;
    while (true) {
        p = fj.skip_ws(mapped.bytes, p);
        if (p < mapped.bytes.len and mapped.bytes[p] == ']') break;
        if (!first) {
            p = try fj.expect_byte(mapped.bytes, p, ',');
            p = fj.skip_ws(mapped.bytes, p);
        }
        const obj_start = p;
        const obj_end = try fj.skip_to_matching_close(mapped.bytes, p + 1, '{', '}');
        var out: [N_FEATURES]f32 = undefined;
        try vectorize(mapped.bytes[obj_start..obj_end], &out);
        for (out, 0..) |x, i| {
            if (i == 5 or i == 6) {
                try std.testing.expect(x == -1 or (x >= 0 and x <= 1));
            } else {
                try std.testing.expect(x >= 0 and x <= 1);
            }
        }
        p = obj_end;
        first = false;
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 50), count);
}

fn slice_nth_object(bytes: []const u8, index: usize) Error![]const u8 {
    var p: usize = 0;
    p = fj.skip_ws(bytes, p);
    p = try fj.expect_byte(bytes, p, '[');
    var i: usize = 0;
    var first = true;
    while (true) {
        p = fj.skip_ws(bytes, p);
        if (p < bytes.len and bytes[p] == ']') return error.UnexpectedEof;
        if (!first) {
            p = try fj.expect_byte(bytes, p, ',');
            p = fj.skip_ws(bytes, p);
        }
        const start = p;
        const end = try fj.skip_to_matching_close(bytes, p + 1, '{', '}');
        if (i == index) return bytes[start..end];
        p = end;
        first = false;
        i += 1;
    }
}
