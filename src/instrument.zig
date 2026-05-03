//! Comptime-gated instrumentation for hot-path stage breakdowns and
//! search-internal counters. Surfaced over a `GET /__metrics` route in
//! `handler.dispatch`. Off by default — `build_options.instrument` (the
//! `-Dinstrument` flag) flips every callsite to the active path; otherwise
//! every helper here is a no-op the optimizer drops.
//!
//! Single-threaded server → plain `u64` counters and module-level state. No
//! atomics needed.
//!
//! Time source: `clock_gettime(CLOCK_MONOTONIC)` via VDSO (Linux) — ~25 ns
//! per call. With 5 timer points per request that's ~250 ns of overhead on
//! a ~5 µs total budget — measurable, which is exactly why this is opt-in.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

pub const enabled: bool = build_options.instrument;

/// Sentinel size for the log2 histogram buckets. 64 covers up to 2^63 ns
/// (~292 years), which is plenty.
const N_BUCKETS: usize = 64;

pub const Hist = struct {
    buckets: [N_BUCKETS]u64 = @splat(0),
    count: u64 = 0,
    sum_ns: u64 = 0,
    max_ns: u64 = 0,

    /// Record one observation in nanoseconds.
    pub inline fn observe(self: *Hist, ns: u64) void {
        self.count += 1;
        self.sum_ns += ns;
        if (ns > self.max_ns) self.max_ns = ns;
        // ns ∈ [2^(b-1), 2^b) → bucket b. ns=0 → bucket 0.
        const b = if (ns == 0) 0 else 64 - @as(usize, @clz(ns));
        self.buckets[b] += 1;
    }

    /// Walk buckets to find the bucket containing the p-quantile sample;
    /// return that bucket's upper bound (overestimate by ≤ 2x — fine for
    /// finding which stage is slow).
    fn quantile(self: *const Hist, p: f64) u64 {
        if (self.count == 0) return 0;
        const target_f = p * @as(f64, @floatFromInt(self.count));
        const target: u64 = @intFromFloat(@max(1.0, target_f));
        var cum: u64 = 0;
        var b: usize = 0;
        while (b < N_BUCKETS) : (b += 1) {
            cum += self.buckets[b];
            if (cum >= target) {
                if (b == 0) return 0;
                return @as(u64, 1) << @as(u6, @intCast(b));
            }
        }
        return self.max_ns;
    }
};

// ─── stage histograms ─────────────────────────────────────────────────────

pub var hist_total: Hist = .{};
pub var hist_parse: Hist = .{};
pub var hist_vectorize: Hist = .{};
pub var hist_search: Hist = .{};
pub var hist_write: Hist = .{};

// ─── counters ─────────────────────────────────────────────────────────────

pub var req_total: u64 = 0;
pub var req_parse_err: u64 = 0;
pub var req_vectorize_err: u64 = 0;

// epoll loop / connection lifecycle
pub var epoll_wakeups: u64 = 0;
pub var epoll_events: u64 = 0;
pub var accepts: u64 = 0;
pub var conn_closes: u64 = 0;
pub var read_eagain: u64 = 0;
pub var write_eagain: u64 = 0;
pub var partial_writes: u64 = 0;

// HTTP fast/slow path
pub var head_fast: u64 = 0;
pub var head_slow: u64 = 0;

// search internals (per-query)
pub var search_clusters_probed: u64 = 0;
pub var search_clusters_bbox_skipped: u64 = 0;
pub var search_clusters_bbox_scanned: u64 = 0;
pub var search_blocks_scanned: u64 = 0;
pub var search_blocks_early_pruned: u64 = 0;
pub var search_sift_ins: u64 = 0;

var start_ns: u64 = 0;

pub fn init() void {
    if (comptime !enabled) return;
    start_ns = now_ns();
}

/// CLOCK_MONOTONIC ns. On Linux uses the VDSO fast path (no syscall). On
/// other targets falls back to libc clock_gettime through std.c.
pub inline fn now_ns() u64 {
    if (comptime !enabled) return 0;
    if (comptime builtin.os.tag == .linux) {
        var ts: std.os.linux.timespec = undefined;
        _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
    }
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(@enumFromInt(0), &ts); // CLOCK_REALTIME fallback
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub inline fn observe_since(hist: *Hist, t0: u64) void {
    if (comptime !enabled) return;
    const t1 = now_ns();
    const d = if (t1 >= t0) t1 - t0 else 0;
    hist.observe(d);
}

pub inline fn inc(counter: *u64, by: u64) void {
    if (comptime !enabled) return;
    counter.* +%= by;
}

// ─── /__metrics report ────────────────────────────────────────────────────

pub const REPORT_BUF_SIZE: usize = 16384;

pub fn render(buf: []u8) []const u8 {
    if (comptime !enabled) {
        const msg = "instrument off — rebuild with -Dinstrument=true\n";
        const n = @min(buf.len, msg.len);
        @memcpy(buf[0..n], msg[0..n]);
        return buf[0..n];
    }
    var w = std.Io.Writer.fixed(buf);
    const uptime_ns = now_ns() -% start_ns;

    w.print("uptime_ns {d}\n", .{uptime_ns}) catch {};

    w.print("req_total {d}\n", .{req_total}) catch {};
    w.print("req_parse_err {d}\n", .{req_parse_err}) catch {};
    w.print("req_vectorize_err {d}\n", .{req_vectorize_err}) catch {};

    w.print("epoll_wakeups {d}\n", .{epoll_wakeups}) catch {};
    w.print("epoll_events {d}\n", .{epoll_events}) catch {};
    w.print("accepts {d}\n", .{accepts}) catch {};
    w.print("conn_closes {d}\n", .{conn_closes}) catch {};
    w.print("read_eagain {d}\n", .{read_eagain}) catch {};
    w.print("write_eagain {d}\n", .{write_eagain}) catch {};
    w.print("partial_writes {d}\n", .{partial_writes}) catch {};

    w.print("head_fast {d}\n", .{head_fast}) catch {};
    w.print("head_slow {d}\n", .{head_slow}) catch {};

    w.print("search_clusters_probed {d}\n", .{search_clusters_probed}) catch {};
    w.print("search_clusters_bbox_skipped {d}\n", .{search_clusters_bbox_skipped}) catch {};
    w.print("search_clusters_bbox_scanned {d}\n", .{search_clusters_bbox_scanned}) catch {};
    w.print("search_blocks_scanned {d}\n", .{search_blocks_scanned}) catch {};
    w.print("search_blocks_early_pruned {d}\n", .{search_blocks_early_pruned}) catch {};
    w.print("search_sift_ins {d}\n", .{search_sift_ins}) catch {};

    write_hist(&w, "hist_total_ns", &hist_total);
    write_hist(&w, "hist_parse_ns", &hist_parse);
    write_hist(&w, "hist_vectorize_ns", &hist_vectorize);
    write_hist(&w, "hist_search_ns", &hist_search);
    write_hist(&w, "hist_write_ns", &hist_write);

    write_buckets(&w, "buckets_total_ns", &hist_total);
    write_buckets(&w, "buckets_parse_ns", &hist_parse);
    write_buckets(&w, "buckets_vectorize_ns", &hist_vectorize);
    write_buckets(&w, "buckets_search_ns", &hist_search);
    write_buckets(&w, "buckets_write_ns", &hist_write);

    return w.buffered();
}

fn write_buckets(w: *std.Io.Writer, name: []const u8, h: *const Hist) void {
    // Emit only non-empty buckets so the probe can diff window-over-window
    // and recompute windowed quantiles without parsing 64 zeros per histogram.
    w.print("{s}", .{name}) catch {};
    var b: usize = 0;
    while (b < N_BUCKETS) : (b += 1) {
        if (h.buckets[b] != 0) {
            w.print(" b{d}={d}", .{ b, h.buckets[b] }) catch {};
        }
    }
    w.print("\n", .{}) catch {};
}

fn write_hist(w: *std.Io.Writer, name: []const u8, h: *const Hist) void {
    const mean: u64 = if (h.count == 0) 0 else h.sum_ns / h.count;
    w.print(
        "{s} count={d} mean={d} p50={d} p90={d} p99={d} p999={d} max={d}\n",
        .{
            name,
            h.count,
            mean,
            h.quantile(0.50),
            h.quantile(0.90),
            h.quantile(0.99),
            h.quantile(0.999),
            h.max_ns,
        },
    ) catch {};
}

// ─── tests ─────────────────────────────────────────────────────────────────

test "Hist: observe + quantile rough order" {
    var h: Hist = .{};
    var v: u64 = 1;
    while (v <= 1024) : (v <<= 1) h.observe(v);
    try std.testing.expectEqual(@as(u64, 11), h.count);
    try std.testing.expect(h.quantile(0.99) >= 1024);
}

test "Hist: empty" {
    const h: Hist = .{};
    try std.testing.expectEqual(@as(u64, 0), h.quantile(0.5));
}
