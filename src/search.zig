//! Euclidean top-K search using fixed-point i16 storage with a single global
//! scale (`FIX_SCALE`).
//!
//! Storage: `int_value = round(float_value * FIX_SCALE)` per feature, written
//! once at prep time. With `FIX_SCALE = 10000`, values in [−1, 1] map to
//! [−10000, 10000] — well inside i16's [−32768, 32767]. Diff range
//! [−20000, 20000] fits i16 too, so subtraction stays in i16 without
//! widening; only the squared-difference accumulator needs to widen.
//!
//! Hot path is **fully integer**: `Σ (q_i − r_i)²`. Order-preserving vs the
//! true float distance (both are scaled by `FIX_SCALE²` per feature), so
//! ranking is exact. The query is quantized once per request; nothing else
//! is precomputed and nothing is dequantized per row.
//!
//! Strategy: W=8 rows in parallel via `@Vector(W, i16)`. Per (row chunk,
//! feature): one i16 load, one i16 sub (no overflow at FIX_SCALE=10000), one
//! i16 → i32 widen for the square, one i32 mul, one i32 → i64 widen + add to
//! the accumulator. 14 features × 8 rows; tail rows handled scalar with the
//! same algebra.

const std = @import("std");
const transform_reference = @import("transform_reference.zig");

pub const N_FEATURES: usize = transform_reference.N_FEATURES;
pub const TOP_K: usize = 5;

/// Global fixed-point scale. Persisted in the v5 blob header so a stale blob
/// built against a different value is rejected at load time.
pub const FIX_SCALE: i32 = 10000;

const W: usize = 8;
const Vec = @Vector(W, f32); // f32 brute-force only
const Vi16 = @Vector(W, i16);
const Vi32 = @Vector(W, i32);
const Vi64 = @Vector(W, i64);

/// Stack-buffer cap. 4 KB at K=1024.
const MAX_K_CLUSTERS: usize = 1024;
const PROBE_CLUSTERS: usize = 8;

/// Quantize a 14-feature float query to i16 once per request.
inline fn quantize_query(q: *const [N_FEATURES]f32, out: *[N_FEATURES]i16) void {
    const fix_scale_f: f32 = @floatFromInt(FIX_SCALE);
    inline for (0..N_FEATURES) |k| {
        out[k] = @intFromFloat(@round(q[k] * fix_scale_f));
    }
}

// ─── public search API ────────────────────────────────────────────────────

/// Find the indices of the `TOP_K` rows nearest to `q` in raw-feature
/// Euclidean distance, against a non-quantized f32 dataset.
///
/// Test/reference path — production uses `euclidean_topk_q_ivf`.
pub fn euclidean_topk(
    ds: transform_reference.Dataset,
    q: *const [N_FEATURES]f32,
    out: *[TOP_K]u32,
) void {
    std.debug.assert(ds.n >= TOP_K);

    var q_vec: [N_FEATURES]Vec = undefined;
    inline for (0..N_FEATURES) |k| q_vec[k] = @splat(q[k]);

    // Top-K kept descending by distance: top_dists[0] is the LARGEST current.
    var top_dists: [TOP_K]f32 = @splat(std.math.inf(f32));
    var top_rows: [TOP_K]u32 = @splat(0);

    const n = ds.n;
    const features = ds.features;

    var row: usize = 0;
    while (row + W <= n) : (row += W) {
        var dist: Vec = @splat(0);
        inline for (0..N_FEATURES) |k| {
            const r_chunk: Vec = features[k * n + row ..][0..W].*;
            const diff = q_vec[k] - r_chunk;
            dist += diff * diff;
        }
        inline for (0..W) |lane| {
            const d = dist[lane];
            if (d < top_dists[0]) {
                sift_in_min_f32(&top_dists, &top_rows, d, @intCast(row + lane));
            }
        }
    }

    while (row < n) : (row += 1) {
        var dist: f32 = 0;
        inline for (0..N_FEATURES) |k| {
            const v = features[k * n + row];
            const diff = q[k] - v;
            dist += diff * diff;
        }
        if (dist < top_dists[0]) {
            sift_in_min_f32(&top_dists, &top_rows, dist, @intCast(row));
        }
    }

    // Reverse: emit ascending by distance so out[0] is the closest.
    inline for (0..TOP_K) |i| out[i] = top_rows[TOP_K - 1 - i];
}

/// Brute-force int top-K over an i16-quantized dataset.
pub fn euclidean_topk_q(
    qds: transform_reference.QuantizedDataset,
    q: *const [N_FEATURES]f32,
    out: *[TOP_K]u32,
) void {
    std.debug.assert(qds.n >= TOP_K);

    var q_int: [N_FEATURES]i16 = undefined;
    quantize_query(q, &q_int);

    var top_dists: [TOP_K]i64 = @splat(std.math.maxInt(i64));
    var top_rows: [TOP_K]u32 = @splat(0);

    scan_range_int(qds.features, qds.n, 0, @intCast(qds.n), &q_int, &top_dists, &top_rows);

    inline for (0..TOP_K) |i| out[i] = top_rows[TOP_K - 1 - i];
}

/// Production IVF Euclidean top-K — **exact**, regardless of PROBE.
///
/// Float Euclidean against the centroids picks `PROBE_CLUSTERS` for the
/// initial int scan (tight upper bound on the K-th best distance). Then a
/// **bbox repair pass** walks every other cluster: for each, compute the
/// axis-aligned lower bound `LB² = Σ_k max(0, max(q − hi, lo − q))²` from
/// the cluster's per-feature `[lo, hi]`. If `LB² ≥ top_dists[0]`, no point
/// in the cluster can improve us — skip. Otherwise scan it. By construction
/// every cluster containing a true top-K neighbour passes the prune (its
/// neighbour's true distance ≤ current K-th best ≤ LB), so the result is
/// exact.
pub fn euclidean_topk_q_ivf(
    qds: transform_reference.IvfQuantizedDataset,
    q: *const [N_FEATURES]f32,
    out: *[TOP_K]u32,
) void {
    std.debug.assert(qds.n >= TOP_K);
    std.debug.assert(qds.k_clusters <= MAX_K_CLUSTERS);
    std.debug.assert(qds.k_clusters >= PROBE_CLUSTERS);

    // Step 1: float Euclidean to every centroid.
    var centroid_dists: [MAX_K_CLUSTERS]f32 = undefined;
    for (0..qds.k_clusters) |c| {
        var d: f32 = 0;
        inline for (0..N_FEATURES) |k| {
            const diff = q[k] - qds.centroids[c * N_FEATURES + k];
            d = @mulAdd(f32, diff, diff, d);
        }
        centroid_dists[c] = d;
    }

    // Step 2: pick top PROBE_CLUSTERS clusters by smallest distance.
    var probe_dists: [PROBE_CLUSTERS]f32 = @splat(std.math.inf(f32));
    var probe_clusters: [PROBE_CLUSTERS]u32 = @splat(0);
    for (0..qds.k_clusters) |c| {
        if (centroid_dists[c] < probe_dists[0]) {
            sift_in_n_min_f32(PROBE_CLUSTERS, &probe_dists, &probe_clusters, centroid_dists[c], @intCast(c));
        }
    }

    // Step 3: int Euclidean over rows in selected clusters → tight top-K bound.
    var q_int: [N_FEATURES]i16 = undefined;
    quantize_query(q, &q_int);

    var top_dists: [TOP_K]i64 = @splat(std.math.maxInt(i64));
    var top_rows: [TOP_K]u32 = @splat(0);

    var probed: [MAX_K_CLUSTERS]bool = @splat(false);
    for (probe_clusters[0..]) |c| {
        probed[c] = true;
        const start = qds.cluster_starts[c];
        const end = qds.cluster_starts[c + 1];
        scan_range_int(qds.features, qds.n, start, end, &q_int, &top_dists, &top_rows);
    }

    // Step 4: bbox repair pass over the remaining clusters. Most prune cheaply
    // (bbox LB > current K-th best); the few that survive get scanned to
    // guarantee exact top-K.
    for (0..qds.k_clusters) |c| {
        if (probed[c]) continue;
        const lb = bbox_lower_bound_sq(&q_int, qds.bbox_lo, qds.bbox_hi, c);
        if (lb >= top_dists[0]) continue;
        const start = qds.cluster_starts[c];
        const end = qds.cluster_starts[c + 1];
        scan_range_int(qds.features, qds.n, start, end, &q_int, &top_dists, &top_rows);
    }

    inline for (0..TOP_K) |i| out[i] = top_rows[TOP_K - 1 - i];
}

/// Axis-aligned squared distance from query to cluster `c`'s bounding box.
/// Per feature, contributes `(below_lo)²` if `q < lo`, `(above_hi)²` if
/// `q > hi`, else 0. Widening pattern matches `scan_range_int`.
inline fn bbox_lower_bound_sq(
    q_int: *const [N_FEATURES]i16,
    bbox_lo: []const i16,
    bbox_hi: []const i16,
    c: usize,
) i64 {
    var lb: i64 = 0;
    inline for (0..N_FEATURES) |k| {
        const lo: i32 = bbox_lo[c * N_FEATURES + k];
        const hi: i32 = bbox_hi[c * N_FEATURES + k];
        const qv: i32 = q_int[k];
        const below_lo: i32 = lo - qv;
        const above_hi: i32 = qv - hi;
        const d: i32 = @max(0, @max(below_lo, above_hi));
        lb += @as(i64, d) * @as(i64, d);
    }
    return lb;
}

/// Test-only: scan every cluster (PROBE = K). Used by `dataset_blob.zig`
/// equivalence tests against `euclidean_topk_q` with K small enough that PROBE
/// would be ≥ K. Iterates `qds.cluster_starts` directly so it's correct for
/// any `k_clusters`.
pub fn euclidean_topk_q_ivf_full(
    qds: transform_reference.IvfQuantizedDataset,
    q: *const [N_FEATURES]f32,
    out: *[TOP_K]u32,
) void {
    std.debug.assert(qds.n >= TOP_K);

    var q_int: [N_FEATURES]i16 = undefined;
    quantize_query(q, &q_int);

    var top_dists: [TOP_K]i64 = @splat(std.math.maxInt(i64));
    var top_rows: [TOP_K]u32 = @splat(0);

    for (0..qds.k_clusters) |c| {
        const start = qds.cluster_starts[c];
        const end = qds.cluster_starts[c + 1];
        scan_range_int(qds.features, qds.n, start, end, &q_int, &top_dists, &top_rows);
    }

    inline for (0..TOP_K) |i| out[i] = top_rows[TOP_K - 1 - i];
}

// ─── int inner loop ───────────────────────────────────────────────────────

inline fn scan_range_int(
    features: []const i16,
    n: usize,
    start: u32,
    end: u32,
    q_int: *const [N_FEATURES]i16,
    top_dists: *[TOP_K]i64,
    top_rows: *[TOP_K]u32,
) void {
    var q_vec: [N_FEATURES]Vi16 = undefined;
    inline for (0..N_FEATURES) |k| q_vec[k] = @splat(q_int[k]);

    var row: usize = start;
    while (row + W <= end) : (row += W) {
        var dist: Vi64 = @splat(0);
        inline for (0..N_FEATURES) |k| {
            const r_i16: Vi16 = features[k * n + row ..][0..W].*;
            const diff_i16: Vi16 = q_vec[k] - r_i16; // safe at FIX_SCALE=10000
            const diff_i32: Vi32 = diff_i16;
            const sq: Vi32 = diff_i32 * diff_i32;
            dist += @as(Vi64, sq);
        }
        inline for (0..W) |lane| {
            const d = dist[lane];
            if (d < top_dists[0]) {
                sift_in_min_i64(top_dists, top_rows, d, @intCast(row + lane));
            }
        }
    }
    while (row < end) : (row += 1) {
        var dist: i64 = 0;
        inline for (0..N_FEATURES) |k| {
            const diff: i32 = @as(i32, q_int[k]) - @as(i32, features[k * n + row]);
            dist += @as(i64, diff * diff);
        }
        if (dist < top_dists[0]) {
            sift_in_min_i64(top_dists, top_rows, dist, @intCast(row));
        }
    }
}

// ─── sift helpers ─────────────────────────────────────────────────────────
//
// Top-K kept descending — `dists[0]` is the LARGEST of the current best, so
// any new entry strictly smaller than it displaces it. After replace we
// bubble down (largest stays at index 0) until the new entry finds its slot.

inline fn sift_in_min_f32(
    dists: *[TOP_K]f32,
    rows: *[TOP_K]u32,
    new_dist: f32,
    new_row: u32,
) void {
    dists[0] = new_dist;
    rows[0] = new_row;
    inline for (0..TOP_K - 1) |i| {
        if (dists[i + 1] <= dists[i]) break;
        const td = dists[i];
        dists[i] = dists[i + 1];
        dists[i + 1] = td;
        const tr = rows[i];
        rows[i] = rows[i + 1];
        rows[i + 1] = tr;
    }
}

inline fn sift_in_min_i64(
    dists: *[TOP_K]i64,
    rows: *[TOP_K]u32,
    new_dist: i64,
    new_row: u32,
) void {
    dists[0] = new_dist;
    rows[0] = new_row;
    inline for (0..TOP_K - 1) |i| {
        if (dists[i + 1] <= dists[i]) break;
        const td = dists[i];
        dists[i] = dists[i + 1];
        dists[i + 1] = td;
        const tr = rows[i];
        rows[i] = rows[i + 1];
        rows[i + 1] = tr;
    }
}

inline fn sift_in_n_min_f32(
    comptime N: usize,
    dists: *[N]f32,
    ids: *[N]u32,
    new_dist: f32,
    new_id: u32,
) void {
    dists[0] = new_dist;
    ids[0] = new_id;
    inline for (0..N - 1) |i| {
        if (dists[i + 1] <= dists[i]) break;
        const td = dists[i];
        dists[i] = dists[i + 1];
        dists[i + 1] = td;
        const tr = ids[i];
        ids[i] = ids[i + 1];
        ids[i + 1] = tr;
    }
}

// ─── tests ─────────────────────────────────────────────────────────────────

const fast_json = @import("fast_json.zig");

/// Build an i16 quantized column-major dataset from f32 row data using the
/// global FIX_SCALE. Test-only helper.
fn quantize_dataset(comptime n: usize, src: *const [N_FEATURES * n]f32, dst: *[N_FEATURES * n]i16) void {
    const fix_scale_f: f32 = @floatFromInt(FIX_SCALE);
    const lo_clamp: f32 = @floatFromInt(std.math.minInt(i16));
    const hi_clamp: f32 = @floatFromInt(std.math.maxInt(i16));
    for (0..N_FEATURES * n) |i| {
        const q = @round(src[i] * fix_scale_f);
        dst[i] = @intFromFloat(@max(lo_clamp, @min(hi_clamp, q)));
    }
}

/// Compute per-(cluster, feature) min/max from an i16 column-major dataset.
/// Test-only helper mirroring `dataset_blob.write`'s bbox accumulation.
fn compute_bboxes(
    comptime n: usize,
    comptime k_clusters: usize,
    features: *const [N_FEATURES * n]i16,
    cluster_starts: *const [k_clusters + 1]u32,
    bbox_lo: *[k_clusters * N_FEATURES]i16,
    bbox_hi: *[k_clusters * N_FEATURES]i16,
) void {
    @memset(bbox_lo, std.math.maxInt(i16));
    @memset(bbox_hi, std.math.minInt(i16));
    for (0..k_clusters) |c| {
        const cs = cluster_starts[c];
        const ce = cluster_starts[c + 1];
        for (0..N_FEATURES) |k| {
            const slot = c * N_FEATURES + k;
            for (cs..ce) |row| {
                const v = features[k * n + row];
                if (v < bbox_lo[slot]) bbox_lo[slot] = v;
                if (v > bbox_hi[slot]) bbox_hi[slot] = v;
            }
        }
    }
}

test "euclidean_topk hand-built tiny dataset" {
    // 5 rows, 14 features, column-major. Query [1,0,0,...,0]. Use unit-norm
    // rows (their construction gives dist² = 2(1 − cos), which is monotonic in
    // cos, so the expected ranking matches the cosine ranking).
    const n: usize = 5;
    const expected_cos = [5]f32{ 1.0, 0.9, 0.5, 0.0, -1.0 };

    var features: [N_FEATURES * n]f32 = @splat(0);
    var labels: [n]bool = @splat(false);

    for (expected_cos, 0..) |c, row| {
        features[0 * n + row] = c;
        features[1 * n + row] = @sqrt(@max(0.0, 1.0 - c * c));
    }

    const ds: transform_reference.Dataset = .{
        .n = n,
        .features = &features,
        .labels = &labels,
    };
    var q: [N_FEATURES]f32 = @splat(0);
    q[0] = 1.0;

    var out: [TOP_K]u32 = undefined;
    euclidean_topk(ds, &q, &out);

    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4 }, &out);
}

test "euclidean_topk SIMD-tail: rows just past a W boundary" {
    // n = 10 = W + 2 → exercises both the SIMD batch and the scalar tail.
    const n: usize = 10;
    var features: [N_FEATURES * n]f32 = @splat(0);
    var labels: [n]bool = @splat(false);

    for (0..n) |row| {
        const c: f32 = @as(f32, @floatFromInt(n - 1 - row)) /
            @as(f32, @floatFromInt(n - 1));
        features[0 * n + row] = c;
        features[1 * n + row] = @sqrt(@max(0.0, 1.0 - c * c));
    }

    const ds: transform_reference.Dataset = .{
        .n = n,
        .features = &features,
        .labels = &labels,
    };
    var q: [N_FEATURES]f32 = @splat(0);
    q[0] = 1.0;

    var out: [TOP_K]u32 = undefined;
    euclidean_topk(ds, &q, &out);

    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4 }, &out);
}

/// Naive O(n log n) reference: compute every Euclidean distance² in f64 over
/// the raw features, sort ascending by distance, return top-K. Independent
/// implementation so the differential test catches any regression in
/// `euclidean_topk`'s SIMD math.
fn naive_euclidean_topk(
    allocator: std.mem.Allocator,
    ds: transform_reference.Dataset,
    q: *const [N_FEATURES]f32,
) ![TOP_K]u32 {
    const Score = struct { dist: f64, row: u32 };
    const scores = try allocator.alloc(Score, ds.n);
    defer allocator.free(scores);

    for (0..ds.n) |row| {
        var d: f64 = 0;
        for (0..N_FEATURES) |k| {
            const v: f64 = ds.features[k * ds.n + row];
            const diff = @as(f64, q[k]) - v;
            d += diff * diff;
        }
        scores[row] = .{ .dist = d, .row = @intCast(row) };
    }

    const cmp = struct {
        fn lt(_: void, a: Score, b: Score) bool {
            // Ascending by distance; ties broken by row asc to be deterministic.
            if (a.dist != b.dist) return a.dist < b.dist;
            return a.row < b.row;
        }
    }.lt;
    std.mem.sort(Score, scores, {}, cmp);

    var out: [TOP_K]u32 = undefined;
    for (0..TOP_K) |i| out[i] = scores[i].row;
    return out;
}

test "euclidean_topk vs naive on example-references.json" {
    const allocator = std.testing.allocator;

    var mapped = try fast_json.mmap_file("./resources/example-references.json");
    defer mapped.deinit();

    const n = transform_reference.count_records(mapped.bytes);
    const features = try allocator.alloc(f32, N_FEATURES * n);
    defer allocator.free(features);
    const labels = try allocator.alloc(bool, n);
    defer allocator.free(labels);

    const ds = try transform_reference.parse_into(mapped.bytes, features, labels);

    var queries: [8][N_FEATURES]f32 = undefined;
    inline for (0..4) |i| {
        for (0..N_FEATURES) |c| queries[i][c] = ds.features[c * n + i * 17];
    }
    queries[4] = .{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    queries[5] = .{ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 };
    queries[6] = .{ 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1 };
    queries[7] = .{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4 };

    for (&queries, 0..) |*q, qi| {
        var got: [TOP_K]u32 = undefined;
        euclidean_topk(ds, q, &got);
        const want = try naive_euclidean_topk(allocator, ds, q);

        std.testing.expectEqualSlices(u32, &want, &got) catch |err| {
            std.debug.print("query #{d}: got {any}, want {any}\n", .{ qi, got, want });
            return err;
        };
    }
}

test "euclidean_topk_q_ivf hand-built tiny clustered dataset" {
    // 10 rows in 14 features. Rows 0..4 cluster around +e0, rows 5..9 around +e1.
    // K=2; query toward +e0 must return rows 0..4 (cluster 0 wins).
    const n: usize = 10;
    const k_clusters: usize = 2;

    var f32_features: [N_FEATURES * n]f32 = @splat(0);
    var labels: [n]bool = @splat(false);
    for (0..5) |row| {
        f32_features[0 * n + row] = 1.0;
        f32_features[2 * n + row] = 0.001 * @as(f32, @floatFromInt(row));
    }
    for (5..10) |row| {
        f32_features[1 * n + row] = 1.0;
        f32_features[3 * n + row] = 0.001 * @as(f32, @floatFromInt(row));
    }

    var i16_features: [N_FEATURES * n]i16 = @splat(0);
    quantize_dataset(n, &f32_features, &i16_features);

    var centroids: [k_clusters * N_FEATURES]f32 = @splat(0);
    centroids[0 * N_FEATURES + 0] = 1.0;
    centroids[1 * N_FEATURES + 1] = 1.0;

    const cluster_starts = [_]u32{ 0, 5, 10 };

    var bbox_lo: [k_clusters * N_FEATURES]i16 = undefined;
    var bbox_hi: [k_clusters * N_FEATURES]i16 = undefined;
    compute_bboxes(n, k_clusters, &i16_features, &cluster_starts, &bbox_lo, &bbox_hi);

    const qds: transform_reference.IvfQuantizedDataset = .{
        .n = n,
        .k_clusters = k_clusters,
        .features = &i16_features,
        .labels = &labels,
        .centroids = &centroids,
        .cluster_starts = &cluster_starts,
        .bbox_lo = &bbox_lo,
        .bbox_hi = &bbox_hi,
    };

    var q: [N_FEATURES]f32 = @splat(0);
    q[0] = 1.0;

    var out: [TOP_K]u32 = undefined;
    euclidean_topk_q_ivf_full(qds, &q, &out);

    // Cluster 0's 5 rows ordered by distance: row 0 (no jitter) closest, 1..4
    // by jitter magnitude.
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4 }, &out);
}

test "euclidean_topk_q hand-built tiny dataset" {
    const n: usize = 5;
    const expected_cos = [5]f32{ 1.0, 0.9, 0.5, 0.0, -1.0 };

    var f32_features: [N_FEATURES * n]f32 = @splat(0);
    var labels: [n]bool = @splat(false);

    for (expected_cos, 0..) |c, row| {
        f32_features[0 * n + row] = c;
        f32_features[1 * n + row] = @sqrt(@max(0.0, 1.0 - c * c));
    }

    var i16_features: [N_FEATURES * n]i16 = @splat(0);
    quantize_dataset(n, &f32_features, &i16_features);

    const qds: transform_reference.QuantizedDataset = .{
        .n = n,
        .features = &i16_features,
        .labels = &labels,
    };
    var q: [N_FEATURES]f32 = @splat(0);
    q[0] = 1.0;

    var out: [TOP_K]u32 = undefined;
    euclidean_topk_q(qds, &q, &out);

    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4 }, &out);
}
