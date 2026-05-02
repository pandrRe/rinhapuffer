//! Cosine top-K search over a column-major SoA `Dataset`.
//!
//! Input invariant: dataset rows are L2-normalized (set up by
//! `transform_reference.parse_into`). Given that, cosine top-K reduces to
//! plain dot-product top-K — `|q|` is the same scalar for every row, so
//! dividing by it is order-preserving and the top-K *indices* are identical.
//! Score values stored internally are raw dot products, not cosines.
//!
//! Strategy: W rows in parallel via `@Vector(W, f32)`. For each batch the inner
//! loop is `inline for (0..N_FEATURES)` of FMAs against query lanes broadcast
//! once into vectors. A small ascending-sorted top-K array tracks the best
//! matches seen so far — insertion-sift on each lane that beats the current
//! minimum. Tail rows (`n % W != 0`) fall back to a scalar pass.

const std = @import("std");
const transform_reference = @import("transform_reference.zig");

pub const N_FEATURES: usize = transform_reference.N_FEATURES;
pub const TOP_K: usize = 5;
const W: usize = 8;

const Vec = @Vector(W, f32);
const Vu16 = @Vector(W, u16);

/// Find the indices of the `TOP_K` rows with the highest cosine similarity to `q`.
///
/// `out[0]` is the closest row, `out[TOP_K - 1]` the K-th. Score ties are
/// broken by lower row index winning (insertion-order — first to reach a
/// given score keeps its slot).
///
/// Requires `ds.n >= TOP_K` and dataset rows L2-normalized (the post-condition
/// of `transform_reference.parse_into`). The query `q` may have any norm; the
/// returned indices are correct cosine top-K since `|q|` is constant across
/// the scan and dividing by it is order-preserving.
pub fn cosine_topk(
    ds: transform_reference.Dataset,
    q: *const [N_FEATURES]f32,
    out: *[TOP_K]u32,
) void {
    std.debug.assert(ds.n >= TOP_K);

    var q_vec: [N_FEATURES]Vec = undefined;
    inline for (0..N_FEATURES) |k| q_vec[k] = @splat(q[k]);

    // Top-K kept ascending: top_scores[0] is the smallest of the current best.
    // Initialise with -inf so the first TOP_K rows always displace the sentinel.
    var top_scores: [TOP_K]f32 = @splat(-std.math.inf(f32));
    var top_rows: [TOP_K]u32 = @splat(0);

    const n = ds.n;
    const features = ds.features;

    var row: usize = 0;
    while (row + W <= n) : (row += W) {
        var dot: Vec = @splat(0);
        inline for (0..N_FEATURES) |k| {
            const r_chunk: Vec = features[k * n + row ..][0..W].*;
            dot = @mulAdd(Vec, q_vec[k], r_chunk, dot);
        }

        inline for (0..W) |lane| {
            const s = dot[lane];
            if (s > top_scores[0]) {
                sift_in(&top_scores, &top_rows, s, @intCast(row + lane));
            }
        }
    }

    while (row < n) : (row += 1) {
        var dot: f32 = 0;
        inline for (0..N_FEATURES) |k| {
            dot = @mulAdd(f32, q[k], features[k * n + row], dot);
        }
        if (dot > top_scores[0]) {
            sift_in(&top_scores, &top_rows, dot, @intCast(row));
        }
    }

    // Reverse: emit descending by score so out[0] is the best match.
    inline for (0..TOP_K) |i| out[i] = top_rows[TOP_K - 1 - i];
}

/// Cosine top-K over a `QuantizedDataset` (u16 features, per-feature
/// `(min, inv_scale)`). Algebraically equivalent to `cosine_topk`:
///
/// ```
/// dot = Σ q[k] * (r_u16[k] * inv_scale[k] + min[k])
///     = Σ q[k] * min[k]                 // const_q, computed once per query
///       + Σ (q[k] * inv_scale[k]) * r_u16[k]   // q_eff[k] folded once per query
/// ```
///
/// So the inner per-row work is one `@floatFromInt(Vu16) -> Vec` convert and
/// one FMA per feature — no per-row affine reconstruction.
pub fn cosine_topk_q(
    qds: transform_reference.QuantizedDataset,
    q: *const [N_FEATURES]f32,
    out: *[TOP_K]u32,
) void {
    std.debug.assert(qds.n >= TOP_K);

    var const_q: f32 = 0;
    var q_eff: [N_FEATURES]Vec = undefined;
    inline for (0..N_FEATURES) |k| {
        const_q += q[k] * qds.mins[k];
        q_eff[k] = @splat(q[k] * qds.inv_scales[k]);
    }

    var top_scores: [TOP_K]f32 = @splat(-std.math.inf(f32));
    var top_rows: [TOP_K]u32 = @splat(0);

    const n = qds.n;
    const features = qds.features;

    var row: usize = 0;
    while (row + W <= n) : (row += W) {
        var dot: Vec = @splat(const_q);
        inline for (0..N_FEATURES) |k| {
            const r_u16: Vu16 = features[k * n + row ..][0..W].*;
            const r_f32: Vec = @floatFromInt(r_u16);
            dot = @mulAdd(Vec, q_eff[k], r_f32, dot);
        }

        inline for (0..W) |lane| {
            const s = dot[lane];
            if (s > top_scores[0]) {
                sift_in(&top_scores, &top_rows, s, @intCast(row + lane));
            }
        }
    }

    while (row < n) : (row += 1) {
        var dot: f32 = const_q;
        inline for (0..N_FEATURES) |k| {
            const r_f32: f32 = @floatFromInt(features[k * n + row]);
            dot = @mulAdd(f32, q[k] * qds.inv_scales[k], r_f32, dot);
        }
        if (dot > top_scores[0]) {
            sift_in(&top_scores, &top_rows, dot, @intCast(row));
        }
    }

    inline for (0..TOP_K) |i| out[i] = top_rows[TOP_K - 1 - i];
}

/// Replace the smallest element of an ascending-sorted top-K with
/// `(new_score, new_row)`, then bubble it up to its correct slot.
inline fn sift_in(
    scores: *[TOP_K]f32,
    rows: *[TOP_K]u32,
    new_score: f32,
    new_row: u32,
) void {
    scores[0] = new_score;
    rows[0] = new_row;
    inline for (0..TOP_K - 1) |i| {
        if (scores[i + 1] >= scores[i]) break;
        const ts = scores[i];
        scores[i] = scores[i + 1];
        scores[i + 1] = ts;
        const tr = rows[i];
        rows[i] = rows[i + 1];
        rows[i + 1] = tr;
    }
}

/// Generic ascending top-N sift-in. Used for PROBE-cluster selection (N=8).
inline fn sift_in_n(
    comptime N: usize,
    scores: *[N]f32,
    ids: *[N]u32,
    new_score: f32,
    new_id: u32,
) void {
    scores[0] = new_score;
    ids[0] = new_id;
    inline for (0..N - 1) |i| {
        if (scores[i + 1] >= scores[i]) break;
        const ts = scores[i];
        scores[i] = scores[i + 1];
        scores[i + 1] = ts;
        const tr = ids[i];
        ids[i] = ids[i + 1];
        ids[i + 1] = tr;
    }
}

/// Stack-buffer cap. 4 KB at K=1024.
const MAX_K_CLUSTERS: usize = 1024;
const PROBE_CLUSTERS: usize = 8;

/// Production IVF cosine top-K. Score query against all centroids, pick the
/// top-`PROBE_CLUSTERS` by score, scan only those slabs with the Phase 4
/// dot-product algebra.
pub fn cosine_topk_q_ivf(
    qds: transform_reference.IvfQuantizedDataset,
    q: *const [N_FEATURES]f32,
    out: *[TOP_K]u32,
) void {
    std.debug.assert(qds.n >= TOP_K);
    std.debug.assert(qds.k_clusters <= MAX_K_CLUSTERS);
    std.debug.assert(qds.k_clusters >= PROBE_CLUSTERS);

    // Step 1: dot(q, centroid[c]) for every c.
    var centroid_scores: [MAX_K_CLUSTERS]f32 = undefined;
    for (0..qds.k_clusters) |c| {
        var dot: f32 = 0;
        inline for (0..N_FEATURES) |k| {
            dot = @mulAdd(f32, q[k], qds.centroids[c * N_FEATURES + k], dot);
        }
        centroid_scores[c] = dot;
    }

    // Step 2: top-PROBE_CLUSTERS clusters by score (ascending sift).
    var probe_scores: [PROBE_CLUSTERS]f32 = @splat(-std.math.inf(f32));
    var probe_clusters: [PROBE_CLUSTERS]u32 = @splat(0);
    for (0..qds.k_clusters) |c| {
        if (centroid_scores[c] > probe_scores[0]) {
            sift_in_n(PROBE_CLUSTERS, &probe_scores, &probe_clusters, centroid_scores[c], @intCast(c));
        }
    }

    // Step 3: scan rows in selected clusters.
    scan_clusters(qds, q, probe_clusters[0..], out);
}

/// Test-only: scan every cluster (PROBE = K). Used by `dataset_blob.zig`
/// equivalence tests against `cosine_topk_q` with K small enough that PROBE
/// would be ≥ K. Iterates `qds.cluster_starts` directly so it's correct for
/// any `k_clusters`.
pub fn cosine_topk_q_ivf_full(
    qds: transform_reference.IvfQuantizedDataset,
    q: *const [N_FEATURES]f32,
    out: *[TOP_K]u32,
) void {
    std.debug.assert(qds.n >= TOP_K);

    var const_q: f32 = 0;
    var q_eff: [N_FEATURES]Vec = undefined;
    inline for (0..N_FEATURES) |k| {
        const_q += q[k] * qds.mins[k];
        q_eff[k] = @splat(q[k] * qds.inv_scales[k]);
    }

    var top_scores: [TOP_K]f32 = @splat(-std.math.inf(f32));
    var top_rows: [TOP_K]u32 = @splat(0);

    const n = qds.n;
    const features = qds.features;

    for (0..qds.k_clusters) |c| {
        const start = qds.cluster_starts[c];
        const end = qds.cluster_starts[c + 1];
        scan_one(features, n, start, end, const_q, q_eff, q, qds.inv_scales, &top_scores, &top_rows);
    }

    inline for (0..TOP_K) |i| out[i] = top_rows[TOP_K - 1 - i];
}

inline fn scan_clusters(
    qds: transform_reference.IvfQuantizedDataset,
    q: *const [N_FEATURES]f32,
    cluster_ids: []const u32,
    out: *[TOP_K]u32,
) void {
    var const_q: f32 = 0;
    var q_eff: [N_FEATURES]Vec = undefined;
    inline for (0..N_FEATURES) |k| {
        const_q += q[k] * qds.mins[k];
        q_eff[k] = @splat(q[k] * qds.inv_scales[k]);
    }

    var top_scores: [TOP_K]f32 = @splat(-std.math.inf(f32));
    var top_rows: [TOP_K]u32 = @splat(0);

    const n = qds.n;
    const features = qds.features;

    for (cluster_ids) |c| {
        const start = qds.cluster_starts[c];
        const end = qds.cluster_starts[c + 1];
        scan_one(features, n, start, end, const_q, q_eff, q, qds.inv_scales, &top_scores, &top_rows);
    }

    inline for (0..TOP_K) |i| out[i] = top_rows[TOP_K - 1 - i];
}

inline fn scan_one(
    features: []const u16,
    n: usize,
    start: u32,
    end: u32,
    const_q: f32,
    q_eff: [N_FEATURES]Vec,
    q: *const [N_FEATURES]f32,
    inv_scales: [N_FEATURES]f32,
    top_scores: *[TOP_K]f32,
    top_rows: *[TOP_K]u32,
) void {
    var row: usize = start;
    while (row + W <= end) : (row += W) {
        var dot: Vec = @splat(const_q);
        inline for (0..N_FEATURES) |k| {
            const r_u16: Vu16 = features[k * n + row ..][0..W].*;
            const r_f32: Vec = @floatFromInt(r_u16);
            dot = @mulAdd(Vec, q_eff[k], r_f32, dot);
        }
        inline for (0..W) |lane| {
            const s = dot[lane];
            if (s > top_scores[0]) sift_in(top_scores, top_rows, s, @intCast(row + lane));
        }
    }
    while (row < end) : (row += 1) {
        var dot: f32 = const_q;
        inline for (0..N_FEATURES) |k| {
            const r_f32: f32 = @floatFromInt(features[k * n + row]);
            dot = @mulAdd(f32, q[k] * inv_scales[k], r_f32, dot);
        }
        if (dot > top_scores[0]) sift_in(top_scores, top_rows, dot, @intCast(row));
    }
}

// ─── tests ──────────────────────────────────────────────────────────────────

const fast_json = @import("fast_json.zig");

test "cosine_topk hand-built tiny dataset" {
    // 5 rows, 14 features, column-major. Query [1,0,0,...,0].
    // Each row is a unit vector with a chosen cosine vs query.
    const n: usize = 5;
    const expected_cos = [5]f32{ 1.0, 0.9, 0.5, 0.0, -1.0 };

    var features: [N_FEATURES * n]f32 = @splat(0);
    var labels: [n]bool = @splat(false);

    // Place feature 0 = expected_cos[row], feature 1 = sqrt(1 - cos²).
    // This gives unit-norm rows, so dot vs q = feature 0 directly.
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
    cosine_topk(ds, &q, &out);

    // Highest-cosine row first.
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4 }, &out);
}

test "cosine_topk SIMD-tail: rows just past a W boundary" {
    // n = 10 = W + 2 → exercises both the SIMD batch and the scalar tail.
    // Construct unit-norm rows with strictly decreasing cosine vs q=[1,0,...]:
    // feature_0 = c, feature_1 = sqrt(1 - c²) → dot(q, r) = c.
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
    cosine_topk(ds, &q, &out);

    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4 }, &out);
}

/// Naive O(n log n) reference: compute every cosine, sort all, return top-K.
/// Computes `|r|` independently in f64 — does **not** trust the dataset's
/// unit-norm post-condition. That way this differential test catches
/// regressions in either `parse_into`'s normalization or `cosine_topk`'s math.
fn naive_cosine_topk(
    allocator: std.mem.Allocator,
    ds: transform_reference.Dataset,
    q: *const [N_FEATURES]f32,
) ![TOP_K]u32 {
    const Score = struct { score: f32, row: u32 };
    const scores = try allocator.alloc(Score, ds.n);
    defer allocator.free(scores);

    var q_sum_sq: f64 = 0;
    for (q) |v| q_sum_sq += @as(f64, v) * @as(f64, v);
    const q_norm: f64 = @sqrt(q_sum_sq);

    for (0..ds.n) |row| {
        var dot: f64 = 0;
        var r_sum_sq: f64 = 0;
        for (0..N_FEATURES) |k| {
            const v: f64 = ds.features[k * ds.n + row];
            dot += v * @as(f64, q[k]);
            r_sum_sq += v * v;
        }
        const r_norm: f64 = @sqrt(r_sum_sq);
        const s: f32 = @floatCast(dot / (q_norm * r_norm));
        scores[row] = .{ .score = s, .row = @intCast(row) };
    }

    const cmp = struct {
        fn lt(_: void, a: Score, b: Score) bool {
            // Descending by score; ties broken by row asc to be deterministic.
            if (a.score != b.score) return a.score > b.score;
            return a.row < b.row;
        }
    }.lt;
    std.mem.sort(Score, scores, {}, cmp);

    var out: [TOP_K]u32 = undefined;
    for (0..TOP_K) |i| out[i] = scores[i].row;
    return out;
}

test "cosine_topk vs naive on example-references.json" {
    const allocator = std.testing.allocator;

    var mapped = try fast_json.mmap_file("./resources/example-references.json");
    defer mapped.deinit();

    const n = transform_reference.count_records(mapped.bytes);
    const features = try allocator.alloc(f32, N_FEATURES * n);
    defer allocator.free(features);
    const labels = try allocator.alloc(bool, n);
    defer allocator.free(labels);

    const ds = try transform_reference.parse_into(mapped.bytes, features, labels);

    // A handful of arbitrary queries. Use rows of the dataset itself as
    // queries (guaranteed non-zero) plus some constructed ones. With 14d
    // floats and 100 records the odds of an exact tie at top-K are nil.
    var queries: [8][N_FEATURES]f32 = undefined;
    // 4 dataset rows verbatim.
    inline for (0..4) |i| {
        for (0..N_FEATURES) |c| queries[i][c] = ds.features[c * n + i * 17];
    }
    // 4 hand-rolled queries.
    queries[4] = .{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    queries[5] = .{ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 };
    queries[6] = .{ 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1 };
    queries[7] = .{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4 };

    for (&queries, 0..) |*q, qi| {
        var got: [TOP_K]u32 = undefined;
        cosine_topk(ds, q, &got);
        const want = try naive_cosine_topk(allocator, ds, q);

        std.testing.expectEqualSlices(u32, &want, &got) catch |err| {
            std.debug.print("query #{d}: got {any}, want {any}\n", .{ qi, got, want });
            return err;
        };
    }
}

test "cosine_topk_q_ivf hand-built tiny clustered dataset" {
    // 10 rows in 14 features. Rows 0..4 cluster around +e0, rows 5..9 around +e1.
    // K=2, PROBE=8 (capped to k_clusters=2 effectively, but we use _full for clarity).
    // Query toward +e0 must return rows 0..4 (the cluster near +e0).
    const n: usize = 10;
    const k_clusters: usize = 2;

    // Build f32 unit-norm rows.
    var f32_features: [N_FEATURES * n]f32 = @splat(0);
    var labels: [n]bool = @splat(false);
    for (0..5) |row| {
        f32_features[0 * n + row] = 1.0;
        f32_features[2 * n + row] = 0.001 * @as(f32, @floatFromInt(row));
        // Renormalize.
        var sum_sq: f32 = 0;
        for (0..N_FEATURES) |k| sum_sq += f32_features[k * n + row] * f32_features[k * n + row];
        const inv = 1.0 / @sqrt(sum_sq);
        for (0..N_FEATURES) |k| f32_features[k * n + row] *= inv;
    }
    for (5..10) |row| {
        f32_features[1 * n + row] = 1.0;
        f32_features[3 * n + row] = 0.001 * @as(f32, @floatFromInt(row));
        var sum_sq: f32 = 0;
        for (0..N_FEATURES) |k| sum_sq += f32_features[k * n + row] * f32_features[k * n + row];
        const inv = 1.0 / @sqrt(sum_sq);
        for (0..N_FEATURES) |k| f32_features[k * n + row] *= inv;
    }

    // Quantize per-column.
    var mins: [N_FEATURES]f32 = undefined;
    var inv_scales: [N_FEATURES]f32 = undefined;
    var u16_features: [N_FEATURES * n]u16 = @splat(0);
    for (0..N_FEATURES) |k| {
        var lo: f32 = f32_features[k * n + 0];
        var hi: f32 = f32_features[k * n + 0];
        for (1..n) |r| {
            const v = f32_features[k * n + r];
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        const range = hi - lo;
        const scale: f32 = if (range > 0) 65535.0 / range else 0.0;
        mins[k] = lo;
        inv_scales[k] = if (scale != 0) 1.0 / scale else 0.0;
        for (0..n) |r| {
            const f = f32_features[k * n + r];
            const q = @round((f - lo) * scale);
            const clamped = @max(0.0, @min(65535.0, q));
            u16_features[k * n + r] = @intFromFloat(clamped);
        }
    }

    // Centroids: hand-pick (would be the kmeans output on this layout).
    // Centroid 0 = [1, 0, 0, ...], Centroid 1 = [0, 1, 0, ...].
    var centroids: [k_clusters * N_FEATURES]f32 = @splat(0);
    centroids[0 * N_FEATURES + 0] = 1.0;
    centroids[1 * N_FEATURES + 1] = 1.0;

    // Cluster starts: rows 0..4 in cluster 0, rows 5..9 in cluster 1.
    const cluster_starts = [_]u32{ 0, 5, 10 };

    const qds: transform_reference.IvfQuantizedDataset = .{
        .n = n,
        .k_clusters = k_clusters,
        .features = &u16_features,
        .labels = &labels,
        .centroids = &centroids,
        .cluster_starts = &cluster_starts,
        .mins = mins,
        .inv_scales = inv_scales,
    };

    // Query toward +e0 — should return rows 0..4 (cluster 0 wins).
    var q: [N_FEATURES]f32 = @splat(0);
    q[0] = 1.0;

    var out: [TOP_K]u32 = undefined;
    cosine_topk_q_ivf_full(qds, &q, &out);

    // The 5 rows of cluster 0 (rows 0..4) — order by score descending. Row 0
    // has 0 jitter on f2 so it's exactly e0 → highest cosine. Rows 1..4 have
    // tiny jitter pulling them slightly off e0, so cosine decreases as jitter
    // grows.
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4 }, &out);
}

test "cosine_topk_q hand-built tiny dataset" {
    // Same setup as `cosine_topk hand-built tiny dataset`, then quantize the
    // f32 columns into a QuantizedDataset built directly (no blob round-trip).
    const n: usize = 5;
    const expected_cos = [5]f32{ 1.0, 0.9, 0.5, 0.0, -1.0 };

    var f32_features: [N_FEATURES * n]f32 = @splat(0);
    var labels: [n]bool = @splat(false);

    for (expected_cos, 0..) |c, row| {
        f32_features[0 * n + row] = c;
        f32_features[1 * n + row] = @sqrt(@max(0.0, 1.0 - c * c));
    }

    // Per-column min/scale, then quantize. Mirrors `dataset_blob.compute_quant_params`
    // + the encode formula in `dataset_blob.write` — kept inline here so this test
    // doesn't drag in `dataset_blob.zig` (which would create a cycle).
    var mins: [N_FEATURES]f32 = undefined;
    var inv_scales: [N_FEATURES]f32 = undefined;
    var u16_features: [N_FEATURES * n]u16 = @splat(0);
    for (0..N_FEATURES) |k| {
        var lo: f32 = f32_features[k * n + 0];
        var hi: f32 = f32_features[k * n + 0];
        for (1..n) |r| {
            const v = f32_features[k * n + r];
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        const range = hi - lo;
        const scale: f32 = if (range > 0) 65535.0 / range else 0.0;
        mins[k] = lo;
        inv_scales[k] = if (scale != 0) 1.0 / scale else 0.0;
        for (0..n) |r| {
            const f = f32_features[k * n + r];
            const q = @round((f - lo) * scale);
            const clamped = @max(0.0, @min(65535.0, q));
            u16_features[k * n + r] = @intFromFloat(clamped);
        }
    }

    const qds: transform_reference.QuantizedDataset = .{
        .n = n,
        .features = &u16_features,
        .labels = &labels,
        .mins = mins,
        .inv_scales = inv_scales,
    };
    var q: [N_FEATURES]f32 = @splat(0);
    q[0] = 1.0;

    var out: [TOP_K]u32 = undefined;
    cosine_topk_q(qds, &q, &out);

    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4 }, &out);
}
