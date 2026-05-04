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
//! Strategy: W=16 rows in parallel via `@Vector(W, i16)` (256-bit YMM under
//! `-Dcpu=haswell`). Per (row chunk, feature): one i16 load, one i16 sub
//! (no overflow at FIX_SCALE=10000), one i16 → i32 widen for the square,
//! one i32 mul, one i32 → i64 widen + add to the accumulator. 14 features
//! × 16 rows. Cluster residue handled by a W=8 chunk (XMM) and then a
//! scalar tail with the same algebra.

const std = @import("std");
const transform_reference = @import("transform_reference.zig");
const instrument = @import("instrument.zig");

pub const N_FEATURES: usize = transform_reference.N_FEATURES;
pub const TOP_K: usize = 5;

/// Number of clusters scanned in nearest-first order before the bbox-pruned
/// repair pass. Each subsequent probe is gated by `lb_sq[c] < top_dists[0]`.
const PROBE_CLUSTERS: usize = 1;

/// Cap on how many of the K-PROBE remaining clusters get visited by the
/// bbox-pruned repair pass — only the top-N smallest by `lb_sq` (closest
/// bounding-box) are considered. Bounds the long-tail "many clusters
/// survive the prune" queries that dominated the eval p99. Safe vs the
/// unbounded version because the unbounded scan only ever scans clusters
/// with `lb_sq < top_dists[0]`, which (after PROBE seeds tight) is a
/// small set living in the closest-bbox region.
const REPAIR_TOP_N: usize = 64;

/// Global fixed-point scale. Persisted in the v5 blob header so a stale blob
/// built against a different value is rejected at load time.
pub const FIX_SCALE: i32 = 10000;

const W: usize = 16;
const Vec = @Vector(W, f32); // f32 brute-force only
const Vi16 = @Vector(W, i16);
const Vi32 = @Vector(W, i32);
const Vi64 = @Vector(W, i64);

// Half-width fallback for cluster residues in [W/2, W) — e.g. clusters of
// 8–15 rows and the ~5-row average tail. Maps to XMM under AVX2.
const HW: usize = W / 2;
const HVi16 = @Vector(HW, i16);
const HVi32 = @Vector(HW, i32);
const HVi64 = @Vector(HW, i64);

/// Stack-buffer cap. 4 KB at K=1024.
const MAX_K_CLUSTERS: usize = 1024;

/// Block-SoA inner-loop width — must match `dataset_blob.BLOCK_W`.
/// Mirrored here to avoid a circular import; comptime-asserted equal in
/// `dataset_blob.zig`'s test suite (the structural round-trip).
const BLOCK_W: usize = 8;
const BVi16 = @Vector(BLOCK_W, i16);
const BVi32 = @Vector(BLOCK_W, i32);
const BVi64 = @Vector(BLOCK_W, i64);

// Padding lanes (last block of a cluster when rc % BLOCK_W != 0) are
// filled with `dataset_blob.BLOCK_PAD_VALUE = 0` and excluded from sift
// via a runtime valid-lane mask in `scan_cluster_blocks`. The pad value
// is intentionally 0 (a real quantized value) so the inner SIMD loop can
// keep its diff in i16 — `q ∈ [-FIX_SCALE, FIX_SCALE]` minus 0 stays
// well inside i16 range, no widen-before-sub needed.

/// Quantize a 14-feature float query to i16 once per request.
inline fn quantize_query(q: *const [N_FEATURES]f32, out: *[N_FEATURES]i16) void {
    const fix_scale_f: f32 = @floatFromInt(FIX_SCALE);
    inline for (0..N_FEATURES) |k| {
        const r: f32 = @round(q[k] * fix_scale_f);
        out[k] = @trunc(r);
    }
}

// ─── public search API ────────────────────────────────────────────────────
//
// Production hot path is `euclidean_topk_q_ivf` only. The other public
// `euclidean_topk*` functions live below the `// ─── reference / bench-only
// paths ───` banner — they exist for the differential tests and `bench.zig`,
// not for prod.

/// Production IVF Euclidean top-K — **exact** by bbox-repair construction.
///
/// One ordered walk: SIMD-compute every centroid distance, sort cluster
/// indices ascending by centroid distance, then traverse in order. Scan the
/// nearest cluster unconditionally to seed `top_dists`; for each subsequent
/// cluster compute the axis-aligned lower bound
/// `LB² = Σ_k max(0, max(q − hi, lo − q))²` against the cluster's per-feature
/// `[lo, hi]` and skip if `LB² ≥ top_dists[0]`. Every cluster containing a
/// true top-K neighbour passes the prune (neighbour's true distance ≤ current
/// K-th best ≤ LB), so the top-K set is exact regardless of how many clusters
/// actually get scanned. The centroid-ascending order tightens `top_dists`
/// fast — most far clusters then prune cheaply on the bbox check.
pub fn euclidean_topk_q_ivf(
    qds: transform_reference.IvfQuantizedDataset,
    q: *const [N_FEATURES]f32,
    out: *[TOP_K]u32,
) void {
    @setEvalBranchQuota(8000);
    std.debug.assert(qds.n >= TOP_K);
    std.debug.assert(qds.k_clusters <= MAX_K_CLUSTERS);
    std.debug.assert(qds.k_clusters >= PROBE_CLUSTERS);

    var q_int: [N_FEATURES]i16 = undefined;
    quantize_query(q, &q_int);

    // Step 1 (combined): per cluster, compute centroid distance² AND bbox
    // lower-bound² in the same loop. Both metadata regions (centroids 56KB,
    // bbox_lo+hi 56KB) are L2-resident and prefetched as the loop streams.
    var centroid_dists: [MAX_K_CLUSTERS]f32 = undefined;
    var lb_sq: [MAX_K_CLUSTERS]i64 = undefined;
    const Vf = @Vector(N_FEATURES, f32);
    const Vi16N = @Vector(N_FEATURES, i16);
    const Vi32N = @Vector(N_FEATURES, i32);
    const Vi64N = @Vector(N_FEATURES, i64);
    const qv_f: Vf = q.*;
    const qv_i32: Vi32N = @as(Vi16N, q_int);
    const zero_i32: Vi32N = @splat(0);
    for (0..qds.k_clusters) |c| {
        const cv: Vf = qds.centroids[c * N_FEATURES ..][0..N_FEATURES].*;
        const diff = qv_f - cv;
        centroid_dists[c] = @reduce(.Add, diff * diff);

        const lo: Vi16N = qds.bbox_lo[c * N_FEATURES ..][0..N_FEATURES].*;
        const hi: Vi16N = qds.bbox_hi[c * N_FEATURES ..][0..N_FEATURES].*;
        const lo_i32: Vi32N = lo;
        const hi_i32: Vi32N = hi;
        const below_lo: Vi32N = lo_i32 - qv_i32;
        const above_hi: Vi32N = qv_i32 - hi_i32;
        const d: Vi32N = @max(zero_i32, @max(below_lo, above_hi));
        const d_i64: Vi64N = d;
        lb_sq[c] = @reduce(.Add, d_i64 * d_i64);
    }

    // Step 2: select PROBE clusters with smallest centroid distance via a
    // sorted insertion-sort accumulator. Result is ascending so the first
    // cluster scanned is the closest (tightens `top_dists[0]` fastest).
    var probe_dists: [PROBE_CLUSTERS]f32 = @splat(std.math.inf(f32));
    var probe_idxs: [PROBE_CLUSTERS]u32 = @splat(0);
    for (0..qds.k_clusters) |c| {
        const d = centroid_dists[c];
        if (d >= probe_dists[PROBE_CLUSTERS - 1]) continue;
        var pos: usize = PROBE_CLUSTERS - 1;
        while (pos > 0 and d < probe_dists[pos - 1]) : (pos -= 1) {}
        var i: usize = PROBE_CLUSTERS - 1;
        while (i > pos) : (i -= 1) {
            probe_dists[i] = probe_dists[i - 1];
            probe_idxs[i] = probe_idxs[i - 1];
        }
        probe_dists[pos] = d;
        probe_idxs[pos] = @intCast(c);
    }

    // Step 3a: scan PROBE clusters in nearest-first order. Mark each by
    // poisoning its `lb_sq` to maxInt so step 3b's top-N pick skips it.
    var top_dists: [TOP_K]i64 = @splat(std.math.maxInt(i64));
    var top_rows: [TOP_K]u32 = @splat(0);

    scan_cluster_blocks(qds, probe_idxs[0], &q_int, &top_dists, &top_rows);
    lb_sq[probe_idxs[0]] = std.math.maxInt(i64);
    instrument.inc(&instrument.search_clusters_probed, 1);
    inline for (1..PROBE_CLUSTERS) |i| {
        const c = probe_idxs[i];
        if (lb_sq[c] < top_dists[0]) {
            scan_cluster_blocks(qds, c, &q_int, &top_dists, &top_rows);
            instrument.inc(&instrument.search_clusters_probed, 1);
        } else {
            instrument.inc(&instrument.search_clusters_bbox_skipped, 1);
        }
        lb_sq[c] = std.math.maxInt(i64);
    }

    // Step 3b: capped bbox repair. Pick the REPAIR_TOP_N un-probed clusters
    // with smallest `lb_sq` via insertion-sort, then walk in ascending
    // order. The unbounded version walked all K-PROBE clusters in arbitrary
    // order with the same prune check — capping bounds long-tail queries
    // where many clusters survive `lb_sq < top_dists[0]`. Safe because
    // every cluster the unbounded version actually scans has small lb_sq
    // (else it'd prune), so it lives in the top-N anyway. Once the array
    // is sorted and we hit `lb_sq >= top_dists[0]`, all subsequent are
    // also pruneable — break.
    var repair_lb: [REPAIR_TOP_N]i64 = @splat(std.math.maxInt(i64));
    var repair_idxs: [REPAIR_TOP_N]u32 = @splat(0);
    for (0..qds.k_clusters) |c| {
        const lb = lb_sq[c];
        if (lb >= repair_lb[REPAIR_TOP_N - 1]) continue;
        var pos: usize = REPAIR_TOP_N - 1;
        while (pos > 0 and lb < repair_lb[pos - 1]) : (pos -= 1) {}
        var i: usize = REPAIR_TOP_N - 1;
        while (i > pos) : (i -= 1) {
            repair_lb[i] = repair_lb[i - 1];
            repair_idxs[i] = repair_idxs[i - 1];
        }
        repair_lb[pos] = lb;
        repair_idxs[pos] = @intCast(c);
    }
    for (0..REPAIR_TOP_N) |i| {
        const lb = repair_lb[i];
        if (lb == std.math.maxInt(i64)) break; // sentinel: ran out of un-probed clusters
        if (lb >= top_dists[0]) {
            instrument.inc(&instrument.search_clusters_bbox_skipped, 1);
            break; // sorted ascending → all subsequent are pruneable too
        }
        scan_cluster_blocks(qds, repair_idxs[i], &q_int, &top_dists, &top_rows);
        instrument.inc(&instrument.search_clusters_bbox_scanned, 1);
    }

    inline for (0..TOP_K) |i| out[i] = top_rows[TOP_K - 1 - i];
}

// ─── block-SoA cluster scan (production IVF inner loop) ──────────────────

/// Scan all blocks of cluster `c`, sifting any row whose squared distance
/// is < `top_dists[0]` into the running top-K. Padding lanes within the
/// last block produce sentinel-large distances and are silently dropped
/// by the sift comparison (the comptime invariant above guarantees this).
inline fn scan_cluster_blocks(
    qds: transform_reference.IvfQuantizedDataset,
    c: usize,
    q_int: *const [N_FEATURES]i16,
    top_dists: *[TOP_K]i64,
    top_rows: *[TOP_K]u32,
) void {
    var q_vec: [N_FEATURES]BVi16 = undefined;
    inline for (0..N_FEATURES) |k| q_vec[k] = @splat(q_int[k]);

    const row_base = qds.cluster_starts[c];
    const ce: u32 = qds.cluster_starts[c + 1];
    const block_start = qds.cluster_block_starts[c];
    const block_end = qds.cluster_block_starts[c + 1];
    const stride = N_FEATURES * BLOCK_W;

    var b: usize = block_start;
    while (b < block_end) : (b += 1) {
        instrument.inc(&instrument.search_blocks_scanned, 1);
        const block_base = b * stride;
        if (b + 1 < block_end) {
            @prefetch(&qds.block_features[(b + 1) * stride], .{ .locality = 1 });
        }

        // Inner loop matches the Phase 9.6 W=16 SoA pattern: i16 sub (safe
        // because pad is 0 ∈ valid range, real features ∈ [-FIX_SCALE,
        // FIX_SCALE], diff ⊂ ±2*FIX_SCALE = ±20000 fits in i16), then
        // widen to i32 for the square (4e8 max per feature fits in i32),
        // then widen to i64 for the running 14-feature sum (5.6e9 fits).
        //
        // Mid-loop prune: after the first `EARLY_OUT_AT` features, every
        // lane's running partial-sum is a true LOWER BOUND on its final
        // distance (each remaining feature can only ADD non-negative).
        // If the smallest lane's partial-sum already exceeds the current
        // K-th best, no lane in this block can win — skip the remaining
        // features and the sift. The Min reduction is 1 horizontal op
        // every block; the saved work is 7 × (load+sub+widen+mul+widen+add)
        // ≈ 42 vector ops on every pruned block.
        const EARLY_OUT_AT: usize = N_FEATURES / 2;
        var dist: BVi64 = @splat(0);
        inline for (0..EARLY_OUT_AT) |k| {
            const r_i16: BVi16 = qds.block_features[block_base + k * BLOCK_W ..][0..BLOCK_W].*;
            const diff_i16: BVi16 = q_vec[k] - r_i16;
            const diff_i32: BVi32 = diff_i16;
            const sq: BVi32 = diff_i32 * diff_i32;
            dist += @as(BVi64, sq);
        }
        if (@reduce(.Min, dist) > top_dists[0]) {
            instrument.inc(&instrument.search_blocks_early_pruned, 1);
            continue;
        }
        inline for (EARLY_OUT_AT..N_FEATURES) |k| {
            const r_i16: BVi16 = qds.block_features[block_base + k * BLOCK_W ..][0..BLOCK_W].*;
            const diff_i16: BVi16 = q_vec[k] - r_i16;
            const diff_i32: BVi32 = diff_i16;
            const sq: BVi32 = diff_i32 * diff_i32;
            dist += @as(BVi64, sq);
        }

        // Padding lanes (last block of a cluster with rc % W != 0) are
        // pad-value rows whose row index would land outside the cluster's
        // canonical range — sifting them in would emit garbage row indices
        // and then `label_at()` reads the labels bitset out of bounds.
        // Mask the sift to valid lanes only. For full blocks
        // (valid_lanes == BLOCK_W) the comparison constant-folds; for the
        // single partial block per cluster, ~3 wasted comparisons.
        const lane_row_base: u32 = @intCast(row_base + (b - block_start) * BLOCK_W);
        const valid_lanes: u32 = if (lane_row_base >= ce) 0 else @min(@as(u32, BLOCK_W), ce - lane_row_base);
        inline for (0..BLOCK_W) |lane| {
            if (@as(u32, @intCast(lane)) < valid_lanes) {
                const d = dist[lane];
                const r = lane_row_base + @as(u32, @intCast(lane));
                if (better_pair_i64(d, r, top_dists[0], top_rows[0])) {
                    instrument.inc(&instrument.search_sift_ins, 1);
                    sift_in_min_i64(top_dists, top_rows, d, r);
                }
            }
        }
    }
}

// ─── sift helpers (production) ───────────────────────────────────────────
//
// Top-K kept descending by (dist, then row): index 0 = worst (largest dist
// or, on tie, largest row). New entry replaces position 0 then bubbles up
// (toward TOP_K-1) while it remains *better* than its right neighbour.

/// Strict lexicographic ordering on (dist, row) — smaller dist wins; ties
/// broken by smaller row index. Used by `sift_in_min_i64` so the top-K is
/// deterministic regardless of cluster scan order. Mirrors the
/// `is_better_pair` comparator in thiagorigonatti #1
/// (rinha-2026/src/ivf_search.c:18).
inline fn better_pair_i64(da: i64, ia: u32, db: i64, ib: u32) bool {
    return da < db or (da == db and ia < ib);
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
        if (!better_pair_i64(dists[i], rows[i], dists[i + 1], rows[i + 1])) break;
        const td = dists[i];
        dists[i] = dists[i + 1];
        dists[i + 1] = td;
        const tr = rows[i];
        rows[i] = rows[i + 1];
        rows[i + 1] = tr;
    }
}

// ─── reference / bench-only paths ────────────────────────────────────────
//
// Below this banner: nothing the production handler touches. Kept around so
// `bench.zig` can A/B against the IVF path and `dataset_blob.zig` tests can
// differentially compare. None of these pull labels via the bitset directly
// — they all return row indices that the caller maps separately.

/// Brute-force f32 top-K against a non-quantized dataset. Bench-only.
pub fn euclidean_topk(
    ds: transform_reference.Dataset,
    q: *const [N_FEATURES]f32,
    out: *[TOP_K]u32,
) void {
    std.debug.assert(ds.n >= TOP_K);

    var q_vec: [N_FEATURES]Vec = undefined;
    inline for (0..N_FEATURES) |k| q_vec[k] = @splat(q[k]);

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

    inline for (0..TOP_K) |i| out[i] = top_rows[TOP_K - 1 - i];
}

/// Brute-force int top-K over an i16-quantized flat-SoA dataset. Bench/test only.
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

/// Test-only: scan every cluster of an IVF dataset (PROBE = K). Used by
/// `dataset_blob.zig` equivalence tests against `euclidean_topk_q` with K
/// small enough that PROBE would be ≥ K.
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
        scan_cluster_blocks(qds, c, &q_int, &top_dists, &top_rows);
    }

    inline for (0..TOP_K) |i| out[i] = top_rows[TOP_K - 1 - i];
}

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
    const PF_AHEAD: usize = W * 8;
    while (row + W <= end) : (row += W) {
        if (row + PF_AHEAD < end) {
            @prefetch(&features[row + PF_AHEAD], .{ .locality = 1 });
        }
        var dist: Vi64 = @splat(0);
        inline for (0..N_FEATURES) |k| {
            const r_i16: Vi16 = features[k * n + row ..][0..W].*;
            const diff_i16: Vi16 = q_vec[k] - r_i16;
            const diff_i32: Vi32 = diff_i16;
            const sq: Vi32 = diff_i32 * diff_i32;
            dist += @as(Vi64, sq);
        }
        inline for (0..W) |lane| {
            const d = dist[lane];
            const r: u32 = @intCast(row + lane);
            if (better_pair_i64(d, r, top_dists[0], top_rows[0])) {
                sift_in_min_i64(top_dists, top_rows, d, r);
            }
        }
    }
    if (row + HW <= end) {
        var dist: HVi64 = @splat(0);
        inline for (0..N_FEATURES) |k| {
            const qv: HVi16 = @splat(q_int[k]);
            const r_i16: HVi16 = features[k * n + row ..][0..HW].*;
            const diff_i16: HVi16 = qv - r_i16;
            const diff_i32: HVi32 = diff_i16;
            const sq: HVi32 = diff_i32 * diff_i32;
            dist += @as(HVi64, sq);
        }
        inline for (0..HW) |lane| {
            const d = dist[lane];
            const r: u32 = @intCast(row + lane);
            if (better_pair_i64(d, r, top_dists[0], top_rows[0])) {
                sift_in_min_i64(top_dists, top_rows, d, r);
            }
        }
        row += HW;
    }
    while (row < end) : (row += 1) {
        var dist: i64 = 0;
        inline for (0..N_FEATURES) |k| {
            const diff: i32 = @as(i32, q_int[k]) - @as(i32, features[k * n + row]);
            dist += @as(i64, diff * diff);
        }
        const r: u32 = @intCast(row);
        if (better_pair_i64(dist, r, top_dists[0], top_rows[0])) {
            sift_in_min_i64(top_dists, top_rows, dist, r);
        }
    }
}

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
        dst[i] = @trunc(@max(lo_clamp, @min(hi_clamp, q)));
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

/// Test-only: build per-cluster block-SoA features + cluster_block_starts
/// from a flat-SoA `[N_FEATURES][n]i16` source. Pads the last block of each
/// cluster with 0 (matches `dataset_blob.BLOCK_PAD_VALUE`). Padding lanes
/// are excluded from sift via the valid-lane mask in `scan_cluster_blocks`.
fn build_block_features_test_only(
    n: usize,
    k_clusters: usize,
    cluster_starts: []const u32,
    flat_features: []const i16,
    out_block_features: []i16,
    out_block_starts: []u32,
) void {
    out_block_starts[0] = 0;
    for (0..k_clusters) |c| {
        const cs = cluster_starts[c];
        const ce = cluster_starts[c + 1];
        const rc = ce - cs;
        const blocks_in_c: u32 = @intCast((@as(usize, rc) + BLOCK_W - 1) / BLOCK_W);
        out_block_starts[c + 1] = out_block_starts[c] + blocks_in_c;
        for (0..blocks_in_c) |b| {
            const block_base = (@as(usize, out_block_starts[c]) + b) * N_FEATURES * BLOCK_W;
            for (0..N_FEATURES) |k| {
                for (0..BLOCK_W) |lane| {
                    const row = cs + @as(u32, @intCast(b * BLOCK_W + lane));
                    out_block_features[block_base + k * BLOCK_W + lane] =
                        if (row < ce) flat_features[k * n + row] else 0;
                }
            }
        }
    }
}

test "euclidean_topk_q_ivf hand-built tiny clustered dataset" {
    // 10 rows in 14 features. Rows 0..4 cluster around +e0, rows 5..9 around +e1.
    // K=2; query toward +e0 must return rows 0..4 (cluster 0 wins).
    const n: usize = 10;
    const k_clusters: usize = 2;

    var f32_features: [N_FEATURES * n]f32 = @splat(0);
    var labels_bits: [(n + 63) / 64]u64 = @splat(0);
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

    // Per-cluster blocks: 5 rows / W=8 = 1 block per cluster; 2 blocks total.
    const total_blocks: usize = 2;
    var block_features: [total_blocks * N_FEATURES * BLOCK_W]i16 = undefined;
    var cluster_block_starts: [k_clusters + 1]u32 = undefined;
    build_block_features_test_only(
        n,
        k_clusters,
        &cluster_starts,
        &i16_features,
        &block_features,
        &cluster_block_starts,
    );

    const qds: transform_reference.IvfQuantizedDataset = .{
        .n = n,
        .k_clusters = k_clusters,
        .block_features = &block_features,
        .labels_bits = &labels_bits,
        .centroids = &centroids,
        .cluster_starts = &cluster_starts,
        .cluster_block_starts = &cluster_block_starts,
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

test "probe-select: smallest PROBE_CLUSTERS over a 32-element stream" {
    // Feed 32 shuffled distances through the same insertion-sort accumulator
    // used in `euclidean_topk_q_ivf`; verify it picks the 8 smallest in order.
    var src: [32]f32 = undefined;
    for (0..32) |i| src[i] = @floatFromInt((i * 37 + 13) % 32); // shuffled 0..31

    var dists: [PROBE_CLUSTERS]f32 = @splat(std.math.inf(f32));
    var idxs: [PROBE_CLUSTERS]u32 = @splat(0);
    for (src, 0..) |d, c| {
        if (d >= dists[PROBE_CLUSTERS - 1]) continue;
        var pos: usize = PROBE_CLUSTERS - 1;
        while (pos > 0 and d < dists[pos - 1]) : (pos -= 1) {}
        var i: usize = PROBE_CLUSTERS - 1;
        while (i > pos) : (i -= 1) {
            dists[i] = dists[i - 1];
            idxs[i] = idxs[i - 1];
        }
        dists[pos] = d;
        idxs[pos] = @intCast(c);
    }

    inline for (0..PROBE_CLUSTERS) |i| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(i)), dists[i]);
        try std.testing.expectEqual(dists[i], src[idxs[i]]);
    }
}

test "euclidean_topk_q hand-built tiny dataset" {
    const n: usize = 5;
    const expected_cos = [5]f32{ 1.0, 0.9, 0.5, 0.0, -1.0 };

    var f32_features: [N_FEATURES * n]f32 = @splat(0);
    var labels_bits: [(n + 63) / 64]u64 = @splat(0);

    for (expected_cos, 0..) |c, row| {
        f32_features[0 * n + row] = c;
        f32_features[1 * n + row] = @sqrt(@max(0.0, 1.0 - c * c));
    }

    var i16_features: [N_FEATURES * n]i16 = @splat(0);
    quantize_dataset(n, &f32_features, &i16_features);

    const qds: transform_reference.QuantizedDataset = .{
        .n = n,
        .features = &i16_features,
        .labels_bits = &labels_bits,
    };
    var q: [N_FEATURES]f32 = @splat(0);
    q[0] = 1.0;

    var out: [TOP_K]u32 = undefined;
    euclidean_topk_q(qds, &q, &out);

    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4 }, &out);
}
