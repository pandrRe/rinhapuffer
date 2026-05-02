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

pub const N_FEATURES: usize = transform_reference.N_FEATURES;
pub const TOP_K: usize = 5;

/// Number of clusters scanned per query in `euclidean_topk_q_ivf`. Mirrors
/// `dataset_blob.PROBE_CLUSTERS` (which lives there for layout commentary);
/// kept private here to avoid a circular import.
const PROBE_CLUSTERS: usize = 8;

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

// Padding sentinel must produce a strictly larger squared distance than
// any real per-row distance, so padding lanes always lose the sift in
// `scan_cluster_blocks` without a per-block valid-lane mask.
comptime {
    const max_real_diff: i64 = 2 * @as(i64, FIX_SCALE);
    const max_real_dist: i64 = N_FEATURES * max_real_diff * max_real_diff;
    const sentinel: i64 = @intCast(std.math.maxInt(i16));
    const min_pad_diff: i64 = sentinel - @as(i64, FIX_SCALE);
    const min_pad_dist: i64 = N_FEATURES * min_pad_diff * min_pad_diff;
    if (min_pad_dist <= max_real_dist) {
        @compileError("BLOCK_PAD_SENTINEL no longer dominates real distances; revisit search.scan_cluster_blocks before bumping FIX_SCALE");
    }
}

/// Quantize a 14-feature float query to i16 once per request.
inline fn quantize_query(q: *const [N_FEATURES]f32, out: *[N_FEATURES]i16) void {
    const fix_scale_f: f32 = @floatFromInt(FIX_SCALE);
    inline for (0..N_FEATURES) |k| {
        const r: f32 = @round(q[k] * fix_scale_f);
        out[k] = @trunc(r);
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

    // Step 1: SIMD centroid distances (one @Vector(14, f32) per cluster).
    var centroid_dists: [MAX_K_CLUSTERS]f32 = undefined;
    const Vf = @Vector(N_FEATURES, f32);
    const qv: Vf = q.*;
    for (0..qds.k_clusters) |c| {
        const cv: Vf = qds.centroids[c * N_FEATURES ..][0..N_FEATURES].*;
        const diff = qv - cv;
        centroid_dists[c] = @reduce(.Add, diff * diff);
    }

    // Step 2: select PROBE clusters with smallest centroid distance via a
    // max-heap of size PROBE. ~10× cheaper than pdq-sorting all 1024 entries
    // when we only need the smallest 8 in distance order. The remaining
    // K - PROBE clusters are NOT sorted — they get walked unsorted in the
    // bbox-prune pass, which keeps exact-top-K (every unprobed cluster that
    // could contain a contender still gets a bbox check).
    //
    // `in_probe` marks the PROBE survivors so the prune pass skips them.
    var probe_dists: [PROBE_CLUSTERS]f32 = undefined;
    var probe_idxs: [PROBE_CLUSTERS]u32 = undefined;
    inline for (0..PROBE_CLUSTERS) |i| {
        probe_dists[i] = centroid_dists[i];
        probe_idxs[i] = @intCast(i);
    }
    heapify_max(&probe_dists, &probe_idxs);
    var c2: usize = PROBE_CLUSTERS;
    while (c2 < qds.k_clusters) : (c2 += 1) {
        if (centroid_dists[c2] < probe_dists[0]) {
            probe_dists[0] = centroid_dists[c2];
            probe_idxs[0] = @intCast(c2);
            sift_down_max(&probe_dists, &probe_idxs, 0);
        }
    }
    // PROBE is small (8) — insertion-sort the survivors ascending so the
    // first cluster scanned is the closest (tightens `top_dists[0]` fastest).
    sort_pairs_asc(&probe_dists, &probe_idxs);

    var in_probe = [_]bool{false} ** MAX_K_CLUSTERS;
    inline for (0..PROBE_CLUSTERS) |i| in_probe[probe_idxs[i]] = true;

    // Step 3: scan PROBE clusters in nearest-first order, then bbox-prune
    // the remaining K - PROBE in arbitrary order. Most fail-prune cheaply.
    var q_int: [N_FEATURES]i16 = undefined;
    quantize_query(q, &q_int);

    var top_dists: [TOP_K]i64 = @splat(std.math.maxInt(i64));
    var top_rows: [TOP_K]u32 = @splat(0);

    scan_cluster_blocks(qds, probe_idxs[0], &q_int, &top_dists, &top_rows);
    inline for (1..PROBE_CLUSTERS) |i| {
        const c = probe_idxs[i];
        const lb = bbox_lower_bound_sq(&q_int, qds.bbox_lo, qds.bbox_hi, c);
        if (lb < top_dists[0]) {
            scan_cluster_blocks(qds, c, &q_int, &top_dists, &top_rows);
        }
    }
    for (0..qds.k_clusters) |c| {
        if (in_probe[c]) continue;
        const lb = bbox_lower_bound_sq(&q_int, qds.bbox_lo, qds.bbox_hi, c);
        if (lb >= top_dists[0]) continue;
        scan_cluster_blocks(qds, c, &q_int, &top_dists, &top_rows);
    }

    inline for (0..TOP_K) |i| out[i] = top_rows[TOP_K - 1 - i];
}

/// Axis-aligned squared distance from query to cluster `c`'s bounding box.
/// Per feature, contributes `(below_lo)²` if `q < lo`, `(above_hi)²` if
/// `q > hi`, else 0. SIMD across all 14 features in one shot —
/// `@Vector(N_FEATURES, ·)` lanes get padded to 16 by LLVM on aarch64
/// (4× int32x4 ops), no waste of meaningful work.
inline fn bbox_lower_bound_sq(
    q_int: *const [N_FEATURES]i16,
    bbox_lo: []const i16,
    bbox_hi: []const i16,
    c: usize,
) i64 {
    const Vi16N = @Vector(N_FEATURES, i16);
    const Vi32N = @Vector(N_FEATURES, i32);
    const Vi64N = @Vector(N_FEATURES, i64);

    const lo: Vi16N = bbox_lo[c * N_FEATURES ..][0..N_FEATURES].*;
    const hi: Vi16N = bbox_hi[c * N_FEATURES ..][0..N_FEATURES].*;
    const qv: Vi16N = q_int.*;

    const lo_i32: Vi32N = lo;
    const hi_i32: Vi32N = hi;
    const qv_i32: Vi32N = qv;

    const below_lo: Vi32N = lo_i32 - qv_i32;
    const above_hi: Vi32N = qv_i32 - hi_i32;
    const zero: Vi32N = @splat(0);
    const d: Vi32N = @max(zero, @max(below_lo, above_hi));

    const d_i64: Vi64N = d;
    const sq: Vi64N = d_i64 * d_i64;
    return @reduce(.Add, sq);
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
        scan_cluster_blocks(qds, c, &q_int, &top_dists, &top_rows);
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
    // Widen the broadcasted query to i32 once per feature: the row vector
    // can be `BLOCK_PAD_SENTINEL` (= 32767) on padding lanes, so the diff
    // would overflow i16 (`q - 32767` ≤ -22767, but real q can also be
    // ±10000 → diff range ⊂ [-42767, +42767]). i32 sub stays in range.
    var q_vec: [N_FEATURES]BVi32 = undefined;
    inline for (0..N_FEATURES) |k| q_vec[k] = @splat(@as(i32, q_int[k]));

    const row_base = qds.cluster_starts[c];
    const block_start = qds.cluster_block_starts[c];
    const block_end = qds.cluster_block_starts[c + 1];
    const stride = N_FEATURES * BLOCK_W;

    var b: usize = block_start;
    while (b < block_end) : (b += 1) {
        const block_base = b * stride;
        if (b + 1 < block_end) {
            @prefetch(&qds.block_features[(b + 1) * stride], .{ .locality = 1 });
        }

        var dist: BVi64 = @splat(0);
        inline for (0..N_FEATURES) |k| {
            const r_i16: BVi16 = qds.block_features[block_base + k * BLOCK_W ..][0..BLOCK_W].*;
            const r_i32: BVi32 = r_i16;
            const diff_i32: BVi32 = q_vec[k] - r_i32;
            const sq: BVi32 = diff_i32 * diff_i32;
            dist += @as(BVi64, sq);
        }

        const lane_row_base: u32 = @intCast(row_base + (b - block_start) * BLOCK_W);
        inline for (0..BLOCK_W) |lane| {
            const d = dist[lane];
            if (d < top_dists[0]) {
                sift_in_min_i64(top_dists, top_rows, d, lane_row_base + @as(u32, @intCast(lane)));
            }
        }
    }
}

// ─── int inner loop (brute-force flat-SoA path; test-only) ───────────────

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
    // Prefetch ~256 B ahead in feature[0]'s column. The HW L1 streamer picks
    // up the other 13 column streams once any byte in each is touched; issuing
    // 14 prefetches per chunk would saturate the prefetch queue and backfire.
    const PF_AHEAD: usize = W * 8;
    while (row + W <= end) : (row += W) {
        if (row + PF_AHEAD < end) {
            @prefetch(&features[row + PF_AHEAD], .{ .locality = 1 });
        }
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
    // W/2 fallback chunk for residues in [W/2, W) — keeps clusters of 8–15
    // rows (and average ~5-row residues after a W=16 main loop) in SIMD
    // instead of falling all the way to the scalar tail.
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
            if (d < top_dists[0]) {
                sift_in_min_i64(top_dists, top_rows, d, @intCast(row + lane));
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

// ─── PROBE-cluster heap helpers ───────────────────────────────────────────
//
// Operate on parallel `dists` (key) + `idxs` (payload) arrays of compile-
// time size `PROBE_CLUSTERS`. Max-heap means root = largest, so a smaller
// candidate displaces it. After all sifting, `sort_pairs_asc` flips it
// to ascending for the nearest-first scan order.

inline fn heap_swap(dists: *[PROBE_CLUSTERS]f32, idxs: *[PROBE_CLUSTERS]u32, a: usize, b: usize) void {
    const td = dists[a];
    dists[a] = dists[b];
    dists[b] = td;
    const ti = idxs[a];
    idxs[a] = idxs[b];
    idxs[b] = ti;
}

inline fn sift_down_max(dists: *[PROBE_CLUSTERS]f32, idxs: *[PROBE_CLUSTERS]u32, start: usize) void {
    var root = start;
    while (true) {
        const left = 2 * root + 1;
        if (left >= PROBE_CLUSTERS) break;
        var swap = root;
        if (dists[swap] < dists[left]) swap = left;
        const right = left + 1;
        if (right < PROBE_CLUSTERS and dists[swap] < dists[right]) swap = right;
        if (swap == root) break;
        heap_swap(dists, idxs, root, swap);
        root = swap;
    }
}

inline fn heapify_max(dists: *[PROBE_CLUSTERS]f32, idxs: *[PROBE_CLUSTERS]u32) void {
    // Floyd's bottom-up: start at last internal node, sift down. For
    // PROBE_CLUSTERS=8 this unrolls to ~3 sifts.
    var i: isize = @as(isize, PROBE_CLUSTERS / 2) - 1;
    while (i >= 0) : (i -= 1) sift_down_max(dists, idxs, @intCast(i));
}

inline fn sort_pairs_asc(dists: *[PROBE_CLUSTERS]f32, idxs: *[PROBE_CLUSTERS]u32) void {
    // Insertion sort, ascending by dist. PROBE_CLUSTERS is small (8) so
    // this beats pdq's overhead and the branch pattern is predictable.
    var i: usize = 1;
    while (i < PROBE_CLUSTERS) : (i += 1) {
        const kd = dists[i];
        const ki = idxs[i];
        var j: usize = i;
        while (j > 0 and dists[j - 1] > kd) : (j -= 1) {
            dists[j] = dists[j - 1];
            idxs[j] = idxs[j - 1];
        }
        dists[j] = kd;
        idxs[j] = ki;
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
/// cluster with `maxInt(i16)` (matches `dataset_blob.BLOCK_PAD_SENTINEL`).
/// Caller pre-sizes `out_block_features` to `total_blocks * N_FEATURES *
/// BLOCK_W` and `out_block_starts` to `[k_clusters + 1]u32`.
fn build_block_features_test_only(
    n: usize,
    k_clusters: usize,
    cluster_starts: []const u32,
    flat_features: []const i16,
    out_block_features: []i16,
    out_block_starts: []u32,
) void {
    const sentinel: i16 = std.math.maxInt(i16);
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
                        if (row < ce) flat_features[k * n + row] else sentinel;
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

test "heap-select helpers: smallest PROBE_CLUSTERS over a 32-element stream" {
    // Feed 32 known distances; verify the heap-select picks the 8 smallest.
    var src: [32]f32 = undefined;
    for (0..32) |i| src[i] = @floatFromInt((i * 37 + 13) % 32); // shuffled 0..31

    var dists: [PROBE_CLUSTERS]f32 = undefined;
    var idxs: [PROBE_CLUSTERS]u32 = undefined;
    inline for (0..PROBE_CLUSTERS) |i| {
        dists[i] = src[i];
        idxs[i] = @intCast(i);
    }
    heapify_max(&dists, &idxs);
    var c: usize = PROBE_CLUSTERS;
    while (c < src.len) : (c += 1) {
        if (src[c] < dists[0]) {
            dists[0] = src[c];
            idxs[0] = @intCast(c);
            sift_down_max(&dists, &idxs, 0);
        }
    }
    sort_pairs_asc(&dists, &idxs);

    // Expected smallest 8 of 0..31 are {0,1,2,3,4,5,6,7} in ascending order.
    inline for (0..PROBE_CLUSTERS) |i| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(i)), dists[i]);
    }
    // Each survivor's idx must point back to a src entry whose value matches.
    inline for (0..PROBE_CLUSTERS) |i| {
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
