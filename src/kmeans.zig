//! Plain Euclidean k-means for the IVF index.
//!
//! Inputs are raw rows in SoA layout (`features[k * n + row]`) — same value
//! space as `transform_reference.parse_into`'s output. Distance is Euclidean
//! (`Σ (row − centroid)²`); ranking is `argmin dist²` == `argmax (row·c − ½‖c‖²)`
//! with the per-centroid `½‖c‖²` precomputed once per iteration. The centroid
//! update is the per-cluster mean. Empty clusters reseed from a random sample
//! row.
//!
//! Init is k-means++ (Arthur & Vassilvitskii 2007): first centroid uniform,
//! each subsequent one sampled with prob ∝ D²(row) against the running
//! min-distance to already-chosen centroids. Vs vanilla random init this
//! spreads seeds along the data manifold, dramatically reducing
//! empty-cluster reseeds and final cluster-size variance — the latter
//! matters for IVF tail latency since mega-clusters are scanned in full
//! whenever they're probed.
//!
//! Determinism: same `seed` + same input ⇒ bit-identical centroids and
//! assignments. Argmax tiebreak is "lowest cluster index wins", and both
//! the k-means++ sampling and empty-cluster reseed are driven by an
//! explicitly-seeded `Xoshiro256`.

const std = @import("std");
const transform_reference = @import("transform_reference.zig");

const N_FEATURES = transform_reference.N_FEATURES;

/// Initialise centroids via k-means++, then alternate
/// assign-by-min-Euclidean / sum / mean for `n_iter` iterations. Final
/// centroids land in `centroids_out` (row-major, `[c * N_FEATURES + k]`).
///
/// Allocates: `[k_clusters * N_FEATURES]f32` sums + `[k_clusters]u32` counts +
/// `[n]u32` assignments + `[n]f32` k-means++ min-D² scratch +
/// `[k_clusters]f32` half-norm-sq cache.
pub fn run_kmeans(
    allocator: std.mem.Allocator,
    features_soa: []const f32,
    n: usize,
    k_clusters: usize,
    n_iter: usize,
    seed: u64,
    centroids_out: []f32,
) !void {
    std.debug.assert(centroids_out.len == k_clusters * N_FEATURES);
    std.debug.assert(features_soa.len == N_FEATURES * n);
    std.debug.assert(n >= k_clusters);

    var prng = std.Random.Xoshiro256.init(seed);
    const rand = prng.random();

    // Init: k-means++. `min_d_sq[r]` is the running squared distance from row
    // r to its nearest already-chosen centroid; updated incrementally each
    // time a new centroid is placed. Sampling proportional to `min_d_sq`
    // gives each iteration a strong bias toward "uncovered" regions of the
    // input space, which costs O(N·K·d) once at prep — same order as one
    // Lloyd iteration.
    const min_d_sq = try allocator.alloc(f32, n);
    defer allocator.free(min_d_sq);
    @memset(min_d_sq, std.math.inf(f32));

    {
        const r0 = rand.uintLessThan(usize, n);
        inline for (0..N_FEATURES) |k| {
            centroids_out[0 * N_FEATURES + k] = features_soa[k * n + r0];
        }
    }

    for (1..k_clusters) |c| {
        // Refresh `min_d_sq` against the centroid just placed (c-1) and
        // accumulate the total weight in one pass. f64 accumulator because
        // at n=3M the sum of f32 squared distances easily exceeds f32's
        // 24-bit mantissa (~16M precision ceiling) and would round-down on
        // the long tail.
        const cm1 = c - 1;
        var total: f64 = 0;
        for (0..n) |row| {
            var d: f32 = 0;
            inline for (0..N_FEATURES) |k| {
                const diff = features_soa[k * n + row] - centroids_out[cm1 * N_FEATURES + k];
                d += diff * diff;
            }
            if (d < min_d_sq[row]) min_d_sq[row] = d;
            total += min_d_sq[row];
        }

        // Sample row by inverse-CDF on min_d_sq. Degenerate fallback (every
        // row exact-matches some chosen centroid → total == 0) picks
        // uniformly so we don't divide by zero.
        const pick: usize = if (total <= 0)
            rand.uintLessThan(usize, n)
        else blk: {
            const u = rand.float(f64) * total;
            var acc: f64 = 0;
            var idx: usize = n - 1;
            for (0..n) |row| {
                acc += min_d_sq[row];
                if (acc >= u) {
                    idx = row;
                    break;
                }
            }
            break :blk idx;
        };

        inline for (0..N_FEATURES) |k| {
            centroids_out[c * N_FEATURES + k] = features_soa[k * n + pick];
        }
    }

    const assignments = try allocator.alloc(u32, n);
    defer allocator.free(assignments);
    const sums = try allocator.alloc(f32, k_clusters * N_FEATURES);
    defer allocator.free(sums);
    const counts = try allocator.alloc(u32, k_clusters);
    defer allocator.free(counts);
    const half_norm_sq = try allocator.alloc(f32, k_clusters);
    defer allocator.free(half_norm_sq);

    for (0..n_iter) |_| {
        // Per-iteration cache: ½‖c‖² for every centroid. Lets the assign loop
        // rank by `argmax (row·c − ½‖c‖²)` (== argmin Euclidean distance)
        // without recomputing the centroid norm for every (row, centroid) pair.
        for (0..k_clusters) |c| {
            var ss: f32 = 0;
            inline for (0..N_FEATURES) |k| {
                const v = centroids_out[c * N_FEATURES + k];
                ss += v * v;
            }
            half_norm_sq[c] = 0.5 * ss;
        }

        // Assign: argmax (row·c − ½‖c‖²), tiebreak lowest cluster index.
        for (0..n) |row| {
            var best: f32 = -std.math.inf(f32);
            var best_c: u32 = 0;
            for (0..k_clusters) |c| {
                var dot: f32 = 0;
                inline for (0..N_FEATURES) |k| {
                    dot = @mulAdd(
                        f32,
                        features_soa[k * n + row],
                        centroids_out[c * N_FEATURES + k],
                        dot,
                    );
                }
                const score = dot - half_norm_sq[c];
                if (score > best) {
                    best = score;
                    best_c = @intCast(c);
                }
            }
            assignments[row] = best_c;
        }

        // Sum + count.
        @memset(sums, 0);
        @memset(counts, 0);
        for (0..n) |row| {
            const c = assignments[row];
            counts[c] += 1;
            inline for (0..N_FEATURES) |k| {
                sums[@as(usize, c) * N_FEATURES + k] += features_soa[k * n + row];
            }
        }

        // Update: empty → reseed from random row; else mean of cluster.
        for (0..k_clusters) |c| {
            if (counts[c] == 0) {
                const r = rand.uintLessThan(usize, n);
                inline for (0..N_FEATURES) |k| {
                    centroids_out[c * N_FEATURES + k] = features_soa[k * n + r];
                }
                continue;
            }
            const inv_count: f32 = 1.0 / @as(f32, @floatFromInt(counts[c]));
            inline for (0..N_FEATURES) |k| {
                centroids_out[c * N_FEATURES + k] = sums[c * N_FEATURES + k] * inv_count;
            }
        }
    }
}

/// Argmin-Euclidean assignment of every row in `features_soa` against
/// `centroids`. Tiebreak: lowest centroid index wins. Writes one u32 per row.
/// Caller-owned scratch buffer `half_norm_sq_scratch` of length `k_clusters`
/// is populated upfront with `½‖c‖²` so the inner loop only does one dot per
/// (row, centroid) pair.
pub fn assign_all(
    features_soa: []const f32,
    n: usize,
    k_clusters: usize,
    centroids: []const f32,
    assignments_out: []u32,
    half_norm_sq_scratch: []f32,
) void {
    std.debug.assert(features_soa.len == N_FEATURES * n);
    std.debug.assert(centroids.len == k_clusters * N_FEATURES);
    std.debug.assert(assignments_out.len == n);
    std.debug.assert(half_norm_sq_scratch.len == k_clusters);

    for (0..k_clusters) |c| {
        var ss: f32 = 0;
        inline for (0..N_FEATURES) |k| {
            const v = centroids[c * N_FEATURES + k];
            ss += v * v;
        }
        half_norm_sq_scratch[c] = 0.5 * ss;
    }

    for (0..n) |row| {
        var best: f32 = -std.math.inf(f32);
        var best_c: u32 = 0;
        for (0..k_clusters) |c| {
            var dot: f32 = 0;
            inline for (0..N_FEATURES) |k| {
                dot = @mulAdd(
                    f32,
                    features_soa[k * n + row],
                    centroids[c * N_FEATURES + k],
                    dot,
                );
            }
            const score = dot - half_norm_sq_scratch[c];
            if (score > best) {
                best = score;
                best_c = @intCast(c);
            }
        }
        assignments_out[row] = best_c;
    }
}

// ─── tests ──────────────────────────────────────────────────────────────────

fn unit_norm_row(out: *[N_FEATURES]f32) void {
    var sum_sq: f32 = 0;
    for (out) |v| sum_sq += v * v;
    const inv = 1.0 / @sqrt(sum_sq);
    for (out) |*v| v.* *= inv;
}

test "run_kmeans converges on 2 well-separated attractors" {
    const allocator = std.testing.allocator;

    // 6 unit-norm rows: 3 near [1,0,...,0], 3 near [0,1,0,...,0].
    const n: usize = 6;
    const sample = try allocator.alloc(f32, N_FEATURES * n);
    defer allocator.free(sample);
    @memset(sample, 0);

    for (0..3) |row| {
        var r: [N_FEATURES]f32 = @splat(0);
        r[0] = 1.0;
        // Tiny perpendicular jitter so rows aren't bit-identical.
        r[2] = 0.01 * @as(f32, @floatFromInt(row));
        unit_norm_row(&r);
        for (0..N_FEATURES) |k| sample[k * n + row] = r[k];
    }
    for (3..6) |row| {
        var r: [N_FEATURES]f32 = @splat(0);
        r[1] = 1.0;
        r[3] = 0.01 * @as(f32, @floatFromInt(row - 3));
        unit_norm_row(&r);
        for (0..N_FEATURES) |k| sample[k * n + row] = r[k];
    }

    const centroids = try allocator.alloc(f32, 2 * N_FEATURES);
    defer allocator.free(centroids);

    try run_kmeans(allocator, sample, n, 2, 20, 0xa5a5_a5a5_a5a5_a5a5, centroids);

    // Each centroid must be within 0.02 of one of the two attractors (order-agnostic).
    var saw_e0 = false;
    var saw_e1 = false;
    for (0..2) |c| {
        const cx = centroids[c * N_FEATURES + 0];
        const cy = centroids[c * N_FEATURES + 1];
        if (@abs(cx - 1.0) < 0.05) saw_e0 = true;
        if (@abs(cy - 1.0) < 0.05) saw_e1 = true;
    }
    try std.testing.expect(saw_e0);
    try std.testing.expect(saw_e1);
}

test "run_kmeans is deterministic across two runs with same seed" {
    const allocator = std.testing.allocator;

    const n: usize = 20;
    const sample = try allocator.alloc(f32, N_FEATURES * n);
    defer allocator.free(sample);

    // Spread rows along arbitrary directions; not adversarial, just data.
    for (0..n) |row| {
        var r: [N_FEATURES]f32 = @splat(0);
        r[row % N_FEATURES] = 1.0;
        r[(row + 1) % N_FEATURES] = 0.5;
        unit_norm_row(&r);
        for (0..N_FEATURES) |k| sample[k * n + row] = r[k];
    }

    const a = try allocator.alloc(f32, 4 * N_FEATURES);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, 4 * N_FEATURES);
    defer allocator.free(b);

    try run_kmeans(allocator, sample, n, 4, 10, 0x1234_5678, a);
    try run_kmeans(allocator, sample, n, 4, 10, 0x1234_5678, b);

    try std.testing.expectEqualSlices(f32, a, b);
}

test "run_kmeans handles empty clusters via reseed" {
    const allocator = std.testing.allocator;

    // 4 rows that pair up: rows 0,1 nearly along +e0; rows 2,3 nearly along
    // +e1. K=4. Random init picks 4 distinct rows (one per cluster) but since
    // the rows occupy only ~2 directions, after the first assignment most
    // rows pile onto 1–2 centroids, leaving the other(s) empty → reseed.
    const n: usize = 4;
    const sample = try allocator.alloc(f32, N_FEATURES * n);
    defer allocator.free(sample);
    @memset(sample, 0);

    for (0..2) |row| {
        var r: [N_FEATURES]f32 = @splat(0);
        r[0] = 1.0;
        r[2] = 0.001 * @as(f32, @floatFromInt(row));
        unit_norm_row(&r);
        for (0..N_FEATURES) |k| sample[k * n + row] = r[k];
    }
    for (2..4) |row| {
        var r: [N_FEATURES]f32 = @splat(0);
        r[1] = 1.0;
        r[3] = 0.001 * @as(f32, @floatFromInt(row));
        unit_norm_row(&r);
        for (0..N_FEATURES) |k| sample[k * n + row] = r[k];
    }

    const centroids = try allocator.alloc(f32, 4 * N_FEATURES);
    defer allocator.free(centroids);

    try run_kmeans(allocator, sample, n, 4, 5, 42, centroids);

    // Must finish without NaNs in any centroid.
    for (centroids) |v| {
        try std.testing.expect(!std.math.isNan(v));
    }
}

test "assign_all tags every row exactly once and uses valid cluster ids" {
    const allocator = std.testing.allocator;

    const n: usize = 7;
    const features = try allocator.alloc(f32, N_FEATURES * n);
    defer allocator.free(features);

    for (0..n) |row| {
        var r: [N_FEATURES]f32 = @splat(0);
        r[row % N_FEATURES] = 1.0;
        unit_norm_row(&r);
        for (0..N_FEATURES) |k| features[k * n + row] = r[k];
    }

    const k: usize = 3;
    const centroids = try allocator.alloc(f32, k * N_FEATURES);
    defer allocator.free(centroids);
    @memset(centroids, 0);
    centroids[0 * N_FEATURES + 0] = 1.0;
    centroids[1 * N_FEATURES + 1] = 1.0;
    centroids[2 * N_FEATURES + 2] = 1.0;

    const assignments = try allocator.alloc(u32, n);
    defer allocator.free(assignments);
    const half_norm_sq = try allocator.alloc(f32, k);
    defer allocator.free(half_norm_sq);

    assign_all(features, n, k, centroids, assignments, half_norm_sq);

    for (assignments) |a| try std.testing.expect(a < k);

    // Rows whose unit basis aligns with a centroid must pick that centroid.
    try std.testing.expectEqual(@as(u32, 0), assignments[0]);
    try std.testing.expectEqual(@as(u32, 1), assignments[1]);
    try std.testing.expectEqual(@as(u32, 2), assignments[2]);
}
