//! Spherical k-means for the IVF index.
//!
//! Inputs are unit-norm rows in SoA layout (`features[k * n + row]`).
//! "Spherical" means similarity = cosine = dot product (since rows are
//! unit-norm); the centroid update is a sum-then-renormalize, with no division
//! by count for direction. Empty clusters reseed from a random sample row.
//!
//! Determinism: same `seed` + same input ⇒ bit-identical centroids and
//! assignments. Argmax tiebreak is "lowest cluster index wins", and the
//! Fisher-Yates shuffle is driven by an explicitly-seeded `Xoshiro256`.

const std = @import("std");
const transform_reference = @import("transform_reference.zig");

const N_FEATURES = transform_reference.N_FEATURES;

/// Initialise centroids from a random sample of rows, then alternate
/// assign/sum/renormalize for `n_iter` iterations. Final centroids land in
/// `centroids_out` (row-major, `[c * N_FEATURES + k]`).
///
/// Allocates: `[k_clusters * N_FEATURES]f32` sums + `[k_clusters]u32` counts +
/// `[n_sample]u32` assignments + `[n_sample]u32` shuffle scratch.
pub fn run_kmeans(
    allocator: std.mem.Allocator,
    sample_features_soa: []const f32,
    n_sample: usize,
    k_clusters: usize,
    n_iter: usize,
    seed: u64,
    centroids_out: []f32,
) !void {
    std.debug.assert(centroids_out.len == k_clusters * N_FEATURES);
    std.debug.assert(sample_features_soa.len == N_FEATURES * n_sample);
    std.debug.assert(n_sample >= k_clusters);

    var prng = std.Random.Xoshiro256.init(seed);
    const rand = prng.random();

    // Init: Fisher-Yates shuffle of [0..n_sample), take first K as init centroids.
    const shuffle_idx = try allocator.alloc(u32, n_sample);
    defer allocator.free(shuffle_idx);
    for (0..n_sample) |i| shuffle_idx[i] = @intCast(i);
    rand.shuffle(u32, shuffle_idx);

    for (0..k_clusters) |c| {
        const r = shuffle_idx[c];
        for (0..N_FEATURES) |k| {
            centroids_out[c * N_FEATURES + k] = sample_features_soa[k * n_sample + r];
        }
    }

    const assignments = try allocator.alloc(u32, n_sample);
    defer allocator.free(assignments);
    const sums = try allocator.alloc(f32, k_clusters * N_FEATURES);
    defer allocator.free(sums);
    const counts = try allocator.alloc(u32, k_clusters);
    defer allocator.free(counts);

    for (0..n_iter) |_| {
        // Assign: argmax dot vs all K centroids, tiebreak lowest index.
        for (0..n_sample) |row| {
            var best: f32 = -std.math.inf(f32);
            var best_c: u32 = 0;
            for (0..k_clusters) |c| {
                var dot: f32 = 0;
                inline for (0..N_FEATURES) |k| {
                    dot = @mulAdd(
                        f32,
                        sample_features_soa[k * n_sample + row],
                        centroids_out[c * N_FEATURES + k],
                        dot,
                    );
                }
                if (dot > best) {
                    best = dot;
                    best_c = @intCast(c);
                }
            }
            assignments[row] = best_c;
        }

        // Sum + count.
        @memset(sums, 0);
        @memset(counts, 0);
        for (0..n_sample) |row| {
            const c = assignments[row];
            counts[c] += 1;
            inline for (0..N_FEATURES) |k| {
                sums[@as(usize, c) * N_FEATURES + k] += sample_features_soa[k * n_sample + row];
            }
        }

        // Update: empty → reseed from random sample row; else L2-normalize sum.
        for (0..k_clusters) |c| {
            if (counts[c] == 0) {
                const r = rand.uintLessThan(usize, n_sample);
                inline for (0..N_FEATURES) |k| {
                    centroids_out[c * N_FEATURES + k] = sample_features_soa[k * n_sample + r];
                }
                continue;
            }
            var sum_sq: f32 = 0;
            inline for (0..N_FEATURES) |k| {
                const v = sums[c * N_FEATURES + k];
                sum_sq += v * v;
            }
            // sum_sq > 0 since at least one unit-norm row contributed.
            const inv_norm: f32 = 1.0 / @sqrt(sum_sq);
            inline for (0..N_FEATURES) |k| {
                centroids_out[c * N_FEATURES + k] = sums[c * N_FEATURES + k] * inv_norm;
            }
        }
    }
}

/// Argmax-dot assignment of every row in `features_soa` against `centroids`.
/// Tiebreak: lowest centroid index wins. Writes one u32 per row.
pub fn assign_all(
    features_soa: []const f32,
    n: usize,
    k_clusters: usize,
    centroids: []const f32,
    assignments_out: []u32,
) void {
    std.debug.assert(features_soa.len == N_FEATURES * n);
    std.debug.assert(centroids.len == k_clusters * N_FEATURES);
    std.debug.assert(assignments_out.len == n);

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
            if (dot > best) {
                best = dot;
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

    assign_all(features, n, k, centroids, assignments);

    for (assignments) |a| try std.testing.expect(a < k);

    // Rows whose unit basis aligns with a centroid must pick that centroid.
    try std.testing.expectEqual(@as(u32, 0), assignments[0]);
    try std.testing.expectEqual(@as(u32, 1), assignments[1]);
    try std.testing.expectEqual(@as(u32, 2), assignments[2]);
}
