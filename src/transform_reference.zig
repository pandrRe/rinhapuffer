const std = @import("std");
const fast_json = @import("fast_json.zig");

pub const N_FEATURES: usize = 14;

/// Fraud-detection dataset in true SoA column-major layout.
///
/// `features` is a flat slice of length `N_FEATURES * n`. The k-th feature of
/// record `r` is stored at `features[k * n + r]`, so each feature column is
/// contiguous across all records — ideal for per-feature SIMD reductions.
///
/// **Rows are L2-normalized**: `Σ_k features[k*n+r]² ≈ 1.0` for every row.
/// `parse_into` establishes this invariant; `search.cosine_topk` relies on it
/// to skip the per-row divide.
///
/// Buffers are caller-owned. `Dataset` only holds aliasing views.
pub const Dataset = struct {
    n: usize,
    features: []const f32,
    labels: []const bool,
};

/// Quantized SoA view over a u16-encoded dataset (the on-disk v2 format).
///
/// Each feature column is stored as `n` u16 values. The original f32 value is
/// reconstructed via `f = u16 * inv_scales[k] + mins[k]`, with `mins` and
/// `inv_scales` cached at load time from the per-feature `(min, scale)` pairs
/// in the blob header.
///
/// `cosine_topk_q` consumes this view directly without materialising f32
/// columns; only `@floatFromInt(u16) → f32` happens per row in the inner loop.
pub const QuantizedDataset = struct {
    n: usize,
    features: []const u16,
    labels: []const bool,
    mins: [N_FEATURES]f32,
    inv_scales: [N_FEATURES]f32,
};

/// IVF-augmented quantized SoA view: rows are reordered cluster-by-cluster
/// (so cluster `c` occupies `[cluster_starts[c], cluster_starts[c+1])` in
/// every feature column), with `centroids` row-major over `[k_clusters][14]f32`.
///
/// Search picks the top-N centroids by `dot(q, centroid)` then scans only
/// those clusters' row slices. The u16 features and per-feature
/// `(min, inv_scale)` are identical in shape and meaning to `QuantizedDataset`.
pub const IvfQuantizedDataset = struct {
    n: usize,
    k_clusters: usize,
    features: []const u16,
    labels: []const bool,
    centroids: []const f32,
    cluster_starts: []const u32,
    mins: [N_FEATURES]f32,
    inv_scales: [N_FEATURES]f32,
};

pub const Error = fast_json.ParseError || error{BufferTooSmall};

/// Number of records in `bytes`. Use to size feature/label buffers
/// before calling `parse_into`.
pub inline fn count_records(bytes: []const u8) usize {
    return fast_json.count_closing_braces(bytes);
}

/// Parse a reference-dataset JSON into caller-provided buffers. Writes
/// `N_FEATURES * n` floats into `features_buf` (column-major, **L2-normalized
/// per row**) and `n` bools into `labels_buf`. Returns a `Dataset` whose
/// slices alias the used prefixes of those buffers.
pub fn parse_into(
    bytes: []const u8,
    features_buf: []f32,
    labels_buf: []bool,
) Error!Dataset {
    const n = count_records(bytes);
    if (features_buf.len < N_FEATURES * n) return error.BufferTooSmall;
    if (labels_buf.len < n) return error.BufferTooSmall;

    const features = features_buf[0 .. N_FEATURES * n];
    const labels = labels_buf[0..n];

    if (n == 0) return .{ .n = 0, .features = features, .labels = labels };

    var p = fast_json.skip_ws(bytes, 0);
    p = try fast_json.expect_byte(bytes, p, '[');

    var row: usize = 0;
    while (row < n) : (row += 1) {
        p = fast_json.skip_ws(bytes, p);
        p = try fast_json.expect_byte(bytes, p, '{');
        p = fast_json.skip_ws(bytes, p);
        if (p + 8 > bytes.len) return error.UnexpectedEof;
        p += 8; // "vector"
        p = fast_json.skip_ws(bytes, p);
        p = try fast_json.expect_byte(bytes, p, ':');
        p = fast_json.skip_ws(bytes, p);
        p = try fast_json.expect_byte(bytes, p, '[');

        // Stash 14 raw f32s in a stack array so they stay register-resident
        // through the L2 normalization pass — avoids round-tripping through
        // the stride-n column-major buffer twice.
        var row_features: [N_FEATURES]f32 = undefined;
        var sum_sq: f32 = 0;
        inline for (0..N_FEATURES) |k| {
            p = fast_json.skip_ws(bytes, p);
            const r = try fast_json.parse_f32_simple(bytes, p);
            row_features[k] = r.value;
            sum_sq += r.value * r.value;
            p = r.next;
            if (k < N_FEATURES - 1) {
                p = fast_json.skip_ws(bytes, p);
                p = try fast_json.expect_byte(bytes, p, ',');
            }
        }
        std.debug.assert(sum_sq > 0);
        const inv_norm: f32 = 1.0 / @sqrt(sum_sq);
        inline for (0..N_FEATURES) |k| {
            features[k * n + row] = row_features[k] * inv_norm;
        }

        p = fast_json.skip_ws(bytes, p);
        p = try fast_json.expect_byte(bytes, p, ']');
        p = fast_json.skip_ws(bytes, p);
        p = try fast_json.expect_byte(bytes, p, ',');
        p = fast_json.skip_ws(bytes, p);
        if (p + 7 > bytes.len) return error.UnexpectedEof;
        p += 7; // "label"
        p = fast_json.skip_ws(bytes, p);
        p = try fast_json.expect_byte(bytes, p, ':');
        p = fast_json.skip_ws(bytes, p);
        p = try fast_json.expect_byte(bytes, p, '"');
        if (p + 5 > bytes.len) return error.UnexpectedEof;
        labels[row] = bytes[p] == 'f'; // 'f'raud vs 'l'egit
        p += 5;
        p = try fast_json.expect_byte(bytes, p, '"');
        p = fast_json.skip_ws(bytes, p);
        p = try fast_json.expect_byte(bytes, p, '}');
        p = fast_json.skip_ws(bytes, p);
        if (row + 1 < n) {
            p = try fast_json.expect_byte(bytes, p, ',');
        }
    }

    p = fast_json.skip_ws(bytes, p);
    p = try fast_json.expect_byte(bytes, p, ']');

    return .{ .n = n, .features = features, .labels = labels };
}

test "parse_into example references" {
    const allocator = std.testing.allocator;

    var mapped = try fast_json.mmap_file("./resources/example-references.json");
    defer mapped.deinit();

    const n = count_records(mapped.bytes);
    const features = try allocator.alloc(f32, N_FEATURES * n);
    defer allocator.free(features);
    const labels = try allocator.alloc(bool, n);
    defer allocator.free(labels);

    const dataset = try parse_into(mapped.bytes, features, labels);

    try std.testing.expectEqual(@as(usize, 100), dataset.n);
    try std.testing.expectEqual(N_FEATURES * dataset.n, dataset.features.len);
    try std.testing.expectEqual(dataset.n, dataset.labels.len);

    // Every row must be unit-norm post-parse (the Dataset invariant).
    for (0..dataset.n) |row| {
        var sum_sq: f64 = 0;
        for (0..N_FEATURES) |c| {
            const v: f64 = dataset.features[c * dataset.n + row];
            sum_sq += v * v;
        }
        try std.testing.expect(@abs(sum_sq - 1.0) <= 1.0e-5);
    }
}

test "parse_into matches std.json on example-references.json" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const file = try std.Io.Dir.cwd().openFile(io, "./resources/example-references.json", .{});
    defer file.close(io);
    const st = try file.stat(io);
    const size: usize = @intCast(st.size);
    const contents = try allocator.alloc(u8, size);
    defer allocator.free(contents);
    const read_n = try file.readPositionalAll(io, contents, 0);
    try std.testing.expectEqual(size, read_n);

    const Entry = struct { vector: [N_FEATURES]f32, label: []const u8 };
    const parsed = try std.json.parseFromSlice([]Entry, allocator, contents, .{});
    defer parsed.deinit();

    const n = count_records(contents);
    const features = try allocator.alloc(f32, N_FEATURES * n);
    defer allocator.free(features);
    const labels = try allocator.alloc(bool, n);
    defer allocator.free(labels);

    const dataset = try parse_into(contents, features, labels);

    try std.testing.expectEqual(parsed.value.len, dataset.n);
    for (parsed.value, 0..) |entry, row| {
        const expected_label = std.mem.eql(u8, entry.label, "fraud");
        try std.testing.expectEqual(expected_label, dataset.labels[row]);

        // Compare against the std.json values after applying the same
        // L2 normalization parse_into does.
        var sum_sq: f64 = 0;
        for (entry.vector) |v| sum_sq += @as(f64, v) * @as(f64, v);
        const inv_norm: f64 = 1.0 / @sqrt(sum_sq);
        for (entry.vector, 0..) |expected_v, c| {
            const got_v = dataset.features[c * dataset.n + row];
            const expected_norm: f32 = @floatCast(@as(f64, expected_v) * inv_norm);
            try std.testing.expect(@abs(got_v - expected_norm) <= 1.0e-5);
        }
    }
}

test "parse_into BufferTooSmall" {
    const allocator = std.testing.allocator;

    var mapped = try fast_json.mmap_file("./resources/example-references.json");
    defer mapped.deinit();

    const n = count_records(mapped.bytes);

    // features too small
    {
        var tiny_features: [10]f32 = undefined;
        const labels = try allocator.alloc(bool, n);
        defer allocator.free(labels);
        try std.testing.expectError(
            error.BufferTooSmall,
            parse_into(mapped.bytes, &tiny_features, labels),
        );
    }
    // labels too small
    {
        const features = try allocator.alloc(f32, N_FEATURES * n);
        defer allocator.free(features);
        var tiny_labels: [10]bool = undefined;
        try std.testing.expectError(
            error.BufferTooSmall,
            parse_into(mapped.bytes, features, &tiny_labels),
        );
    }
}
