const std = @import("std");
const fast_json = @import("fast_json.zig");

pub const N_FEATURES: usize = 14;

/// Fraud-detection dataset in true SoA column-major layout.
///
/// `features` is a flat slice of length `N_FEATURES * n`. The k-th feature of
/// record `r` is stored at `features[k * n + r]`, so each feature column is
/// contiguous across all records — ideal for per-feature SIMD reductions.
///
/// **Rows are raw [0,1] ∪ {−1}**: same value space as `payload.vectorize`'s
/// output (per-feature normalized to [0,1] with a sentinel −1 for null
/// `last_transaction` on features 5 and 6). No per-row L2 normalization —
/// `search.euclidean_topk` computes plain Euclidean distance.
///
/// Buffers are caller-owned. `Dataset` only holds aliasing views.
pub const Dataset = struct {
    n: usize,
    features: []const f32,
    labels: []const bool,
};

/// Quantized SoA view over an i16-encoded dataset.
///
/// Each feature column is stored as `n` i16 values, encoded with a single
/// global scale `search.FIX_SCALE`: `int_value = round(float_value * FIX_SCALE)`.
/// The hot path NEVER dequantizes — search ranks by `Σ (q_int − r_int)²` in
/// integer space, which is `FIX_SCALE²` times the true float distance and
/// therefore order-preserving.
///
/// `labels_bits` is a packed bitset of length `(n + 63) / 64` u64s, little-
/// endian within each word. Indexed via `dataset_blob.label_at(bits, row)`.
/// 8× tighter than `[]const bool` so the random TOP_K gather stays in L1.
pub const QuantizedDataset = struct {
    n: usize,
    features: []const i16,
    labels_bits: []const u64,
};

/// IVF-augmented quantized view with **block-SoA features** (v8 layout).
/// Rows are reordered cluster-by-cluster (cluster `c`'s canonical row range
/// is `[cluster_starts[c], cluster_starts[c+1])`), and within each cluster
/// the rows are stored in BLOCK_W-row blocks of `[N_FEATURES][BLOCK_W]i16`
/// col-major-within-block. Walking a cluster's blocks is one prefetcher
/// stream instead of 14 (vs the prior pure-SoA layout) — the layout
/// shipped by the rinha leaderboard's #1 thiagorigonatti and #4 joojf.
///
/// `bbox_lo` and `bbox_hi` are per-cluster, per-feature axis-aligned bounds in
/// i16 quantized units. Search picks the top-N centroids by min Euclidean
/// distance to the query, scans those clusters' blocks in int, then runs
/// a **bbox repair pass** over the remaining clusters: for each, compute the
/// axis-aligned lower-bound distance `LB² = Σ_k max(0, max(q_k − hi_k,
/// lo_k − q_k))²`. Skip if `LB² ≥ current K-th best`; scan otherwise. The
/// pass guarantees exact top-K regardless of PROBE.
pub const IvfQuantizedDataset = struct {
    n: usize,
    k_clusters: usize,
    /// Block-SoA features. Length = `total_blocks * N_FEATURES * BLOCK_W`.
    /// Block b of cluster c lives at
    ///   `[(cluster_block_starts[c] + b) * N_FEATURES * BLOCK_W ..]
    ///    [0 .. N_FEATURES * BLOCK_W]`
    /// Within a block, features are col-major: feature k's BLOCK_W lanes
    /// at offset `k * BLOCK_W`. Lane l corresponds to canonical row
    /// `cluster_starts[c] + b * BLOCK_W + l`, padded with sentinel on the
    /// last block of each cluster when its row count isn't a multiple of W.
    block_features: []const i16,
    /// Packed bitset of length `(n + 63) / 64` u64s, little-endian within
    /// each word. Indexed via `dataset_blob.label_at(bits, row)`.
    labels_bits: []const u64,
    centroids: []const f32,
    /// Canonical row count per cluster (no padding). `cluster_starts[K] == n`.
    cluster_starts: []const u32,
    /// Cumulative block count per cluster. Used to address `block_features`.
    /// `cluster_block_starts[K]` is the total block count.
    cluster_block_starts: []const u32,
    /// `[k_clusters * N_FEATURES]i16` — per-(cluster, feature) min, AoS by cluster.
    bbox_lo: []const i16,
    /// `[k_clusters * N_FEATURES]i16` — per-(cluster, feature) max, AoS by cluster.
    bbox_hi: []const i16,
};

pub const Error = fast_json.ParseError || error{BufferTooSmall};

/// Number of records in `bytes`. Use to size feature/label buffers
/// before calling `parse_into`.
pub inline fn count_records(bytes: []const u8) usize {
    return fast_json.count_closing_braces(bytes);
}

/// Parse a reference-dataset JSON into caller-provided buffers. Writes
/// `N_FEATURES * n` floats into `features_buf` (column-major, **raw values
/// in [0,1] ∪ {−1}** — same value space as `payload.vectorize`'s output)
/// and `n` bools into `labels_buf`. Returns a `Dataset` whose slices alias
/// the used prefixes of those buffers.
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

        inline for (0..N_FEATURES) |k| {
            p = fast_json.skip_ws(bytes, p);
            const r = try fast_json.parse_f32_simple(bytes, p);
            features[k * n + row] = r.value;
            p = r.next;
            if (k < N_FEATURES - 1) {
                p = fast_json.skip_ws(bytes, p);
                p = try fast_json.expect_byte(bytes, p, ',');
            }
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

    // Every value lives in [0, 1] except features 5 and 6 which can be the
    // null-`last_transaction` sentinel −1.
    for (0..dataset.n) |row| {
        for (0..N_FEATURES) |c| {
            const v = dataset.features[c * dataset.n + row];
            if (c == 5 or c == 6) {
                try std.testing.expect(v == -1.0 or (v >= 0.0 and v <= 1.0));
            } else {
                try std.testing.expect(v >= 0.0 and v <= 1.0);
            }
        }
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

        // parse_into stores raw values from the JSON; compare directly.
        for (entry.vector, 0..) |expected_v, c| {
            const got_v = dataset.features[c * dataset.n + row];
            try std.testing.expectEqual(expected_v, got_v);
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
