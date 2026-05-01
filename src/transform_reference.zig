const std = @import("std");
const fast_json = @import("fast_json.zig");

pub const N_FEATURES: usize = 14;

/// Fraud-detection dataset in true SoA column-major layout.
///
/// `features` is a flat slice of length `N_FEATURES * n`. The k-th feature of
/// record `r` is stored at `features[k * n + r]`, so each feature column is
/// contiguous across all records — ideal for per-feature SIMD reductions.
pub const Dataset = struct {
    n: usize,
    features: []f32,
    labels: []bool,

    pub fn deinit(self: *Dataset, allocator: std.mem.Allocator) void {
        allocator.free(self.features);
        allocator.free(self.labels);
    }

    pub fn col(self: Dataset, c: usize) []f32 {
        return self.features[c * self.n .. (c + 1) * self.n];
    }

    pub fn feature(self: Dataset, row: usize, c: usize) f32 {
        return self.features[c * self.n + row];
    }
};

/// Load a reference dataset from `path` into a SoA `Dataset`.
///
/// mmaps the file, counts records via a SIMD '}' scan, then parses with a
/// hand-rolled loop specialised for the
/// `[{ "vector": [...14 f32...], "label": "..." }, ...]` shape.
/// No general-purpose JSON parser is used.
pub fn load_dataset(allocator: std.mem.Allocator, path: []const u8) !Dataset {
    var mapped = try fast_json.mmap_file(path);
    defer mapped.deinit();
    const b = mapped.bytes;

    const n = fast_json.count_closing_braces(b);
    if (n == 0) {
        const empty_f = try allocator.alloc(f32, 0);
        const empty_l = try allocator.alloc(bool, 0);
        return .{ .n = 0, .features = empty_f, .labels = empty_l };
    }

    const features = try allocator.alloc(f32, N_FEATURES * n);
    errdefer allocator.free(features);
    const labels = try allocator.alloc(bool, n);
    errdefer allocator.free(labels);

    var p = fast_json.skip_ws(b, 0);
    p = try fast_json.expect_byte(b, p, '[');

    var row: usize = 0;
    while (row < n) : (row += 1) {
        p = fast_json.skip_ws(b, p);
        p = try fast_json.expect_byte(b, p, '{');
        p = fast_json.skip_ws(b, p);
        if (p + 8 > b.len) return error.UnexpectedEof;
        p += 8; // "vector"
        p = fast_json.skip_ws(b, p);
        p = try fast_json.expect_byte(b, p, ':');
        p = fast_json.skip_ws(b, p);
        p = try fast_json.expect_byte(b, p, '[');

        inline for (0..N_FEATURES) |k| {
            p = fast_json.skip_ws(b, p);
            const r = try fast_json.parse_f32_simple(b, p);
            features[k * n + row] = r.value;
            p = r.next;
            if (k < N_FEATURES - 1) {
                p = fast_json.skip_ws(b, p);
                p = try fast_json.expect_byte(b, p, ',');
            }
        }

        p = fast_json.skip_ws(b, p);
        p = try fast_json.expect_byte(b, p, ']');
        p = fast_json.skip_ws(b, p);
        p = try fast_json.expect_byte(b, p, ',');
        p = fast_json.skip_ws(b, p);
        if (p + 7 > b.len) return error.UnexpectedEof;
        p += 7; // "label"
        p = fast_json.skip_ws(b, p);
        p = try fast_json.expect_byte(b, p, ':');
        p = fast_json.skip_ws(b, p);
        p = try fast_json.expect_byte(b, p, '"');
        if (p + 5 > b.len) return error.UnexpectedEof;
        labels[row] = b[p] == 'f'; // 'f'raud vs 'l'egit
        p += 5;
        p = try fast_json.expect_byte(b, p, '"');
        p = fast_json.skip_ws(b, p);
        p = try fast_json.expect_byte(b, p, '}');
        p = fast_json.skip_ws(b, p);
        if (row + 1 < n) {
            p = try fast_json.expect_byte(b, p, ',');
        }
    }

    p = fast_json.skip_ws(b, p);
    p = try fast_json.expect_byte(b, p, ']');

    return .{ .n = n, .features = features, .labels = labels };
}

test "load reference dataset" {
    const allocator = std.testing.allocator;
    var dataset = try load_dataset(allocator, "./resources/example-references.json");
    defer dataset.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 100), dataset.n);
    try std.testing.expectEqual(N_FEATURES * dataset.n, dataset.features.len);
    try std.testing.expectEqual(dataset.n, dataset.labels.len);

    var fraud_count: usize = 0;
    for (dataset.labels) |is_fraud| {
        if (is_fraud) fraud_count += 1;
    }
    std.debug.print("\nloaded {} records, {} fraud, {} legit\n", .{
        dataset.n,
        fraud_count,
        dataset.n - fraud_count,
    });
}

test "load_dataset matches std.json on example-references.json" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // Reference parse using std.json — only used to validate the fast loader.
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

    var dataset = try load_dataset(allocator, "./resources/example-references.json");
    defer dataset.deinit(allocator);

    try std.testing.expectEqual(parsed.value.len, dataset.n);
    for (parsed.value, 0..) |entry, row| {
        const expected_label = std.mem.eql(u8, entry.label, "fraud");
        try std.testing.expectEqual(expected_label, dataset.labels[row]);
        for (entry.vector, 0..) |expected_v, c| {
            const got_v = dataset.features[c * dataset.n + row];
            try std.testing.expect(@abs(got_v - expected_v) <= 1.0e-6);
        }
    }
}
