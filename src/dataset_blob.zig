//! On-disk format for the prepped reference dataset, plus mmap-only `load`
//! and `write` (used by the `prep` build step).
//!
//! Layout (little-endian, native f32 IEEE 754):
//!
//!     offset    size                          field
//!     0         4                             magic = "RBP1"
//!     4         4                             version: u32 = 1
//!     8         4                             n: u32   (record count)
//!     12        4                             _pad: u32 = 0  (aligns features to 16)
//!     16        N_FEATURES * n * 4            features: f32 column-major, L2-normalized
//!     ..        n                             labels: u8 (0=legit, 1=fraud)
//!
//! Endianness is native LE — both dev (macOS arm64) and target (x86_64 Linux)
//! are LE. Cross-arch deployment is out of scope and would require a version
//! bump if it ever mattered.

const std = @import("std");
const transform_reference = @import("transform_reference.zig");
const fast_json = @import("fast_json.zig");

const N_FEATURES = transform_reference.N_FEATURES;

pub const MAGIC: u32 = std.mem.readInt(u32, "RBP1", .little);
pub const VERSION: u32 = 1;

pub const Header = extern struct {
    magic: u32,
    version: u32,
    n: u32,
    _pad: u32,
};

pub const HEADER_SIZE = @sizeOf(Header); // 16

/// Bytes on disk for a dataset with `n` records.
pub fn blob_size(n: u32) usize {
    return HEADER_SIZE + N_FEATURES * @as(usize, n) * @sizeOf(f32) + @as(usize, n);
}

pub const LoadError = error{
    BadMagic,
    UnsupportedVersion,
    Truncated,
} || fast_json.MmapError;

pub const WriteError = error{TooManyRecords} || std.Io.File.OpenError || std.Io.File.Writer.Error;

/// mmap-backed Dataset. `deinit` munmaps.
pub const Blob = struct {
    mapped: fast_json.Mapped,
    dataset: transform_reference.Dataset,

    pub fn deinit(self: Blob) void {
        self.mapped.deinit();
    }
};

/// mmap `path`, validate the header, return a Dataset view aliasing the mapping.
/// No body parsing — the kernel faults pages in lazily on first access.
pub fn load(path: []const u8) LoadError!Blob {
    var mapped = try fast_json.mmap_file(path);
    errdefer mapped.deinit();

    if (mapped.bytes.len < HEADER_SIZE) return error.Truncated;
    const hdr: *const Header = @ptrCast(@alignCast(mapped.bytes.ptr));
    if (hdr.magic != MAGIC) return error.BadMagic;
    if (hdr.version != VERSION) return error.UnsupportedVersion;

    const n = hdr.n;
    if (mapped.bytes.len < blob_size(n)) return error.Truncated;

    const features_count = @as(usize, n) * N_FEATURES;
    const features_ptr: [*]const f32 = @ptrCast(@alignCast(mapped.bytes.ptr + HEADER_SIZE));
    const features = features_ptr[0..features_count];

    const labels_offset = HEADER_SIZE + features_count * @sizeOf(f32);
    // Labels are written as 0/1 bytes by `write`; Zig bool ABI is 1 byte with
    // values {0, 1}, so reinterpreting the slice as []const bool is well-defined
    // as long as the producer is `write` (it is).
    const labels_ptr: [*]const bool = @ptrCast(mapped.bytes.ptr + labels_offset);
    const labels = labels_ptr[0..n];

    return .{
        .mapped = mapped,
        .dataset = .{ .n = n, .features = features, .labels = labels },
    };
}

/// Serialize a Dataset to `<dir>/<sub_path>`, truncating an existing file.
/// Used by the `prep` build step.
pub fn write(
    io: std.Io,
    dir: std.Io.Dir,
    sub_path: []const u8,
    ds: transform_reference.Dataset,
) WriteError!void {
    if (ds.n > std.math.maxInt(u32)) return error.TooManyRecords;
    const n: u32 = @intCast(ds.n);

    const file = try dir.createFile(io, sub_path, .{});
    defer file.close(io);

    const hdr: Header = .{ .magic = MAGIC, .version = VERSION, .n = n, ._pad = 0 };
    try file.writeStreamingAll(io, std.mem.asBytes(&hdr));
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(ds.features));

    // Labels: bool → u8 (0/1) in a 4 KiB stack buffer, no temp alloc.
    var buf: [4096]u8 = undefined;
    var i: usize = 0;
    while (i < ds.labels.len) {
        const chunk_end = @min(i + buf.len, ds.labels.len);
        for (ds.labels[i..chunk_end], 0..) |l, j| buf[j] = @intFromBool(l);
        try file.writeStreamingAll(io, buf[0 .. chunk_end - i]);
        i = chunk_end;
    }
}

// ─── tests ──────────────────────────────────────────────────────────────────

const search = @import("search.zig");

fn tmp_abs_path(
    allocator: std.mem.Allocator,
    tmp: *std.testing.TmpDir,
    sub_path: []const u8,
) ![]u8 {
    var dir_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const dir_len = try tmp.dir.realPath(std.testing.io, &dir_buf);
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir_buf[0..dir_len], sub_path });
}

test "write/load round-trip on example-references.json" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var src = try fast_json.mmap_file("./resources/example-references.json");
    defer src.deinit();

    const n = transform_reference.count_records(src.bytes);
    const features = try allocator.alloc(f32, N_FEATURES * n);
    defer allocator.free(features);
    const labels = try allocator.alloc(bool, n);
    defer allocator.free(labels);
    const ds = try transform_reference.parse_into(src.bytes, features, labels);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try write(io, tmp.dir, "dataset.bin", ds);

    const path = try tmp_abs_path(allocator, &tmp, "dataset.bin");
    defer allocator.free(path);
    var blob = try load(path);
    defer blob.deinit();

    try std.testing.expectEqual(ds.n, blob.dataset.n);
    try std.testing.expectEqualSlices(f32, ds.features, blob.dataset.features);
    try std.testing.expectEqualSlices(bool, ds.labels, blob.dataset.labels);
}

test "loaded blob produces same top-K as in-memory dataset" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var src = try fast_json.mmap_file("./resources/example-references.json");
    defer src.deinit();

    const n = transform_reference.count_records(src.bytes);
    const features = try allocator.alloc(f32, N_FEATURES * n);
    defer allocator.free(features);
    const labels = try allocator.alloc(bool, n);
    defer allocator.free(labels);
    const ds_mem = try transform_reference.parse_into(src.bytes, features, labels);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(io, tmp.dir, "dataset.bin", ds_mem);
    const path = try tmp_abs_path(allocator, &tmp, "dataset.bin");
    defer allocator.free(path);
    var blob = try load(path);
    defer blob.deinit();

    // Three queries: a row of the dataset, a hand-rolled vector, and a constructed mix.
    var queries: [3][N_FEATURES]f32 = undefined;
    for (0..N_FEATURES) |c| queries[0][c] = ds_mem.features[c * n + 7];
    queries[1] = .{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    queries[2] = .{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4 };

    for (&queries) |*q| {
        var got_mem: [search.TOP_K]u32 = undefined;
        var got_blob: [search.TOP_K]u32 = undefined;
        search.cosine_topk(ds_mem, q, &got_mem);
        search.cosine_topk(blob.dataset, q, &got_blob);
        try std.testing.expectEqualSlices(u32, &got_mem, &got_blob);
    }
}

test "load rejects bad magic" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    {
        var f = try tmp.dir.createFile(io, "bad.bin", .{});
        defer f.close(io);
        var bad: [HEADER_SIZE]u8 = @splat(0);
        try f.writeStreamingAll(io, &bad);
    }

    const path = try tmp_abs_path(allocator, &tmp, "bad.bin");
    defer allocator.free(path);
    try std.testing.expectError(error.BadMagic, load(path));
}

test "load rejects unsupported version" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    {
        var f = try tmp.dir.createFile(io, "wrongver.bin", .{});
        defer f.close(io);
        const hdr: Header = .{ .magic = MAGIC, .version = 99, .n = 0, ._pad = 0 };
        try f.writeStreamingAll(io, std.mem.asBytes(&hdr));
    }

    const path = try tmp_abs_path(allocator, &tmp, "wrongver.bin");
    defer allocator.free(path);
    try std.testing.expectError(error.UnsupportedVersion, load(path));
}

test "load rejects truncated body" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    {
        var f = try tmp.dir.createFile(io, "trunc.bin", .{});
        defer f.close(io);
        // Header claims n=10 but file contains only the header.
        const hdr: Header = .{ .magic = MAGIC, .version = VERSION, .n = 10, ._pad = 0 };
        try f.writeStreamingAll(io, std.mem.asBytes(&hdr));
    }

    const path = try tmp_abs_path(allocator, &tmp, "trunc.bin");
    defer allocator.free(path);
    try std.testing.expectError(error.Truncated, load(path));
}
