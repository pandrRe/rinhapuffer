//! On-disk format for the prepped reference dataset, plus mmap-only `load`
//! and `write` (used by the `prep` build step).
//!
//! Layout v2 (little-endian, native f32 IEEE 754):
//!
//!     offset    size                          field
//!     0         4                             magic = "RBP1"
//!     4         4                             version: u32 = 2
//!     8         4                             n: u32   (record count)
//!     12        4                             _pad: u32 = 0
//!     16        N_FEATURES * 8                quant_params: [N_FEATURES] of {min: f32, scale: f32}
//!     128       N_FEATURES * n * 2            features: u16 column-major, quantized
//!     ..        n                             labels: u8 (0=legit, 1=fraud)
//!
//! Quantization: `q_u16 = clamp(round((f - min) * scale), 0, 65535)` per column.
//! Decode: `f ≈ q_u16 * inv_scale + min`, where `inv_scale = 1/scale` is cached
//! at load time. With per-column `(min, scale)` and 16 bits of resolution, each
//! feature retains ~5e-5 relative precision against the L2-normalised input.
//!
//! Endianness is native LE — both dev (macOS arm64) and target (x86_64 Linux)
//! are LE. Cross-arch deployment is out of scope; a future endian-aware format
//! would bump VERSION.

const std = @import("std");
const transform_reference = @import("transform_reference.zig");
const fast_json = @import("fast_json.zig");

const N_FEATURES = transform_reference.N_FEATURES;

pub const MAGIC: u32 = std.mem.readInt(u32, "RBP1", .little);
pub const VERSION: u32 = 2;

pub const QuantParam = extern struct {
    min: f32,
    scale: f32,
};

pub const Header = extern struct {
    magic: u32,
    version: u32,
    n: u32,
    _pad: u32,
    quant_params: [N_FEATURES]QuantParam,
};

pub const HEADER_SIZE = @sizeOf(Header); // 16 + 14*8 = 128

/// Bytes on disk for a dataset with `n` records.
pub fn blob_size(n: u32) usize {
    return HEADER_SIZE + N_FEATURES * @as(usize, n) * @sizeOf(u16) + @as(usize, n);
}

pub const LoadError = error{
    BadMagic,
    UnsupportedVersion,
    Truncated,
} || fast_json.MmapError;

pub const WriteError = error{TooManyRecords} || std.Io.File.OpenError || std.Io.File.Writer.Error;

/// Per-column min/scale derived from an f32 dataset. `scale = 65535 / range`
/// (or 0 for a constant column, which dequantises to `min` for every row).
pub fn compute_quant_params(features: []const f32, n: usize) [N_FEATURES]QuantParam {
    var out: [N_FEATURES]QuantParam = undefined;
    for (0..N_FEATURES) |k| {
        const col = features[k * n .. (k + 1) * n];
        var lo: f32 = col[0];
        var hi: f32 = col[0];
        for (col[1..]) |v| {
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        const range = hi - lo;
        const scale: f32 = if (range > 0) 65535.0 / range else 0.0;
        out[k] = .{ .min = lo, .scale = scale };
    }
    return out;
}

/// mmap-backed quantized dataset. `deinit` munmaps.
pub const QuantizedBlob = struct {
    mapped: fast_json.Mapped,
    dataset: transform_reference.QuantizedDataset,

    pub fn deinit(self: QuantizedBlob) void {
        self.mapped.deinit();
    }
};

/// mmap `path`, validate the header, return a QuantizedDataset view aliasing
/// the mapping. No body parsing — the kernel faults pages in lazily on first
/// access. Per-feature `inv_scales = 1/scales` are computed once here and
/// stashed in the returned view.
pub fn load(path: []const u8) LoadError!QuantizedBlob {
    var mapped = try fast_json.mmap_file(path);
    errdefer mapped.deinit();

    if (mapped.bytes.len < HEADER_SIZE) return error.Truncated;
    const hdr: *const Header = @ptrCast(@alignCast(mapped.bytes.ptr));
    if (hdr.magic != MAGIC) return error.BadMagic;
    if (hdr.version != VERSION) return error.UnsupportedVersion;

    const n = hdr.n;
    if (mapped.bytes.len < blob_size(n)) return error.Truncated;

    const features_count = @as(usize, n) * N_FEATURES;
    const features_ptr: [*]const u16 = @ptrCast(@alignCast(mapped.bytes.ptr + HEADER_SIZE));
    const features = features_ptr[0..features_count];

    const labels_offset = HEADER_SIZE + features_count * @sizeOf(u16);
    // Labels written as 0/1 bytes by `write`; Zig bool ABI is 1 byte with
    // values {0, 1}, so reinterpreting as []const bool is well-defined as
    // long as the producer is `write` (it is).
    const labels_ptr: [*]const bool = @ptrCast(mapped.bytes.ptr + labels_offset);
    const labels = labels_ptr[0..n];

    var mins: [N_FEATURES]f32 = undefined;
    var inv_scales: [N_FEATURES]f32 = undefined;
    inline for (0..N_FEATURES) |k| {
        mins[k] = hdr.quant_params[k].min;
        const s = hdr.quant_params[k].scale;
        inv_scales[k] = if (s != 0) 1.0 / s else 0.0;
    }

    return .{
        .mapped = mapped,
        .dataset = .{
            .n = n,
            .features = features,
            .labels = labels,
            .mins = mins,
            .inv_scales = inv_scales,
        },
    };
}

/// f32 dataset materialised by dequantizing a v2 blob. Owns the f32 feature
/// and label buffers; the source mmap is closed before this returns.
/// Provided for benchmarking parity against `cosine_topk` — production never
/// pays the 168 MB allocation.
pub const UnquantBlob = struct {
    allocator: std.mem.Allocator,
    features: []f32,
    labels: []bool,
    dataset: transform_reference.Dataset,

    pub fn deinit(self: *UnquantBlob) void {
        self.allocator.free(self.features);
        self.allocator.free(self.labels);
    }
};

/// Load the v2 blob and dequantize every feature into a fresh f32 buffer.
/// Allocates `N_FEATURES * n * 4` + `n` bytes (~171 MB at full size). Used by
/// the bench harness to pit `cosine_topk` (f32) against `cosine_topk_q` (u16)
/// over the same on-disk dataset.
pub fn load_unquant(allocator: std.mem.Allocator, path: []const u8) !UnquantBlob {
    var blob = try load(path);
    defer blob.deinit();
    const qds = blob.dataset;

    const features = try allocator.alloc(f32, N_FEATURES * qds.n);
    errdefer allocator.free(features);
    const labels = try allocator.alloc(bool, qds.n);
    errdefer allocator.free(labels);

    for (0..N_FEATURES) |k| {
        const inv_scale = qds.inv_scales[k];
        const min = qds.mins[k];
        for (0..qds.n) |row| {
            const u = qds.features[k * qds.n + row];
            features[k * qds.n + row] = @as(f32, @floatFromInt(u)) * inv_scale + min;
        }
    }
    @memcpy(labels, qds.labels);

    return .{
        .allocator = allocator,
        .features = features,
        .labels = labels,
        .dataset = .{ .n = qds.n, .features = features, .labels = labels },
    };
}

/// Serialize a Dataset to `<dir>/<sub_path>`, truncating an existing file.
/// Quantizes per-column to u16 against `(min, scale)` derived from the input.
/// Used by the `prep` build step.
pub fn write(
    io: std.Io,
    dir: std.Io.Dir,
    sub_path: []const u8,
    ds: transform_reference.Dataset,
) WriteError!void {
    if (ds.n > std.math.maxInt(u32)) return error.TooManyRecords;
    const n_u32: u32 = @intCast(ds.n);

    const params = compute_quant_params(ds.features, ds.n);

    const file = try dir.createFile(io, sub_path, .{});
    defer file.close(io);

    const hdr: Header = .{
        .magic = MAGIC,
        .version = VERSION,
        .n = n_u32,
        ._pad = 0,
        .quant_params = params,
    };
    try file.writeStreamingAll(io, std.mem.asBytes(&hdr));

    // Features: 14 columns × n rows, quantized, written column-major in
    // 2048-row chunks (a 4 KiB stack buffer of u16).
    var u16_chunk: [2048]u16 = undefined;
    for (0..N_FEATURES) |k| {
        const min = params[k].min;
        const scale = params[k].scale;
        const col = ds.features[k * ds.n .. (k + 1) * ds.n];
        var i: usize = 0;
        while (i < col.len) {
            const end = @min(i + u16_chunk.len, col.len);
            for (col[i..end], 0..) |f, j| {
                const q = @round((f - min) * scale);
                const clamped = @max(0.0, @min(65535.0, q));
                u16_chunk[j] = @intFromFloat(clamped);
            }
            try file.writeStreamingAll(io, std.mem.sliceAsBytes(u16_chunk[0 .. end - i]));
            i = end;
        }
    }

    // Labels: bool → u8 (0/1) in a 4 KiB stack buffer, no temp alloc.
    var byte_chunk: [4096]u8 = undefined;
    var i: usize = 0;
    while (i < ds.labels.len) {
        const end = @min(i + byte_chunk.len, ds.labels.len);
        for (ds.labels[i..end], 0..) |l, j| byte_chunk[j] = @intFromBool(l);
        try file.writeStreamingAll(io, byte_chunk[0 .. end - i]);
        i = end;
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

test "write/load round-trip with quantization precision check" {
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

    const qds = blob.dataset;
    try std.testing.expectEqual(ds.n, qds.n);
    try std.testing.expectEqualSlices(bool, ds.labels, qds.labels);

    // Header `quant_params` should match recomputed params exactly, and
    // `inv_scales` should be the reciprocals.
    const recomputed = compute_quant_params(ds.features, ds.n);
    for (0..N_FEATURES) |k| {
        try std.testing.expectEqual(recomputed[k].min, qds.mins[k]);
        const expected_inv = if (recomputed[k].scale != 0) 1.0 / recomputed[k].scale else 0.0;
        try std.testing.expectEqual(expected_inv, qds.inv_scales[k]);
    }

    // Per-element precision: |f_orig - dequant(q)| <= 1 quantization step.
    for (0..N_FEATURES) |k| {
        const range = if (recomputed[k].scale != 0) 65535.0 / recomputed[k].scale else 0.0;
        const tol: f32 = @floatCast(1.0001 * range / 65535.0);
        for (0..ds.n) |row| {
            const f_orig = ds.features[k * ds.n + row];
            const q_u16 = qds.features[k * ds.n + row];
            const f_dec: f32 = @as(f32, @floatFromInt(q_u16)) * qds.inv_scales[k] + qds.mins[k];
            try std.testing.expect(@abs(f_orig - f_dec) <= tol + 1.0e-7);
        }
    }
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

    // Three queries: a row of the dataset, a hand-rolled vector, a constructed mix.
    var queries: [3][N_FEATURES]f32 = undefined;
    for (0..N_FEATURES) |c| queries[0][c] = ds_mem.features[c * n + 7];
    queries[1] = .{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    queries[2] = .{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4 };

    for (&queries) |*q| {
        var got_mem: [search.TOP_K]u32 = undefined;
        var got_blob: [search.TOP_K]u32 = undefined;
        search.cosine_topk(ds_mem, q, &got_mem);
        search.cosine_topk_q(blob.dataset, q, &got_blob);
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
        const hdr: Header = .{
            .magic = MAGIC,
            .version = 99,
            .n = 0,
            ._pad = 0,
            .quant_params = @splat(.{ .min = 0, .scale = 0 }),
        };
        try f.writeStreamingAll(io, std.mem.asBytes(&hdr));
    }

    const path = try tmp_abs_path(allocator, &tmp, "wrongver.bin");
    defer allocator.free(path);
    try std.testing.expectError(error.UnsupportedVersion, load(path));
}

test "load rejects v1 (old format)" {
    // Writes a header that looks like v1: 16 bytes, magic + version=1 + n + pad.
    // Our v2 reader must reject it via the version check.
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    {
        var f = try tmp.dir.createFile(io, "v1.bin", .{});
        defer f.close(io);
        // Write a v2-shaped header on disk (so the file is at least HEADER_SIZE)
        // but with version=1; v2 reader rejects.
        const hdr: Header = .{
            .magic = MAGIC,
            .version = 1,
            .n = 0,
            ._pad = 0,
            .quant_params = @splat(.{ .min = 0, .scale = 0 }),
        };
        try f.writeStreamingAll(io, std.mem.asBytes(&hdr));
    }

    const path = try tmp_abs_path(allocator, &tmp, "v1.bin");
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
        const hdr: Header = .{
            .magic = MAGIC,
            .version = VERSION,
            .n = 10,
            ._pad = 0,
            .quant_params = @splat(.{ .min = 0, .scale = 0 }),
        };
        try f.writeStreamingAll(io, std.mem.asBytes(&hdr));
    }

    const path = try tmp_abs_path(allocator, &tmp, "trunc.bin");
    defer allocator.free(path);
    try std.testing.expectError(error.Truncated, load(path));
}
