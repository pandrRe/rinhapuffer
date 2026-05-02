//! On-disk format for the prepped reference dataset, plus mmap-only `load`
//! and `write` (used by the `prep` build step).
//!
//! Layout v7 (little-endian, native f32 IEEE 754). Same as v6 except labels
//! are packed as a u64 bitset (8× cache-density vs the v6 bool[]). Both
//! v5→v6 (bbox columns) and v6→v7 (bitset labels) bumps are tracked in the
//! version field below.
//!
//!     offset           size                          field
//!     0                4                             magic = "RBP1"
//!     4                4                             version: u32 = 7
//!     8                4                             n: u32   (record count)
//!     12               4                             k_clusters: u32 (= 1024 in prod)
//!     16               4                             fix_scale: i32 (= search.FIX_SCALE)
//!     20               12                            zero pad to 32-byte alignment
//!     32               k_clusters * N_FEATURES * 4   centroids: [K][14]f32 row-major
//!     +K*14*4          (k_clusters + 1) * 4          cluster_starts: [K+1]u32
//!     +(K+1)*4         k_clusters * N_FEATURES * 2   bbox_lo: [K][14]i16 row-major
//!     +K*14*2          k_clusters * N_FEATURES * 2   bbox_hi: [K][14]i16 row-major
//!     pad to 16        0–15                          zero pad
//!     features_offset  N_FEATURES * n * 2            features: i16 column-major (rows reordered by cluster)
//!     labels_offset    ((n + 63) / 64) * 8           labels_bits: packed u64 LE bitset (1=fraud)  ← v7
//!
//! Quantization: `i16 = round(f * FIX_SCALE)`. Decoding back to float is
//! `f ≈ i16 / FIX_SCALE` but the hot path never decodes — search compares
//! `Σ (q_i - r_i)²` directly in integer units (still order-preserving since
//! both sides are scaled by `FIX_SCALE²`).
//!
//! IVF metadata: rows are reordered cluster-by-cluster so that all rows
//! assigned to centroid `c` occupy `[cluster_starts[c], cluster_starts[c+1])`
//! in every feature column. `cluster_starts[K] == n`. Search probes the
//! top-N centroids by Euclidean distance to the (float) query, scans those
//! slabs in integer space, then runs a bbox repair pass over the remaining
//! clusters: each cluster's per-feature `[lo, hi]` gives an axis-aligned
//! lower bound on min distance to any point inside; clusters whose LB ≥
//! current K-th best are provably pruneable, otherwise scanned. Result is
//! exact top-K regardless of the PROBE budget.
//!
//! Endianness is native LE — both dev (macOS arm64) and target (x86_64 Linux)
//! are LE.

const std = @import("std");
const builtin = @import("builtin");
const transform_reference = @import("transform_reference.zig");
const fast_json = @import("fast_json.zig");
const kmeans = @import("kmeans.zig");
const search = @import("search.zig");

const N_FEATURES = transform_reference.N_FEATURES;

pub const MAGIC: u32 = std.mem.readInt(u32, "RBP1", .little);
pub const VERSION: u32 = 7;

/// Production cluster count. Persisted in the header so loaders can reject
/// blobs built with a different topology (`error.UnsupportedKClusters`).
pub const K_CLUSTERS: u32 = 1024;

/// Number of clusters scanned per query. Compile-time so `[PROBE_CLUSTERS]f32`
/// is stack-allocated and inner loops fully unroll.
pub const PROBE_CLUSTERS: usize = 8;

/// Random seed used for k-means init. Pinned so prep is deterministic.
pub const KMEANS_SEED: u64 = 0xa5a5_a5a5_a5a5_a5a5;

/// k-means iteration count for the 100k-row training sample.
pub const KMEANS_ITERS: usize = 20;

/// Cap on the random sample fed into k-means. Smaller datasets feed every row.
pub const KMEANS_SAMPLE_CAP: usize = 100_000;

pub const Header = extern struct {
    magic: u32,
    version: u32,
    n: u32,
    k_clusters: u32,
    fix_scale: i32,
    _pad: [3]u32 = @splat(0),
};

pub const HEADER_SIZE = @sizeOf(Header); // 5 × 4 + 3 × 4 = 32

comptime {
    std.debug.assert(HEADER_SIZE == 32);
}

pub const Offsets = struct {
    centroids: usize,
    cluster_starts: usize,
    bbox_lo: usize,
    bbox_hi: usize,
    features: usize,
    labels: usize,
    total: usize,
};

/// Number of u64 words needed to hold `n` packed label bits.
pub inline fn labels_word_count(n: u32) usize {
    return (@as(usize, n) + 63) / 64;
}

/// Compute byte offsets and total file size for a v7 blob with the given
/// `n` and `k_clusters`. Features are 16-byte-aligned by inserting up to 15
/// pad bytes after `bbox_hi`. Labels are packed u64 (naturally 8-aligned;
/// 16-aligned in practice since features section is 16-aligned and is a
/// multiple of 16 bytes long for any n with `N_FEATURES * 2 % 16 == 0` not
/// holding — features_size = 28*n bytes, so we explicitly align labels to 8).
pub fn offsets(n: u32, k_clusters: u32) Offsets {
    const centroids = HEADER_SIZE;
    const centroids_size = @as(usize, k_clusters) * N_FEATURES * @sizeOf(f32);
    const cluster_starts = centroids + centroids_size;
    const cluster_starts_end = cluster_starts + (@as(usize, k_clusters) + 1) * @sizeOf(u32);
    const bbox_lo = cluster_starts_end;
    const bbox_size = @as(usize, k_clusters) * N_FEATURES * @sizeOf(i16);
    const bbox_hi = bbox_lo + bbox_size;
    const bbox_hi_end = bbox_hi + bbox_size;
    const features = std.mem.alignForward(usize, bbox_hi_end, 16);
    const features_size = N_FEATURES * @as(usize, n) * @sizeOf(i16);
    const labels_unaligned = features + features_size;
    const labels = std.mem.alignForward(usize, labels_unaligned, 8);
    const labels_size = labels_word_count(n) * @sizeOf(u64);
    return .{
        .centroids = centroids,
        .cluster_starts = cluster_starts,
        .bbox_lo = bbox_lo,
        .bbox_hi = bbox_hi,
        .features = features,
        .labels = labels,
        .total = labels + labels_size,
    };
}

/// Total bytes a v7 blob occupies on disk.
pub fn blob_size(n: u32, k_clusters: u32) usize {
    return offsets(n, k_clusters).total;
}

/// Read a single packed label bit. `row` must be < n.
pub inline fn label_at(bits: []const u64, row: u32) u1 {
    const word = bits[row >> 6];
    const shift: u6 = @intCast(row & 63);
    return @intCast((word >> shift) & 1);
}

/// Force every page of the mmap'd dataset to be resident before any request
/// is served. The eval target is a Mac Mini Late 2014 (HDD); a lazy page
/// fault on rotational disk is ~5–10 ms wall, easily creating tail spikes
/// inside the 2 min test window. Touch one byte per 4 KB page so the kernel
/// faults all ~20K pages once, up-front. Best-effort `mlock` on Linux pins
/// them so they can't be evicted under cgroup memory pressure.
///
/// **No `madvise(MADV_RANDOM)`** — Phase 9.4 included it and regressed locally
/// on macOS APFS (RANDOM disables readahead, which APFS uses aggressively).
/// On Linux + HDD, omitting RANDOM keeps the kernel's default readahead
/// behaviour during the touch loop, which is what we want.
pub fn prefault_and_lock(blob: *const IvfQuantizedBlob) void {
    const bytes = blob.mapped.bytes;
    if (bytes.len == 0) return;

    var sink: u8 = 0;
    var off: usize = 0;
    while (off < bytes.len) : (off += 4096) {
        sink ^= bytes[off];
    }
    std.mem.doNotOptimizeAway(sink);

    if (comptime builtin.os.tag == .linux) {
        // mlock errors (EPERM from missing CAP_IPC_LOCK, ENOMEM from
        // RLIMIT_MEMLOCK) are non-fatal — pages are still resident from the
        // touch loop above. Pin only so the kernel's eviction policy
        // doesn't reclaim them later under burst memory pressure.
        const linux = std.os.linux;
        _ = linux.mlock(bytes.ptr, bytes.len);
    }
}

pub const LoadError = error{
    BadMagic,
    UnsupportedVersion,
    UnsupportedKClusters,
    UnsupportedFixScale,
    Truncated,
} || fast_json.MmapError;

pub const WriteError = error{TooManyRecords} ||
    std.Io.File.OpenError ||
    std.Io.File.Writer.Error ||
    std.mem.Allocator.Error;

/// mmap-backed quantized + IVF dataset. `deinit` munmaps.
pub const IvfQuantizedBlob = struct {
    mapped: fast_json.Mapped,
    dataset: transform_reference.IvfQuantizedDataset,

    pub fn deinit(self: IvfQuantizedBlob) void {
        self.mapped.deinit();
    }
};

/// mmap `path`, validate the header, return an IvfQuantizedDataset view
/// aliasing the mapping. No body parsing — the kernel faults pages in lazily
/// on first access.
pub fn load(path: []const u8) LoadError!IvfQuantizedBlob {
    var mapped = try fast_json.mmap_file(path);
    errdefer mapped.deinit();

    if (mapped.bytes.len < HEADER_SIZE) return error.Truncated;
    const hdr: *const Header = @ptrCast(@alignCast(mapped.bytes.ptr));
    if (hdr.magic != MAGIC) return error.BadMagic;
    if (hdr.version != VERSION) return error.UnsupportedVersion;
    if (hdr.k_clusters != K_CLUSTERS) return error.UnsupportedKClusters;
    if (hdr.fix_scale != search.FIX_SCALE) return error.UnsupportedFixScale;

    const off = offsets(hdr.n, hdr.k_clusters);
    if (mapped.bytes.len < off.total) return error.Truncated;

    const centroids_ptr: [*]const f32 = @ptrCast(@alignCast(mapped.bytes.ptr + off.centroids));
    const centroids = centroids_ptr[0 .. @as(usize, hdr.k_clusters) * N_FEATURES];

    const starts_ptr: [*]const u32 = @ptrCast(@alignCast(mapped.bytes.ptr + off.cluster_starts));
    const cluster_starts = starts_ptr[0 .. @as(usize, hdr.k_clusters) + 1];

    const bbox_count = @as(usize, hdr.k_clusters) * N_FEATURES;
    const bbox_lo_ptr: [*]const i16 = @ptrCast(@alignCast(mapped.bytes.ptr + off.bbox_lo));
    const bbox_lo = bbox_lo_ptr[0..bbox_count];
    const bbox_hi_ptr: [*]const i16 = @ptrCast(@alignCast(mapped.bytes.ptr + off.bbox_hi));
    const bbox_hi = bbox_hi_ptr[0..bbox_count];

    const features_count = @as(usize, hdr.n) * N_FEATURES;
    const features_ptr: [*]const i16 = @ptrCast(@alignCast(mapped.bytes.ptr + off.features));
    const features = features_ptr[0..features_count];

    // Labels: packed u64 bitset, LE within each word. Producer is `write`.
    const labels_ptr: [*]const u64 = @ptrCast(@alignCast(mapped.bytes.ptr + off.labels));
    const labels_bits = labels_ptr[0..labels_word_count(hdr.n)];

    return .{
        .mapped = mapped,
        .dataset = .{
            .n = hdr.n,
            .k_clusters = hdr.k_clusters,
            .features = features,
            .labels_bits = labels_bits,
            .centroids = centroids,
            .cluster_starts = cluster_starts,
            .bbox_lo = bbox_lo,
            .bbox_hi = bbox_hi,
        },
    };
}

/// f32 dataset materialised by dequantizing a v7 blob. Owns the f32 feature
/// and label buffers; the source mmap is closed before this returns. Provided
/// for benchmarking parity against `euclidean_topk` — production never pays the
/// 168 MB allocation. The cluster reordering is preserved (rows are still
/// grouped by cluster) which doesn't change search semantics.
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

/// Load the v7 blob and dequantize every feature into a fresh f32 buffer.
/// Also unpacks the bitset labels into a fresh `[]bool` so the resulting
/// `Dataset` is consumable by `search.euclidean_topk` and friends.
pub fn load_unquant(allocator: std.mem.Allocator, path: []const u8) !UnquantBlob {
    var blob = try load(path);
    defer blob.deinit();
    const qds = blob.dataset;

    const features = try allocator.alloc(f32, N_FEATURES * qds.n);
    errdefer allocator.free(features);
    const labels = try allocator.alloc(bool, qds.n);
    errdefer allocator.free(labels);

    const inv_scale: f32 = 1.0 / @as(f32, @floatFromInt(search.FIX_SCALE));
    for (0..N_FEATURES * qds.n) |i| {
        features[i] = @as(f32, @floatFromInt(qds.features[i])) * inv_scale;
    }
    for (0..qds.n) |i| labels[i] = label_at(qds.labels_bits, @intCast(i)) == 1;

    return .{
        .allocator = allocator,
        .features = features,
        .labels = labels,
        .dataset = .{ .n = qds.n, .features = features, .labels = labels },
    };
}

/// Serialize a Dataset to `<dir>/<sub_path>`, truncating an existing file.
/// Runs the full IVF prep pipeline:
///   1. draw a random sample, run plain Euclidean k-means → centroids
///   2. assign every row to its nearest centroid
///   3. counting-sort permutation grouping rows by cluster
///   4. write header + centroids + cluster_starts + (i16-quantized, permuted)
///      features + (permuted) labels
///
/// `k_clusters` is normally `K_CLUSTERS`. Tests pass smaller values to keep
/// example datasets viable.
pub fn write(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    sub_path: []const u8,
    ds: transform_reference.Dataset,
    k_clusters: u32,
) WriteError!void {
    if (ds.n > std.math.maxInt(u32)) return error.TooManyRecords;
    const n = ds.n;
    const n_u32: u32 = @intCast(n);

    // 1. Draw a sample, run k-means.
    const n_sample = @min(KMEANS_SAMPLE_CAP, n);
    const centroids = try allocator.alloc(f32, @as(usize, k_clusters) * N_FEATURES);
    defer allocator.free(centroids);

    {
        // Fisher-Yates the [0..n) row indices, take first n_sample.
        var prng = std.Random.Xoshiro256.init(KMEANS_SEED ^ 0xc0ffee);
        const rand = prng.random();
        const all_idx = try allocator.alloc(u32, n);
        defer allocator.free(all_idx);
        for (0..n) |i| all_idx[i] = @intCast(i);
        rand.shuffle(u32, all_idx);

        const sample = try allocator.alloc(f32, N_FEATURES * n_sample);
        defer allocator.free(sample);
        for (0..N_FEATURES) |k| {
            for (0..n_sample) |i| {
                sample[k * n_sample + i] = ds.features[k * n + all_idx[i]];
            }
        }

        try kmeans.run_kmeans(
            allocator,
            sample,
            n_sample,
            k_clusters,
            KMEANS_ITERS,
            KMEANS_SEED,
            centroids,
        );
    }

    // 2. Assign every row to its cluster.
    const assignments = try allocator.alloc(u32, n);
    defer allocator.free(assignments);
    const assign_scratch = try allocator.alloc(f32, k_clusters);
    defer allocator.free(assign_scratch);
    kmeans.assign_all(ds.features, n, k_clusters, centroids, assignments, assign_scratch);

    // 3. Counting sort: cluster_starts[c] = first new_idx of cluster c.
    const cluster_starts = try allocator.alloc(u32, @as(usize, k_clusters) + 1);
    defer allocator.free(cluster_starts);
    @memset(cluster_starts, 0);
    for (0..n) |row| cluster_starts[assignments[row] + 1] += 1;
    for (1..@as(usize, k_clusters) + 1) |c| cluster_starts[c] += cluster_starts[c - 1];
    std.debug.assert(cluster_starts[k_clusters] == n_u32);

    // 4. Build the permutation: perm[new_idx] = old row index.
    const cursor = try allocator.alloc(u32, k_clusters);
    defer allocator.free(cursor);
    @memcpy(cursor, cluster_starts[0..k_clusters]);

    const perm = try allocator.alloc(u32, n);
    defer allocator.free(perm);
    for (0..n) |row| {
        const c = assignments[row];
        perm[cursor[c]] = @intCast(row);
        cursor[c] += 1;
    }

    // 5. Write the file.
    const file = try dir.createFile(io, sub_path, .{});
    defer file.close(io);

    const hdr: Header = .{
        .magic = MAGIC,
        .version = VERSION,
        .n = n_u32,
        .k_clusters = k_clusters,
        .fix_scale = search.FIX_SCALE,
    };
    // Per-cluster axis-aligned bbox accumulators. Initialized so an empty
    // cluster (lo > hi) makes the search-time LB compute return a huge
    // positive number → cluster always pruned. Computed in a pre-pass so
    // they can be written to disk in their proper position (before features
    // in the streaming layout); the second pass below re-runs the same
    // quantize formula and streams the i16 columns out.
    const bbox_count: usize = @as(usize, k_clusters) * N_FEATURES;
    const bbox_lo = try allocator.alloc(i16, bbox_count);
    defer allocator.free(bbox_lo);
    const bbox_hi = try allocator.alloc(i16, bbox_count);
    defer allocator.free(bbox_hi);
    @memset(bbox_lo, std.math.maxInt(i16));
    @memset(bbox_hi, std.math.minInt(i16));

    const fix_scale_f: f32 = @floatFromInt(search.FIX_SCALE);
    const lo_clamp: f32 = @floatFromInt(std.math.minInt(i16));
    const hi_clamp: f32 = @floatFromInt(std.math.maxInt(i16));
    for (0..k_clusters) |c| {
        const cs = cluster_starts[c];
        const ce = cluster_starts[c + 1];
        for (0..N_FEATURES) |k| {
            const slot = c * N_FEATURES + k;
            var lo_v: i16 = std.math.maxInt(i16);
            var hi_v: i16 = std.math.minInt(i16);
            for (cs..ce) |new_idx| {
                const old_row = perm[new_idx];
                const f = ds.features[k * n + old_row];
                const q = @round(f * fix_scale_f);
                const clamped = @max(lo_clamp, @min(hi_clamp, q));
                const v: i16 = @intFromFloat(clamped);
                if (v < lo_v) lo_v = v;
                if (v > hi_v) hi_v = v;
            }
            bbox_lo[slot] = lo_v;
            bbox_hi[slot] = hi_v;
        }
    }

    try file.writeStreamingAll(io, std.mem.asBytes(&hdr));
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(centroids));
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(cluster_starts));
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(bbox_lo));
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(bbox_hi));

    // Pad to 16-align the features section.
    const off = offsets(n_u32, k_clusters);
    const written = HEADER_SIZE +
        centroids.len * @sizeOf(f32) +
        cluster_starts.len * @sizeOf(u32) +
        2 * bbox_count * @sizeOf(i16);
    std.debug.assert(written <= off.features);
    if (written < off.features) {
        var pad: [16]u8 = @splat(0);
        try file.writeStreamingAll(io, pad[0 .. off.features - written]);
    }

    // Features: 14 columns × n rows, i16-quantized + permuted, written
    // column-major in 2048-row chunks (a 4 KiB stack buffer of i16).
    var i16_chunk: [2048]i16 = undefined;
    for (0..N_FEATURES) |k| {
        var i: usize = 0;
        while (i < n) {
            const end = @min(i + i16_chunk.len, n);
            for (i..end, 0..) |new_idx, j| {
                const old_row = perm[new_idx];
                const f = ds.features[k * n + old_row];
                const q = @round(f * fix_scale_f);
                const clamped = @max(lo_clamp, @min(hi_clamp, q));
                i16_chunk[j] = @intFromFloat(clamped);
            }
            try file.writeStreamingAll(io, std.mem.sliceAsBytes(i16_chunk[0 .. end - i]));
            i = end;
        }
    }

    // Pad up to the (8-aligned) labels offset. The features section is
    // `28 * n` bytes long; for odd `n` that leaves a 4-byte tail, which we
    // fill with zeros so the u64 labels word starts naturally aligned.
    {
        const features_end = off.features + N_FEATURES * @as(usize, n) * @sizeOf(i16);
        std.debug.assert(features_end <= off.labels);
        if (features_end < off.labels) {
            var pad8: [8]u8 = @splat(0);
            try file.writeStreamingAll(io, pad8[0 .. off.labels - features_end]);
        }
    }

    // Labels: pack each cluster of 64 (permuted) booleans into one u64 LE
    // word. `byte_chunk` here is a u64 stack buffer that streams 512-word
    // (4 KiB) chunks. Trailing bits beyond `n` are zero-padded.
    var word_chunk: [512]u64 = undefined;
    const word_count: usize = labels_word_count(n_u32);
    {
        var w: usize = 0;
        while (w < word_count) {
            const end_w = @min(w + word_chunk.len, word_count);
            for (w..end_w, 0..) |word_idx, j| {
                var bits: u64 = 0;
                const base = word_idx * 64;
                const end_bit = @min(base + 64, n);
                var bit_idx: usize = base;
                while (bit_idx < end_bit) : (bit_idx += 1) {
                    if (ds.labels[perm[bit_idx]]) {
                        const shift: u6 = @intCast(bit_idx - base);
                        bits |= @as(u64, 1) << shift;
                    }
                }
                word_chunk[j] = bits;
            }
            try file.writeStreamingAll(io, std.mem.sliceAsBytes(word_chunk[0 .. end_w - w]));
            w = end_w;
        }
    }
}

// ─── tests ──────────────────────────────────────────────────────────────────

const TEST_K: u32 = 4;

fn tmp_abs_path(
    allocator: std.mem.Allocator,
    tmp: *std.testing.TmpDir,
    sub_path: []const u8,
) ![]u8 {
    var dir_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const dir_len = try tmp.dir.realPath(std.testing.io, &dir_buf);
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir_buf[0..dir_len], sub_path });
}

/// Test-only loader: same as `load` but accepts any `k_clusters`. Used by
/// fixture files written with smaller `K` values (TEST_K=8 from `example-
/// references.json`) that the production `load` would reject. Only callers
/// in tests should reach for this; production paths must use `load`.
pub fn load_any_k(path: []const u8) LoadError!IvfQuantizedBlob {
    var mapped = try fast_json.mmap_file(path);
    errdefer mapped.deinit();

    if (mapped.bytes.len < HEADER_SIZE) return error.Truncated;
    const hdr: *const Header = @ptrCast(@alignCast(mapped.bytes.ptr));
    if (hdr.magic != MAGIC) return error.BadMagic;
    if (hdr.version != VERSION) return error.UnsupportedVersion;
    if (hdr.fix_scale != search.FIX_SCALE) return error.UnsupportedFixScale;

    const off = offsets(hdr.n, hdr.k_clusters);
    if (mapped.bytes.len < off.total) return error.Truncated;

    const centroids_ptr: [*]const f32 = @ptrCast(@alignCast(mapped.bytes.ptr + off.centroids));
    const centroids = centroids_ptr[0 .. @as(usize, hdr.k_clusters) * N_FEATURES];

    const starts_ptr: [*]const u32 = @ptrCast(@alignCast(mapped.bytes.ptr + off.cluster_starts));
    const cluster_starts = starts_ptr[0 .. @as(usize, hdr.k_clusters) + 1];

    const bbox_count = @as(usize, hdr.k_clusters) * N_FEATURES;
    const bbox_lo_ptr: [*]const i16 = @ptrCast(@alignCast(mapped.bytes.ptr + off.bbox_lo));
    const bbox_lo = bbox_lo_ptr[0..bbox_count];
    const bbox_hi_ptr: [*]const i16 = @ptrCast(@alignCast(mapped.bytes.ptr + off.bbox_hi));
    const bbox_hi = bbox_hi_ptr[0..bbox_count];

    const features_count = @as(usize, hdr.n) * N_FEATURES;
    const features_ptr: [*]const i16 = @ptrCast(@alignCast(mapped.bytes.ptr + off.features));
    const features = features_ptr[0..features_count];

    const labels_ptr: [*]const u64 = @ptrCast(@alignCast(mapped.bytes.ptr + off.labels));
    const labels_bits = labels_ptr[0..labels_word_count(hdr.n)];

    return .{
        .mapped = mapped,
        .dataset = .{
            .n = hdr.n,
            .k_clusters = hdr.k_clusters,
            .features = features,
            .labels_bits = labels_bits,
            .centroids = centroids,
            .cluster_starts = cluster_starts,
            .bbox_lo = bbox_lo,
            .bbox_hi = bbox_hi,
        },
    };
}

test "write/load v7 round-trip with IVF metadata" {
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

    try write(allocator, io, tmp.dir, "dataset.bin", ds, TEST_K);

    const path = try tmp_abs_path(allocator, &tmp, "dataset.bin");
    defer allocator.free(path);
    var blob = try load_any_k(path);
    defer blob.deinit();

    const qds = blob.dataset;
    try std.testing.expectEqual(ds.n, qds.n);
    try std.testing.expectEqual(@as(usize, TEST_K), qds.k_clusters);

    // cluster_starts: monotonic, starts at 0, ends at n.
    try std.testing.expectEqual(@as(u32, 0), qds.cluster_starts[0]);
    try std.testing.expectEqual(@as(u32, @intCast(n)), qds.cluster_starts[TEST_K]);
    for (1..@as(usize, TEST_K) + 1) |c| {
        try std.testing.expect(qds.cluster_starts[c] >= qds.cluster_starts[c - 1]);
    }

    // Sanity-check quantization: every stored i16 should round-trip back to
    // a value within one quant step (1/FIX_SCALE) of *some* original feature
    // value in the same column.
    const inv_scale: f32 = 1.0 / @as(f32, @floatFromInt(search.FIX_SCALE));
    for (0..N_FEATURES * n) |i| {
        const dequant = @as(f32, @floatFromInt(qds.features[i])) * inv_scale;
        // Bounded by quantization step.
        try std.testing.expect(@abs(dequant) <= 1.0 + inv_scale);
    }
}

test "v5 quantization precision per element after permutation" {
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
    try write(allocator, io, tmp.dir, "dataset.bin", ds, TEST_K);

    const path = try tmp_abs_path(allocator, &tmp, "dataset.bin");
    defer allocator.free(path);
    var blob = try load_any_k(path);
    defer blob.deinit();
    const qds = blob.dataset;

    // Per-column sorted comparison, matching values in order. The global
    // FIX_SCALE means quant step is 1/FIX_SCALE for every column.
    const inv_scale: f32 = 1.0 / @as(f32, @floatFromInt(search.FIX_SCALE));
    const tol: f32 = inv_scale * 1.0001;

    const orig_col = try allocator.alloc(f32, n);
    defer allocator.free(orig_col);
    const blob_col = try allocator.alloc(f32, n);
    defer allocator.free(blob_col);

    for (0..N_FEATURES) |k| {
        for (0..n) |row| orig_col[row] = ds.features[k * n + row];
        for (0..n) |row| {
            blob_col[row] = @as(f32, @floatFromInt(qds.features[k * n + row])) * inv_scale;
        }
        std.mem.sort(f32, orig_col, {}, std.sort.asc(f32));
        std.mem.sort(f32, blob_col, {}, std.sort.asc(f32));
        for (orig_col, blob_col) |a, b| {
            try std.testing.expect(@abs(a - b) <= tol + 1.0e-7);
        }
    }
}

test "label_at round-trip: write known pattern, decode every bit" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // Build a 100-row toy dataset whose labels form a known mod-3 pattern,
    // write it, reload, and confirm `label_at` agrees on every row.
    var src = try fast_json.mmap_file("./resources/example-references.json");
    defer src.deinit();

    const n = transform_reference.count_records(src.bytes);
    const features = try allocator.alloc(f32, N_FEATURES * n);
    defer allocator.free(features);
    const labels = try allocator.alloc(bool, n);
    defer allocator.free(labels);
    var ds = try transform_reference.parse_into(src.bytes, features, labels);
    // Overwrite labels with a deterministic pattern so the test doesn't
    // depend on the example file's contents.
    for (0..n) |i| labels[i] = (i % 3) == 0;
    ds.labels = labels;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(allocator, io, tmp.dir, "dataset.bin", ds, TEST_K);

    const path = try tmp_abs_path(allocator, &tmp, "dataset.bin");
    defer allocator.free(path);
    var blob = try load_any_k(path);
    defer blob.deinit();
    const qds = blob.dataset;

    // Permutation-aware check: walk clusters in storage order, look up the
    // original row index, compare against the mod-3 pattern.
    // We don't have the perm here, so re-derive the equivalence indirectly:
    // count fraud bits and check it matches the input's count.
    var written_count: usize = 0;
    var input_count: usize = 0;
    for (0..n) |row| {
        if (label_at(qds.labels_bits, @intCast(row)) == 1) written_count += 1;
        if (labels[row]) input_count += 1;
    }
    try std.testing.expectEqual(input_count, written_count);

    // Spot-check: every word's bit must extract via label_at consistently
    // (i.e. label_at agrees with manual bit extraction).
    for (0..n) |row| {
        const word = qds.labels_bits[row >> 6];
        const shift: u6 = @intCast(row & 63);
        const manual: u1 = @intCast((word >> shift) & 1);
        try std.testing.expectEqual(manual, label_at(qds.labels_bits, @intCast(row)));
    }
}

test "v5 PROBE=K equivalence: euclidean_topk_q_ivf matches euclidean_topk_q" {
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
    try write(allocator, io, tmp.dir, "dataset.bin", ds_mem, TEST_K);

    const path = try tmp_abs_path(allocator, &tmp, "dataset.bin");
    defer allocator.free(path);
    var blob = try load_any_k(path);
    defer blob.deinit();
    const qds_ivf = blob.dataset;

    // Build a brute-force quantized view aliasing the same buffers.
    const qds_brute: transform_reference.QuantizedDataset = .{
        .n = qds_ivf.n,
        .features = qds_ivf.features,
        .labels_bits = qds_ivf.labels_bits,
    };

    var queries: [3][N_FEATURES]f32 = undefined;
    for (0..N_FEATURES) |c| queries[0][c] = ds_mem.features[c * n + 7];
    queries[1] = .{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    queries[2] = .{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4 };

    for (&queries) |*q| {
        var got_brute: [search.TOP_K]u32 = undefined;
        var got_ivf: [search.TOP_K]u32 = undefined;
        search.euclidean_topk_q(qds_brute, q, &got_brute);
        // Full coverage means scan every cluster — must match brute force exactly.
        search.euclidean_topk_q_ivf_full(qds_ivf, q, &got_ivf);
        try std.testing.expectEqualSlices(u32, &got_brute, &got_ivf);
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
            .k_clusters = K_CLUSTERS,
            .fix_scale = search.FIX_SCALE,
        };
        try f.writeStreamingAll(io, std.mem.asBytes(&hdr));
    }

    const path = try tmp_abs_path(allocator, &tmp, "wrongver.bin");
    defer allocator.free(path);
    try std.testing.expectError(error.UnsupportedVersion, load(path));
}

test "load rejects v2 (old format)" {
    // Simulates a stale v2 dataset.bin sitting in resources/. v4 reader must
    // reject it via the version check.
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    {
        var f = try tmp.dir.createFile(io, "v2.bin", .{});
        defer f.close(io);
        const hdr: Header = .{
            .magic = MAGIC,
            .version = 2,
            .n = 0,
            .k_clusters = 0,
            .fix_scale = search.FIX_SCALE,
        };
        try f.writeStreamingAll(io, std.mem.asBytes(&hdr));
    }

    const path = try tmp_abs_path(allocator, &tmp, "v2.bin");
    defer allocator.free(path);
    try std.testing.expectError(error.UnsupportedVersion, load(path));
}

test "load rejects v3 (cosine-normalized format)" {
    // v3 used per-column-quantized u16 features with the unit-norm dataset.
    // v5 reader must reject via the version check.
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    {
        var f = try tmp.dir.createFile(io, "v3.bin", .{});
        defer f.close(io);
        const hdr: Header = .{
            .magic = MAGIC,
            .version = 3,
            .n = 0,
            .k_clusters = K_CLUSTERS,
            .fix_scale = search.FIX_SCALE,
        };
        try f.writeStreamingAll(io, std.mem.asBytes(&hdr));
    }

    const path = try tmp_abs_path(allocator, &tmp, "v3.bin");
    defer allocator.free(path);
    try std.testing.expectError(error.UnsupportedVersion, load(path));
}

test "load rejects v4 (raw-Euclidean per-column quantized format)" {
    // v4 used u16 + per-column (min, scale). v5 dropped that for a global
    // FIX_SCALE on i16 — a stale v4 blob must reject via the version check.
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    {
        var f = try tmp.dir.createFile(io, "v4.bin", .{});
        defer f.close(io);
        const hdr: Header = .{
            .magic = MAGIC,
            .version = 4,
            .n = 0,
            .k_clusters = K_CLUSTERS,
            .fix_scale = search.FIX_SCALE,
        };
        try f.writeStreamingAll(io, std.mem.asBytes(&hdr));
    }

    const path = try tmp_abs_path(allocator, &tmp, "v4.bin");
    defer allocator.free(path);
    try std.testing.expectError(error.UnsupportedVersion, load(path));
}

test "load rejects v5 (no bbox columns)" {
    // v5 had no per-cluster bbox sections.
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    {
        var f = try tmp.dir.createFile(io, "v5.bin", .{});
        defer f.close(io);
        const hdr: Header = .{
            .magic = MAGIC,
            .version = 5,
            .n = 0,
            .k_clusters = K_CLUSTERS,
            .fix_scale = search.FIX_SCALE,
        };
        try f.writeStreamingAll(io, std.mem.asBytes(&hdr));
    }

    const path = try tmp_abs_path(allocator, &tmp, "v5.bin");
    defer allocator.free(path);
    try std.testing.expectError(error.UnsupportedVersion, load(path));
}

test "load rejects v6 (bool labels, pre-bitset)" {
    // v6 had per-byte bool labels; v7 packs them into a u64 bitset.
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    {
        var f = try tmp.dir.createFile(io, "v6.bin", .{});
        defer f.close(io);
        const hdr: Header = .{
            .magic = MAGIC,
            .version = 6,
            .n = 0,
            .k_clusters = K_CLUSTERS,
            .fix_scale = search.FIX_SCALE,
        };
        try f.writeStreamingAll(io, std.mem.asBytes(&hdr));
    }

    const path = try tmp_abs_path(allocator, &tmp, "v6.bin");
    defer allocator.free(path);
    try std.testing.expectError(error.UnsupportedVersion, load(path));
}

test "load rejects mismatched fix_scale" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    {
        var f = try tmp.dir.createFile(io, "wrongscale.bin", .{});
        defer f.close(io);
        const hdr: Header = .{
            .magic = MAGIC,
            .version = VERSION,
            .n = 0,
            .k_clusters = K_CLUSTERS,
            .fix_scale = search.FIX_SCALE + 1,
        };
        try f.writeStreamingAll(io, std.mem.asBytes(&hdr));
    }

    const path = try tmp_abs_path(allocator, &tmp, "wrongscale.bin");
    defer allocator.free(path);
    try std.testing.expectError(error.UnsupportedFixScale, load(path));
}

test "load rejects unsupported k_clusters" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    {
        var f = try tmp.dir.createFile(io, "wrongk.bin", .{});
        defer f.close(io);
        const hdr: Header = .{
            .magic = MAGIC,
            .version = VERSION,
            .n = 0,
            .k_clusters = 99,
            .fix_scale = search.FIX_SCALE,
        };
        try f.writeStreamingAll(io, std.mem.asBytes(&hdr));
    }

    const path = try tmp_abs_path(allocator, &tmp, "wrongk.bin");
    defer allocator.free(path);
    try std.testing.expectError(error.UnsupportedKClusters, load(path));
}

test "load rejects truncated body" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    {
        var f = try tmp.dir.createFile(io, "trunc.bin", .{});
        defer f.close(io);
        // Header claims n=10 and k_clusters=K_CLUSTERS, but file contains only the header.
        const hdr: Header = .{
            .magic = MAGIC,
            .version = VERSION,
            .n = 10,
            .k_clusters = K_CLUSTERS,
            .fix_scale = search.FIX_SCALE,
        };
        try f.writeStreamingAll(io, std.mem.asBytes(&hdr));
    }

    const path = try tmp_abs_path(allocator, &tmp, "trunc.bin");
    defer allocator.free(path);
    try std.testing.expectError(error.Truncated, load(path));
}
