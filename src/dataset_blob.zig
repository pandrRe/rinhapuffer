//! On-disk format for the prepped reference dataset, plus mmap-only `load`
//! and `write` (used by the `prep` build step).
//!
//! Layout v8 (little-endian, native f32 IEEE 754). v7→v8 reorganises the
//! features section into per-cluster blocks of `BLOCK_W=8` rows × N_FEATURES,
//! col-major within each block (a.k.a. block-SoA / PDX). Adds a new
//! `cluster_block_starts[K+1]u32` section for block-range lookup. Walking
//! a cluster's blocks is one prefetcher stream instead of 14 — top of the
//! rinha leaderboard ships this exact shape (thiagorigonatti #1, joojf #4).
//!
//!     offset           size                                field
//!     0                4                                   magic = "RBP1"
//!     4                4                                   version: u32 = 8
//!     8                4                                   n: u32   (record count)
//!     12               4                                   k_clusters: u32 (= 1024 in prod)
//!     16               4                                   fix_scale: i32 (= search.FIX_SCALE)
//!     20               12                                  zero pad to 32-byte alignment
//!     32               k_clusters * N_FEATURES * 4         centroids: [K][14]f32 row-major
//!     +                (k_clusters + 1) * 4                cluster_starts: [K+1]u32 (rows)
//!     +                (k_clusters + 1) * 4                cluster_block_starts: [K+1]u32 (blocks) ← NEW
//!     +                k_clusters * N_FEATURES * 2         bbox_lo: [K][14]i16 row-major
//!     +                k_clusters * N_FEATURES * 2         bbox_hi: [K][14]i16 row-major
//!     pad to 16        0–15                                zero pad
//!     features_offset  total_blocks * N_FEATURES * W * 2   block_features: i16 block-SoA ← LAYOUT CHANGED
//!     labels_offset    ((n + 63) / 64) * 8                 labels_bits: packed u64 LE bitset (1=fraud)
//!
//! Block layout: cluster c's blocks occupy
//!   `block_features[cluster_block_starts[c] * N_FEATURES * W
//!                .. cluster_block_starts[c+1] * N_FEATURES * W]`.
//! Within block b of cluster c, feature k's W lanes are at offset
//!   `(cluster_block_starts[c] + b) * N_FEATURES * W + k * W`.
//! Lane l corresponds to canonical row `cluster_starts[c] + b * W + l`,
//! padded to W lanes with `BLOCK_PAD_VALUE` (= 0) on the last block of
//! each cluster. Padding lanes are skipped at sift time via a per-cluster
//! valid-lane mask in `search.scan_cluster_blocks`, so the padding value
//! itself doesn't affect correctness — 0 is chosen to keep the inner i16
//! subtraction safely in range without forcing an i32 widen-before-sub.
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
const build_options = @import("build_options");
const transform_reference = @import("transform_reference.zig");
const fast_json = @import("fast_json.zig");
const kmeans = @import("kmeans.zig");
const search = @import("search.zig");

const N_FEATURES = transform_reference.N_FEATURES;

pub const MAGIC: u32 = std.mem.readInt(u32, "RBP1", .little);
pub const VERSION: u32 = 8;

/// Production cluster count. Persisted in the header so loaders can reject
/// blobs built with a different topology (`error.UnsupportedKClusters`).
pub const K_CLUSTERS: u32 = 1024;

/// Block width for the per-cluster block-SoA features section. 8 rows per
/// block matches Haswell AVX2 i32 ops (one YMM `vpmulld` per feature) and
/// matches the layout shipped by the rinha leaderboard's #1 thiagorigonatti
/// and #4 joojf submissions. Block bytes = N_FEATURES * BLOCK_W * 2 = 224 B.
pub const BLOCK_W: usize = 8;

/// Value written into padding lanes of the last block of each cluster
/// (when row count isn't a multiple of BLOCK_W). The choice doesn't affect
/// correctness — `search.scan_cluster_blocks` masks the sift to the
/// per-cluster valid lane count, so padding lanes never enter top-K. Pad
/// with 0 specifically because it stays inside the i16 subtraction range
/// when paired with any real query feature (q ∈ [-FIX_SCALE, FIX_SCALE]),
/// letting the inner SIMD loop keep its diff in i16 (and only widen to
/// i32 for the square — matches the Phase 9.6 W=16 SoA pattern, ~14
/// fewer YMM widen ops per block than a sentinel-dominates-distance
/// approach would need).
pub const BLOCK_PAD_VALUE: i16 = 0;

/// Number of clusters scanned per query. Compile-time so `[PROBE_CLUSTERS]f32`
/// is stack-allocated and inner loops fully unroll.
pub const PROBE_CLUSTERS: usize = 8;

/// Random seed used for k-means init. Pinned so prep is deterministic.
pub const KMEANS_SEED: u64 = 0xa5a5_a5a5_a5a5_a5a5;

/// k-means Lloyd-iteration count. Default 40 — k-means++ delivers a usable
/// starting configuration that benefits from extra refinement (with vanilla
/// random init the marginal gain past iter 20 was negligible — stuck in
/// local minima). Production: full dataset, each iter O(N·K·d). Drop via
/// `-Dkmeans-iters=N` for faster local prep.
pub const KMEANS_ITERS: usize = build_options.kmeans_iters;

/// Fast-path stride-sample size. 0 = full-dataset k-means++ (production).
/// Nonzero = `kmeans.run_kmeans_fast` over a deterministic stride sample
/// — looser clusters, much faster prep. Search remains exact via the
/// bbox-pruned repair pass; only p99 is affected. See `-Dkmeans-sample`.
pub const KMEANS_SAMPLE: usize = build_options.kmeans_sample;

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
    cluster_block_starts: usize,
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

/// Number of BLOCK_W-row blocks needed to hold `rc` rows of one cluster
/// (ceiling division — last block is right-padded with `BLOCK_PAD_VALUE`).
pub inline fn block_count_for_rows(rc: u32) usize {
    return (@as(usize, rc) + BLOCK_W - 1) / BLOCK_W;
}

/// Compute byte offsets and total file size for a v8 blob. `total_blocks`
/// is the sum of `block_count_for_rows(rc[c])` across all clusters — the
/// writer computes it once after k-means assignment; the loader reads
/// `cluster_block_starts[k_clusters]` to recover it.
pub fn offsets(n: u32, k_clusters: u32, total_blocks: usize) Offsets {
    const centroids = HEADER_SIZE;
    const centroids_size = @as(usize, k_clusters) * N_FEATURES * @sizeOf(f32);
    const cluster_starts = centroids + centroids_size;
    const cluster_starts_end = cluster_starts + (@as(usize, k_clusters) + 1) * @sizeOf(u32);
    const cluster_block_starts = cluster_starts_end;
    const cluster_block_starts_end = cluster_block_starts + (@as(usize, k_clusters) + 1) * @sizeOf(u32);
    const bbox_lo = cluster_block_starts_end;
    const bbox_size = @as(usize, k_clusters) * N_FEATURES * @sizeOf(i16);
    const bbox_hi = bbox_lo + bbox_size;
    const bbox_hi_end = bbox_hi + bbox_size;
    const features = std.mem.alignForward(usize, bbox_hi_end, 16);
    const features_size = total_blocks * N_FEATURES * BLOCK_W * @sizeOf(i16);
    const labels_unaligned = features + features_size;
    const labels = std.mem.alignForward(usize, labels_unaligned, 8);
    const labels_size = labels_word_count(n) * @sizeOf(u64);
    return .{
        .centroids = centroids,
        .cluster_starts = cluster_starts,
        .cluster_block_starts = cluster_block_starts,
        .bbox_lo = bbox_lo,
        .bbox_hi = bbox_hi,
        .features = features,
        .labels = labels,
        .total = labels + labels_size,
    };
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
/// inside the 2 min test window. Best-effort `mlock` on Linux pins resident
/// pages so they can't be evicted under cgroup memory pressure.
///
/// On Linux: `madvise(WILLNEED)` kicks off async readahead, then
/// `madvise(POPULATE_READ)` (Linux 5.14+) does the prefault in-kernel —
/// batched, fewer minor faults, larger I/Os than the per-page touch loop.
/// On older kernels POPULATE_READ returns EINVAL and the touch loop below
/// picks up the slack. The touch loop is also the only prefault mechanism
/// on macOS (dev path).
///
/// **No `madvise(MADV_RANDOM)`** — Phase 9.4 included it and regressed locally
/// on macOS APFS (RANDOM disables readahead, which APFS uses aggressively).
/// `WILLNEED` is different: it requests readahead rather than disabling it,
/// so the APFS regression doesn't apply (and we gate it to Linux anyway).
pub fn prefault_and_lock(blob: *const IvfQuantizedBlob) void {
    const bytes = blob.mapped.bytes;
    if (bytes.len == 0) return;

    if (comptime builtin.os.tag == .linux) {
        const linux = std.os.linux;
        // MADV_POPULATE_READ landed in Linux 5.14; not yet in zig 0.16's
        // std.os.linux.MADV. Hardcode the stable ABI value.
        const MADV_POPULATE_READ: u32 = 22;
        // madvise takes [*]u8; the mmap is PRIVATE+READ so the cast is
        // kernel-side only (no actual write to the mapping).
        const ptr = @constCast(bytes.ptr);
        // HUGEPAGE first, then populate. Hint must precede the fault-in so
        // the kernel maps in 2 MB chunks instead of 4 KB pages — collapses
        // ~80 MB into ~40 PMD entries instead of ~20k PTEs, slashing TLB
        // misses on cold IVF probes. Best-effort: file-backed THP needs
        // CONFIG_READ_ONLY_THP_FOR_FS; older/stripped kernels return EINVAL
        // and the populate path below still gives us 4 KB pages — no harm.
        _ = linux.madvise(ptr, bytes.len, linux.MADV.HUGEPAGE);
        _ = linux.madvise(ptr, bytes.len, linux.MADV.WILLNEED);
        _ = linux.madvise(ptr, bytes.len, MADV_POPULATE_READ);
    }

    // Cross-platform fallback: touch one byte per 4 KB page. On modern
    // Linux this walks already-resident memory after POPULATE_READ — cheap
    // (sequential RAM read of ~80 MB). On macOS / pre-5.14 Linux it's the
    // mechanism that actually faults the pages in.
    var sink: u8 = 0;
    var off: usize = 0;
    while (off < bytes.len) : (off += 4096) {
        sink ^= bytes[off];
    }
    std.mem.doNotOptimizeAway(sink);

    if (comptime builtin.os.tag == .linux) {
        // mlock errors (EPERM from missing CAP_IPC_LOCK, ENOMEM from
        // RLIMIT_MEMLOCK) are non-fatal — pages are still resident from
        // the steps above. Pin only so the kernel's eviction policy
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
/// on first access. Two-pass section sizing because v8's features section
/// length depends on `total_blocks = cluster_block_starts[K]`, which lives
/// inside the file: read the cluster_block_starts array first (using a
/// preliminary offsets() call with total_blocks=0 to locate it), recover
/// total_blocks from its last word, then re-call offsets() to size the
/// rest.
pub fn load(path: []const u8) LoadError!IvfQuantizedBlob {
    var mapped = try fast_json.mmap_file(path);
    errdefer mapped.deinit();

    if (mapped.bytes.len < HEADER_SIZE) return error.Truncated;
    const hdr: *const Header = @ptrCast(@alignCast(mapped.bytes.ptr));
    if (hdr.magic != MAGIC) return error.BadMagic;
    if (hdr.version != VERSION) return error.UnsupportedVersion;
    if (hdr.k_clusters != K_CLUSTERS) return error.UnsupportedKClusters;
    if (hdr.fix_scale != search.FIX_SCALE) return error.UnsupportedFixScale;

    return load_inner(mapped, hdr);
}

/// Shared loader body for `load` and `load_any_k`. Assumes the header has
/// already been validated for magic/version/fix_scale; `load` additionally
/// requires `k_clusters == K_CLUSTERS`.
fn load_inner(mapped: fast_json.Mapped, hdr: *const Header) LoadError!IvfQuantizedBlob {
    // First pass: locate cluster_block_starts to recover total_blocks.
    const off_pre = offsets(hdr.n, hdr.k_clusters, 0);
    const cbs_end = off_pre.cluster_block_starts + (@as(usize, hdr.k_clusters) + 1) * @sizeOf(u32);
    if (mapped.bytes.len < cbs_end) return error.Truncated;

    const cbs_ptr: [*]const u32 = @ptrCast(@alignCast(mapped.bytes.ptr + off_pre.cluster_block_starts));
    const cluster_block_starts = cbs_ptr[0 .. @as(usize, hdr.k_clusters) + 1];
    const total_blocks: usize = cluster_block_starts[hdr.k_clusters];

    const off = offsets(hdr.n, hdr.k_clusters, total_blocks);
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

    const block_features_count = total_blocks * N_FEATURES * BLOCK_W;
    const block_features_ptr: [*]const i16 = @ptrCast(@alignCast(mapped.bytes.ptr + off.features));
    const block_features = block_features_ptr[0..block_features_count];

    // Labels: packed u64 bitset, LE within each word. Producer is `write`.
    const labels_ptr: [*]const u64 = @ptrCast(@alignCast(mapped.bytes.ptr + off.labels));
    const labels_bits = labels_ptr[0..labels_word_count(hdr.n)];

    return .{
        .mapped = mapped,
        .dataset = .{
            .n = hdr.n,
            .k_clusters = hdr.k_clusters,
            .block_features = block_features,
            .labels_bits = labels_bits,
            .centroids = centroids,
            .cluster_starts = cluster_starts,
            .cluster_block_starts = cluster_block_starts,
            .bbox_lo = bbox_lo,
            .bbox_hi = bbox_hi,
        },
    };
}

/// Serialize a Dataset to `<dir>/<sub_path>`, truncating an existing file.
/// Runs the full IVF prep pipeline:
///   1. draw a random sample, run plain Euclidean k-means → centroids
///   2. assign every row to its nearest centroid
///   3. counting-sort permutation grouping rows by cluster
///   4. compute per-cluster bbox over canonical (un-padded) rows
///   5. write header + centroids + cluster_starts + cluster_block_starts +
///      bboxes + (i16-quantized, block-SoA, sentinel-padded) block_features +
///      bitset labels
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

    // 1. Run k-means. Production: full-dataset k-means++ + Lloyd iters.
    //    Fast path (`-Dkmeans-sample=N`): stride-sample init + Lloyd over
    //    sample only. Looser clusters but search stays exact via bbox repair.
    const centroids = try allocator.alloc(f32, @as(usize, k_clusters) * N_FEATURES);
    defer allocator.free(centroids);

    if (KMEANS_SAMPLE != 0) {
        const sample_n = @min(KMEANS_SAMPLE, n);
        try kmeans.run_kmeans_fast(
            allocator,
            ds.features,
            n,
            k_clusters,
            sample_n,
            KMEANS_ITERS,
            centroids,
        );
    } else {
        try kmeans.run_kmeans(
            allocator,
            ds.features,
            n,
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

    // Compute cluster_block_starts: cumulative block count per cluster
    // (each cluster ceil(rc/W) blocks). Total blocks = cluster_block_starts[K].
    const cluster_block_starts = try allocator.alloc(u32, @as(usize, k_clusters) + 1);
    defer allocator.free(cluster_block_starts);
    cluster_block_starts[0] = 0;
    for (0..k_clusters) |c| {
        const rc: u32 = cluster_starts[c + 1] - cluster_starts[c];
        cluster_block_starts[c + 1] = cluster_block_starts[c] + @as(u32, @intCast(block_count_for_rows(rc)));
    }
    const total_blocks: usize = cluster_block_starts[k_clusters];

    try file.writeStreamingAll(io, std.mem.asBytes(&hdr));
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(centroids));
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(cluster_starts));
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(cluster_block_starts));
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(bbox_lo));
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(bbox_hi));

    // Pad to 16-align the block_features section.
    const off = offsets(n_u32, k_clusters, total_blocks);
    const written = HEADER_SIZE +
        centroids.len * @sizeOf(f32) +
        cluster_starts.len * @sizeOf(u32) +
        cluster_block_starts.len * @sizeOf(u32) +
        2 * bbox_count * @sizeOf(i16);
    std.debug.assert(written <= off.features);
    if (written < off.features) {
        var pad: [16]u8 = @splat(0);
        try file.writeStreamingAll(io, pad[0 .. off.features - written]);
    }

    // block_features: per cluster c, per block b ∈ [0, ceil(rc/W)), write
    // [N_FEATURES][BLOCK_W]i16 — col-major within block. Padding lanes
    // (when rc % W != 0 on the last block) get BLOCK_PAD_VALUE so the
    // search-time sift comparison drops them. One block at a time fits in
    // the 224-byte stack buffer below; we stream each block as one write.
    var block_buf: [N_FEATURES * BLOCK_W]i16 = undefined;
    for (0..k_clusters) |c| {
        const cs = cluster_starts[c];
        const ce = cluster_starts[c + 1];
        const blocks_in_c = block_count_for_rows(ce - cs);
        for (0..blocks_in_c) |b| {
            const row_base = cs + @as(u32, @intCast(b * BLOCK_W));
            inline for (0..N_FEATURES) |k| {
                inline for (0..BLOCK_W) |lane| {
                    const new_idx = row_base + @as(u32, @intCast(lane));
                    if (new_idx < ce) {
                        const old_row = perm[new_idx];
                        const f = ds.features[k * n + old_row];
                        const q = @round(f * fix_scale_f);
                        const clamped = @max(lo_clamp, @min(hi_clamp, q));
                        block_buf[k * BLOCK_W + lane] = @intFromFloat(clamped);
                    } else {
                        block_buf[k * BLOCK_W + lane] = BLOCK_PAD_VALUE;
                    }
                }
            }
            try file.writeStreamingAll(io, std.mem.sliceAsBytes(block_buf[0..]));
        }
    }

    // Pad up to the (8-aligned) labels offset. Each block is `N_FEATURES *
    // BLOCK_W * 2` = 224 bytes; for total_blocks * 224, the modulo-8
    // remainder is 0 (224 = 28 * 8). So the pad is always 0 in practice,
    // but we keep the alignment forward in case BLOCK_W changes.
    {
        const features_end = off.features + total_blocks * N_FEATURES * BLOCK_W * @sizeOf(i16);
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

// ─── reference / bench-only paths ────────────────────────────────────────
//
// `UnquantBlob` + `load_unquant` materialise the dataset as flat f32 SoA so
// `bench.zig` can A/B against the IVF path. Production never pays the
// 168 MB allocation.

/// f32 dataset materialised by dequantizing a v8 blob. Owns the f32 feature
/// and label buffers; the source mmap is closed before this returns.
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

/// Load the v8 blob and dequantize every feature into a fresh flat-SoA f32
/// buffer (column-major, `features[k * n + row]`), undoing the block-SoA
/// layout. Also unpacks the bitset labels into a fresh `[]bool`. Result is
/// consumable by `search.euclidean_topk` and friends. Bench-only.
pub fn load_unquant(allocator: std.mem.Allocator, path: []const u8) !UnquantBlob {
    var blob = try load(path);
    defer blob.deinit();
    const qds = blob.dataset;

    const features = try allocator.alloc(f32, N_FEATURES * qds.n);
    errdefer allocator.free(features);
    const labels = try allocator.alloc(bool, qds.n);
    errdefer allocator.free(labels);

    const inv_scale: f32 = 1.0 / @as(f32, @floatFromInt(search.FIX_SCALE));
    for (0..qds.k_clusters) |c| {
        const cs = qds.cluster_starts[c];
        const ce = qds.cluster_starts[c + 1];
        const block_start = qds.cluster_block_starts[c];
        const blocks_in_c = qds.cluster_block_starts[c + 1] - block_start;
        for (0..blocks_in_c) |b| {
            const block_base = (@as(usize, block_start) + b) * N_FEATURES * BLOCK_W;
            for (0..BLOCK_W) |lane| {
                const row = cs + @as(u32, @intCast(b * BLOCK_W + lane));
                if (row >= ce) break;
                for (0..N_FEATURES) |k| {
                    const v = qds.block_features[block_base + k * BLOCK_W + lane];
                    features[k * qds.n + row] = @as(f32, @floatFromInt(v)) * inv_scale;
                }
            }
        }
    }
    for (0..qds.n) |i| labels[i] = label_at(qds.labels_bits, @intCast(i)) == 1;

    return .{
        .allocator = allocator,
        .features = features,
        .labels = labels,
        .dataset = .{ .n = qds.n, .features = features, .labels = labels },
    };
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

    return load_inner(mapped, hdr);
}

/// Test-only: dequantize a block-SoA IvfQuantizedDataset back to a flat
/// canonical-ordered i16 SoA buffer (`[N_FEATURES][n]i16`, k * n + row
/// addressing). Used by tests that need to compare against the brute-force
/// flat-SoA path (`search.euclidean_topk_q`). Caller frees.
fn flatten_block_features_test_only(
    allocator: std.mem.Allocator,
    qds: transform_reference.IvfQuantizedDataset,
) ![]i16 {
    const out = try allocator.alloc(i16, N_FEATURES * qds.n);
    for (0..qds.k_clusters) |c| {
        const cs = qds.cluster_starts[c];
        const ce = qds.cluster_starts[c + 1];
        const block_start = qds.cluster_block_starts[c];
        const blocks_in_c = qds.cluster_block_starts[c + 1] - block_start;
        for (0..blocks_in_c) |b| {
            const block_base = (@as(usize, block_start) + b) * N_FEATURES * BLOCK_W;
            for (0..BLOCK_W) |lane| {
                const row = cs + @as(u32, @intCast(b * BLOCK_W + lane));
                if (row >= ce) break;
                for (0..N_FEATURES) |k| {
                    out[k * qds.n + row] = qds.block_features[block_base + k * BLOCK_W + lane];
                }
            }
        }
    }
    return out;
}

test "write/load v8 round-trip with IVF metadata" {
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

    // cluster_block_starts: monotonic; per-cluster block count matches
    // ceil(rc / BLOCK_W).
    try std.testing.expectEqual(@as(u32, 0), qds.cluster_block_starts[0]);
    for (0..@as(usize, TEST_K)) |c| {
        const rc = qds.cluster_starts[c + 1] - qds.cluster_starts[c];
        const got = qds.cluster_block_starts[c + 1] - qds.cluster_block_starts[c];
        try std.testing.expectEqual(@as(u32, @intCast(block_count_for_rows(rc))), got);
    }

    // Every stored i16 is either a real quantized value (|·| ≤ FIX_SCALE)
    // or the padding value (0). Padding lanes are skipped at sift time.
    const max_real: i16 = @as(i16, @intCast(search.FIX_SCALE));
    for (qds.block_features) |v| {
        try std.testing.expect(v == BLOCK_PAD_VALUE or (v >= -max_real and v <= max_real));
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

    // Block-SoA → flat-SoA i16 buffer for the per-column comparison below.
    const flat = try flatten_block_features_test_only(allocator, qds);
    defer allocator.free(flat);

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
            blob_col[row] = @as(f32, @floatFromInt(flat[k * n + row])) * inv_scale;
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

    // Build a flat-SoA i16 buffer for the brute-force baseline. Cannot
    // alias the IVF block-SoA buffer (different layout); allocate fresh.
    const flat = try flatten_block_features_test_only(std.testing.allocator, qds_ivf);
    defer std.testing.allocator.free(flat);
    const qds_brute: transform_reference.QuantizedDataset = .{
        .n = qds_ivf.n,
        .features = flat,
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

test "load rejects v7 (flat-SoA features, pre-block-SoA)" {
    // v7 stored features as flat column-major SoA; v8 reorganises into
    // per-cluster blocks of BLOCK_W rows × N_FEATURES.
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    {
        var f = try tmp.dir.createFile(io, "v7.bin", .{});
        defer f.close(io);
        const hdr: Header = .{
            .magic = MAGIC,
            .version = 7,
            .n = 0,
            .k_clusters = K_CLUSTERS,
            .fix_scale = search.FIX_SCALE,
        };
        try f.writeStreamingAll(io, std.mem.asBytes(&hdr));
    }

    const path = try tmp_abs_path(allocator, &tmp, "v7.bin");
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
