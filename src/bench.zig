//! Bench harness. Each `bench_*` function reports on one workload; main runs them
//! all and prints results. Add new ones by writing another `bench_*` and calling
//! it from `main`.

const std = @import("std");
const rinhapuffer = @import("rinhapuffer");
const transform_reference = rinhapuffer.transform_reference;
const fast_json = rinhapuffer.fast_json;
const payload = rinhapuffer.payload;
const search = rinhapuffer.search;
const dataset_blob = rinhapuffer.dataset_blob;

const REFERENCE_PATH = "./resources/references.json";
const PAYLOADS_PATH = "./resources/example-payloads.json";
const DATASET_BIN_PATH = "./resources/dataset.bin";
const LOAD_RUNS = 50;
const MMAP_RUNS = 5;
const STDJSON_RUNS = 3; // std.json is slow; fewer runs to keep total wall time reasonable.
const PAYLOAD_RUNS = 7;
const PAYLOAD_BATCH_REPEATS = 1000; // amortise per-payload measurement: vectorize the whole batch this many times per timed run.
const MAX_PAYLOADS = 128;
const SEARCH_QUERIES = 32; // distinct queries used in cosine_topk distribution.
const SEARCH_PASSES = 200; // each pass scans the full dataset once per query.

fn now_ns() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

const Stats = struct {
    cold_ns: u64,
    runs: []u64, // sorted ascending after print_stats
    bytes_in: usize, // file size on disk
    bytes_out: usize, // dataset memory footprint
    records: usize,
};

fn dataset_memory_bytes(ds: transform_reference.Dataset) usize {
    return ds.features.len * @sizeOf(f32) +
        ds.labels.len * @sizeOf(bool);
}

const ReferenceBuffers = struct {
    mapped: fast_json.Mapped,
    features: []f32,
    labels: []bool,

    fn deinit(self: *ReferenceBuffers, allocator: std.mem.Allocator) void {
        allocator.free(self.features);
        allocator.free(self.labels);
        self.mapped.deinit();
    }
};

fn setup_reference_buffers(allocator: std.mem.Allocator, path: []const u8) !ReferenceBuffers {
    var mapped = try fast_json.mmap_file(path);
    errdefer mapped.deinit();
    const n = transform_reference.count_records(mapped.bytes);
    const features = try allocator.alloc(f32, transform_reference.N_FEATURES * n);
    errdefer allocator.free(features);
    const labels = try allocator.alloc(bool, n);
    errdefer allocator.free(labels);
    return .{ .mapped = mapped, .features = features, .labels = labels };
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1.0e6;
}

fn mib(b: usize) f64 {
    return @as(f64, @floatFromInt(b)) / (1024.0 * 1024.0);
}

fn print_stats(label: []const u8, s: *Stats) void {
    std.mem.sort(u64, s.runs, {}, std.sort.asc(u64));
    const min_ns = s.runs[0];
    const med_ns = s.runs[s.runs.len / 2];
    const min_s = @as(f64, @floatFromInt(min_ns)) / 1.0e9;
    const records_f = @as(f64, @floatFromInt(s.records));

    std.debug.print(
        "  {s}\n" ++
            "    cold (first run): {d:.3} ms\n" ++
            "    warm min:         {d:.3} ms ({d} runs)\n" ++
            "    warm median:      {d:.3} ms\n" ++
            "    file size:        {d} bytes ({d:.2} MiB)\n" ++
            "    dataset memory:   {d} bytes ({d:.2} MiB)\n" ++
            "    records:          {d}\n" ++
            "    throughput (min): {d:.1} MiB/s, {d:.2}M records/s\n",
        .{
            label,
            ms(s.cold_ns),
            ms(min_ns),
            s.runs.len,
            ms(med_ns),
            s.bytes_in,
            mib(s.bytes_in),
            s.bytes_out,
            mib(s.bytes_out),
            s.records,
            mib(s.bytes_in) / min_s,
            (records_f / min_s) / 1.0e6,
        },
    );
}

// ─── reference dataset: mmap fast loader ────────────────────────────────────

fn bench_reference_mmap(allocator: std.mem.Allocator, runs_buf: []u64) !Stats {
    // Buffers and mmap are set up once and reused across runs — this matches
    // production deployment, where the dataset is loaded into a long-lived
    // arena. We measure only `parse_into`, which is what repeats in real use.
    var bufs = try setup_reference_buffers(allocator, REFERENCE_PATH);
    defer bufs.deinit(allocator);

    const t0 = now_ns();
    const ds_cold = try transform_reference.parse_into(bufs.mapped.bytes, bufs.features, bufs.labels);
    const t1 = now_ns();
    const cold_ns = t1 - t0;
    const bytes_out = dataset_memory_bytes(ds_cold);
    const records = ds_cold.n;

    for (runs_buf) |*t| {
        const a = now_ns();
        _ = try transform_reference.parse_into(bufs.mapped.bytes, bufs.features, bufs.labels);
        const b = now_ns();
        t.* = b - a;
    }

    return .{
        .cold_ns = cold_ns,
        .runs = runs_buf,
        .bytes_in = bufs.mapped.bytes.len,
        .bytes_out = bytes_out,
        .records = records,
    };
}

// ─── reference dataset: std.json comparison ─────────────────────────────────

fn parse_stdjson_into(
    allocator: std.mem.Allocator,
    contents: []const u8,
    features: []f32,
    labels: []bool,
) !transform_reference.Dataset {
    const Entry = struct {
        vector: [transform_reference.N_FEATURES]f32,
        label: []const u8,
    };
    const parsed = try std.json.parseFromSlice([]Entry, allocator, contents, .{});
    defer parsed.deinit();

    const n = parsed.value.len;
    if (features.len < transform_reference.N_FEATURES * n) return error.BufferTooSmall;
    if (labels.len < n) return error.BufferTooSmall;

    const f = features[0 .. transform_reference.N_FEATURES * n];
    const l = labels[0..n];
    // Mirror parse_into's L2-normalization so the std.json baseline produces
    // an identical dataset shape and consumes the same memory.
    for (parsed.value, 0..) |entry, row| {
        l[row] = std.mem.eql(u8, entry.label, "fraud");
        var sum_sq: f32 = 0;
        for (entry.vector) |v| sum_sq += v * v;
        const inv_norm: f32 = 1.0 / @sqrt(sum_sq);
        for (entry.vector, 0..) |v, c| {
            f[c * n + row] = v * inv_norm;
        }
    }

    return .{ .n = n, .features = f, .labels = l };
}

fn bench_reference_stdjson(
    allocator: std.mem.Allocator,
    runs_buf: []u64,
) !Stats {
    var bufs = try setup_reference_buffers(allocator, REFERENCE_PATH);
    defer bufs.deinit(allocator);

    const t0 = now_ns();
    const ds_cold = try parse_stdjson_into(allocator, bufs.mapped.bytes, bufs.features, bufs.labels);
    const t1 = now_ns();
    const cold_ns = t1 - t0;
    const bytes_out = dataset_memory_bytes(ds_cold);
    const records = ds_cold.n;

    for (runs_buf) |*t| {
        const a = now_ns();
        _ = try parse_stdjson_into(allocator, bufs.mapped.bytes, bufs.features, bufs.labels);
        const b = now_ns();
        t.* = b - a;
    }

    return .{
        .cold_ns = cold_ns,
        .runs = runs_buf,
        .bytes_in = bufs.mapped.bytes.len,
        .bytes_out = bytes_out,
        .records = records,
    };
}

// ─── example payloads: vectorize hot path ──────────────────────────────────

fn slice_payloads(bytes: []const u8, out: []([]const u8)) !usize {
    var n: usize = 0;
    var p: usize = 0;
    p = fast_json.skip_ws(bytes, p);
    p = try fast_json.expect_byte(bytes, p, '[');
    var first = true;
    while (true) {
        p = fast_json.skip_ws(bytes, p);
        if (p < bytes.len and bytes[p] == ']') break;
        if (!first) {
            p = try fast_json.expect_byte(bytes, p, ',');
            p = fast_json.skip_ws(bytes, p);
        }
        const start = p;
        const end = try fast_json.skip_to_matching_close(bytes, p + 1, '{', '}');
        if (n >= out.len) return error.TooManyPayloads;
        out[n] = bytes[start..end];
        n += 1;
        p = end;
        first = false;
    }
    return n;
}

fn print_payload_stats(label: []const u8, s: *Stats, batch_size: usize, batch_repeats: usize) void {
    std.mem.sort(u64, s.runs, {}, std.sort.asc(u64));
    const min_ns = s.runs[0];
    const med_ns = s.runs[s.runs.len / 2];
    const min_s = @as(f64, @floatFromInt(min_ns)) / 1.0e9;
    const total_calls = batch_size * batch_repeats;
    const ns_per_call = @as(f64, @floatFromInt(min_ns)) / @as(f64, @floatFromInt(total_calls));
    const calls_per_s = @as(f64, @floatFromInt(total_calls)) / min_s;

    std.debug.print(
        "  {s}\n" ++
            "    cold (first batch): {d:.3} ms\n" ++
            "    warm min:           {d:.3} ms ({d} runs of {d}×{d} = {d} parses)\n" ++
            "    warm median:        {d:.3} ms\n" ++
            "    file size:          {d} bytes ({d:.2} KiB)\n" ++
            "    payloads in file:   {d}\n" ++
            "    per-payload (min):  {d:.1} ns\n" ++
            "    throughput (min):   {d:.2}M payloads/s, {d:.1} MiB/s\n",
        .{
            label,
            ms(s.cold_ns),
            ms(min_ns),
            s.runs.len,
            batch_size,
            batch_repeats,
            total_calls,
            ms(med_ns),
            s.bytes_in,
            @as(f64, @floatFromInt(s.bytes_in)) / 1024.0,
            batch_size,
            ns_per_call,
            calls_per_s / 1.0e6,
            mib(s.bytes_in * batch_repeats) / min_s,
        },
    );
}

const PerCallStats = struct {
    passes: usize,
    calls_per_pass: usize,
    bytes_avg: f64,
    // Per-call (= pass_ns / calls_per_pass) at each percentile.
    min_ns: f64,
    p50_ns: f64,
    p95_ns: f64,
    p99_ns: f64,
    max_ns: f64,
    mean_ns: f64,
    stddev_ns: f64,
};

fn percentile(sorted: []const u64, p: f64) u64 {
    if (sorted.len == 0) return 0;
    const idx_f = p * @as(f64, @floatFromInt(sorted.len - 1));
    const idx: usize = @intFromFloat(@round(idx_f));
    return sorted[idx];
}

/// Time each pass over all payloads as one sample (passes of ~20 µs are well
/// above the ~1 µs `clock_gettime` resolution on macOS, so individual samples
/// are meaningful). Each sample is divided by `calls_per_pass` to express
/// per-call ns. Captures throughput-under-load distribution, not per-payload
/// type variance.
fn bench_payload_per_call(allocator: std.mem.Allocator, slices: []const []const u8, passes: usize) !PerCallStats {
    const samples = try allocator.alloc(u64, passes);
    defer allocator.free(samples);

    var feats: [payload.N_FEATURES]f32 = undefined;
    var anchor: f32 = 0;

    // Warm caches/branch predictors with one full pass before timing.
    for (slices) |s| {
        try payload.vectorize(s, &feats);
        anchor += feats[0];
    }

    for (samples) |*sample| {
        const a = now_ns();
        for (slices) |s| {
            try payload.vectorize(s, &feats);
            anchor += feats[0];
        }
        const b = now_ns();
        sample.* = b - a;
    }
    std.mem.doNotOptimizeAway(anchor);

    std.mem.sort(u64, samples, {}, std.sort.asc(u64));

    const calls_per_pass = slices.len;
    const calls_f = @as(f64, @floatFromInt(calls_per_pass));

    var sum: u128 = 0;
    var sum_sq: u128 = 0;
    for (samples) |x| {
        sum += x;
        sum_sq += @as(u128, x) * @as(u128, x);
    }
    const n_f = @as(f64, @floatFromInt(samples.len));
    const pass_mean = @as(f64, @floatFromInt(sum)) / n_f;
    const pass_var = @as(f64, @floatFromInt(sum_sq)) / n_f - pass_mean * pass_mean;
    const pass_stddev = @sqrt(@max(pass_var, 0));

    var bytes_total: usize = 0;
    for (slices) |s| bytes_total += s.len;
    const bytes_avg = @as(f64, @floatFromInt(bytes_total)) / calls_f;

    return .{
        .passes = samples.len,
        .calls_per_pass = calls_per_pass,
        .bytes_avg = bytes_avg,
        .min_ns = @as(f64, @floatFromInt(samples[0])) / calls_f,
        .p50_ns = @as(f64, @floatFromInt(percentile(samples, 0.50))) / calls_f,
        .p95_ns = @as(f64, @floatFromInt(percentile(samples, 0.95))) / calls_f,
        .p99_ns = @as(f64, @floatFromInt(percentile(samples, 0.99))) / calls_f,
        .max_ns = @as(f64, @floatFromInt(samples[samples.len - 1])) / calls_f,
        .mean_ns = pass_mean / calls_f,
        .stddev_ns = pass_stddev / calls_f,
    };
}

fn print_per_call_stats(label: []const u8, s: PerCallStats) void {
    const peak_mib_s = s.bytes_avg / s.min_ns * 1.0e9 / (1024.0 * 1024.0);
    std.debug.print(
        "  {s} (per-call latency from {d} passes × {d} calls; per-pass timing ÷ calls_per_pass)\n" ++
            "    avg payload size:  {d:.0} bytes\n" ++
            "    per-call min:      {d:.1} ns  ({d:.1} MiB/s peak)\n" ++
            "    per-call p50:      {d:.1} ns\n" ++
            "    per-call p95:      {d:.1} ns\n" ++
            "    per-call p99:      {d:.1} ns\n" ++
            "    per-call max:      {d:.1} ns\n" ++
            "    per-call mean:     {d:.1} ns ± {d:.1} (stddev)\n",
        .{
            label,
            s.passes,
            s.calls_per_pass,
            s.bytes_avg,
            s.min_ns,
            peak_mib_s,
            s.p50_ns,
            s.p95_ns,
            s.p99_ns,
            s.max_ns,
            s.mean_ns,
            s.stddev_ns,
        },
    );
}

fn bench_payload_vectorize(runs_buf: []u64) !Stats {
    var mapped = try fast_json.mmap_file(PAYLOADS_PATH);
    defer mapped.deinit();

    var slices: [MAX_PAYLOADS][]const u8 = undefined;
    const n = try slice_payloads(mapped.bytes, &slices);

    // Side-effect anchor that the optimiser can't fold away.
    var anchor: f32 = 0;
    var feats: [payload.N_FEATURES]f32 = undefined;

    // Cold run = first batch.
    const t0 = now_ns();
    var rep: usize = 0;
    while (rep < PAYLOAD_BATCH_REPEATS) : (rep += 1) {
        for (slices[0..n]) |s| {
            try payload.vectorize(s, &feats);
            anchor += feats[0];
        }
    }
    const t1 = now_ns();
    const cold_ns = t1 - t0;

    for (runs_buf) |*t| {
        const a = now_ns();
        rep = 0;
        while (rep < PAYLOAD_BATCH_REPEATS) : (rep += 1) {
            for (slices[0..n]) |s| {
                try payload.vectorize(s, &feats);
                anchor += feats[0];
            }
        }
        const b = now_ns();
        t.* = b - a;
    }

    // Force `anchor` to be observable so the loop body isn't dead-code-eliminated.
    std.mem.doNotOptimizeAway(anchor);

    return .{
        .cold_ns = cold_ns,
        .runs = runs_buf,
        .bytes_in = mapped.bytes.len,
        .bytes_out = n * payload.N_FEATURES * @sizeOf(f32),
        .records = n,
    };
}

// ─── cosine top-K over the full reference dataset ──────────────────────────

const SearchStats = struct {
    n_rows: usize,
    queries: usize,
    passes: usize,
    // ns per cosine_topk call.
    min_ns: f64,
    p50_ns: f64,
    p95_ns: f64,
    p99_ns: f64,
    max_ns: f64,
    mean_ns: f64,
    stddev_ns: f64,
};

/// f32 query builder for `bench_cosine_topk_f32`. Mirrors `build_search_queries_q`
/// but reads f32 features directly.
fn build_search_queries_f32(
    ds: transform_reference.Dataset,
    queries: *[SEARCH_QUERIES][search.N_FEATURES]f32,
) void {
    const stride = if (ds.n > SEARCH_QUERIES) ds.n / SEARCH_QUERIES else 1;
    for (0..SEARCH_QUERIES) |i| {
        const row = (i * stride) % ds.n;
        for (0..search.N_FEATURES) |c| {
            const v = ds.features[c * ds.n + row];
            const jitter: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(c)) - 7)) * 0.001;
            queries[i][c] = v + jitter;
        }
    }
}

/// Build `SEARCH_QUERIES` queries by sampling rows from the quantized dataset
/// (dequantizing on the fly) and perturbing them slightly so cosine_topk_q
/// can't shortcut to score=1. Works against either `QuantizedDataset` or
/// `IvfQuantizedDataset` (both have the same `features`/`mins`/`inv_scales`
/// shape for raw row reads).
fn build_search_queries_q(
    qds: transform_reference.IvfQuantizedDataset,
    queries: *[SEARCH_QUERIES][search.N_FEATURES]f32,
) void {
    const stride = if (qds.n > SEARCH_QUERIES) qds.n / SEARCH_QUERIES else 1;
    for (0..SEARCH_QUERIES) |i| {
        const row = (i * stride) % qds.n;
        for (0..search.N_FEATURES) |c| {
            const q_u16 = qds.features[c * qds.n + row];
            const v: f32 = @as(f32, @floatFromInt(q_u16)) * qds.inv_scales[c] + qds.mins[c];
            // Tiny per-feature perturbation to defeat any "exact match" fast path.
            const jitter: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(c)) - 7)) * 0.001;
            queries[i][c] = v + jitter;
        }
    }
}

/// Construct a brute-force `QuantizedDataset` view aliasing the same
/// buffers as the IVF dataset. Used to bench the brute-force path on the
/// v3 blob and to compute recall@5 ground-truth.
fn brute_view(qds: transform_reference.IvfQuantizedDataset) transform_reference.QuantizedDataset {
    return .{
        .n = qds.n,
        .features = qds.features,
        .labels = qds.labels,
        .mins = qds.mins,
        .inv_scales = qds.inv_scales,
    };
}

fn bench_cosine_topk_f32(allocator: std.mem.Allocator) !SearchStats {
    // Dequantize the v2 blob into a 168 MB f32 buffer so we can bench the
    // unquantized inner loop against the quantized one over the same data.
    var blob = try dataset_blob.load_unquant(allocator, DATASET_BIN_PATH);
    defer blob.deinit();
    const ds = blob.dataset;

    var queries: [SEARCH_QUERIES][search.N_FEATURES]f32 = undefined;
    build_search_queries_f32(ds, &queries);

    var anchor: u32 = 0;
    var out: [search.TOP_K]u32 = undefined;

    for (&queries) |*q| {
        search.cosine_topk(ds, q, &out);
        anchor +%= out[0];
    }

    const total_calls = SEARCH_PASSES * SEARCH_QUERIES;
    const samples = try allocator.alloc(u64, total_calls);
    defer allocator.free(samples);

    var idx: usize = 0;
    for (0..SEARCH_PASSES) |_| {
        for (&queries) |*q| {
            const a = now_ns();
            search.cosine_topk(ds, q, &out);
            const b = now_ns();
            samples[idx] = b - a;
            idx += 1;
            anchor +%= out[0];
        }
    }
    std.mem.doNotOptimizeAway(anchor);

    std.mem.sort(u64, samples, {}, std.sort.asc(u64));

    var sum: u128 = 0;
    var sum_sq: u128 = 0;
    for (samples) |x| {
        sum += x;
        sum_sq += @as(u128, x) * @as(u128, x);
    }
    const n_f = @as(f64, @floatFromInt(samples.len));
    const mean = @as(f64, @floatFromInt(sum)) / n_f;
    const variance = @as(f64, @floatFromInt(sum_sq)) / n_f - mean * mean;
    const stddev = @sqrt(@max(variance, 0));

    return .{
        .n_rows = ds.n,
        .queries = SEARCH_QUERIES,
        .passes = SEARCH_PASSES,
        .min_ns = @floatFromInt(samples[0]),
        .p50_ns = @floatFromInt(percentile(samples, 0.50)),
        .p95_ns = @floatFromInt(percentile(samples, 0.95)),
        .p99_ns = @floatFromInt(percentile(samples, 0.99)),
        .max_ns = @floatFromInt(samples[samples.len - 1]),
        .mean_ns = mean,
        .stddev_ns = stddev,
    };
}

/// IVF top-K bench. Production path. Reports recall@5 vs brute-force
/// `cosine_topk_q` on the same query set.
fn bench_cosine_topk_q_ivf(allocator: std.mem.Allocator, recall_out: *f64) !SearchStats {
    var blob = try dataset_blob.load(DATASET_BIN_PATH);
    defer blob.deinit();
    const qds = blob.dataset;
    const brute = brute_view(qds);

    var queries: [SEARCH_QUERIES][search.N_FEATURES]f32 = undefined;
    build_search_queries_q(qds, &queries);

    // Compute brute-force ground-truth top-5 per query, then mean recall@5.
    var brute_tops: [SEARCH_QUERIES][search.TOP_K]u32 = undefined;
    for (&queries, 0..) |*q, qi| search.cosine_topk_q(brute, q, &brute_tops[qi]);

    var ivf_tops: [SEARCH_QUERIES][search.TOP_K]u32 = undefined;
    for (&queries, 0..) |*q, qi| search.cosine_topk_q_ivf(qds, q, &ivf_tops[qi]);

    var hits_total: usize = 0;
    for (0..SEARCH_QUERIES) |qi| {
        for (ivf_tops[qi]) |r| {
            for (brute_tops[qi]) |b| if (r == b) {
                hits_total += 1;
                break;
            };
        }
    }
    recall_out.* = @as(f64, @floatFromInt(hits_total)) /
        @as(f64, @floatFromInt(SEARCH_QUERIES * search.TOP_K));

    var anchor: u32 = 0;
    var out: [search.TOP_K]u32 = undefined;

    // Warm pass.
    for (&queries) |*q| {
        search.cosine_topk_q_ivf(qds, q, &out);
        anchor +%= out[0];
    }

    const total_calls = SEARCH_PASSES * SEARCH_QUERIES;
    const samples = try allocator.alloc(u64, total_calls);
    defer allocator.free(samples);

    var idx: usize = 0;
    for (0..SEARCH_PASSES) |_| {
        for (&queries) |*q| {
            const a = now_ns();
            search.cosine_topk_q_ivf(qds, q, &out);
            const b = now_ns();
            samples[idx] = b - a;
            idx += 1;
            anchor +%= out[0];
        }
    }
    std.mem.doNotOptimizeAway(anchor);

    std.mem.sort(u64, samples, {}, std.sort.asc(u64));

    var sum: u128 = 0;
    var sum_sq: u128 = 0;
    for (samples) |x| {
        sum += x;
        sum_sq += @as(u128, x) * @as(u128, x);
    }
    const n_f = @as(f64, @floatFromInt(samples.len));
    const mean = @as(f64, @floatFromInt(sum)) / n_f;
    const variance = @as(f64, @floatFromInt(sum_sq)) / n_f - mean * mean;
    const stddev = @sqrt(@max(variance, 0));

    return .{
        .n_rows = qds.n,
        .queries = SEARCH_QUERIES,
        .passes = SEARCH_PASSES,
        .min_ns = @floatFromInt(samples[0]),
        .p50_ns = @floatFromInt(percentile(samples, 0.50)),
        .p95_ns = @floatFromInt(percentile(samples, 0.95)),
        .p99_ns = @floatFromInt(percentile(samples, 0.99)),
        .max_ns = @floatFromInt(samples[samples.len - 1]),
        .mean_ns = mean,
        .stddev_ns = stddev,
    };
}

/// Brute-force quantized top-K bench. Kept for side-by-side comparison
/// against `cosine_topk_q_ivf`.
fn bench_cosine_topk_q_brute(allocator: std.mem.Allocator) !SearchStats {
    var blob = try dataset_blob.load(DATASET_BIN_PATH);
    defer blob.deinit();
    const brute = brute_view(blob.dataset);

    var queries: [SEARCH_QUERIES][search.N_FEATURES]f32 = undefined;
    build_search_queries_q(blob.dataset, &queries);

    var anchor: u32 = 0;
    var out: [search.TOP_K]u32 = undefined;

    for (&queries) |*q| {
        search.cosine_topk_q(brute, q, &out);
        anchor +%= out[0];
    }

    const total_calls = SEARCH_PASSES * SEARCH_QUERIES;
    const samples = try allocator.alloc(u64, total_calls);
    defer allocator.free(samples);

    var idx: usize = 0;
    for (0..SEARCH_PASSES) |_| {
        for (&queries) |*q| {
            const a = now_ns();
            search.cosine_topk_q(brute, q, &out);
            const b = now_ns();
            samples[idx] = b - a;
            idx += 1;
            anchor +%= out[0];
        }
    }
    std.mem.doNotOptimizeAway(anchor);

    std.mem.sort(u64, samples, {}, std.sort.asc(u64));

    var sum: u128 = 0;
    var sum_sq: u128 = 0;
    for (samples) |x| {
        sum += x;
        sum_sq += @as(u128, x) * @as(u128, x);
    }
    const n_f = @as(f64, @floatFromInt(samples.len));
    const mean = @as(f64, @floatFromInt(sum)) / n_f;
    const variance = @as(f64, @floatFromInt(sum_sq)) / n_f - mean * mean;
    const stddev = @sqrt(@max(variance, 0));

    return .{
        .n_rows = brute.n,
        .queries = SEARCH_QUERIES,
        .passes = SEARCH_PASSES,
        .min_ns = @floatFromInt(samples[0]),
        .p50_ns = @floatFromInt(percentile(samples, 0.50)),
        .p95_ns = @floatFromInt(percentile(samples, 0.95)),
        .p99_ns = @floatFromInt(percentile(samples, 0.99)),
        .max_ns = @floatFromInt(samples[samples.len - 1]),
        .mean_ns = mean,
        .stddev_ns = stddev,
    };
}

fn print_search_stats(label: []const u8, s: SearchStats) void {
    const total = s.passes * s.queries;
    const rows_f = @as(f64, @floatFromInt(s.n_rows));
    const rows_per_us_min = rows_f / (s.min_ns / 1.0e3);
    std.debug.print(
        "  {s} (per-call latency from {d} samples = {d} passes × {d} queries; n_rows = {d})\n" ++
            "    per-call min:    {d:.1} µs  ({d:.1}M rows/s scan rate)\n" ++
            "    per-call p50:    {d:.1} µs\n" ++
            "    per-call p95:    {d:.1} µs\n" ++
            "    per-call p99:    {d:.1} µs\n" ++
            "    per-call max:    {d:.1} µs\n" ++
            "    per-call mean:   {d:.1} µs ± {d:.1} (stddev)\n",
        .{
            label,
            total,
            s.passes,
            s.queries,
            s.n_rows,
            s.min_ns / 1.0e3,
            rows_per_us_min,
            s.p50_ns / 1.0e3,
            s.p95_ns / 1.0e3,
            s.p99_ns / 1.0e3,
            s.max_ns / 1.0e3,
            s.mean_ns / 1.0e3,
            s.stddev_ns / 1.0e3,
        },
    );
}

// ─── dataset.bin mmap-only load ─────────────────────────────────────────────

const LoadStats = struct {
    runs: usize,
    file_bytes: usize,
    records: usize,
    min_ns: f64,
    p50_ns: f64,
    p99_ns: f64,
    mean_ns: f64,
};

/// Time the warm `dataset_blob.load` path: open + fstat + mmap + header
/// validation. No body parsing, so this is essentially syscalls. Cold-state
/// page-fault timing is not measured here — that's a Phase 7 concern.
fn bench_dataset_load() !LoadStats {
    var samples: [LOAD_RUNS]u64 = undefined;
    var n_records: usize = 0;
    var file_bytes: usize = 0;

    for (&samples) |*s| {
        const a = now_ns();
        var blob = try dataset_blob.load(DATASET_BIN_PATH);
        const b = now_ns();
        s.* = b - a;
        n_records = blob.dataset.n;
        file_bytes = blob.mapped.bytes.len;
        blob.deinit();
    }

    std.mem.sort(u64, &samples, {}, std.sort.asc(u64));

    var sum: u128 = 0;
    for (samples) |x| sum += x;
    const mean = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(samples.len));

    return .{
        .runs = samples.len,
        .file_bytes = file_bytes,
        .records = n_records,
        .min_ns = @floatFromInt(samples[0]),
        .p50_ns = @floatFromInt(percentile(&samples, 0.50)),
        .p99_ns = @floatFromInt(percentile(&samples, 0.99)),
        .mean_ns = mean,
    };
}

fn print_load_stats(label: []const u8, s: LoadStats) void {
    std.debug.print(
        "  {s} (warm-only, {d} runs)\n" ++
            "    file size:    {d} bytes ({d:.2} MiB)\n" ++
            "    records:      {d}\n" ++
            "    per-call min: {d:.1} µs\n" ++
            "    per-call p50: {d:.1} µs\n" ++
            "    per-call p99: {d:.1} µs\n" ++
            "    per-call mean:{d:.1} µs\n",
        .{
            label,
            s.runs,
            s.file_bytes,
            mib(s.file_bytes),
            s.records,
            s.min_ns / 1.0e3,
            s.p50_ns / 1.0e3,
            s.p99_ns / 1.0e3,
            s.mean_ns / 1.0e3,
        },
    );
}

// ─── main ───────────────────────────────────────────────────────────────────

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // Probe for the dataset; skip the whole bench if absent.
    var probe = fast_json.mmap_file(REFERENCE_PATH) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("bench: skipped ({s} not found)\n", .{REFERENCE_PATH});
            return;
        },
        else => return err,
    };
    probe.deinit();

    std.debug.print("=== reference dataset ===\n\n", .{});

    var runs_mmap: [MMAP_RUNS]u64 = undefined;
    var stats_mmap = try bench_reference_mmap(allocator, &runs_mmap);
    print_stats("mmap fast loader", &stats_mmap);

    std.debug.print("\n", .{});

    var runs_json: [STDJSON_RUNS]u64 = undefined;
    var stats_json = try bench_reference_stdjson(allocator, &runs_json);
    print_stats("std.json baseline", &stats_json);

    const speedup = @as(f64, @floatFromInt(stats_json.runs[0])) /
        @as(f64, @floatFromInt(stats_mmap.runs[0]));
    std.debug.print("\nspeedup (warm min vs warm min): {d:.1}x\n", .{speedup});

    std.debug.print("\n=== example payloads ===\n\n", .{});

    // Probe for the payload file too.
    var probe2 = fast_json.mmap_file(PAYLOADS_PATH) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("payload bench: skipped ({s} not found)\n", .{PAYLOADS_PATH});
            return;
        },
        else => return err,
    };
    const payload_count_in_file = blk: {
        var slices: [MAX_PAYLOADS][]const u8 = undefined;
        const n = try slice_payloads(probe2.bytes, &slices);
        break :blk n;
    };
    probe2.deinit();

    var runs_payload: [PAYLOAD_RUNS]u64 = undefined;
    var stats_payload = try bench_payload_vectorize(&runs_payload);
    print_payload_stats("payload.vectorize", &stats_payload, payload_count_in_file, PAYLOAD_BATCH_REPEATS);

    std.debug.print("\n", .{});

    // Per-call distribution: re-mmap and slice once for the latency pass.
    var pc_mapped = try fast_json.mmap_file(PAYLOADS_PATH);
    defer pc_mapped.deinit();
    var pc_slices: [MAX_PAYLOADS][]const u8 = undefined;
    const pc_n = try slice_payloads(pc_mapped.bytes, &pc_slices);
    const per_call = try bench_payload_per_call(allocator, pc_slices[0..pc_n], PAYLOAD_BATCH_REPEATS);
    print_per_call_stats("payload.vectorize", per_call);

    // Probe dataset.bin once — both benches below consume it. Run the
    // dataset_blob.load bench BEFORE cosine_topk: the load path doesn't fault
    // any feature pages (header-only access), so it measures cold-cache state,
    // which is what the boot path will actually see in production.
    var probe3 = dataset_blob.load(DATASET_BIN_PATH) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print(
                "\nskipped (run `zig build prep` first): dataset.bin load + cosine top-K\n",
                .{},
            );
            return;
        },
        else => return err,
    };
    probe3.deinit();

    std.debug.print("\n=== dataset.bin mmap-only load ===\n\n", .{});

    const load_stats = try bench_dataset_load();
    print_load_stats("dataset_blob.load", load_stats);

    std.debug.print("\n=== cosine top-K (quantized u16, IVF probe={d}/{d}) ===\n\n", .{
        dataset_blob.PROBE_CLUSTERS,
        dataset_blob.K_CLUSTERS,
    });

    var recall: f64 = 0;
    const search_stats_ivf = try bench_cosine_topk_q_ivf(allocator, &recall);
    print_search_stats("search.cosine_topk_q_ivf", search_stats_ivf);
    std.debug.print("    mean recall@5:    {d:.3} (vs cosine_topk_q brute force)\n", .{recall});

    std.debug.print("\n=== cosine top-K (quantized u16, brute force) ===\n\n", .{});

    const search_stats_q = try bench_cosine_topk_q_brute(allocator);
    print_search_stats("search.cosine_topk_q", search_stats_q);

    std.debug.print("\n=== cosine top-K (unquantized f32, brute force) ===\n\n", .{});

    const search_stats_f32 = try bench_cosine_topk_f32(allocator);
    print_search_stats("search.cosine_topk", search_stats_f32);
}
