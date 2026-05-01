//! Bench harness. Each `bench_*` function reports on one workload; main runs them
//! all and prints results. Add new ones by writing another `bench_*` and calling
//! it from `main`.

const std = @import("std");
const Io = std.Io;
const rinhapuffer = @import("rinhapuffer");
const transform_reference = rinhapuffer.transform_reference;
const fast_json = rinhapuffer.fast_json;

const REFERENCE_PATH = "./resources/references.json";
const MMAP_RUNS = 5;
const STDJSON_RUNS = 3; // std.json is slow; fewer runs to keep total wall time reasonable.

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
    return ds.features.len * @sizeOf(f32) + ds.labels.len * @sizeOf(bool);
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
    // First run = cold: page cache state depends on prior usage of this file.
    const t0 = now_ns();
    var ds = try transform_reference.load_dataset(allocator, REFERENCE_PATH);
    const t1 = now_ns();
    const cold_ns = t1 - t0;
    const bytes_out = dataset_memory_bytes(ds);
    const records = ds.n;
    ds.deinit(allocator);

    var m = try fast_json.mmap_file(REFERENCE_PATH);
    const bytes_in = m.bytes.len;
    m.deinit();

    for (runs_buf) |*t| {
        const a = now_ns();
        var d = try transform_reference.load_dataset(allocator, REFERENCE_PATH);
        const b = now_ns();
        d.deinit(allocator);
        t.* = b - a;
    }

    return .{
        .cold_ns = cold_ns,
        .runs = runs_buf,
        .bytes_in = bytes_in,
        .bytes_out = bytes_out,
        .records = records,
    };
}

// ─── reference dataset: std.json comparison ─────────────────────────────────

fn load_dataset_stdjson(
    allocator: std.mem.Allocator,
    io: Io,
    path: []const u8,
) !transform_reference.Dataset {
    const file = try Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const st = try file.stat(io);
    const size: usize = @intCast(st.size);
    const contents = try allocator.alloc(u8, size);
    defer allocator.free(contents);
    const got = try file.readPositionalAll(io, contents, 0);
    if (got != size) return error.ShortRead;

    const Entry = struct {
        vector: [transform_reference.N_FEATURES]f32,
        label: []const u8,
    };
    const parsed = try std.json.parseFromSlice([]Entry, allocator, contents, .{});
    defer parsed.deinit();

    const n = parsed.value.len;
    const features = try allocator.alloc(f32, transform_reference.N_FEATURES * n);
    errdefer allocator.free(features);
    const labels = try allocator.alloc(bool, n);
    errdefer allocator.free(labels);

    for (parsed.value, 0..) |entry, row| {
        labels[row] = std.mem.eql(u8, entry.label, "fraud");
        for (entry.vector, 0..) |v, c| {
            features[c * n + row] = v;
        }
    }

    return .{ .n = n, .features = features, .labels = labels };
}

fn bench_reference_stdjson(
    allocator: std.mem.Allocator,
    io: Io,
    runs_buf: []u64,
) !Stats {
    const t0 = now_ns();
    var ds = try load_dataset_stdjson(allocator, io, REFERENCE_PATH);
    const t1 = now_ns();
    const cold_ns = t1 - t0;
    const bytes_out = dataset_memory_bytes(ds);
    const records = ds.n;
    ds.deinit(allocator);

    var m = try fast_json.mmap_file(REFERENCE_PATH);
    const bytes_in = m.bytes.len;
    m.deinit();

    for (runs_buf) |*t| {
        const a = now_ns();
        var d = try load_dataset_stdjson(allocator, io, REFERENCE_PATH);
        const b = now_ns();
        d.deinit(allocator);
        t.* = b - a;
    }

    return .{
        .cold_ns = cold_ns,
        .runs = runs_buf,
        .bytes_in = bytes_in,
        .bytes_out = bytes_out,
        .records = records,
    };
}

// ─── main ───────────────────────────────────────────────────────────────────

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

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
    var stats_json = try bench_reference_stdjson(allocator, io, &runs_json);
    print_stats("std.json baseline", &stats_json);

    const speedup = @as(f64, @floatFromInt(stats_json.runs[0])) /
        @as(f64, @floatFromInt(stats_mmap.runs[0]));
    std.debug.print("\nspeedup (warm min vs warm min): {d:.1}x\n", .{speedup});
}
