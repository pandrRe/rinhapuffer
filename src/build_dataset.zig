//! Build-time tool: parse `resources/references.json`, L2-normalize, write
//! `resources/dataset.bin`. Wired in `build.zig` as the `prep` step.
//!
//! Run with `zig build prep`. Re-run whenever `references.json` changes.

const std = @import("std");
const rinhapuffer = @import("rinhapuffer");

const REFERENCES_PATH = "./resources/references.json";
const OUTPUT_SUB_PATH = "resources/dataset.bin";

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var mapped = try rinhapuffer.fast_json.mmap_file(REFERENCES_PATH);
    defer mapped.deinit();

    const n = rinhapuffer.transform_reference.count_records(mapped.bytes);
    const features = try allocator.alloc(f32, rinhapuffer.transform_reference.N_FEATURES * n);
    defer allocator.free(features);
    const labels = try allocator.alloc(bool, n);
    defer allocator.free(labels);

    const ds = try rinhapuffer.transform_reference.parse_into(mapped.bytes, features, labels);

    const cwd = std.Io.Dir.cwd();
    try rinhapuffer.dataset_blob.write(
        allocator,
        init.io,
        cwd,
        OUTPUT_SUB_PATH,
        ds,
        rinhapuffer.dataset_blob.K_CLUSTERS,
    );

    std.debug.print(
        "wrote {s}: {d} records, {d} bytes (K={d} clusters)\n",
        .{
            OUTPUT_SUB_PATH,
            ds.n,
            rinhapuffer.dataset_blob.blob_size(@intCast(ds.n), rinhapuffer.dataset_blob.K_CLUSTERS),
            rinhapuffer.dataset_blob.K_CLUSTERS,
        },
    );
}
