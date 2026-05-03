//! Phase 7: the wired-up `/fraud-score` route.
//!
//! State (mmap blob, query buffer, top-K buffer) is module-level and lives in
//! BSS — same stance as `http.zig`. Single-threaded server → at most one in-
//! flight request → safe to share across the program lifetime.
//!
//! The handler signature stays `*const fn (Request) Response`; the dataset is
//! reachable via the file-private global. `init_dataset` must be called once
//! before any `dispatch` invocation; `main` enforces this by loading the blob
//! before opening the listen socket.
//!
//! Hot-path budget: zero allocations, zero syscalls (the dataset mmap is
//! faulted in lazily by the kernel on first scan; no `read`/`write` here).
//! Vectorize: ~1 µs. IVF top-K: p99 ~64 µs (Phase 5). Response is one of six
//! precomputed static strings — no `std.fmt`.

const std = @import("std");
const http = @import("http.zig");
const payload = @import("payload.zig");
const search = @import("search.zig");
const dataset_blob = @import("dataset_blob.zig");
const instrument = @import("instrument.zig");

const N_FEATURES = payload.N_FEATURES;
const TOP_K = search.TOP_K;

// Owned state. Loaded by `init_dataset`, never released (process-lifetime mmap).
var blob: ?dataset_blob.IvfQuantizedBlob = null;

// Per-request scratch. Single-threaded server → safe to share.
var q_buf: [N_FEATURES]f32 = undefined;
var top_rows: [TOP_K]u32 = undefined;

// Backing buffer for the /__metrics text response. Module-level → outlives
// the dispatch return so the response body slice stays valid through the
// epoll write drain. Single-threaded server → safe to share. Only written
// when instrument is enabled; in non-instrument builds the helper returns
// a tiny static "off" message so the buffer is unused but cheap.
var metrics_buf: [instrument.REPORT_BUF_SIZE]u8 = undefined;

// Keep-alive policy is a comptime decision driven by the `-Dkeep-alive` build
// option. Default off — k6 ramping load showed that honouring keep-alive head-
// of-lines the single-threaded accept loop on idle VU connections; closing
// per response keeps the accept queue draining. Build with
// `zig build -Dkeep-alive=true` to flip it back on.
const build_options = @import("build_options");
pub const KEEP_ALIVE: bool = build_options.keep_alive;

inline fn ka(req: http.Request) bool {
    return if (KEEP_ALIVE) req.keep_alive else false;
}

/// `RESPONSES[fraud_count]` is the on-the-wire JSON body for that count.
/// Score = `fraud_count / 5` ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}; per spec
/// `approved = score < 0.6`, so counts 0–2 are approved and 3–5 are not.
const RESPONSES = [6][]const u8{
    "{\"approved\":true,\"fraud_score\":0.0}",
    "{\"approved\":true,\"fraud_score\":0.2}",
    "{\"approved\":true,\"fraud_score\":0.4}",
    "{\"approved\":false,\"fraud_score\":0.6}",
    "{\"approved\":false,\"fraud_score\":0.8}",
    "{\"approved\":false,\"fraud_score\":1.0}",
};

pub const InitError = dataset_blob.LoadError;

/// Mmap and validate the dataset blob, then prefault every page + best-
/// effort `mlock`. Must be called once before `dispatch`. Asserts no prior
/// dataset is loaded so a duplicate init is a loud panic, not a silent leak
/// of the previous mmap.
pub fn init_dataset(path: []const u8) InitError!void {
    std.debug.assert(blob == null);
    blob = try dataset_blob.load(path);
    dataset_blob.prefault_and_lock(&blob.?);
}

/// Run `iters` synthetic queries through `search.euclidean_topk_q_ivf` to
/// warm i-cache, branch predictor, and the IVF traversal code paths before
/// the first real request lands. Cheap (~5 µs/iter). Caller must have
/// already invoked `init_dataset`.
pub fn warmup(iters: usize) void {
    std.debug.assert(blob != null);
    const ds = blob.?.dataset;

    var prng = std.Random.Xoshiro256.init(0xa5a5_a5a5_a5a5_a5a5);
    const rand = prng.random();
    var q: [N_FEATURES]f32 = undefined;
    var top_rows_local: [TOP_K]u32 = undefined;

    var i: usize = 0;
    while (i < iters) : (i += 1) {
        inline for (0..N_FEATURES) |k| q[k] = rand.float(f32);
        search.euclidean_topk_q_ivf(ds, &q, &top_rows_local);
        std.mem.doNotOptimizeAway(top_rows_local);
    }
}

/// Route table.
///   `GET  /ready`         → 200, empty body
///   `POST /fraud-score`   → vectorize → IVF top-5 → fraud-count → response[count]
///   `GET  /__metrics`     → instrument report
///   anything else         → 404
/// Routes have unique path lengths (6, 12, 10) → key the dispatch on
/// `path.len` so the compiler emits one cmp + jump instead of three
/// sequential `mem.eql` chains. Vectorize errors → 400, keep_alive preserved.
pub fn dispatch(req: http.Request) http.Response {
    switch (req.path.len) {
        6 => {
            if (req.method == .get and std.mem.eql(u8, req.path, "/ready")) {
                return .{ .status = 200, .body = "", .content_type = "text/plain", .keep_alive = ka(req) };
            }
        },
        12 => {
            if (req.method == .post and std.mem.eql(u8, req.path, "/fraud-score")) {
                return handle_fraud_score(req);
            }
        },
        10 => {
            if (req.method == .get and std.mem.eql(u8, req.path, "/__metrics")) {
                const body = instrument.render(&metrics_buf);
                return .{ .status = 200, .body = body, .content_type = "text/plain", .keep_alive = ka(req) };
            }
        },
        else => {},
    }
    return .{ .status = 404, .body = "", .content_type = "text/plain", .keep_alive = ka(req) };
}

fn handle_fraud_score(req: http.Request) http.Response {
    instrument.inc(&instrument.req_total, 1);

    const t_vec = instrument.now_ns();
    payload.vectorize(req.body, &q_buf) catch {
        instrument.inc(&instrument.req_vectorize_err, 1);
        return .{ .status = 400, .body = "", .content_type = "text/plain", .keep_alive = ka(req) };
    };
    instrument.observe_since(&instrument.hist_vectorize, t_vec);

    const qds = blob.?.dataset;
    const t_search = instrument.now_ns();
    search.euclidean_topk_q_ivf(qds, &q_buf, &top_rows);
    instrument.observe_since(&instrument.hist_search, t_search);

    var fraud_count: u8 = 0;
    inline for (0..TOP_K) |i| {
        fraud_count += dataset_blob.label_at(qds.labels_bits, top_rows[i]);
    }
    return .{
        .status = 200,
        .body = RESPONSES[fraud_count],
        .content_type = "application/json",
        .keep_alive = ka(req),
    };
}

// ─── test helpers ──────────────────────────────────────────────────────────

/// Test-only: install an externally-loaded blob. Asserts the global is empty
/// so test ordering bugs surface as a panic, not silent state carryover.
pub fn set_blob_for_test(b: dataset_blob.IvfQuantizedBlob) void {
    std.debug.assert(blob == null);
    blob = b;
}

/// Test-only: release the current blob (munmap + close fd) and reset the
/// global. Idempotent.
pub fn deinit_for_test() void {
    if (blob) |b| {
        b.deinit();
        blob = null;
    }
}

// ─── tests ─────────────────────────────────────────────────────────────────

const transform_reference = @import("transform_reference.zig");
const fast_json = @import("fast_json.zig");

// Fixture K must be ≥ `search` PROBE_CLUSTERS so the production
// `euclidean_topk_q_ivf` invariant holds. The example-references dataset has
// 100 rows so K up to ~50 leaves enough rows/cluster for k-means to converge.
const TEST_K: u32 = 8;

fn tmp_abs_path(
    allocator: std.mem.Allocator,
    tmp: *std.testing.TmpDir,
    sub_path: []const u8,
) ![]u8 {
    var dir_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const dir_len = try tmp.dir.realPath(std.testing.io, &dir_buf);
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir_buf[0..dir_len], sub_path });
}

/// Build a temp `dataset.bin` from `./resources/example-references.json` and
/// install it via `set_blob_for_test`. Returns the owned tmp dir; caller must
/// `defer tmp.cleanup()` and `defer deinit_for_test()`.
fn install_test_blob(allocator: std.mem.Allocator) !std.testing.TmpDir {
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
    errdefer tmp.cleanup();

    try dataset_blob.write(allocator, io, tmp.dir, "dataset.bin", ds, TEST_K);

    const path = try tmp_abs_path(allocator, &tmp, "dataset.bin");
    defer allocator.free(path);
    const b = try dataset_blob.load_any_k(path);
    set_blob_for_test(b);
    return tmp;
}

fn slice_nth_payload(bytes: []const u8, index: usize) ![]const u8 {
    var p: usize = 0;
    p = fast_json.skip_ws(bytes, p);
    p = try fast_json.expect_byte(bytes, p, '[');
    var i: usize = 0;
    var first = true;
    while (true) {
        p = fast_json.skip_ws(bytes, p);
        if (p < bytes.len and bytes[p] == ']') return error.OutOfRange;
        if (!first) {
            p = try fast_json.expect_byte(bytes, p, ',');
            p = fast_json.skip_ws(bytes, p);
        }
        const start = p;
        const end = try fast_json.skip_to_matching_close(bytes, p + 1, '{', '}');
        if (i == index) return bytes[start..end];
        p = end;
        first = false;
        i += 1;
    }
}

fn is_one_of_responses(body: []const u8) bool {
    for (RESPONSES) |r| {
        if (std.mem.eql(u8, body, r)) return true;
    }
    return false;
}

test "dispatch — /ready honours comptime keep-alive policy" {
    const resp_ka = dispatch(.{ .method = .get, .path = "/ready", .body = "", .keep_alive = true });
    try std.testing.expectEqual(@as(u16, 200), resp_ka.status);
    try std.testing.expectEqualStrings("", resp_ka.body);
    // KEEP_ALIVE off → always close. KEEP_ALIVE on → mirror the request.
    try std.testing.expectEqual(KEEP_ALIVE, resp_ka.keep_alive);

    const resp_close = dispatch(.{ .method = .get, .path = "/ready", .body = "", .keep_alive = false });
    try std.testing.expect(!resp_close.keep_alive);
}

test "dispatch — unknown route → 404" {
    const resp = dispatch(.{ .method = .get, .path = "/nope", .body = "", .keep_alive = true });
    try std.testing.expectEqual(@as(u16, 404), resp.status);
    try std.testing.expectEqualStrings("", resp.body);
    try std.testing.expectEqual(KEEP_ALIVE, resp.keep_alive);
}

test "dispatch — /fraud-score vectorize failure → 400" {
    const resp = dispatch(.{ .method = .post, .path = "/fraud-score", .body = "{}", .keep_alive = true });
    try std.testing.expectEqual(@as(u16, 400), resp.status);
    try std.testing.expectEqualStrings("", resp.body);
    try std.testing.expectEqual(KEEP_ALIVE, resp.keep_alive);
}

test "dispatch — /fraud-score on example payload[0]" {
    const allocator = std.testing.allocator;

    var tmp = try install_test_blob(allocator);
    defer tmp.cleanup();
    defer deinit_for_test();

    var payloads = try fast_json.mmap_file("./resources/example-payloads.json");
    defer payloads.deinit();

    const body = try slice_nth_payload(payloads.bytes, 0);
    const resp = dispatch(.{ .method = .post, .path = "/fraud-score", .body = body, .keep_alive = true });

    try std.testing.expectEqual(@as(u16, 200), resp.status);
    try std.testing.expect(is_one_of_responses(resp.body));
    try std.testing.expectEqualStrings("application/json", resp.content_type);
    try std.testing.expectEqual(KEEP_ALIVE, resp.keep_alive);
}

test "dispatch — /fraud-score across all example payloads stays in spec" {
    const allocator = std.testing.allocator;

    var tmp = try install_test_blob(allocator);
    defer tmp.cleanup();
    defer deinit_for_test();

    var payloads = try fast_json.mmap_file("./resources/example-payloads.json");
    defer payloads.deinit();

    var i: usize = 0;
    while (i < 50) : (i += 1) {
        const body = try slice_nth_payload(payloads.bytes, i);
        const resp = dispatch(.{ .method = .post, .path = "/fraud-score", .body = body, .keep_alive = true });
        try std.testing.expectEqual(@as(u16, 200), resp.status);
        try std.testing.expect(is_one_of_responses(resp.body));
    }
}
