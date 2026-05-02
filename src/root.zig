//! Library root for the rinhapuffer package. Re-exports the modules that
//! consumers (the `prep` exe, the bench harness, and the eventual HTTP server
//! in Phase 6) link against.

const builtin = @import("builtin");

pub const fast_json = @import("fast_json.zig");
pub const transform_reference = @import("transform_reference.zig");
pub const payload = @import("payload.zig");
pub const search = @import("search.zig");
pub const dataset_blob = @import("dataset_blob.zig");
pub const kmeans = @import("kmeans.zig");
pub const http = @import("http.zig");
pub const handler = @import("handler.zig");
// http_async is Linux-only (epoll). On other targets we expose an empty
// namespace so this module still compiles; main.zig comptime-branches the
// call site so the empty namespace is never used on Mac/etc.
pub const http_async = if (builtin.os.tag == .linux) @import("http_async.zig") else struct {};

test {
    _ = fast_json;
    _ = transform_reference;
    _ = payload;
    _ = search;
    _ = dataset_blob;
    _ = kmeans;
    _ = http;
    _ = handler;
    if (builtin.os.tag == .linux) _ = http_async;
}
