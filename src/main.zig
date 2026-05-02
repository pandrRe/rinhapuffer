//! Default exe entry point. The HTTP server lands in Phase 6; until then this
//! binary exists so `zig build` produces *something* runnable, and to give the
//! exe-test step a root module to attach to.

const std = @import("std");

pub fn main(_: std.process.Init) !void {
    std.debug.print(
        "rinhapuffer: HTTP server lands in Phase 6. Use `zig build prep` and `zig build bench`.\n",
        .{},
    );
}
