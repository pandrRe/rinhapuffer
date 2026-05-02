//! Server entry point. Phase 7: load `dataset.bin` once, then spin the
//! blocking accept loop. The handler logic + dataset state lives in
//! `handler.zig`; this file is just load / bind / accept / serve.
//!
//! Keep-alive policy is comptime — set via `zig build -Dkeep-alive=true`,
//! defaults off (see `handler.KEEP_ALIVE`).
//!
//! Listen mode is env-driven: `RINHAPUFFER_SOCKET=/path/api.sock` switches
//! to a Unix domain socket (Phase 8 — haproxy fronts two API instances
//! over UDS). Otherwise the server binds TCP `0.0.0.0:9999`.

const std = @import("std");
const rinhapuffer = @import("rinhapuffer");
const http = rinhapuffer.http;
const handler = rinhapuffer.handler;
const libc = std.c;

const PORT: u16 = 9999;
const DATASET_PATH: []const u8 = "./resources/dataset.bin";
const SOCKET_ENV: [*:0]const u8 = "RINHAPUFFER_SOCKET";

pub fn main(_: std.process.Init) !void {
    try handler.init_dataset(DATASET_PATH);

    const listen_fd = blk: {
        if (libc.getenv(SOCKET_ENV)) |raw| {
            const path = std.mem.span(@as([*:0]const u8, raw));
            const fd = try http.bind_listen_unix(path);
            std.debug.print("rinhapuffer listening on unix:{s} (keep-alive {s})\n", .{
                path,
                if (handler.KEEP_ALIVE) "on" else "off",
            });
            break :blk fd;
        }
        const fd = try http.bind_listen(PORT);
        std.debug.print("rinhapuffer listening on 0.0.0.0:{d} (keep-alive {s})\n", .{
            PORT,
            if (handler.KEEP_ALIVE) "on" else "off",
        });
        break :blk fd;
    };
    defer _ = libc.close(listen_fd);

    while (true) {
        const client_fd = try http.accept_one(listen_fd);
        defer _ = libc.close(client_fd);
        http.serve(client_fd, &handler.dispatch);
    }
}
