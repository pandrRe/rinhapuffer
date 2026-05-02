//! Server entry point. Phase 7: load `dataset.bin` once, then spin the
//! blocking accept loop. The handler logic + dataset state lives in
//! `handler.zig`; this file is just bind/accept/serve.

const std = @import("std");
const rinhapuffer = @import("rinhapuffer");
const http = rinhapuffer.http;
const handler = rinhapuffer.handler;
const libc = std.c;

const PORT: u16 = 9999;
const DATASET_PATH: []const u8 = "./resources/dataset.bin";

pub fn main(_: std.process.Init) !void {
    try handler.init_dataset(DATASET_PATH);

    const listen_fd = try http.bind_listen(PORT);
    defer _ = libc.close(listen_fd);

    std.debug.print("rinhapuffer listening on 0.0.0.0:{d}\n", .{PORT});

    while (true) {
        const client_fd = try http.accept_one(listen_fd);
        defer _ = libc.close(client_fd);
        http.serve(client_fd, &handler.dispatch);
    }
}
