//! Server entry point. Phase 6: HTTP/1.1 skeleton with two routes returning
//! canned responses. Phase 7 will load `dataset.bin` once at startup and
//! plug `payload.vectorize` + `cosine_topk_q_ivf` into the `/fraud-score`
//! handler.

const std = @import("std");
const rinhapuffer = @import("rinhapuffer");
const http = rinhapuffer.http;
const libc = std.c;

const PORT: u16 = 9999;

const FRAUD_STUB: []const u8 = "{\"approved\":true,\"fraud_score\":0.0}";

fn dispatch(req: http.Request) http.Response {
    if (req.method == .get and std.mem.eql(u8, req.path, "/ready")) {
        return .{ .status = 200, .body = "", .content_type = "text/plain", .keep_alive = req.keep_alive };
    }
    if (req.method == .post and std.mem.eql(u8, req.path, "/fraud-score")) {
        return .{ .status = 200, .body = FRAUD_STUB, .keep_alive = req.keep_alive };
    }
    return .{ .status = 404, .body = "", .content_type = "text/plain", .keep_alive = req.keep_alive };
}

pub fn main(_: std.process.Init) !void {
    const listen_fd = try http.bind_listen(PORT);
    defer _ = libc.close(listen_fd);

    std.debug.print("rinhapuffer listening on 0.0.0.0:{d}\n", .{PORT});

    while (true) {
        const client_fd = try http.accept_one(listen_fd);
        defer _ = libc.close(client_fd);
        http.serve(client_fd, &dispatch);
    }
}
