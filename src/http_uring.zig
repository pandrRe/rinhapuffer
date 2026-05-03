//! Linux-only async HTTP/1.1 server using io_uring. Designed to replace the
//! epoll loop in `http_async.zig`. All allocation is at startup; the hot path
//! is one `io_uring_enter` syscall per loop tick, amortized across the batch
//! of CQEs drained that tick.
//!
//! Milestones (incremental, see plan in `docs/`):
//!   M1 (this commit) — skeleton: multishot accept, close on accept.
//!   M2 — one-shot recv + writev + close (functional correctness).
//!   M3 — registered files (fixed-file table).
//!   M4 — BUF_RING for recv.
//!   M5 — multishot recv.
//!   M6 — TCP_NODELAY, generation-tag stale-CQE drop, comptime wiring polish.
//!
//! Kernel floor is 5.19 (multishot accept, BUF_RING). Multishot recv (6.0)
//! lands in M5 behind a comptime fallback.

const std = @import("std");
const builtin = @import("builtin");
const http = @import("http.zig");
const instrument = @import("instrument.zig");

comptime {
    if (builtin.os.tag != .linux) {
        @compileError("http_uring is Linux-only — main.zig must comptime-branch");
    }
}

const linux = std.os.linux;

pub const MAX_CONNS: u32 = 256;
pub const READ_BUF_SIZE: usize = 8192;
pub const HEAD_BUF_SIZE: usize = 256;

// SQ depth holds at most: 1 multishot accept + 1 recv per conn + 1 writev per
// in-flight write + 1 close per conn-being-closed = ~3·MAX_CONNS upper bound.
// Round up to the next power of two with headroom.
const SQ_DEPTH: u16 = 1024;
const CQES_PER_BATCH: usize = 256;

// user_data tag layout (u64):
//   bits 0..15   slot index (0..MAX_CONNS-1)
//   bits 16..23  op tag      (ACCEPT, RECV, WRITEV, CLOSE)
//   bits 24..31  generation  (bumped on slot reuse — used for stale-CQE drop)
//   bits 32..63  reserved
const OP_ACCEPT: u8 = 0;
// OP_RECV/OP_WRITEV/OP_CLOSE land in M2.

inline fn make_user_data(slot: u16, op: u8, gen: u8) u64 {
    return (@as(u64, gen) << 24) | (@as(u64, op) << 16) | @as(u64, slot);
}

inline fn ud_op(ud: u64) u8 {
    return @intCast((ud >> 16) & 0xff);
}

inline fn ud_slot(ud: u64) u16 {
    return @intCast(ud & 0xffff);
}

pub const RunError = error{
    InitFailed,
    SubmitFailed,
    AcceptMultishotLost,
};

pub fn run(listen_fd: i32, dispatch: http.Handler) RunError!noreturn {
    _ = dispatch; // M1: data path not wired yet — future M2.

    // Plain io_uring init for M1; SETUP flags (COOP_TASKRUN/SINGLE_ISSUER/
    // DEFER_TASKRUN) layered later once the basic flow works.
    var params: linux.io_uring_params = std.mem.zeroes(linux.io_uring_params);
    var ring = linux.IoUring.init_params(SQ_DEPTH, &params) catch return error.InitFailed;
    defer ring.deinit();

    instrument.init();

    arm_accept(&ring, listen_fd) catch return error.SubmitFailed;

    var cqes: [CQES_PER_BATCH]linux.io_uring_cqe = undefined;
    while (true) {
        // submit_and_wait flushes pending SQEs to the kernel AND blocks for
        // ≥1 CQE — one io_uring_enter per loop tick. `copy_cqes` alone would
        // never flush the SQ so the accept SQE would sit in user space.
        _ = ring.submit_and_wait(1) catch return error.SubmitFailed;

        while (true) {
            const n = ring.copy_cqes(&cqes, 0) catch return error.SubmitFailed;
            if (n == 0) break;
            for (cqes[0..n]) |cqe| {
                switch (ud_op(cqe.user_data)) {
                    OP_ACCEPT => try handle_accept(&ring, listen_fd, cqe),
                    else => {
                        // Unknown op — should be impossible at M1 since only
                        // accept is submitted. Drop silently.
                    },
                }
            }
        }
    }
}

fn arm_accept(ring: *linux.IoUring, listen_fd: i32) !void {
    _ = try ring.accept_multishot(make_user_data(0, OP_ACCEPT, 0), listen_fd, null, null, 0);
}

fn handle_accept(ring: *linux.IoUring, listen_fd: i32, cqe: linux.io_uring_cqe) RunError!void {
    if (cqe.res >= 0) {
        const fd: i32 = cqe.res;
        instrument.inc(&instrument.accepts, 1);
        // M1: hold no per-conn state; close immediately so the kernel can
        // tear down. M2 replaces this with `pool_alloc + submit_recv`.
        _ = linux.close(fd);
        instrument.inc(&instrument.conn_closes, 1);
    }
    // else: per-accept error (e.g. client RST'd before accept queued).
    // Multishot stays armed via F_MORE — only re-arm when the kernel says
    // it's done with this multishot SQE.

    if (cqe.flags & linux.IORING_CQE_F_MORE == 0) {
        // Multishot terminated (typically only on real listen-fd error).
        // Re-arm once; if even that fails, propagate.
        arm_accept(ring, listen_fd) catch return error.AcceptMultishotLost;
    }
}
