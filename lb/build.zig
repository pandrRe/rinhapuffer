const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const default_target: std.Target.Query = if (builtin.os.tag == .linux)
        .{ .abi = .musl }
    else
        .{};
    const target = b.standardTargetOptions(.{ .default_target = default_target });
    const optimize = b.standardOptimizeOption(.{});

    const exe_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .strip = if (optimize == .Debug) null else true,
        .omit_frame_pointer = if (optimize == .Debug) null else true,
    });

    const cflags = [_][]const u8{
        "-std=gnu11",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
        "-fno-stack-protector",
        "-fno-asynchronous-unwind-tables",
        "-fno-unwind-tables",
    };

    exe_mod.addCSourceFiles(.{
        .files = &.{ "src/main.c", "src/ring.c" },
        .flags = &cflags,
    });

    const exe = b.addExecutable(.{
        .name = "rinhalb",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);
}
