#!/usr/bin/env bash
# Phase 8 host-side build.
#
# Cross-compiles the rinhapuffer binary to a static linux-musl target while
# running `prep` natively on the host. Optionally builds the container image
# and pushes it to GitHub Container Registry.
#
# `dataset.bin` is portable across LE targets (header is u32 LE, payload is
# fixed-width i16/f32 LE), so the macOS-built file works unchanged inside the
# linux-amd64 image.
#
# Usage:
#   ./build.sh                            # zig only: amd64 binary + dataset
#   TARGET_ARCH=arm64 ./build.sh          # native arm64 (fast local podman dev)
#   SKIP_PREP=1 ./build.sh                # reuse existing resources/dataset.bin
#   IMAGE=1 ./build.sh                    # also `podman build` the image
#   PUSH=1 GHCR_USER=pedroandre9877 ./build.sh
#                                         # build + tag + push to ghcr.io
#   IMAGE_TAG=v0.1 PUSH=1 GHCR_USER=... ./build.sh
#                                         # custom tag (default: latest)
#
# `podman login ghcr.io` is a one-time user setup — this script does NOT
# manage credentials. See https://docs.github.com/en/packages.

set -euo pipefail
cd "$(dirname "$0")"

TARGET_ARCH="${TARGET_ARCH:-amd64}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

case "$TARGET_ARCH" in
    amd64) ZIG_TARGET="x86_64-linux-musl" ;;
    arm64) ZIG_TARGET="aarch64-linux-musl" ;;
    *) echo "unknown TARGET_ARCH=$TARGET_ARCH (want amd64 or arm64)"; exit 2 ;;
esac

# `PUSH=1` implies `IMAGE=1`. Validate GHCR_USER up front so we don't run a
# 50-second prep before discovering the env var is missing.
if [ -n "${PUSH:-}" ]; then
    IMAGE=1
    if [ -z "${GHCR_USER:-}" ]; then
        echo "PUSH=1 requires GHCR_USER (your GitHub username)"
        exit 2
    fi
fi

# ─── 1. dataset.bin (native zig, LE-portable output) ────────────────────────

if [ -z "${SKIP_PREP:-}" ]; then
    echo "==> prep dataset.bin (native zig — output is LE-portable)"
    zig build prep -Doptimize=ReleaseFast
else
    echo "==> SKIP_PREP=1 → reusing $(ls -lh resources/dataset.bin | awk '{print $5}') existing dataset.bin"
fi

# ─── 2. rinhapuffer binary (cross-compile to linux-musl) ────────────────────

echo "==> build rinhapuffer for ${ZIG_TARGET}"
# Linux artifact uses the epoll async accept loop (Phase 9.2) which holds
# many persistent conns concurrently — HoL on idle keep-alive is no longer
# a concern. Mac native build keeps the blocking accept loop and defaults
# `-Dkeep-alive=false` (per `build.zig`).
zig build -Doptimize=ReleaseFast -Dtarget="${ZIG_TARGET}" -Dkeep-alive=true

echo
echo "==> artifacts"
ls -lh zig-out/bin/rinhapuffer resources/dataset.bin
file zig-out/bin/rinhapuffer 2>/dev/null || true

# ─── 3. (optional) container image build ────────────────────────────────────

if [ -z "${IMAGE:-}" ]; then
    exit 0
fi

LOCAL_TAG="localhost/rinhapuffer:${TARGET_ARCH}"
echo
echo "==> podman build --platform=linux/${TARGET_ARCH} -t ${LOCAL_TAG} ."
podman build --platform="linux/${TARGET_ARCH}" -t "${LOCAL_TAG}" .

# ─── 4. (optional) tag + push to GHCR ───────────────────────────────────────

if [ -z "${PUSH:-}" ]; then
    echo
    echo "==> built ${LOCAL_TAG} (set PUSH=1 GHCR_USER=<username> to publish)"
    exit 0
fi

REMOTE_TAG="ghcr.io/${GHCR_USER}/rinhapuffer:${IMAGE_TAG}"
echo
echo "==> podman tag ${LOCAL_TAG} ${REMOTE_TAG}"
podman tag "${LOCAL_TAG}" "${REMOTE_TAG}"
echo "==> podman push ${REMOTE_TAG}"
podman push "${REMOTE_TAG}"
echo
echo "==> published ${REMOTE_TAG}"
echo "    update docker-compose.yml \`image:\` lines to ${REMOTE_TAG} for the rinha submission."
