#!/usr/bin/env bash
# rinhalb host-side build.
#
# Cross-compiles the load balancer to a static linux-musl target via `zig cc`
# (driven by build.zig). Optionally builds the container image and pushes it
# to GHCR. Mirrors the top-level build.sh interface.
#
# Usage:
#   ./build.sh                            # zig only: amd64 binary
#   TARGET_ARCH=arm64 ./build.sh          # native arm64
#   IMAGE=1 ./build.sh                    # also build container
#   PUSH=1 GHCR_USER=pandrre ./build.sh   # build + tag + push to ghcr.io

set -euo pipefail
cd "$(dirname "$0")"

TARGET_ARCH="${TARGET_ARCH:-amd64}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_TOOL="${CONTAINER_TOOL:-$(command -v podman 2>/dev/null || command -v docker)}"

case "$TARGET_ARCH" in
    amd64) ZIG_TARGET="x86_64-linux-musl" ;;
    arm64) ZIG_TARGET="aarch64-linux-musl" ;;
    *) echo "unknown TARGET_ARCH=$TARGET_ARCH (want amd64 or arm64)"; exit 2 ;;
esac

if [ -n "${PUSH:-}" ]; then
    IMAGE=1
    if [ -z "${GHCR_USER:-}" ]; then
        echo "PUSH=1 requires GHCR_USER (your GitHub username)"
        exit 2
    fi
fi

echo "==> build rinhalb for ${ZIG_TARGET}"
zig build -Doptimize=ReleaseFast -Dtarget="${ZIG_TARGET}"

echo
echo "==> artifact"
ls -lh zig-out/bin/rinhalb
file zig-out/bin/rinhalb 2>/dev/null || true

if [ -z "${IMAGE:-}" ]; then
    exit 0
fi

LOCAL_TAG="localhost/rinhalb:${TARGET_ARCH}"
echo
echo "==> ${CONTAINER_TOOL} build --platform=linux/${TARGET_ARCH} -t ${LOCAL_TAG} ."
"${CONTAINER_TOOL}" build --platform="linux/${TARGET_ARCH}" -t "${LOCAL_TAG}" .

if [ -z "${PUSH:-}" ]; then
    echo
    echo "==> built ${LOCAL_TAG} (set PUSH=1 GHCR_USER=<username> to publish)"
    exit 0
fi

REMOTE_TAG="ghcr.io/${GHCR_USER}/rinhalb:${IMAGE_TAG}"
echo
echo "==> ${CONTAINER_TOOL} tag ${LOCAL_TAG} ${REMOTE_TAG}"
"${CONTAINER_TOOL}" tag "${LOCAL_TAG}" "${REMOTE_TAG}"
echo "==> ${CONTAINER_TOOL} push ${REMOTE_TAG}"
"${CONTAINER_TOOL}" push "${REMOTE_TAG}"
echo
echo "==> published ${REMOTE_TAG}"
