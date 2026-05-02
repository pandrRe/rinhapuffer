# syntax=docker/dockerfile:1.7
#
# Single-stage scratch image for rinhapuffer (Phase 8).
#
# Both artifacts (statically-linked musl binary + dataset.bin) are produced on
# the host by `./build.sh` — no in-container compile, no qemu emulation cost.
# This file is just the packaging layer.
#
# Build:
#   ./build.sh                                       # cross-compile amd64 + prep
#   podman build -t rinhapuffer:amd64 .              # package
#
# At runtime the server picks transport from RINHAPUFFER_SOCKET (unset → TCP :9999).

FROM scratch

COPY zig-out/bin/rinhapuffer /rinhapuffer
COPY resources/dataset.bin /resources/dataset.bin

WORKDIR /

EXPOSE 9999

ENTRYPOINT ["/rinhapuffer"]
