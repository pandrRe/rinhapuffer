# syntax=docker/dockerfile:1.7
#
# rinhalb — round-robin TCP→UDS L4 splicer.
#
# Built on the host by ./build.sh (cross-compile via zig cc to static
# linux-musl). This file just packages the binary into a scratch image.

FROM scratch

COPY zig-out/bin/rinhalb /rinhalb

EXPOSE 9999

ENTRYPOINT ["/rinhalb"]
