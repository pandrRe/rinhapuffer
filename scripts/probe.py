#!/usr/bin/env python3
"""
Probe the instrumented rinhapuffer stack: poll /__metrics-1 and /__metrics-2
through the LB, diff successive snapshots, and print per-window deltas +
windowed quantiles recomputed from histogram bucket diffs.

Cumulative percentiles (over the whole process lifetime) are also shown for
reference, but the window numbers are what actually move with each iteration
of a load test.

Usage:
  scripts/probe.py                       # default: 5 samples, 5s apart
  scripts/probe.py -i 2 -n 30            # 30 samples, 2s apart
  scripts/probe.py --once                # single snapshot, pretty-printed
  scripts/probe.py --url http://host:9999

Run a load generator (k6, hey, oha, etc.) in another terminal — this script
only observes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field

DEFAULT_URL = "http://localhost:9999"

HIST_NAMES = (
    "hist_total_ns",
    "hist_parse_ns",
    "hist_vectorize_ns",
    "hist_search_ns",
    "hist_write_ns",
)
BUCKET_NAMES = tuple(n.replace("hist_", "buckets_") for n in HIST_NAMES)


@dataclass
class Snapshot:
    counters: dict[str, int] = field(default_factory=dict)
    hist_summary: dict[str, dict[str, int]] = field(default_factory=dict)
    buckets: dict[str, dict[int, int]] = field(default_factory=dict)
    raw: str = ""

    def counter(self, k: str) -> int:
        return self.counters.get(k, 0)


def fetch(url: str, timeout: float = 5.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


_BUCKET_LINE = re.compile(r"\bb(\d+)=(\d+)")


def parse(text: str) -> Snapshot:
    s = Snapshot(raw=text)
    for line in text.splitlines():
        if not line:
            continue
        parts = line.split()
        name = parts[0]
        if name.startswith("hist_") and "=" in line:
            kv = {}
            for tok in parts[1:]:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        kv[k] = int(v)
                    except ValueError:
                        pass
            s.hist_summary[name] = kv
        elif name.startswith("buckets_"):
            d: dict[int, int] = {}
            for m in _BUCKET_LINE.finditer(line):
                d[int(m.group(1))] = int(m.group(2))
            s.buckets[name] = d
        elif len(parts) == 2:
            try:
                s.counters[name] = int(parts[1])
            except ValueError:
                pass
    return s


def diff_buckets(prev: dict[int, int], cur: dict[int, int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for b, v in cur.items():
        d = v - prev.get(b, 0)
        if d > 0:
            out[b] = d
    return out


def quantile_from_buckets(buckets: dict[int, int], p: float) -> int:
    """Walk the log2 buckets and return the upper bound of the bucket
    containing the p-quantile. Buckets are u64-clz-indexed: bucket b covers
    [2^(b-1), 2^b)."""
    total = sum(buckets.values())
    if total == 0:
        return 0
    target = max(1, int(p * total))
    cum = 0
    for b in sorted(buckets):
        cum += buckets[b]
        if cum >= target:
            return 0 if b == 0 else 1 << b
    return 0


def fmt_ns(ns: float) -> str:
    if ns <= 0:
        return "—"
    if ns < 1_000:
        return f"{ns:.0f}ns"
    if ns < 1_000_000:
        return f"{ns/1_000:.1f}µs"
    if ns < 1_000_000_000:
        return f"{ns/1_000_000:.2f}ms"
    return f"{ns/1_000_000_000:.2f}s"


def fmt_int(n: int) -> str:
    return f"{n:>10,d}"


def render_window(label: str, prev: Snapshot, cur: Snapshot, dt_s: float) -> str:
    out: list[str] = []
    out.append(f"── {label}  (window: {dt_s:.2f}s) ─────────────────────────")

    def cdiff(k: str) -> int:
        return cur.counter(k) - prev.counter(k)

    req = cdiff("req_total")
    rps = req / dt_s if dt_s > 0 else 0
    out.append(
        f"requests          {fmt_int(req)}    rps {rps:>9,.1f}    "
        f"parse_err {cdiff('req_parse_err')}    vec_err {cdiff('req_vectorize_err')}"
    )

    wakeups = cdiff("epoll_wakeups")
    events = cdiff("epoll_events")
    epr = events / wakeups if wakeups > 0 else 0
    out.append(
        f"epoll             wakeups {fmt_int(wakeups)}  events/wakeup {epr:5.2f}    "
        f"accepts {cdiff('accepts')}  closes {cdiff('conn_closes')}"
    )
    out.append(
        f"io                read_eagain {cdiff('read_eagain')}    "
        f"write_eagain {cdiff('write_eagain')}    partial_writes {cdiff('partial_writes')}"
    )

    fast = cdiff("head_fast")
    slow = cdiff("head_slow")
    fast_pct = 100.0 * fast / max(1, fast + slow)
    out.append(f"http head         fast {fmt_int(fast)}  slow {slow}  ({fast_pct:.1f}% fast)")

    probed = cdiff("search_clusters_probed")
    skipped = cdiff("search_clusters_bbox_skipped")
    bbox_scan = cdiff("search_clusters_bbox_scanned")
    blocks = cdiff("search_blocks_scanned")
    early = cdiff("search_blocks_early_pruned")
    sifts = cdiff("search_sift_ins")
    per_q = (lambda x: x / req) if req > 0 else (lambda _: 0.0)
    out.append(
        f"search/req        probed {per_q(probed):4.2f}   bbox-skip {per_q(skipped):5.2f}   "
        f"bbox-scan {per_q(bbox_scan):4.2f}   blocks {per_q(blocks):6.1f}   "
        f"early-prune {per_q(early):5.1f} ({100.0*early/max(1,blocks):.1f}%)   "
        f"sifts {per_q(sifts):4.2f}"
    )

    out.append("")
    out.append(
        f"  {'stage':<14}  {'count':>8}  {'p50':>10}  {'p90':>10}  {'p99':>10}  {'p999':>10}"
    )
    for hn, bn in zip(HIST_NAMES, BUCKET_NAMES):
        pb = prev.buckets.get(bn, {})
        cb = cur.buckets.get(bn, {})
        win = diff_buckets(pb, cb)
        n = sum(win.values())
        if n == 0:
            out.append(f"  {hn:<14}  {fmt_int(0)}  {'—':>10}  {'—':>10}  {'—':>10}  {'—':>10}")
            continue
        p50 = quantile_from_buckets(win, 0.50)
        p90 = quantile_from_buckets(win, 0.90)
        p99 = quantile_from_buckets(win, 0.99)
        p999 = quantile_from_buckets(win, 0.999)
        out.append(
            f"  {hn:<14}  {fmt_int(n)}  {fmt_ns(p50):>10}  {fmt_ns(p90):>10}  "
            f"{fmt_ns(p99):>10}  {fmt_ns(p999):>10}"
        )

    return "\n".join(out)


def render_once(label: str, s: Snapshot) -> str:
    out: list[str] = []
    out.append(f"── {label}  (cumulative since process start) ─────────────")
    uptime_ns = s.counter("uptime_ns")
    out.append(
        f"uptime {fmt_ns(uptime_ns)}    "
        f"req_total {fmt_int(s.counter('req_total'))}    "
        f"parse_err {s.counter('req_parse_err')}    "
        f"vec_err {s.counter('req_vectorize_err')}"
    )
    fast = s.counter("head_fast")
    slow = s.counter("head_slow")
    out.append(
        f"head fast {fmt_int(fast)}  slow {slow}  "
        f"({100.0*fast/max(1,fast+slow):.1f}% fast)"
    )
    out.append(
        f"epoll wakeups {fmt_int(s.counter('epoll_wakeups'))}  "
        f"events {fmt_int(s.counter('epoll_events'))}  "
        f"events/wakeup {s.counter('epoll_events')/max(1,s.counter('epoll_wakeups')):.2f}"
    )
    req = max(1, s.counter("req_total"))
    out.append(
        f"search/req  probed {s.counter('search_clusters_probed')/req:4.2f}  "
        f"bbox-skip {s.counter('search_clusters_bbox_skipped')/req:5.2f}  "
        f"bbox-scan {s.counter('search_clusters_bbox_scanned')/req:4.2f}  "
        f"blocks {s.counter('search_blocks_scanned')/req:6.1f}  "
        f"early-prune {100.0*s.counter('search_blocks_early_pruned')/max(1,s.counter('search_blocks_scanned')):.1f}%"
    )
    out.append("")
    out.append(
        f"  {'stage':<14}  {'count':>8}  {'mean':>10}  {'p50':>10}  {'p99':>10}  {'p999':>10}  {'max':>10}"
    )
    for hn in HIST_NAMES:
        h = s.hist_summary.get(hn, {})
        n = h.get("count", 0)
        out.append(
            f"  {hn:<14}  {fmt_int(n)}  "
            f"{fmt_ns(h.get('mean', 0)):>10}  "
            f"{fmt_ns(h.get('p50', 0)):>10}  "
            f"{fmt_ns(h.get('p99', 0)):>10}  "
            f"{fmt_ns(h.get('p999', 0)):>10}  "
            f"{fmt_ns(h.get('max', 0)):>10}"
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL, help="LB base URL (default: %(default)s)")
    ap.add_argument("--once", action="store_true", help="single cumulative snapshot, no diffing")
    ap.add_argument("-i", "--interval", type=float, default=5.0, help="seconds between samples")
    ap.add_argument("-n", "--samples", type=int, default=5, help="number of windowed samples")
    ap.add_argument("--raw", action="store_true", help="dump raw /__metrics text and exit")
    ap.add_argument("--instance", choices=["1", "2", "both"], default="both")
    ap.add_argument("--save", help="write JSON timeline to this path")
    args = ap.parse_args()

    targets: list[tuple[str, str]] = []
    if args.instance in ("1", "both"):
        targets.append(("api1", f"{args.url}/__metrics-1"))
    if args.instance in ("2", "both"):
        targets.append(("api2", f"{args.url}/__metrics-2"))

    if args.raw:
        for label, url in targets:
            try:
                txt = fetch(url)
            except Exception as e:
                print(f"# {label} {url} FAILED: {e}", file=sys.stderr)
                continue
            print(f"# === {label} {url} ===")
            print(txt)
        return 0

    timeline: list[dict] = []

    if args.once:
        for label, url in targets:
            try:
                cur = parse(fetch(url))
            except Exception as e:
                print(f"# {label} {url} FAILED: {e}", file=sys.stderr)
                continue
            print(render_once(label, cur))
            print()
            if args.save is not None:
                timeline.append({"label": label, "ts": time.time(), "counters": cur.counters, "hist_summary": cur.hist_summary})
        if args.save:
            with open(args.save, "w") as f:
                json.dump(timeline, f, indent=2)
        return 0

    prev: dict[str, Snapshot] = {}
    for label, url in targets:
        prev[label] = parse(fetch(url))
    t_prev = time.monotonic()

    for sample in range(1, args.samples + 1):
        time.sleep(args.interval)
        t_now = time.monotonic()
        dt = t_now - t_prev
        print(f"\n========== sample {sample}/{args.samples}  ({time.strftime('%H:%M:%S')}) ==========")
        for label, url in targets:
            try:
                cur = parse(fetch(url))
            except Exception as e:
                print(f"# {label} fetch failed: {e}", file=sys.stderr)
                continue
            print(render_window(label, prev[label], cur, dt))
            print()
            if args.save is not None:
                timeline.append({
                    "label": label,
                    "ts": time.time(),
                    "sample": sample,
                    "dt_s": dt,
                    "counters": cur.counters,
                    "buckets": {k: dict(v) for k, v in cur.buckets.items()},
                })
            prev[label] = cur
        t_prev = t_now

    if args.save:
        with open(args.save, "w") as f:
            json.dump(timeline, f, indent=2)
        print(f"# wrote timeline to {args.save}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
