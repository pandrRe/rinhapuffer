#!/usr/bin/env python3
"""
Diagnose where our FP/FN come from.

For every test entry:
  1. POST to the running server (whatever ranking it does today).
  2. Compute brute-force raw-Euclidean (no normalization) over the original
     references.json features and labels.

For each entry we got wrong, check whether raw-Euclidean would have been right.
If yes → the dataset L2-normalization in transform_reference is the culprit and
the refactor is justified. If no → it's noise/edge-cases and refactoring won't
help.

Run with the rinhapuffer server live on :9999. Takes ~1–2 min total
(load + 54k HTTPs + ~300 numpy brute-force scans).
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_PATH = ROOT / "rinha-de-backend-2026" / "test" / "test-data.json"
REFERENCES_PATH = ROOT / "resources" / "references.json"
SERVER_URL = "http://localhost:9999/fraud-score"
TOP_K = 5
THRESHOLD = 0.6
VERBOSE_LIMIT = 10


def load_references_raw(path: Path):
    """Stream references.json into (features [N x 14] f32, labels [N] bool)."""
    print(f"loading references.json ({path.stat().st_size / 1e6:.1f} MB)...")
    t0 = time.perf_counter()
    with path.open("rb") as f:
        data = json.load(f)
    n = len(data)
    features = np.empty((n, 14), dtype=np.float32)
    labels = np.empty(n, dtype=bool)
    for i, row in enumerate(data):
        features[i] = row["vector"]
        labels[i] = row["label"] == "fraud"
    t1 = time.perf_counter()
    print(f"  parsed n={n} in {t1 - t0:.1f}s")
    return features, labels


def raw_euclidean_topk_labels(features: np.ndarray, labels: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the labels of the top-K closest references (raw Euclidean)."""
    diff = features - q                       # broadcast
    d2 = np.einsum("ij,ij->i", diff, diff)    # squared distance per row
    top_idx = np.argpartition(d2, TOP_K)[:TOP_K]
    return labels[top_idx]


def predicted_approved_from_count(fraud_count: int) -> bool:
    return (fraud_count / TOP_K) < THRESHOLD


def vectorize_via_server(session: requests.Session, payload: dict) -> tuple[bool, float]:
    """POST to the running server. Returns (approved, fraud_score)."""
    r = session.post(SERVER_URL, json=payload, timeout=5.0)
    r.raise_for_status()
    body = r.json()
    return body["approved"], body["fraud_score"]


def main() -> int:
    # Sanity-check the server is up before doing anything heavy.
    try:
        ready = requests.get("http://localhost:9999/ready", timeout=2.0)
        if ready.status_code != 200:
            print(f"server /ready returned {ready.status_code}", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"server not reachable on :9999 — start it first: {e}", file=sys.stderr)
        return 1

    features, labels = load_references_raw(REFERENCES_PATH)

    print(f"loading test-data.json...")
    with TEST_DATA_PATH.open("rb") as f:
        td = json.load(f)
    entries = td["entries"]
    print(f"  {len(entries)} test entries")

    session = requests.Session()
    # Force per-request close so the diagnostic mirrors how k6 hammers the server
    # in default keep-alive-off mode.
    session.headers.update({"Connection": "close"})

    ours_correct = 0
    ours_wrong = 0
    raw_correct_when_wrong = 0
    raw_wrong_when_wrong = 0
    raw_same_verdict_when_wrong = 0
    raw_correct_total = 0
    verbose_printed = 0

    t0 = time.perf_counter()
    for i, entry in enumerate(entries):
        req = entry["request"]
        expected = entry["expected_approved"]
        try:
            ours_app, ours_fs = vectorize_via_server(session, req)
        except Exception as e:
            print(f"entry #{i}: server error {e}", file=sys.stderr)
            continue

        we_correct = (ours_app == expected)
        if we_correct:
            ours_correct += 1
        else:
            ours_wrong += 1

        # Always compute raw Euclidean so we know overall raw accuracy too.
        # (Vectorize again via the server to get the canonical 14-feature query.
        # But we don't actually have that — we'd need our own vectorizer. So we
        # use the request directly and run our own minimal vectorizer below.)
        # Simpler path: hit the server's response which already encodes a
        # fraud_score, but that's still just OUR ranking. To get raw Euclidean
        # we need the query vector. The server doesn't expose it.
        #
        # Workaround: replicate the vectorizer in Python. Too much code for a
        # diagnostic. Cheaper: trust the server's ranking is wrong-vs-spec only
        # at the boundary, and compute raw Euclidean using a separate Python
        # vectorizer.
        #
        # Actually — see below: we DO need a Python vectorizer. Punt for now
        # and only run the raw-Euclidean comparison when our ranking disagrees
        # with the spec. Skipping the "all entries" raw-correct number.
        if not we_correct:
            q = vectorize_request_py(req)
            top_lbls = raw_euclidean_topk_labels(features, labels, q)
            raw_fc = int(top_lbls.sum())
            raw_app = predicted_approved_from_count(raw_fc)
            raw_correct = (raw_app == expected)
            if raw_correct:
                raw_correct_when_wrong += 1
            else:
                raw_wrong_when_wrong += 1
            if raw_app == ours_app:
                raw_same_verdict_when_wrong += 1
            if verbose_printed < VERBOSE_LIMIT:
                print(
                    f"#{i}: expected_approved={expected} (fs={entry['expected_fraud_score']})  "
                    f"ours={ours_app} (fs={ours_fs})  raw={raw_app} (fc={raw_fc}/5)  "
                    f"{'✓ raw-fixes' if raw_correct else '✗ raw-also-wrong'}"
                )
                verbose_printed += 1

        if (i + 1) % 5000 == 0:
            print(f"  ... {i + 1}/{len(entries)}  ours={ours_correct}/{ours_correct + ours_wrong}")
    t1 = time.perf_counter()

    total = ours_correct + ours_wrong
    print()
    print(f"=== summary (took {t1 - t0:.1f}s) ===")
    print(f"total entries:                              {total}")
    print(f"ours correct / wrong:                       {ours_correct} / {ours_wrong}  (acc {ours_correct/total:.4f})")
    print(f"on entries we got wrong ({ours_wrong} total):")
    print(f"  raw-Euclidean was correct:                {raw_correct_when_wrong}  ← refactor would help by this much")
    print(f"  raw-Euclidean also wrong:                 {raw_wrong_when_wrong}  ← inherent noise / edge-case")
    print(f"  raw-Euclidean returned same verdict:      {raw_same_verdict_when_wrong}")
    return 0


# ─── inline replica of payload.zig::vectorize ──────────────────────────────────

# Mirrors src/payload.zig + resources/normalization.json + resources/mcc_risk.json.
# Kept inline so the diagnostic is a single file with no extra deps.

NORM = {
    "max_amount": 10000.0,
    "max_installments": 12.0,
    "amount_vs_avg_ratio": 10.0,
    "max_minutes": 1440.0,
    "max_km": 1000.0,
    "max_tx_count_24h": 20.0,
    "max_merchant_avg_amount": 10000.0,
}

MCC_DEFAULT = 0.5
MCC_RISK = {
    "5411": 0.15, "5812": 0.30, "5912": 0.20, "5944": 0.45,
    "7801": 0.80, "7802": 0.75, "7995": 0.85, "4511": 0.35,
    "5311": 0.25, "5999": 0.50,
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _parse_iso(ts: str) -> tuple[int, int, int, int, int, int]:
    # YYYY-MM-DDTHH:MM:SSZ
    return (
        int(ts[0:4]), int(ts[5:7]), int(ts[8:10]),
        int(ts[11:13]), int(ts[14:16]), int(ts[17:19]),
    )


def _days_from_civil(y: int, m: int, d: int) -> int:
    if m <= 2:
        y -= 1
    era = y // 400
    yoe = y - era * 400
    m_adj = m - 3 if m > 2 else m + 9
    doy = (153 * m_adj + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _minutes_between(last: str, current: str) -> int:
    ly, lm, ld, lh, lmi, ls = _parse_iso(last)
    cy, cm, cd, ch, cmi, cs = _parse_iso(current)
    last_t = _days_from_civil(ly, lm, ld) * 86400 + lh * 3600 + lmi * 60 + ls
    cur_t = _days_from_civil(cy, cm, cd) * 86400 + ch * 3600 + cmi * 60 + cs
    return abs(cur_t - last_t) // 60


def _day_of_week_mon0(y: int, m: int, d: int) -> int:
    t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
    yy = y - 1 if m < 3 else y
    sun0 = (yy + yy // 4 - yy // 100 + yy // 400 + t[m - 1] + d) % 7
    return (sun0 + 6) % 7


def vectorize_request_py(req: dict) -> np.ndarray:
    """Mirror of payload.zig::vectorize. Returns a (14,) f32 array in [0,1] ∪ {-1}."""
    out = np.empty(14, dtype=np.float32)
    tx = req["transaction"]
    cust = req["customer"]
    merch = req["merchant"]
    term = req["terminal"]
    last = req.get("last_transaction")

    amount = float(tx["amount"])
    out[0] = _clamp01(amount / NORM["max_amount"])
    out[1] = _clamp01(float(tx["installments"]) / NORM["max_installments"])

    req_at = tx["requested_at"]
    y, m, d, hh, _mm, _ss = _parse_iso(req_at)
    out[3] = hh / 23.0
    out[4] = _day_of_week_mon0(y, m, d) / 6.0

    avg_amount = float(cust["avg_amount"])
    ratio = 0.0 if avg_amount == 0 else amount / avg_amount
    out[2] = _clamp01(ratio / NORM["amount_vs_avg_ratio"])

    out[8] = _clamp01(float(cust["tx_count_24h"]) / NORM["max_tx_count_24h"])

    merchant_id = merch["id"]
    known = cust["known_merchants"]
    out[11] = 0.0 if merchant_id in known else 1.0

    out[12] = MCC_RISK.get(merch["mcc"], MCC_DEFAULT)
    out[13] = _clamp01(float(merch["avg_amount"]) / NORM["max_merchant_avg_amount"])

    out[9] = 1.0 if term["is_online"] else 0.0
    out[10] = 1.0 if term["card_present"] else 0.0
    out[7] = _clamp01(float(term["km_from_home"]) / NORM["max_km"])

    if last is None:
        out[5] = -1.0
        out[6] = -1.0
    else:
        out[5] = _clamp01(_minutes_between(last["timestamp"], req_at) / NORM["max_minutes"])
        out[6] = _clamp01(float(last["km_from_current"]) / NORM["max_km"])

    return out


if __name__ == "__main__":
    sys.exit(main())
