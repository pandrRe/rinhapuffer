# Implementation plan

End-to-end roadmap from current state (payload vectorizer + reference loader done) to a running fraud-score service. Constraints: ≤ 150 MB RSS, ~0.25 CPU, single-process Zig binary, blocking sockets.

Each phase is a landable unit: code + tests + (where it makes sense) a bench. One thing at a time.

---

## Phase 1 — Cosine top-K over the in-memory dataset (no quantization, no IVF)

Goal: prove the search math against the existing `Dataset` view (f32 SoA, already in memory). Brute-force, correctness-first.

- [ ] **1.1** Add `norms: []f32` to `Dataset`, populated by `parse_into` (one extra pass, `sqrt(Σ f²)` per row).
  - Caller-owned buffer like `features` and `labels`. Update `Error` and the `BufferTooSmall` checks.
- [ ] **1.2** Add `src/search.zig` with `cosine_topk(ds: Dataset, q: *const [14]f32, out: *[5]u32) void`.
  - Single pass, W=8 SIMD lanes, FMA per feature (`inline for (0..14)`), 5-element insertion-sift heap on `(score, row_idx)`.
  - Score = `q·r / (|q| · |r|)`; precompute `1/|q|`, multiply by `1/norms[row]` per row.
- [ ] **1.3** Test against a hand-built tiny dataset (5 rows, 14 features, known answer).
- [ ] **1.4** Differential test: brute-force `cosine_topk` vs a naive O(n log n) sort on `example-references.json`.
- [ ] **1.5** Bench `cosine_topk` over full `references.json` (latency distribution like the payload bench). Record numbers.
- [ ] **1.6** Commit.

**Exit criterion**: top-5 indices match the naive reference; bench prints µs/query.

---

## Phase 2 — L2-normalize at parse time, switch search to plain dot product

Goal: simplify and speed up the inner loop. Same neighbors, less arithmetic.

- [ ] **2.1** In `parse_into`, after writing each row, divide its 14 features by `|r|`. Drop `norms` (or repurpose to a debug check that all rows have unit norm).
- [ ] **2.2** Update `cosine_topk` to a plain dot product (no per-row divide). Re-test against the differential check (must still match the naive reference within 1e-5).
- [ ] **2.3** Re-bench. Should drop a few % and remove the norms stream.
- [ ] **2.4** Commit.

**Exit criterion**: same top-5 as Phase 1, faster, simpler code.

---

## Phase 3 — Build-time artifact: serialize the prepped dataset to disk

Goal: move parsing + normalization out of boot. Produce a single `dataset.bin` blob the runtime just `mmap`s.

- [ ] **3.1** Define on-disk layout in `src/dataset_blob.zig`:
  ```
  Header { magic, version, n: u32 }
  features: [14 columns of f32, column-major, length n each] (L2-normalized)
  labels: bitset of length n
  ```
  All naturally aligned, fixed offsets derivable from `n`.
- [ ] **3.2** Add `src/build_dataset.zig` — a standalone executable wired in `build.zig` as a `prep` step.
  - Reads `resources/references.json` via the existing fast loader.
  - L2-normalizes (already done in Phase 2 if `parse_into` does it; otherwise do it here).
  - Writes `dataset.bin` to `resources/`.
- [ ] **3.3** Add `dataset_blob.load(path) Blob` — `mmap` only, no parsing, returns a `Dataset`-shaped view over the f32 columns + labels bitset.
- [ ] **3.4** Test: `prep` then `load` round-trips to the same top-5 as Phase 2 on `example-references.json`.
- [ ] **3.5** Bench: boot time = time from `main` entry to "ready" flag flipped. Target: < 50 ms cold.
- [ ] **3.6** Commit.

**Exit criterion**: `zig build prep` produces the file; runtime no longer parses JSON for the dataset.

---

## Phase 4 — u16 quantization to fit in the RAM budget

Goal: drop dataset RSS from 168 MB f32 → 84 MB u16. Required to fit 150 MB cap.

- [ ] **4.1** During `prep`: per-feature `(min, scale)` such that `q = round((f − min) · 65535 / range)`. Store the 14 `(min, scale)` pairs in the header.
- [ ] **4.2** Switch the on-disk feature columns to `u16`. Header gains a `quant_params: [14]struct{min: f32, scale: f32}` field.
- [ ] **4.3** In the search inner loop, dequantize on the fly: `f = u16_val * scale + min` per feature. `inline for (0..14)` makes the constants per-column hoist. Use FMA where possible.
- [ ] **4.4** Differential test: top-5 from the u16 path matches the f32 path on `example-references.json` (allow up to 1 swap among ties; tighten if all match exactly).
- [ ] **4.5** Re-bench. Bandwidth halves; expect ~1.5–2× scan speedup.
- [ ] **4.6** Verify RSS on the full dataset: should be ~84 MB resident features + 0.4 MB labels.
- [ ] **4.7** Commit.

**Exit criterion**: `dataset.bin` ≤ 90 MB on disk; runtime peak RSS ≤ 100 MB.

---

## Phase 5 — IVF index for sub-millisecond search

Goal: cut the per-query scan from 3M rows to ~24k–48k by clustering and probing.

- [ ] **5.1** In `prep`, run k-means (K=1024, ~20 iters) on a 100k random sample. Spherical k-means since data is unit-normalized (renormalize centroids each iter).
  - Hand-rolled, no deps. Deterministic seed.
- [ ] **5.2** Assign every row to its nearest centroid (one pass over the dataset, single dot product per (row, centroid) pair using the same SIMD pattern).
- [ ] **5.3** Reorder rows by cluster ID. Within each cluster, keep SoA column-major. Build `cluster_starts: [1025]u32`.
- [ ] **5.4** Reorder `labels` to match.
- [ ] **5.5** Update on-disk layout: header gains `k_clusters`, `centroids: [K][14]f32`, `cluster_starts: [K+1]u32`. Features stay column-major within each cluster.
- [ ] **5.6** Update `search.zig`:
  - Score query against all 1024 centroids (small enough to fully scan).
  - Pick top-N probe clusters (N=8, configurable).
  - Scan only those clusters' rows; same heap.
- [ ] **5.7** Recall test: top-5 IVF vs top-5 brute on `example-references.json`. Expect ≥ 4/5 match average; tune N if not.
- [ ] **5.8** Bench. Target: ≤ 500 µs p99 per `cosine_topk` at 0.25 CPU.
- [ ] **5.9** Commit.

**Exit criterion**: search latency drops by ~50× vs Phase 4; recall@5 ≥ 0.8 on a held-out sample.

---

## Phase 6 — HTTP server skeleton (blocking, handrolled)

Goal: accept HTTP/1.1 on a TCP socket, route `/ready` and `/fraud-score`, no logic yet.

- [x] **6.1** Add `src/http.zig`. Single accept loop, blocking `read`, fixed 4 KB request buffer per connection.
- [x] **6.2** Parse request line + headers in place: method, path, `Content-Length`. Reject anything else with 400.
- [x] **6.3** Route table: `GET /ready` → 200, `POST /fraud-score` → handler stub returning `{"approved":true,"fraud_score":0.0}`.
- [x] **6.4** Keep-alive on by default. Response template via `writev` (status + headers + body).
- [x] **6.5** Integration test: spawn the server in a test, hit both endpoints with `std.http.Client`, assert responses.
- [x] **6.6** Commit.

**Exit criterion**: `curl localhost:9999/ready` → 200; `curl -X POST … /fraud-score` → stub JSON.

---

## Phase 7 — Wire `payload.vectorize` + `cosine_topk` into `/fraud-score`

Goal: end-to-end. Real fraud scoring through the HTTP path.

- [x] **7.1** On boot: `dataset_blob.load("resources/dataset.bin")`, flip the `ready` flag.
- [x] **7.2** `/fraud-score` handler:
  1. Parse body length, validate it fits in the buffer.
  2. `payload.vectorize(body, &q)` — already zero-alloc.
  3. L2-normalize `q` in place.
  4. `cosine_topk(ds, &q, &top5_rows)`.
  5. `fraud_count = popcount(labels[top5])`.
  6. `score = fraud_count / 5.0`; `approved = score < 0.6`.
  7. Write a precomputed JSON template (six possible scores: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0).
- [x] **7.3** Integration test against the example payloads + spec golden.
- [x] **7.4** End-to-end bench: payloads/sec at 0.25 CPU under `taskset -c 0` + `cgroups` CPU cap.
- [x] **7.5** Commit.

**Exit criterion**: server returns spec-compliant `fraud-score` on the example payloads; sustained throughput measured.

---

## Phase 7.5 — Switch search from L2-normalized cosine to raw-feature Euclidean

Goal: eliminate the 298/299 FP+FN caused by dataset L2-normalization (diagnosed by `scripts/diagnose.py`). Switch the entire search pipeline to plain Euclidean over raw [0,1] ∪ {−1} features. RAM unchanged (u16 quantization stays).

- [x] **7.5.1** Bulk rename `cosine_topk*` → `euclidean_topk*` (no behavior change).
- [x] **7.5.2** Drop L2-normalize in `transform_reference.parse_into`. Switch `kmeans` to plain Euclidean (no centroid renormalize, `assign_all` argmin-Euclidean). Bump `dataset_blob.VERSION` 3 → 4. Rewrite `search` inner loops as `score = q·r − ½‖r‖²` (per-query precompute: `score_const`, `lin_eff`, `neg_half_quad`). Centroid scoring switches to `dot(q, c) − ½‖c‖²`. `naive_cosine_topk` → `naive_euclidean_topk` (f64 brute-force).
- [x] **7.5.3** `zig build prep` regenerates `resources/dataset.bin` (v4, same 87 MB).
- [x] **7.5.4** Smoke + diagnose + k6.

**Result**: errors 299 → 8 (99.45% → 99.99% accuracy). k6 final score **4,122 → 5,647.17** (+37%). p99 = 0.91 ms (slightly better than before despite 2× inner-loop ops). 0 HTTP errors. Both `p99_score` and `detection_score.rate_component` hit their formula maxima.

---

## Phase 7.6 — Int-only Euclidean hot path (i16 + global FIX_SCALE, v5 blob)

Goal: drop dequantization from the row scan loop entirely, following the pattern from [thiagorigonatti/rinha-2026](https://github.com/thiagorigonatti/rinha-2026). One global `FIX_SCALE` constant replaces the per-column `(min, scale)` table; ranking happens in integer units (`Σ (q_i - r_i)²`); the per-query precompute (`QueryPlan`) collapses to one 14-element quantize.

- [x] **7.6.1** `search.FIX_SCALE = 10000`. Storage `u16 → i16`. Drop `mins`/`inv_scales` from `QuantizedDataset`/`IvfQuantizedDataset`. Drop `QuantParam`/`compute_quant_params` from `dataset_blob`.
- [x] **7.6.2** Bump `dataset_blob.VERSION` 4 → 5. Header shrinks 128 → 32 bytes (only `magic, version, n, k_clusters, fix_scale, [3]u32 pad`). Add `error.UnsupportedFixScale`. Add `load rejects v4` and `load rejects mismatched fix_scale` tests.
- [x] **7.6.3** Rewrite `search.zig` with int hot path: i16 load → i16 sub → i32 widen → i32 mul → i64 widen + accumulate. Centroid PROBE selection stays in float.
- [x] **7.6.4** Re-prep, smoke, diagnose, k6.

**Result**: errors **8 → 7** (FP=5, FN=2). k6 final score **5,647.17 → 5,676.25**. **p99 0.91 ms → 0.62 ms (−32%)** — int hot path is faster on Apple Silicon despite extra widens, because no fp/int register-bridge and 3 ops/feature vs 4. 0 HTTP errors. Net `-131` lines of code.

---

## Phase 7.7 — Bbox repair pass for exact top-K (v6 blob)

Goal: 100% accuracy via correctness-by-construction. Per-cluster axis-aligned `[lo, hi]` i16 bboxes give an exact lower-bound distance from query to any point in the cluster; skip clusters that can't beat the current K-th best.

- [x] **7.7.1** Add `bbox_lo`, `bbox_hi: []const i16` to `IvfQuantizedDataset`. Bump VERSION 5 → 6. Insert two `[K][14]i16` sections in the on-disk layout. `+56 KB`, no header change.
- [x] **7.7.2** `dataset_blob.write` pre-pass to compute per-(cluster, feature) min/max before streaming features.
- [x] **7.7.3** `search.euclidean_topk_q_ivf` adds a fourth step: bbox repair pass over unprobed clusters. PROBE stays at 8.
- [x] **7.7.4** Re-prep, smoke, diagnose, k6.

**Result**: **errors 7 → 0** (100% accuracy, FP=0, FN=0). `detection_score` pegged at formula max 3000. But **p99 0.62 ms → 4.72 ms** under k6's 250-VU concurrency — bbox prune costs more under contention than expected. **Final score 5,676.25 → 5,326.02** (-350). Architecture correct (exact by construction) but needs latency optimization to recover the score.

---

## Phase 7.7.1 — SIMD `bbox_lower_bound_sq`

Goal: recover the p99 hit from Phase 7.7. The new bbox LB compute was the only scalar function on the hot path; `scan_range_int` was already SIMD'd at W=8.

- [x] **7.7.1.1** Vectorize `bbox_lower_bound_sq` across all 14 features using `@Vector(N_FEATURES, ·)`. LLVM pads to 16 lanes on aarch64; per cluster goes from ~84 scalar i32 ops to ~8 SIMD ops.

**Result**: **k6 final score 5,326.02 → 6,000.00 (max possible)**. FP=0, FN=0, http_errors=0. **p99 4.72 ms → 0.78 ms** (-83%). Both `p99_score` and `detection_score` pegged at the formula maximum 3000 each. **Six perfect zeros** (errors, http_errors, failure_rate, weighted_errors, error_rate_epsilon, absolute_penalty).

---

## Phase 8 — Containerize against the rinha spec

Goal: hand a docker image that the rinha harness can run.

- [ ] **8.1** Multi-stage `Dockerfile`: stage 1 builds Zig + runs `prep` to produce `dataset.bin`; stage 2 is a `scratch`/`alpine` runtime image with the binary + the blob.
- [ ] **8.2** `docker-compose.yaml` with the 0.25 CPU + 150 MB cap matching the spec.
- [ ] **8.3** Smoke-test under the cap: `docker compose up`, run the example payloads, verify `/ready` flips < 5 s.
- [ ] **8.4** Confirm RSS stays under 150 MB during a sustained load run.
- [ ] **8.5** Commit.

**Exit criterion**: image runs the rinha harness end-to-end without OOM or timeout.

---

## Phase 9 — Tuning pass (only if numbers warrant it)

Pick from this list based on profiling, not speculation:

- [ ] Prefetch (`@prefetch`) one cluster ahead during IVF probe.
- [ ] Switch `u16` quant to `u8` if recall holds.
- [ ] Move IVF probe count `N` from compile-time to a tuned constant per recall target.
- [ ] HNSW small-world overlay on top of IVF if p99 still misses target (memory budget permitting).
- [ ] `SO_REUSEPORT` + multiple workers if the spec ever lifts the CPU cap.
- [ ] `io_uring` if the syscall path ever shows up in a profile (it won't at 0.25 CPU).

Only land tuning changes that show up in a bench delta + a recall check. No speculative micro-opts.

---

## Out-of-scope (don't do unless asked)

- TLS / HTTPS
- Multi-tenant routing, auth, rate limiting
- Persistence of incoming requests
- Online learning / dataset updates
- Any feature engineering beyond what `payload.vectorize` already does
- Cross-platform fallbacks (Linux x86_64 is the target; macOS arm64 is the dev box — both already handled by `@Vector` and posix mmap)
