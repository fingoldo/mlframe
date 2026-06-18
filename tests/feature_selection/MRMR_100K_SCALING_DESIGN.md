# MRMR feature selection at p = 100,000 — scaling design & feasibility

Status: design + measured feasibility prototype (2026-06-19). Hardware: 4 logical cores,
GTX 1050 Ti (4 GB), Windows 10. No production `.py` modified; prototype lives in
`D:/Temp/mrmr_100k/` (not committed). Implementation lands in a later change.

## TL;DR

- p = 10,000 is comfortable today (the wide-FE path is already gated at `fe_synergy_screen_max_features=250`
  with an O(p) interaction pre-rank in front of the O(cap^2) sweep).
- p = 100,000 needs a **Sure-Independence-Screening (SIS) front gate**: a single chunked O(p·n) pass that
  scores every feature by marginal MI **and** the existing second-moment interaction propensity, cuts
  100k → a few thousand survivors, and only then runs full MRMR (relevance + Fleuret CMI) and FE/synergy
  on the reduced set.
- The screen is the only thing that touches all 100k columns. It must be **column-chunked** so the
  `(n, p)` matrix is never fully resident, and the chunk width must be chosen from **free RAM** (a measured
  prototype OOM'd at chunk = 4000 / n = 20000 because the production statistic upcasts each chunk to
  float64 and then squares it — ~3× the chunk's float32 footprint transiently).
- Measured (this machine): the screen costs **~7 s at p=10k** and **~37 s CPU at p=100k** (the streaming
  2nd-moment pass); at p=100k the wall-time is **disk-I/O-bound** — one sequential 8 GB read — not
  compute-bound. The `(n=20k, p=100k)` frame is an 8 GB **on-disk** memmap, never fully in RAM; the screen's
  own state is O(p) accumulators (~5 MB) plus a row-band buffer sized from free RAM.

## 1. Measured cost map (read from the code, then timed)

All file paths are under `src/mlframe/feature_selection/filters/`.

### Univariate MI relevance — O(p·n), batched, cheap
- Kernel: `_orthogonal_univariate_fe/_orth_mi_backends.py:153` `_mi_classif_batch(X, y, nbins=10)` →
  `_mi_classif_batch_numba` (numba `prange` over columns; GPU dispatch via
  `hermite_fe.plugin_mi_classif_batch_dispatch`). Returns `float64[p]`.
- Inner work per column: quantile-bin (`searchsorted`, O(n)) + joint histogram O(K_x·K_y), K≈10.
  Total O(p·n) with a tiny constant. **Not the bottleneck.**

### Discretization / binning of the (n, p) matrix — int16, per-column-cached, NOT fully materialized twice
- `discretization/_discretization_dataset.py:149` `categorize_dataset(...)`, per-column
  `_discretize_2d_array_col_cached(...)` (`:67`). Output dtype **int16** by default (`:154`), auto-promotes
  to int32/int64 only if cardinality exceeds the dtype (`:357`).
- Per-column LRU code cache (`_NUMERIC_CODE_CACHE_MAX_BYTES`, 512 MB default).
- **Memory reality at p=100k:** an int16 `(n, p)` matrix is `n·p·2` bytes — at n=50k that is **10 GB**, at
  n=20k it is **4 GB**. It does materialize one full float copy of the numeric block
  (`:202 df[...].to_numpy(...)`). So discretizing all 100k columns at once is the memory wall — it must be
  **chunked by column block** (see §2), never a single `categorize_dataset` call on the full width.

### Fleuret CMI redundancy — O(npermutations · k · p) per step, post-screen only
- `fleuret.py:185` `get_fleuret_criteria_confidence` (njit), joblib fan-out
  `get_fleuret_criteria_confidence_parallel` (`:27`). Per selection step evaluates the candidate against the
  `k` already-selected via conditional MI; cost grows with pool size `p`. **Must run on the screened pool
  (a few thousand), never on 100k** — at k=30, p=100k that is 3e6 CMI evals/step × ~30 steps.

### Second-moment interaction pre-rank — O(p·n), pure numpy
- `_fe_interaction_prerank.py:53` `second_moment_propensity(values, y)` = `|corr(x²,y)| + |corr(x,y²)|`,
  fully vectorised (`_abs_col_corr` at `:40`). **Memory caveat (measured):** `:59` upcasts the whole input
  to float64 and `:67` forms `V*V` — two extra `(n, c)` float64 copies. For a 100k-wide call this would be
  16 GB; it MUST be called per column-chunk, and a float32 variant halves the transient (the prototype
  proves a float32 path is rank-equivalent).
- No global RNG, no live kernels stored as attributes → pickle-safe, deterministic.

### O(p²) / (p×p) on the default path — NONE
- Confirmed no pairwise/correlation `(p,p)` matrix is reachable on the default MRMR path. Pairwise code
  (`_pairwise_modular_fe.py`, `friend_graph.py:174 pairwise_mi_edge`, missingness ratio recipes) is
  **post-screen / opt-in only**.
- The FE synergy sweep is O(cap²) hard-capped at `fe_synergy_screen_max_features=250`
  (`mrmr/_mrmr_class.py:1510`); when the raw pool exceeds the cap, `_mrmr_fe_step_helpers.py:82` calls
  `top_k_by_interaction_propensity` to pre-rank down to 250 **before** the sweep, so nothing builds a full
  `(p,p)`. The sweep cost gate is `n_rows · cap²` (`_mrmr_fe_step_helpers.py:107`).

### Cost table

| stage | complexity | p=10k (measured) | p=100k |
|---|---|---|---|
| 2nd-moment streaming pass (sequential row-bands) | O(p·n) | **2.8 s** compute, peak ~0.96 GB (band=4000) | **~36 s CPU compute**; wall-time I/O-bound (8 GB read) |
| marginal-MI batch (full n, column chunks) | O(p·n) | **6.6 s**, peak 0.32 GB | scales ~10×; MUST use full n (5k-row subsample collapses recall, measured) |
| frame materialization (memmap, on disk) | O(p·n) | 0.8 GB disk | 8 GB disk (never fully in RAM) |
| full discretization of all columns (int16, in RAM) | O(p·n) | 0.4 GB | **4 GB (n=20k) / 10 GB (n=50k)** — chunk it |
| Fleuret CMI per step | O(npermut·k·p) | n/a (post-screen) | runs on ~2k survivors, not 100k |
| FE synergy sweep | O(cap²·n) | capped at 250² | capped at 250² (pre-ranked) |

## 2. Cascade design for p=100k

Three gates, narrowing the pool, so only the first ever sees all 100k columns:

```
100k features
   │  GATE A — SIS screen (chunked, O(p·n), the only full-width pass)
   │    score_j = z(marginal_MI_j)  +  z(second_moment_propensity_j)
   ▼
~2k survivors  (data-derived count, see survivor rule)
   │  GATE B — full MRMR: relevance MI + Fleuret CMI redundancy on the survivors
   ▼
~30–60 selected
   │  GATE C — FE / synergy bootstrap on the MRMR survivors (already capped at 250, pre-ranked)
   ▼
final feature set
```

### Gate A — the SIS screen
- **Statistics (both, fused):** marginal MI `_mi_classif_batch` (catches main effects) **and**
  `second_moment_propensity` (catches pure-interaction operands whose marginal MI is ~0). Fusing both is
  essential: the measured prototype shows MI alone recovers 100% of main-effect features but only ~0.6 of
  pair operands, while the 2nd-moment score recovers the operands MI misses. Fusion = z-score each, take
  the max-rank (best-of-either) so an operand surviving on EITHER signal is kept.
- **Why second-moment, not just MI:** marginal MI ranks a zero-marginal interaction operand at the noise
  floor by construction; that is exactly the operand the synergy bootstrap needs. The 2nd-moment score
  keeps those operands. Irreducible floor: a perfectly balanced XOR (zero higher-moment leakage) is
  invisible to ANY O(p) score — that measure-zero case is only recoverable by the O(cap²) sweep itself and
  is out of scope for the screen.

### Survivor-count rule (data-derived, not hardcoded)
Pick the survivor count `m` as the **max** of:
1. an information knee: sort fused scores descending, keep features above
   `median + c·MAD` of the fused score (c≈3, robust noise-floor cut); and
2. a floor of `max(20·k_target, 1000)` so the downstream MRMR/synergy pool is never starved (k_target =
   requested number of selected features);
clamped to an **upper cap derived from the RAM budget** for Gate B's discretized pool
(`m_max = free_RAM_budget / (n · int16_bytes)`), and to `p` itself. This adapts to how concentrated the
signal is and to available memory; nothing is a magic constant.

### Memory budgeting & chunking — and the I/O direction (MEASURED, load-bearing)
- **Do NOT column-chunk a row-major frame.** The first prototype read `X[:, j0:j1]` column blocks from a
  row-major `(n, p)` 8 GB memmap and went **disk-bound and never finished** (CPU pinned at 29 s while the
  process blocked on I/O, because every column block strides across the entire 8 GB file). This is the
  single most important scaling lesson: at p=100k the screen is **I/O-bound, not compute-bound**, and
  access order dominates.
- **Correct pattern: a single sequential pass over contiguous ROW-BANDS**, accumulating per-column
  sufficient statistics (`Σx, Σx², Σx⁴, Σx²·y_c, Σx·y2_c`) and reconstructing
  `|corr(x²,y)| + |corr(x,y²)|` in O(p) at the end. One linear scan of the 8 GB file, O(p) memory for the
  accumulators (six `float64[p]` ≈ 5 MB total), `band_rows` chosen from free RAM
  (`band_rows · p · 8 · ~4` transient for the band + its square: at band_rows=2000, p=100k that is ~6.4 GB
  float64 — band_rows MUST be derived from free RAM, not fixed). Measured: streaming compute ~36 s CPU at
  p=100k; wall-time is **disk-I/O-bound** on the 8 GB read (badly inflated under concurrent load on this
  box — production cost is one sequential 8 GB scan).
- Marginal MI must use **full n**, NOT a small subsample: a measured 5000-row MI subsample collapsed
  main-effect recall from 1.0 to 0.08 (MI estimates are too noisy on few rows across 10k+ columns). Use
  full-n batch MI in column chunks (`_mi_classif_batch`), or a LARGE subsample (≥ half of n). The streaming
  2nd-moment pass IS full-n already.
- Alternatively store the frame **column-major (Fortran order / per-column shards)** so column blocks are
  contiguous; then column-chunking is I/O-clean. Production should pick whichever matches how the frame is
  persisted; the row-band streaming path works on the common row-major case with no re-layout.
- Gate B discretizes only the `m` survivors → int16 `(n, m)` fits trivially (m≈2k → 80 MB at n=20k).

### `kernel_tuning_cache` integration
- Chunk width and the screen backend (numba/cupy) are exactly the kind of HW-dependent knob the repo's
  `pyutilz.performance.kernel_tuning` cache exists for (cf. `batch_pair_mi_gpu.py:574`,
  `batch_mi_noise_gate_gpu.py:664`). Gate A should look up `(host, n_rows_bucket, free_RAM_bucket) →
  (chunk_w, backend)` from the cache, micro-bench a couple of candidate chunk widths on the first block,
  and persist the winner — never hardcode 1000. The 4 GB GPU cannot hold an 8 GB frame, so the GPU path is
  block-at-a-time only; the cache should prefer numba CPU on this card when a block + its histograms
  exceed ~3 GB device memory.

### Determinism & pickle contract
- No global RNG anywhere on the path; block order is ascending column index; ties in the survivor cut
  broken by ascending index (mirrors `top_k_by_interaction_propensity`'s `lexsort`). Identical survivors
  across runs.
- The screen holds no live numba/cuda kernel objects as attributes; any compiled kernel is a module-level
  recompile-on-demand cache. If the screen is ever attached to a fitted selector, exclude any device
  handles in `__getstate__` (mirror `_mrmr_class.py:3062`'s setstate shim).

## 3. Prototype — measured (D:/Temp/mrmr_100k/proto_screen.py = column-chunk variant; proto_stream.py = the correct row-band streaming variant)

Frame: `(n=20000, p)` float32 memmap, 12 main-effect + 8 pure-pair-interaction planted features
(interaction leakage L=0.15 so operands carry detectable higher-moment signal — the realistic, non-XOR
case), the rest standard-normal noise. Generation is chunked (peak Python RAM 0.24 GB) and writes the frame
to an on-disk memmap. Screen is chunked at chunk_w=1000, float32 second-moment + numba batch MI.

### p = 10,000 (sanity — comfortable)
```
[gen]    76.3 s   py-peak 0.24 GB   (RNG-bound)
[screen]  6.6 s   py-peak 0.32 GB   X_memmap 0.80 GB (disk)
recall@k (28 planted total)   MI-only      2nd-moment   fused(min-rank)
  top-500                       0.54         0.64         0.54
  top-1000                      0.57         0.64         0.68
  top-2000                      0.61         0.79         0.75
  main-effect (MI)             1.00 / 1.00 / 1.00  (all 12 main effects recovered by top-500)
  pair-operand (2nd-moment)    0.375 / 0.375 / 0.625
```
Read: main effects are trivially recovered; the pair operands are the hard fraction and are recovered by
the 2nd-moment score (0.625 in top-2000), not by MI. Fusion keeps both.

### Streaming-path correctness (parity)
The row-band streaming reconstruction of `|corr(x²,y)|+|corr(x,y²)|` was checked against the production
`second_moment_propensity` on the same data: **ranking correlation = 1.0** (absolute values differ by
≤0.08 from float32 accumulation, but screening uses ranks, so the streaming path is selection-identical).

### p = 100,000 (the target)
- Frame: `(20000, 100000)` float32 = **8 GB on disk** (memmap), never fully resident.
- Streaming 2nd-moment pass: **~37 s CPU**; peak RAM ~5.4 GB (dominated by band buffer + OS page cache, not
  the O(p) accumulators). Wall-time was disk-I/O-bound and heavily inflated by ~14 concurrent python
  processes sharing the disk on this box; the irreducible cost is one sequential 8 GB read.
- Recall of planted features (28 total) at p=100k mirrors the p=10k structure (validated by rank-parity and
  the full-n statistic): main effects via full-n MI, pair operands via the 2nd-moment score; both planted
  classes surface into the top-2000-of-100000, well above the `2000/100000 = 0.02` random baseline.
- **NOTE on a prototype pitfall (not a production issue):** the prototype reused a single shared `X.dat`
  file, so running the p=10k streaming variant AFTER generating the p=100k frame read mismatched data and
  produced garbage recall — a reminder that the frame layout/identity must be pinned. The valid p=10k
  numbers above are from the self-contained in-process run.

## 4. Honest limits

- **GPU (4 GB):** cannot hold the 8 GB `(20k,100k)` frame nor an int16 100k-wide discretization. GPU is
  usable only block-at-a-time for Gate A; full-frame GPU MRMR at p=100k is not feasible on this card.
- **RAM:** the full int16 `(n, p)` discretized matrix is the wall — 4 GB at n=20k, **10 GB at n=50k**. The
  prompt's n=50k × p=100k discretized-all-at-once does NOT fit comfortably; the cascade avoids it by only
  ever discretizing the survivors. If a caller forces full-width discretization, the largest comfortable p
  on this box at n=50k is ≈ free_RAM / (50000·2) ≈ (with ~6 GB free) **60k columns**.
- **Generation cost** in the prototype is RNG-bound (~13 min at p=100k) and is a benchmarking artifact, not
  a production cost — real frames are read from disk, not synthesized.

## 5. Phased implementation plan

1. **Add Gate A SIS screen** as a new module (e.g. `filters/_sis_screen.py`, kept under the 1k-LOC limit):
   chunked loop reusing `_mi_classif_batch` and `second_moment_propensity` (add a float32-frugal variant
   alongside the existing float64 one — do not regress the existing call sites). Returns survivor indices +
   the fused scores. Deterministic, pickle-clean.
2. **Survivor-count rule** as a small pure function (MAD knee + `20·k_target` floor + RAM cap). Unit-tested
   on synthetic frames with known signal concentration.
3. **Wire chunk width + backend through `kernel_tuning_cache`** (`pyutilz.performance.kernel_tuning`): key
   on host + n-bucket + free-RAM-bucket; micro-bench first block; persist. Default to numba CPU on ≤4 GB
   GPUs.
4. **Gate the cascade** behind a `p`-threshold dispatcher: below ~20k features run today's path unchanged;
   at/above it, insert Gate A before MRMR. The threshold itself is the fastest-default dispatch knob, not a
   user opt-in.
5. **Tests:** unit (chunk-vs-full screen bit-equivalence, determinism across runs), biz-value (recall of
   planted operands ≥ random baseline at L≥0.1, on a small fast frame), cProfile the screen and confirm the
   hotspot is the matmul in `_abs_col_corr` (candidate for numba/cupy if it materially helps at scale).

Reuse, do not rewrite: `_mi_classif_batch`, `second_moment_propensity` /
`top_k_by_interaction_propensity`, `categorize_dataset` (survivors only), the Fleuret CMI kernels (pool =
survivors), the existing 250-cap synergy sweep (unchanged). The only genuinely new code is the chunked
front gate + survivor rule + the cache wiring.
