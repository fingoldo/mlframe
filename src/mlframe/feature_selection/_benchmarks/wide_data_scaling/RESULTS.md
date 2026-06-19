# Wide-data scaling of the MRMR-FE synergy screen — benchmark results (2026-06-18)

Hardware: GTX 1050 Ti (cc 6.1, 4 GB VRAM), 4 physical CPU cores. Data: n=8000, 8 quantile bins,
binary target, multi-seed, fixed data per seed. Settles whether the O(p^2) all-pairs joint-MI synergy
sweep (capped at `fe_synergy_screen_max_features=250`) can scale to wide frames two ways.

Scripts (re-runnable; all write throwaway state, none modify the package):
- `h1_bench.py` — exhaustive C(p,2) joint-MI wall-time vs p, CPU njit-prange vs CUDA vs cupy.
- `h1_gpu_large.py` — the p=5000 / p=10000 CUDA points (long-running; captured via live monitor).
- `h2_bench.py` — interaction-propensity ranking recall of planted pure-pair operands.

## H1 — exhaustive C(p,2) joint-MI sweep, wall-time vs p

| p | pairs | CPU njit | CUDA | CUDA pairs/s |
|------|----------|----------|--------|--------------|
| 250 | 31,125 | 6.07 s | 0.74 s | 4.2e4 |
| 500 | 124,750 | (contended) | 2.51 s | 5.0e4 |
| 1000 | 499,500 | 167.3 s | 9.61 s | 5.2e4 |
| 2000 | 1,999,000 | 679.3 s | 37.97 s | 5.3e4 |
| 5000 | 12,497,500 | ~3390 s (extrap) | 241.2 s | 5.2e4 |
| 10000 | 49,995,000 | ~13,560 s (extrap) | **1004 s (~16.7 min)** | 5.0e4 |

Clean O(p^2): CUDA quadruples per p-doubling; measured p=5000/10000 within 6% of the extrapolation.
CUDA is a flat ~13.5x over CPU. cupy was measured and is **non-competitive** (per-pair Python bincount
loop: 45 s at p=250). Memory at p=10000: device ~1.5 GB (codes 320 MB + pair-index 800 MB + out 400 MB),
fits the 4 GB card; host pair-index (800 MB) is the larger pressure. Correctness: a planted XOR pair
ranked **#0 of 49,995,000** with MI = ln2 — the exhaustive sweep finds the needle exactly.

Wall-time crossovers (this GPU): <30 s to p~=1715, <120 s to p~=3430, <600 s to p~=7670.

**H1 verdict:** exhaustive p=10000 IS feasible on this GPU (~17 min, fits VRAM) but too slow to default.
A GPU-exhaustive path lifts the practical cap to ~2000–3500 at a 30–120 s budget; keep it force/opt-in for
moderate p when balanced-interaction correctness outranks wall-time.

## H2 — O(p) interaction-propensity ranking recall (top-m of planted operands)

n=8000, p=2000, K=6 planted pure-pair interactions (12 operands), 5 seeds. L = main-effect leakage.

| criterion | L0/250 | L.1/100 | L.1/250 | L.3/250 |
|-----------|--------|---------|---------|---------|
| marginal_MI (baseline) | 0.10 | 0.45 | 0.68 | 1.00 |
| **2nd_moment** | 0.22 | 0.70 | **0.88** | 1.00 |
| cond_resp_var | 0.10 | 0.45 | 0.68 | 1.00 |
| dcor (sampled) | 0.12 | 0.23 | 0.27 | 1.00 |
| **gbm_splits** | 0.35 | **0.92** | 0.93 | 1.00 |
| random_base | 0.12 | 0.05 | 0.12 | 0.12 |

Timing (O(p*n)): at p=10000 — 2nd_moment 5.2 s, marginal 12 s, dcor 20.7 s, gbm 87.7 s.

**H2 verdict:**
- **L=0.0 (perfectly balanced): irreducible** — every criterion sits at the random baseline. No O(p)
  per-variable score recovers a zero-higher-moment interaction; only the exhaustive sweep can.
- **L>=0.1:** `gbm_splits` is best but 18x costlier; `2nd_moment` is the cheap sweet spot (0.88 @ top-250,
  beats marginal MI 0.68 by a wide margin). `cond_resp_var` ties marginal MI (no lift); `dcor` underperforms.

## Recommendation (implemented + planned)

Two-stage funnel, not GPU-everything:
1. **SHIPPED** (`fe_synergy_prerank`, default ON; `_fe_interaction_prerank.py`): above the cap, pre-rank all
   p by `2nd_moment` and keep the top `fe_synergy_screen_max_features` for the existing O(cap^2) sweep,
   instead of skipping the bootstrap (the legacy "engineer nothing on a wide zero-marginal frame"). At
   top-250 the sweep is ~0.7 s GPU / 6 s CPU.
2. **SHIPPED** (`fe_synergy_exhaustive`, default `"auto"`; `_fe_synergy_exhaustive.py`): `"auto"` ESCALATES to
   the FULL exhaustive C(p,2) CUDA sweep over ALL raw numeric columns (bypassing the cap, the pre-rank, and the
   n*p^2 cost gate) WHEN a CUDA GPU is available AND the predicted wall-time fits the budget -- so the DEFAULT
   gets the complete result (incl. the balanced L=0 case) for free at small/moderate p, and only frames too
   wide to sweep in budget fall back to the pre-rank. `"force"`/`True` runs it regardless of budget; `"never"`/
   `False` always uses the pre-rank. **Budget source:** MRMR's OWN `max_runtime_mins` (x60);
   `fe_synergy_exhaustive_max_seconds` is an optional override (default None); when NEITHER is set the budget is
   UNLIMITED (auto escalates regardless of p). Reuses the existing `batch_pair_mi_cuda` kernel (no new kernel)
   via `dispatch_batch_pair_mi(force_backend="cuda")`. CUDA throughput is measured-and-cached per host + (n, p)
   via `pyutilz.performance.kernel_tuning` (~5e4 pairs/s is only the cold-cache fallback, never hardcoded).

   **Measured crossover** (@ ~5e4 pairs/s fallback): p=400 -> ~1.6 s, p=2000 -> ~38 s, p=5000 -> ~241 s,
   p=10000 -> ~1004 s -- so with a 180 s budget auto escalates up to p~=4400, unlimited budget always.
   **Parity:** below the cap, forcing exhaustive selects the SAME raw operands as the capped path (pinned).
   **Balanced-case recovery proof:** on a wide perfectly-balanced (L=0) frame the exhaustive sweep recovers the
   planted pair as `add(sign(f7),sign(f95))` (BOTH operands), and the DEFAULT `"auto"` recovers it too when the
   frame is affordable; only `"never"` (or auto over-budget) misses it -- the pre-rank's irreducible blind spot.

3. **SHIPPED** -- pre-rank kernel + recall (`_fe_interaction_prerank.py` + `_fe_interaction_prerank_kernels.py`):
   the discrete one-hot score is computed by hoisting the per-class standardization out of the K-loop and
   reformulating it as a single `(p,n)@(n,K)` GEMM dispatched numpy/numba/cupy by work size (numpy kept as the
   bit-reference). Correct for ALL target types (binary / nominal / ordinal multiclass / continuous + binned
   regression / boolean / non-numeric labels). An opt-in `criterion="fused"` rank-fuses 2nd-moment with the
   LightGBM split-frequency signal to lift recall ~0.88 -> ~0.92 at L=0.1 (at ~18x the prerank cost -- the cheap
   `second_moment` stays the default).

4. **SHIPPED** -- SIS front gate for p>=100k (`_mrmr_sis_screen.py`, `sis_screen_threshold` default 20000): a
   single column-CHUNKED O(p*n) pass scores every column by fused `max(z(marginal_MI), z(2nd_moment))`, cuts
   100k -> a few thousand survivors (data-derived MAD-knee rule), then full MRMR runs on the reduced set; the
   (n,p) matrix is never fully resident (memmap, column blocks; chunk width from `kernel_tuning_cache`).
   Measured p=20000/n=4000: 10.85 s, 1000 survivors (5% of p), recall_main=recall_op=1.0. Below the threshold
   the path is byte-identical to before.

5. **SHIPPED** -- real-scale end-to-end test (`tests/feature_selection/test_mrmr_real_100k.py`): a hard 100k x
   10k memmap frame (mixed main effects + pure interactions + redundant cluster + heavy-tail + categorical-like).
   Fast mode (20k x 2k): MRMR selected 13/2000, base_recall 0.667, redundant cluster deduped 5->2, peak RSS
   +406 MB, and selected-feature held-out AUC 0.925 vs equal-count random-subset 0.504 (+0.42).

## Post-campaign loop iterations (2026-06-19) -- shipped wins + bench-rejected no-wins

After the wide-data + critique campaign, the self-paced /loop shipped these (all bit-identical / parity-gated):
- **MDLP per-column edges parallelized** (`per_feature_edges`, njit-nogil kernels -> ThreadPoolExecutor over
  columns, cache-safe 3-phase): **3.15x** (p=2000 n=20k 10.10s->3.20s; p=500 3.14x), default-on (n_jobs=-1,
  >=128 cols), narrow-frame no-regression, edges bit-identical. (9754e79e)
- **Engineered-MI OOM class closed** across ALL 6 FE MI-uplift scorers (univariate / pair-cross / triplet /
  quadruplet / adaptive-arity / mi-greedy) via a shared `mi_classif_batch_chunked`: per-column MI is
  bit-identical, peak RAM O(n*chunk) not O(n*n_engineered) -- fixes the (16000,20000) float64 = 2.38 GiB
  MemoryError. (000f2e8f, 15a1b611, 26f9f674)
- **Chunked-MI chunk width RAM-aware** (bound block BYTES to ~10% free RAM, not a fixed col count) -> safe at
  large n too, not just wide p; no hardcoded threshold. (64604bc6)

Bench-rejected / no-win (recorded so they are not re-attempted):
- **Threading the chunked-MI block loop**: NO-WIN -- `_mi_classif_batch` is already numba-prange-parallel
  across the block's columns, so each per-block call saturates cores; threading blocks would oversubscribe.
  The chunking is a MEMORY bound, not a parallelism lever. (9cd2086b)
- A balanced-L=0 exhaustive test's strict `pre-rank recovers fewer` premise was a FINITE-n knife-edge
  (the capped sweep can recover an approximately-balanced pair); reframed to the robust invariant
  (exhaustive reliably recovers BOTH operands and is never worse than pre-rank). (66551bee)

Net: the FS/FE perf surface is mature on this hardware -- the OOM class is closed (robust in p and n), the
dominant Fleuret-CMI cost is on the batched CUDA path, and discretization is column-parallel. Further wins
need genuinely new code or an uncontended large-n machine, not re-grinding the existing kernels.

## Remaining FS/FE backlog (2026-06-19 scan) -- for a fresh-context session / large-n machine

A grep for explicit deferred markers (TODO/future-work) found only research-scale or narrow-opt-in items --
the default-path perf surface is mature, so these are the genuine next levers, NOT a re-grind:
- **JMIM joint-MI cache** (`evaluation.py` ~562): the JMIM aggregator (`use_jmim`, opt-in) recomputes
  `I({X,Z};Y)` per (candidate, Z-combo) with NO cache, while the plain Fleuret-CMI path has `cached_cond_MIs`.
  A multiset-{X,Z}-keyed cache would speed JMIM mode. Risk: it lives inside the hot `@njit evaluate_gain`;
  needs a numba-typed-dict cache mirroring the arr2str CMI cache. Bounded but careful work.
- **Fleuret synergy rejection** (`fleuret.py:4`): `gain = I(X;Y) - max_k I(X;Y|S_k)` rejects synergistic
  features; a JMIM/CMIM-style aggregator addresses it. Flagged as a separate RESEARCH PR (quality, not perf).
- **k-way target encoding** (`_cat_interactions_step.py:705`): TE is pair-only; k-way is deferred feature work.
- **Weighted MRMR screening / RFECV** (`_cat_target_encoding_and_weighted.py:155`): needs a weighted
  `merge_vars` (500+ callers) -- project-level future work.
The GPU-CMI win is realized in both worker paths; the OOM class is closed (robust in p and n); discretization
is column-parallel. Net: no further default-path /loop win without one of the above (new code) or an
uncontended large-n box to re-measure the kernel crossovers.

## OOM-pattern sweep: codebase-complete (2026-06-19)

A codebase-wide grep for `_mi_classif_batch(... .to_numpy(float64))` full-matrix materializations confirms
ALL production FE MI-uplift scorers now route through `mi_classif_batch_chunked` (the 6 orth-FE scorers +
the shared helper). The only remaining full-matrix `mutual_info_classif` call is in a throwaway benchmark
script (`fs_hybrid/round4_fe_accept_bench.py`), not shipped code. The engineered-MI OOM class is closed.

## Synergy-aware aggregator (2026-06-19)

`redundancy_aggregator='auto'` shipped as a no-regression-gated OPT-IN (f76a4dba): a per-pair
interaction-information-vs-permuted-null detector routes to JMIM when synergy is detected, else stays plain
Fleuret. HARD GATE: reproduces plain-Fleuret selection EXACTLY on additive data and JMIM EXACTLY on
synergistic data (10 tests). Opt-in not default -- JMIM's synergy recovery is real but modest (balanced XOR at
n<=8k near the AUC~0.5 floor; recall 4W/15T/1L vs default) while it over-selects on additive data (F1 0.50 vs
0.94), so the bit-stable Fleuret default stays; `auto` is the strictly-safe way to get JMIM's synergy gain
without its additive over-selection. The default-path FS/FE optimization surface is now mature on this hardware.
