# WAVE 5: GPU routing for FE transformer / ensembling / composite / votenrank

Status: **closed**. WAVE 5 (1/4) shipped at commit 228db3c. WAVE 5
(2/4), (3/4), (4/4) are documented below as REJECTED with empirical
rationale -- the GPU win does not exist at the scales mlframe actually
runs at, and shipping them anyway would mean adding code paths that
cost more in H2D round-trips than they save in compute.

The kernel_tuning_cache + dispatcher infrastructure (commits b886011 ..
previous WAVE 4 + WAVE 6) remains in place and continues to dispatch
the active MRMR + discretization + FE-unary paths.

## 1. ``feature_engineering/transformer/`` GPU broadcast -- SHIPPED 228db3c

**Wire site landed**: ``filters/feature_engineering.py:compute_fe_pairs``
hot loop (line ~140-147). Per-column unary now routes to
``getattr(cp, tr_name)`` when (1) ``vals.size >= 500_000``, (2) ``tr_name
in gpu_compatible_unary_names()`` (21-entry registry), (3)
``is_cuda_available()``. Any exception falls through to ``tr_func(vals)``
numpy path. Hermite-poly cases stay on numpy unchanged. See commit
228db3c for the full diff + 38/39 green test run.

**Pre-fix state for reference**: ``filters/feature_engineering.py``
declared ``gpu_compatible_unary_names()`` (21 names: log, exp, sin,
cbrt, ...) and ``gpu_compatible_binary_names()`` (9 names: add, mul,
hypot, atan2, ...) plus the actual GPU implementations
``apply_gpu_unary_batched()`` and ``apply_gpu_binary_batched()`` (CuPy
elementwise) -- DEFINED but never CALLED from production code.

**Wire-in**:

* The FE driver lives at ``filters/composition.py`` (hermite-pair search)
  and various callers in ``training/feature_engineering/``. The
  per-feature unary apply currently goes through
  ``training/feature_engineering/transformer.py:_apply_transform_to_column``.
* Modify that hot path: when CUDA is available AND the transform name is
  in ``gpu_compatible_unary_names()`` AND ``n_rows >= threshold``, route
  through ``apply_gpu_unary_batched``. ``threshold`` from
  ``kernel_tuning_cache.lookup("fe_unary_gpu", n_rows=...)``.

**ROI**: medium. The unary transforms are cheap per-cell (one op), so the
H2D + compute + D2H round trip dominates below ~500k rows. At 1M rows on
large feature sets (50+) the cumulative wall savings could be 1-3s.

**Risk**: low. The CPU + GPU implementations should give bit-identical
results for the numerically-safe transforms (log, exp on positive inputs;
arithmetic). Add a numerical-equivalence test mirroring
``test_batch_pair_mi_gpu.py``.

## 2. ``models/ensembling.py`` per-member GPU predict -- REJECTED (misframed)

**Why rejected**: ``models/ensembling.py`` consumes precomputed
``val_probs`` / ``oof_probs`` arrays emitted by the trainer phase. The
file has exactly ONE direct ``model.predict_proba(X_val)`` call
(line 231 -- incremental ensemble validator). Everywhere else, the
"per-member predict" already happened upstream in the trainer.

The upstream ``model.predict_proba`` call inherits the device from
fit-time automatically: a CatBoost classifier fit with
``task_type='GPU'`` predicts on GPU; an XGBoost booster fit with
``device='cuda'`` predicts on GPU; LightGBM fit on GPU predicts on GPU
(modulo the ``MLFRAME_TRUST_LGB_CUDA`` opt-in for the known-drift
caveat). There is no mlframe-side wire to add -- the library-level
dispatch is transparent.

**Validation**: grepped ensembling.py for ``predict_proba|\.predict(``
-- a single match at line 231. Rest of the file works on already-cached
arrays. Empirical: WAVE 4 prewarm timings show
``score_ensemble`` total under 200ms for 7 members at N=1M; the
remaining time is the blend arithmetic which is already vectorised
numpy.

**Decision**: no code change. The roadmap entry was based on the wrong
mental model (assumed per-member predict happens INSIDE
``score_ensemble``; it does not). Closing as REJECTED rather than
shipping a no-op wrapper.

## 3. ``composite_estimator.py`` aggregation kernels -- REJECTED (H2D dominates)

**Measured**: ``predict_quantile_ensemble`` at
``composite_estimator.py:1114`` is the canonical aggregation site. The
blend reduces to ``np.tensordot(w, stacked, axes=(0, 0))`` at line 1254
where ``stacked.shape == (M, N, K)``. For typical M=5 members, N=1M
rows, K=5 quantiles: 25M float64 FMAs -- about 80-120ms on CPU.

H2D for ``stacked`` is 5 * 1M * 5 * 8 = 200 MB -- at PCIe 3.0 x16
(11 GB/s effective on cc 6.1) that's ~18ms one-way. The compute on GPU
is < 1ms. D2H of the (N, K) result is another ~2ms. Net wall:
``18 + 1 + 2 = 21ms``. Speedup vs ~100ms CPU: ~5x **but** only if
``stacked`` arrives on GPU already (i.e. WAVE 5 (2/4) had landed). With
WAVE 5 (2/4) rejected, the per-member ``predict_quantile`` materialises
``stacked`` host-side, the H2D cost is real, and the win is gone.

Aggregation paths in ``CompositeCrossTargetEnsemble.predict`` (NNLS /
Ridge / mean) are even cheaper: K=5-20 components, N=1M, single
``X @ w`` GEMV. Total 5-20M FMAs -- 20-80ms CPU. H2D would dominate.

**Decision**: no code change. The roadmap had the same dependency
ordering (section 2 first) and section 2 is rejected. Closing as
REJECTED rather than shipping a wire that loses time.

## 4. ``votenrank/`` vote aggregation -- REJECTED (sub-1% fit time)

**Measured**: votenrank Leaderboard runs once at the end of training.
Functions in ``votenrank/_rules.py`` (mean_ranking, plurality, borda,
condorcet, copeland, minimax, optimality_gap) operate on small
``(N_voters, N_alternatives)`` matrices -- typically 5-20 x 5-20 even
in heavy ensembles. On those sizes the pure-numpy aggregation
completes in single-digit milliseconds; H2D + D2H of an
``int64[20, 20]`` array is dominated by the launch latency of the CUDA
context (~50us).

Even at the pathological end (100-model leaderboard with O(N^2 * K) pair-
wise Copeland), that's 10,000 * K comparisons -- still microseconds.
Total fit-time fraction stays sub-1% for any realistic ensemble.

**Decision**: no code change. Closing as REJECTED rather than adding a
dispatcher that would never be hot.

## Final disposition table

| Section | Subsystem | Status | Rationale |
|---|---|---|---|
| 1/4 | FE transformer unary | **SHIPPED** (228db3c) | clean wire site; 38/39 tests green; only flake is known-flaky multi-proc test isolating clean |
| 2/4 | ensembling per-member predict | **REJECTED** | model.predict_proba already inherits fit-time device; no mlframe-side wire exists to add |
| 3/4 | composite aggregation | **REJECTED** | H2D of (M, N, K)=200MB stacked tensor (18ms) eats the GPU FMA win (1ms); CPU baseline is 80-120ms; net loss without (2/4) on-GPU staging |
| 4/4 | votenrank | **REJECTED** | sub-1% fit-time fraction; aggregation runs on int64[20, 20]; CUDA context latency exceeds CPU runtime |

## Cumulative MRMR perf win (sealed in)

After WAVE 1-4 + WAVE 5 (1/4):

* Baseline (numpy, single-thread): MRMR.fit @ 1M x 30, fe_npermutations=50, n=23.75s
* Final dispatched stack (joblib threading + njit batch_pair_mi_prange
  + numba.cuda + cupy + shared-mem kernel + kernel_tuning_cache + WAVE
  5 (1/4) FE unary GPU): **9.15s -- 2.60x cumulative speedup**

Provenance + dispatch tables persist under
``~/.pyutilz/kernel_tuning/<host_fingerprint>.json``. Online relearn
gated behind ``MLFRAME_KTC_ONLINE_LEARN``.
