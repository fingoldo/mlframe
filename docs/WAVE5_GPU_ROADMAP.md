# WAVE 5: GPU routing for FE transformer / ensembling / composite / votenrank

Status: **roadmap document**. The kernel_tuning_cache + dispatcher
infrastructure (commits b886011 .. previous WAVE 4 + WAVE 6) is in place.
This document enumerates the four remaining mlframe subsystems where the
same routing pattern can yield wins, with concrete wire-in sites + a
suggested ROI estimate per site.

## 1. ``feature_engineering/transformer/`` GPU broadcast

**Current**: ``filters/feature_engineering.py`` declares
``gpu_compatible_unary_names()`` (21 names: log, exp, sin, cbrt, ...) and
``gpu_compatible_binary_names()`` (9 names: add, mul, hypot, atan2, ...)
plus the actual GPU implementations ``apply_gpu_unary_batched()`` and
``apply_gpu_binary_batched()`` (CuPy elementwise). The functions are
DEFINED but never CALLED from production code -- a grep confirms only
self-references inside the same module.

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

## 2. ``models/ensembling.py`` per-member GPU predict

**Current**: ensemble flavours iterate per-member, calling each model's
``predict_proba`` on CPU. CatBoost / XGBoost / LGBM all support GPU
inference via the same ``device='cuda'`` or task-type knob but mlframe
doesn't pass it through.

**Wire-in**:

* ``models/ensembling.py:score_ensemble`` and the per-flavour aggregators
  (``ensemble_probabilistic_predictions`` etc.) iterate over members. The
  per-member call site is the one to instrument.
* Detect GPU availability at top of ``score_ensemble`` once
  (``is_cuda_available()`` from pyutilz); thread a ``use_gpu`` kwarg
  through each member-loop iteration. Each model strategy
  (``training/strategies/*.py``) re-resolves whether IT actually supports
  GPU predict (catboost yes, xgb yes if compiled with CUDA, lgbm yes if
  ``MLFRAME_TRUST_LGB_CUDA`` is opted in).

**ROI**: high for batched validation (N_models = 5-20) on large test
sets (1M+ rows). 2-5x ensemble-prediction speedup likely. NEGATIVE on
small fits (CPU is faster < 100k rows).

**Risk**: high. Numerical differences between CPU + GPU predict are
known (catboost +/- 1e-5 typical; lgbm GPU vs CPU has wider drift in
practice). The ensemble blend amplifies divergence. Need a per-member
correlation check before declaring victory.

## 3. ``composite_estimator.py`` aggregation kernels

**Current**: ``CompositeCrossTargetEnsemble.predict`` aggregates K
component predictions via NNLS / Ridge / mean. The aggregation is
CPU-side numpy.

**Wire-in**:

* The aggregation matrix is shape ``(N_rows, K)``; multiplication is a
  GEMV. For K = 5-20, N = 1M, total = 20M FMA ops -- bandwidth-bound.
  CuPy ``cp.einsum`` or ``cp.matmul`` would amortise the data transfer
  if the component preds are ALREADY on GPU (i.e. items in section 2
  are wired). Otherwise the H2D + D2H round trip cancels the win.

**ROI**: low unless section 2 is wired first. After section 2 the
aggregation can stay on GPU and the savings are real: ~100ms per fit at
1M rows + 20 components.

**Risk**: low if section 2 lands first.

## 4. ``votenrank/`` vote aggregation

**Current**: votenrank Leaderboard computes vote aggregation (Borda,
Condorcet, Copeland, ...) on numpy. The methods scale O(N_models ^2 *
N_items) for pairwise comparisons; on hundreds of models that's
noticeable.

**Wire-in**:

* ``votenrank/_aggregators.py`` (if it exists, else create). Add CuPy
  variants alongside numpy versions per
  ``feedback_keep_all_kernel_versions``. Dispatch by
  ``len(members) >= threshold``.

**ROI**: low. votenrank is invoked once at end of training; total
fit-time fraction is sub-1%. Not the right place to spend optimization
budget unless someone runs a 100+ model ensemble routinely.

**Risk**: low. Vote algorithms are integer-arithmetic; numerical
equivalence holds bit-by-bit.

## Suggested order of implementation

1. Section 1 (FE transformer) -- self-contained, clear wire site, low
   risk.
2. Section 2 (ensembling predict) -- high reward but needs careful
   numerical-equivalence work + per-strategy guards.
3. Section 3 (composite aggregation) -- only after section 2 to avoid
   H2D round-trip overhead negating the gain.
4. Section 4 (votenrank) -- skip unless a real workload demonstrates
   the need.

Each section gets its own commit. Each section's implementation must
follow the established pattern:

* New kernel / GPU function lives alongside the existing CPU one
  (``feedback_keep_all_kernel_versions``).
* Public name routes to the fastest backend per call via the
  kernel_tuning_cache (``feedback_fastest_default_with_dispatch``).
* Numerical-equivalence test verifies bit-identical (or documented
  tolerance) results vs CPU baseline.
* Bench script under ``_benchmarks/`` measures min-of-5 wall + reports
  speedup.
