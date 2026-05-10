# mlframe — project conventions

## Fuzz / combo tests are bug DETECTORS, not bug HIDERS (CRITICAL)

**The fuzz/combo suite (`tests/training/test_fuzz_suite.py` +
`_fuzz_combo.py`) exists to find production bugs. Every test failure is a
real prod bug unless you can prove otherwise. NEVER paper a failing combo
over with any of the following without first fixing the underlying
production code:**

1. **Canonicalisation in `_fuzz_combo.py`** — `_canonical_*` methods and
   `canonical_key` rules must collapse only **semantically-equivalent**
   combos (e.g. `imbalance="balanced"` is the same regardless of the
   imbalance-mode flag because at 50/50 the mode is meaningless). They
   must NOT collapse a combo "because it crashes". A canon rule whose
   justification is "fuzz cXXXX hangs / raises / produces empty val" is
   **prohibited** — fix the prod code instead.
2. **Runtime canonicalisations in `test_fuzz_suite.py`** — the
   `*_eff = ... if condition else ...` rewrites at the top of
   `test_fuzz_train_mlframe_models_suite` (before the suite call). Same
   rule: legitimate only when the rewrite preserves semantics; never as
   a guard against a real prod crash.
3. **`pytest.mark.xfail` / `pytest.skip`** — reserved for genuine
   third-party / OS / unfixable issues (Windows symlinks, optional
   dep missing, sklearn API limitation). A test that surfaces an
   mlframe-internal bug must be FIXED, not xfailed-with-TODO.
4. **Defensive guards in production** (trainer.py `_apply_pre_pipeline_-
   transforms`, wrappers.py CV-fold loop, etc.) — `if len(...) == 0:
   skip` patterns are acceptable only when the empty path is a
   legitimate user scenario. When the empty arises from an upstream
   bug, the guard is a band-aid that hides the bug forever — fix
   upstream.

### Concrete examples from this codebase (do not repeat)

- **Bad** (2026-04-26): `_canonical_text_col_count` zeroed text columns
  for CB+small-n+heavy-NaN combos because CB's `occurrence_lower_bound=50`
  produces an empty TF-IDF dictionary on tiny inner-CV folds. Hid a
  real prod hang.
  **Good** (2026-04-27): replaced with
  `training/helpers.compute_cb_text_processing` which scales
  `occurrence_lower_bound` proportionally to the fit-time row count;
  wired into `trainer._train_model_with_fallback` and
  `feature_selection/wrappers.py` RFECV inner-fold. Canon retired.
- **Bad** (still active): `canonical_key:327-335` collapses
  `inject_degenerate_cols=True` to `False` on CB+multilabel because CB
  mis-detects `num_const`/`num_null` as cat features. Hides c0062.
  **Owed fix**: explicit `cat_features=` arg in CB wrapper (or
  type-cast guard) so CB doesn't auto-detect numeric columns as
  categorical when `inject_degenerate_cols=True`.
- **Bad** (still active): `canonical_key:406` forces
  `remove_constant_columns_cfg=True` whenever degenerate / all-NaN
  columns are injected. Hides polars-ds RobustScaler crashing on
  zero-IQR (c0008/c0116).
  **Owed fix**: zero-IQR guard in the polars-ds robust scaler wrapper
  (skip / clip / fall back).
- **Bad** (still active): four layered "0-row val" tolerances in
  `trainer.py` (`_apply_pre_pipeline_transforms` ×2, `_setup_eval_set`,
  `_compute_split_metrics`). All point at the same upstream defect:
  outlier detection / aging-limit collapsing val to 0 rows silently.
  **Owed fix**: `_apply_outlier_detection_global` should raise on
  empty val (mirroring the train-side `min_keep` guard at core.py:1021).
- **Bad** (deferred): two `_rule_cb_*` functions defined in
  `_fuzz_combo.py` lines 833-868 with TODOs but NOT registered in
  `KNOWN_XFAIL_RULES`. Either bugs are fixed (delete dead code) or
  unfixed (silent fail in fuzz). Both were resolved 2026-04-27.

### Process when a fuzz combo fails

1. Read the traceback. Identify the prod-code line.
2. Decide: is this a legitimate user-facing bug (yes → fix in prod) or
   a genuine third-party limitation (yes → xfail with detailed reason
   + open issue / link)?
3. If you find yourself reaching for `_canonical_*`, ask: "would a real
   user with these settings hit this same crash?" If yes, you are
   masking a prod bug — STOP and fix prod instead.
4. Fixing prod often retires multiple canon rules / guards / xfails at
   once (e.g. fixing the splitter empty-val edge retires 4 trainer
   guards + 1 runtime canon + 1 prod-config validator gap).

## Memory / RAM constraints (CRITICAL)

**Frames in mlframe can be 100+ GB.** Never copy them to work around a bug.
Copying a prod DataFrame doubles peak RAM, which on a 200 GB+ workload means
OOM — the user observed this in 2026-04-22 prod logs.

Avoid:
- `df.copy()` (pandas) or `df.clone()` (polars) inside hot paths
- `df[cols] = df[cols].astype(...)` when `df` is the caller's frame (pandas
  broadcasts-copies the sub-frame)
- Constructing a fresh `pd.DataFrame(df)` / `pl.DataFrame(df)` to "get a new
  reference"
- Any fit-transform pattern that returns a mutated input

Prefer:
- Work on views (`.iloc`, column selection, slices)
- Mutate-and-restore: `X[col] = new; try: ... finally: del X[col]`
- Use `with` / context managers that revert the mutation on exit
- Lazy eval via polars `lazy()` + `.collect()` at the leaf call
- Pass `inplace` options where sklearn / the transformer supports them

Fuzz-caught example: MRMR.fit needed to temporarily inject a `targ_<id>`
column into X for MI computation. Original code mutated caller's X
in place, leaked the injected column into downstream sklearn steps, and
tripped `validate_data` on the next transform. Fix in
`feature_selection/filters.py:~2895` must inject + remove the column in a
try/finally (never call `X.copy()`).

If you find a bug that genuinely needs a copy, escalate — the user would
rather ship a design change than accept an unconditional copy on a hot
DataFrame path.

### Frame-type conversions are caller responsibility, NOT wrapper auto-magic

Helpers / wrappers / estimators MUST NOT silently down-convert polars
to pandas (or pandas to ndarray, or any other format-shift) on a hot
path "to make the inner library happy". A 100+ GB polars frame turning
into a 100+ GB pandas frame doubles RAM and bypasses every zero-copy
Arrow optimisation the rest of the codebase paid for.

The mlframe convention is unambiguous:

- `train_mlframe_models_suite` and the strategies layer
  (`training/strategies.py`) decide once at the suite boundary whether
  a given inner estimator wants polars or pandas, then pass the
  correct flavour through.
- Downstream wrappers (`CompositeTargetEstimator`, the new pre-pipeline
  estimators, anything else that holds a `base_estimator`) MUST NOT
  re-convert. If the inner estimator chokes on what it gets, that is
  a strategy-layer wiring issue to fix at the suite boundary, not
  inside the wrapper's `fit` / `predict` hot path.
- The ONLY allowed in-wrapper read of a polars column is the targeted
  `_extract_base`-style narrow column pull (one ndarray, not a frame
  copy). Even then the surrounding row mask / row subset must use the
  format-native `.filter(...)` / `.iloc[...]` -- never `to_pandas()`
  on the whole frame.

Concrete example (2026-05-10): `CompositeTargetEstimator` originally
did `X.to_pandas()` inside both `fit` and `predict` to work around
LightGBM 4.5 + sklearn 1.6 not accepting polars (the `feature_names_in_`
write-path bug). On a 100 GB polars frame this would have OOM'd the
host. Removed; the test suite now demonstrates the in-suite pattern
(caller does the conversion at the boundary if the inner needs it).

## Profile every new feature with cProfile + optimize hotspots (CRITICAL)

Each new feature added to mlframe must be profiled and its hotspots
optimized before being declared done. This is non-negotiable: features
without an explicit profile pass accumulate latent overhead that only
surfaces as a regression on prod. The standard pipeline:

1. **Write a cProfile harness** for the feature (small-shape + production-
   shape inputs). Save it inside the feature's package — never to
   `D:/Temp` or `/tmp` — so any maintainer can rerun. See
   `mlframe/training/_profile_dummy_baselines.py` as template.
2. **Sort by cumulative time, top 20-30**. The list is the optimization
   plan.
3. **Optimize hotspots where it materially helps**, considering:
   - `numba.njit` (typed JIT for numpy-heavy hot loops)
   - `numba.njit(parallel=True)` + `numba.prange` (multi-core scaling)
   - `numba.cuda.jit` (GPU kernels for very heavy elementwise / reduction
     workloads)
   - `cupy` (drop-in numpy on GPU — often the simplest GPU path when the
     work IS already vectorized numpy)
   - Vectorize loops; replace `apply` with `to_numpy() + ufunc`; cache
     repeated computations; bound input size for unbounded passes (e.g.
     ACF tail-cap pattern in `dummy_baselines._detect_acf_periods`).
4. **Calibrate against cProfile attribution overhead.** cProfile inflates
   pandas/sklearn deep-stack call timings ~10-13× vs standalone wall-
   time microbench. If cProfile flags a hotspot but a wall-time
   microbench on the same function shows <1ms, it's attribution noise
   — document this in the profile harness docstring so future re-runs
   don't re-flag.
5. **Document `no actionable speedup`** when the conclusion is "no
   optimization needed" — same docstring location. Future maintainers
   should see the trail and not redo the analysis.

This rule applies to: new training stages, new metric paths, new
diagnostic passes, new pre-/post-processing transforms, new model
wrappers, and new pipeline integration points. It does NOT apply to
trivial helper additions or pure refactors that don't change the
hot path.

## Numerical-kernel acceleration ladder + size-aware dispatch

For numerical hot kernels (polynomial eval, MI estimation, distance
computation, kernel matrix builds, etc.), the project follows a
**measured ladder** of backends and a **size-aware dispatcher** that
picks the right one per call. Do not assume "numpy is already
optimized" — verify with a microbench across `n` first.

### The four backends (in priority order)

1. **numpy / scipy** — the reference. Always implement this first as
   correctness baseline. Often the per-call dispatch overhead
   dominates the actual compute at n<10k, even though the C code
   itself is fast.
2. **`numba.njit` (single-thread)** — wins for n in 100..50k. Removes
   per-call dispatch overhead, inlines into surrounding njit
   functions, runs in machine code with no Python frame transitions.
   Empirically 3-6x over numpy at n=2k for polynomial Horner eval.
   `cache=True, fastmath=True` for max speed; safe when we don't need
   strict IEEE-754 (most ML code).
3. **`numba.njit(parallel=True)` + `prange`** — wins for n in 50k..500k.
   `prange` over array elements parallelizes across cores. Spawn
   overhead (~50us) makes this LOSE to single-thread njit at small n.
   Sweet spot: when each element's work is large enough to amortise
   the spawn (e.g. higher-degree polynomials, MI batches).
4. **CUDA (cupy or RawKernel)** — wins for n >= 500k once
   host->device transfer is amortised. Three flavours, ranked by
   measured throughput on this repo's reference hardware:
   - **`cp.RawKernel`** (custom CUDA C++ inline) -- **WINNER**. One
     thread per output element with state in registers. Compiled by
     NVCC at first call (~30ms), launches in ~30us thereafter.
     Worth the LOC.
   - **cupy elementwise** (drop-in `import cupy as cp` rewrite of
     numpy code): simplest, but allocates intermediate arrays per
     operation. **2-3x slower** than RawKernel for recurrence
     kernels. Use only when the work is genuinely elementwise (no
     temporaries) -- then it ties RawKernel.
   - **`numba.cuda.jit`** -- empirically the LOSER of the three on
     this hardware. Same kernel logic compiled by numba's PTX backend
     ran **6-10x slower than cupy RawKernel** in the polynomial-eval
     bench (1700us per launch baseline at all n<=100k vs ~50us for
     cupy RawKernel; numba's per-call dispatch overhead dominates).
     The Python-as-CUDA syntax is more ergonomic, but the runtime
     cost is real. **Prefer cupy RawKernel for new kernels.** Only
     reach for `numba.cuda.jit` if the kernel must be called from
     inside another `numba.cuda.jit` device function (no FFI between
     numba.cuda and cupy device functions). See bench:
     `feature_selection/_benchmarks/bench_poly_eval_backends.py`.

### Pre-implementation rule: BENCH first, dispatch second

**Always microbench all four backends across `n in {500, 2000, 10000,
100000, 1_000_000}` BEFORE writing the dispatcher.** Save the bench in
`*/feature_*/_benchmarks/bench_<kernel>_backends.py`. Output a JSON
results table to `_results/`. Only then write the dispatcher with
crossover thresholds derived from the actual measurements — never
guess from "GPU is fast".

Reference implementation: `mlframe/feature_selection/_benchmarks/bench_poly_eval_backends.py`
benches numpy / njit / njit_par / cupy / cuda_kernel for 4 polynomial
families. Output → `_results/poly_eval_backends_<basis>.json`.

### Dispatcher contract

The dispatcher is **stateless** and **per-call size-aware**:

```python
def kernel_dispatch(name: str, x: np.ndarray, *args) -> np.ndarray:
    forced = os.environ.get("MLFRAME_<KERNEL>_BACKEND", "")
    n = x.shape[0]
    if forced == "njit" or n < _PAR_THRESHOLD:
        return _NJIT_FUNCS[name](x, *args)
    if (forced == "cuda" or
            (forced == "" and n >= _CUDA_THRESHOLD and _CUDA_AVAILABLE)):
        if _CUDA_AVAILABLE:
            return _CUDA_FUNCS[name](x, *args)
    return _NJIT_PAR_FUNCS[name](x, *args)
```

Conventions:
- Threshold constants exposed as module globals AND override env vars
  (`MLFRAME_<KERNEL>_PAR_THRESHOLD`, `_CUDA_THRESHOLD`). Lets users
  retune for their hardware without code change.
- `MLFRAME_<KERNEL>_BACKEND=njit|njit_par|cuda` forces a specific
  backend (testing, A/B compare).
- Lazy CUDA init: compile RawKernels on first call, not at import.
  `_ensure_<kernel>_kernels()` pattern.
- CUDA path must auto-fallback to njit_par if cupy is unavailable
  OR if the GPU is OOM (catch `cp.cuda.runtime.CUDARuntimeError` and
  log once).

### Hoist dispatch out of hot loops

Per-call dispatch overhead is ~4us (env-var get, size check, dict
lookup, function call). At 1000 calls per outer iteration this is
4ms — not free. **In hot loops, hoist the dispatch decision out**:
pick the right backend ONCE before the loop based on the (known,
fixed) array size, then call directly through the loop. See
`optimise_hermite_pair` in `feature_selection/filters/hermite_fe.py`
for the pattern.

### When to skip the ladder

The four-backend ladder has real maintenance cost (4 implementations,
4 cross-correctness tests, the dispatcher, the bench). Skip it
unless the kernel is ALL of:

1. Called >100 times per fit / per request
2. Profile-confirmed >5% of wall on production-shape inputs
3. Has well-defined per-element work (kernels that need cross-element
   coordination — e.g. sorts, k-NN — don't fit the prange model
   cleanly)

Pure-numpy kernels that are <1% of wall do NOT get the ladder.
A trivial helper that runs once per fit does NOT get the ladder.
Document the decision in the kernel's docstring so future maintainers
don't re-raise.

### Reference implementations to copy from

- **Polynomial eval** (4 bases x 4 backends + dispatcher):
  `feature_selection/filters/hermite_fe.py` — `_<basis>val_njit`,
  `_<basis>val_njit_parallel`, `_polyeval_cuda`, `polyeval_dispatch`.
- **Joint histogram + MI** (CPU njit + CUDA RawKernel):
  `feature_selection/filters/gpu.py` — `mi_direct_gpu_batched`,
  `_GpuBufferPool` for persistent device buffers across calls.
- **Permutation MI** (njit single + parallel):
  `feature_selection/filters/permutation.py` — `parallel_mi`,
  `parallel_mi_prange`, `parallel_mi_besag_clifford` (early-stop
  variant).

### Don't over-engineer

If your kernel is called 10 times per fit and runs in 50us, the
ladder buys you nothing. Cache the result, use numpy, move on.
The `feedback_perf_measure_first.md` rule applies first: measure,
then optimize, then dispatch.

## Accuracy / performance over legacy / compat / deps

When choosing **defaults** or making **API decisions** in mlframe,
prioritise accuracy and runtime performance over backward-compatibility,
dependency-count minimisation, or "safe" legacy behaviour. The
framework is allowed to evolve aggressively in service of better
results.

**How this applies in mlframe specifically:**

1. **Default knobs flip when a new path measurably wins.** A new
   estimator / strategy / hyperparameter that beats the current default
   on a benchmark becomes the new default. Don't gate behind a feature
   flag "for safety" -- ship it on as the default and mention the
   change in CHANGELOG. Users who relied on the old behaviour can pin
   it explicitly.
   - Example (2026-05-10, R10b): switched
     `CompositeTargetDiscoveryConfig.mi_estimator` default from
     `"knn"` -> `"bin"` because bin is bias-free under monotone
     transforms; switched `screening` from `"mi"` -> `"hybrid"`
     because Phase B catches "wrong base" cases the MI-only path
     misses. Both flips landed without feature-flag gates.

2. **Extra optional deps are fine if they're faster or more accurate.**
   `numba`, `cupy`, `torch`, `lightgbm`, `xgboost`, `catboost`, `dill`,
   `polars`, `scipy.stats.wilcoxon` -- all already pulled in and
   used; users get them via the project's `pip install mlframe[all]`
   extras. A new feature that requires a single additional optional
   dep is welcome if it provides meaningful speedup or accuracy
   improvement. (See user memory `feedback_speed_over_deps`.)

3. **Tighten test assertions to the new path's stronger guarantee.**
   When a test passes on the OLD path with a weaker bound (e.g. RMSE
   tolerance 1.10) and the NEW path delivers 1.05, raise the bound to
   match the new behaviour. Don't keep the loose bound just because
   the old path happens to pass it.

4. **Don't ship "validated only on this fixture" defaults.** When a
   benchmark shows the current default is suboptimal on the
   measured fixture, do the wider benchmark (S1-S16, multi-seed,
   real-data proxy) NOW. Don't ship the suboptimal default with a
   "todo: validate broadly" note.

5. **Hard breakages still need care.** Data-format-incompatibility
   (a saved-model loader that no longer reads v1 pickles) and
   security regressions still warrant the deliberate path with
   explicit fallback / migration. But default-knob flips on quality
   metrics are fair game.

This rule pairs with the user's general-memory entry
`feedback_accuracy_perf_over_legacy.md`.

## Open work items

(Nothing tracked here currently. Polars support for MRMR — both the
selector core and feature engineering — landed 2026-04-22. See tests:
`tests/training/test_mrmr_polars_fe.py`,
`tests/training/test_bizvalue_feature_selection.py::test_mrmr_drops_uninformative_features_on_polars_input`,
regression sensors in `tests/training/test_fuzz_regression_sensors.py`.)
