# models/ (incl. models/ensembling) -- mlframe audit

## Scope

All 41 `.py` files found recursively under `src/mlframe/models/**` were opened and read in full (none skipped, none partially covered):

- `src/mlframe/models/__init__.py`
- `src/mlframe/models/tuning.py`
- `src/mlframe/models/selection.py`
- `src/mlframe/models/optimization.py`
- `src/mlframe/models/rf_proximity.py`
- `src/mlframe/models/lgbm_defaults.py`
- `src/mlframe/models/masked_multilabel_objective.py`
- `src/mlframe/models/additive_interaction_diagnostic.py`
- `src/mlframe/models/_optimization_shared.py`
- `src/mlframe/models/_optimization_search.py`
- `src/mlframe/models/_benchmarks/__init__.py`
- `src/mlframe/models/_benchmarks/bench_lgbm_defaults_dart_heuristic.py`
- `src/mlframe/models/_benchmarks/bench_masked_multilabel_objective.py`
- `src/mlframe/models/_benchmarks/bench_additive_interaction_diagnostic.py`
- `src/mlframe/models/_benchmarks/bench_cpx15_selection.py`
- `src/mlframe/models/_benchmarks/bench_cpx16_optimization.py`
- `src/mlframe/models/_benchmarks/_bench_cpx16_old_driver.py`
- `src/mlframe/models/_benchmarks/_old_selection_cpx15.py`
- `src/mlframe/models/_benchmarks/bench_optimizer_history_growth.py`
- `src/mlframe/models/_benchmarks/bench_favorize_unexplored_cat_only_loop.py`
- `src/mlframe/models/ensembling/__init__.py`
- `src/mlframe/models/ensembling/base.py`
- `src/mlframe/models/ensembling/member_metrics.py`
- `src/mlframe/models/ensembling/per_member_tuning.py`
- `src/mlframe/models/ensembling/float_aggregation.py`
- `src/mlframe/models/ensembling/quality_gate.py`
- `src/mlframe/models/ensembling/score_validate.py`
- `src/mlframe/models/ensembling/score.py`
- `src/mlframe/models/ensembling/score_gate.py`
- `src/mlframe/models/ensembling/score_flavours.py`
- `src/mlframe/models/ensembling/predict.py`
- `src/mlframe/models/ensembling/process_method.py`
- `src/mlframe/models/ensembling/selection.py`
- `src/mlframe/models/ensembling/_benchmarks/bench_backward_ensemble_elimination.py`
- `src/mlframe/models/ensembling/_benchmarks/bench_stepwise_ensemble_selection.py`
- `src/mlframe/models/ensembling/_benchmarks/bench_caruana_vs_nnls.py`
- `src/mlframe/models/ensembling/_benchmarks/bench_ensemble_chooser_rank_metric.py`
- `src/mlframe/models/ensembling/_benchmarks/bench_pairwise_corr_scatter.py`
- `src/mlframe/models/ensembling/_benchmarks/bench_mad_factor_sweep.py`
- `src/mlframe/models/ensembling/_benchmarks/bench_score_ensemble.py`
- `src/mlframe/models/ensembling/_benchmarks/bench_quality_gate_groupcollapse_iter132.py`

Total files reviewed: 41. Total LOC reviewed (per `wc -l`, sum across the 41 files): 9266.

General note: this cluster is unusually well-hardened already -- the majority of files carry dense inline comments documenting prior "Wave N" bug-fix passes (assert->ValueError conversions, NaN-safe reductions, RNG-discipline fixes, weighted-aggregation fixes, etc.), and most public functions have biz_value / regression tests in `tests/models/`. The findings below are the residue that survived that prior hardening, found by tracing actual runtime behaviour (including two live numpy/dtype experiments) rather than static pattern-matching.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P1 | correctness / silent-wrong-value | `src/mlframe/models/_optimization_search.py:411-419` | `MBHOptimizer`'s dedup-known-evaluations path casts `+np.inf`/`-np.inf` sentinels into `_ys.dtype`; when `known_evaluations` is an integer-dtype array, **both** signs of infinity silently become `INT64_MIN`, corrupting the `Minimize`-direction aggregation. |
| F2 | P1 | robustness / missing validation | `src/mlframe/models/_optimization_search.py:298-326` | `MBHOptimizer.__init__`'s `model_name` dispatch (`"CBQ"`/`"CB"`/`"ETR"`) has no `else`/validation branch, unlike every other constructor parameter in the same method (which were deliberately converted from `assert` to `ValueError` in "Wave 31"). An unrecognized `model_name` leaves `self.model` unset and only surfaces as a confusing `AttributeError` on the first `suggest_candidate()` call. |
| F3 | P2 | ML best practice / silent weight loss | `src/mlframe/models/ensembling/base.py:700-704` | `combine_probs`'s `"median"` flavour silently falls back to an **unweighted** median (dropping `sample_weight` entirely) on `TypeError` from `np.quantile(..., weights=...)`, with no log line -- a caller on an older numpy silently loses the weighting they asked for. |
| F4 | P2 | ML best practice / silent weight loss | `src/mlframe/models/ensembling/base.py:752-763` | `combine_probs`'s NaN/inf fallback recomputes `_arith = np.mean(stacked, axis=0)` **unweighted**, even when `precomputed_weights` (NNLS/Caruana weights) were supplied for the main reduction -- any row with a NaN member value in a weighted blend silently reverts to unweighted arithmetic mean for that row only. |
| F5 | P2 | docs/API mismatch | `src/mlframe/models/ensembling/float_aggregation.py:19,51` | `combine_float_predictions`'s own default is `flavour="robust"`, but the module docstring explicitly states "the production resolver keeps `flavour='mean'`" and the real production caller (`_resolve_float_ensemble_flavour` in `training/core/_predict_main_suite.py`) does default to `"mean"`. Any *direct* caller of `combine_float_predictions()` that omits `flavour=` (e.g. a test, a notebook, a future integrator) silently gets the opt-in robust behaviour the module's own rationale says should not be the default. |
| F6 | P2 | code quality / mojibake | `src/mlframe/models/ensembling/process_method.py:428` | A comment contains a corrupted-encoding artifact (`РІР‚вЂќ`, a mis-decoded em dash) -- harmless at runtime but a readability/hygiene defect that should be re-typed as plain ASCII/UTF-8 per this repo's own encoding conventions. |
| F7 | P2 | edge case | `src/mlframe/models/additive_interaction_diagnostic.py:108-111` | `ratio = additive_score / full_score if full_score > 0 else nan` treats *any* non-positive `full_score` (including a legitimately signal-bearing but negative R^2/metric) as "undefined", forcing `recommend_interaction_engineering=False` even when the docstring's own rationale ("undefined only when the full model has no signal") would not apply to a negative-but-informative score. |
| F8 | P2 | code quality / fragile default | `src/mlframe/models/ensembling/score.py:88` | `ensembling_methods=SIMPLE_ENSEMBLING_METHODS` uses the shared module-level list object as the default argument. Today no code path mutates it in place (confirmed: `score_flavours.py`'s helpers always rebind to a new list), so there is no live bug, but it is a classic Python foot-gun -- a future edit that does `ensembling_methods.append(...)`/`.remove(...)` instead of rebinding would silently corrupt the shared default for every subsequent call in the process. |

### F1 -- `MBHOptimizer` dedup sentinel corrupts integer-dtype evaluations (Minimize direction)

`suggest_candidate()`'s dedup-known-evaluations branch (default `dedup_known_evaluations=True`) builds a same-dtype sentinel array via `np.full(len(_unique_x), -np.inf, dtype=_ys.dtype)` (Maximize) / `np.full(..., np.inf, dtype=_ys.dtype)` (Minimize) before reducing duplicate-x evaluations with `np.maximum`/`np.minimum`. When a caller constructs `MBHOptimizer` with an **integer**-typed `known_evaluations` array (e.g. `known_evaluations=[1, 2, 3]`, a perfectly natural thing to pass per the module's own docstring: "Function F can also be a numerical sequence in form of some y scores array"), `_ys.dtype` is `int64`. Live-verified on this environment (numpy 2.3.5):

```python
>>> np.full(3, -np.inf, dtype=np.int64)
[-9223372036854775808 -9223372036854775808 -9223372036854775808]   # RuntimeWarning, no exception
>>> np.full(3, np.inf, dtype=np.int64)
[-9223372036854775808 -9223372036854775808 -9223372036854775808]   # SAME value, no exception
```

Both `+inf` and `-inf` silently collapse to `INT64_MIN` under numpy's "unsafe" cast (only a `RuntimeWarning` is emitted, easily lost in normal logging). For `direction=Maximize` this happens to still work (the sentinel needs to be "very negative", and `INT64_MIN` is exactly that). For `direction=Minimize`, the code needs a "very positive" sentinel so `np.minimum` converges to the real minimum -- but it silently gets `INT64_MIN` instead, the smallest possible value. Since `np.minimum(INT64_MIN, anything) == INT64_MIN` always, `_agg[_i]` for every duplicated x **never updates away from `INT64_MIN`**, and the surrogate model is silently fit on `INT64_MIN` targets for every deduplicated point -- a silent, no-exception data-corruption bug matching exactly the checklist's "implicit truncating cast" bug class, just via `np.full`'s cast instead of `.astype`. No test in `tests/models/` passes integer-typed `known_evaluations` with duplicate `known_candidates` under `Minimize`, so this gap was never caught. Suggested fix: normalize `known_evaluations` to `float64` in `__init__` (mirroring the existing `known_candidates`/`.astype(int)` normalization pattern already used elsewhere in this class), or use `np.finfo`/`np.iinfo`-aware sentinels instead of a dtype-preserving `np.full`.

### F2 -- `MBHOptimizer.__init__` never validates `model_name`

Lines ~146-161 of the same file explicitly convert six other constructor checks (`quantile`, `search_space`, `acquisition_method`, `skip_best_candidate_prob`, `dist_scaling_coefficient`, `exploitation_probability`) from bare `assert` to `raise ValueError(...)`, with an inline comment explaining that asserts silently vanish under `-O`. The `model_name` dispatch a few dozen lines later (`if ... == "CBQ": ... elif ... == "CB": ... elif ... == "ETR": ...`) has no matching validation and no `else` branch — passing e.g. `model_name="CB Q"` (a typo) leaves `self.model` never assigned. The failure only surfaces later, inside `suggest_candidate()`, as an opaque `AttributeError: 'MBHOptimizer' object has no attribute 'model'` — exactly the kind of deferred, hard-to-diagnose failure the sibling checks were added to prevent. Suggested fix: add `if self.model_name not in ("CBQ", "CB", "ETR"): raise ValueError(...)` alongside the other Wave-31 checks.

### F3 / F4 -- `combine_probs` silently drops `sample_weight` in two edge branches

F3: the `"median"` flavour's `try: np.quantile(..., weights=sample_weight, method="inverted_cdf") except TypeError: combined = np.median(...)` silently discards the caller's weighting on any numpy version (or any other `TypeError`-raising condition) where the `weights=` kwarg isn't accepted -- no `logger.warning`, unlike almost every other silent-fallback path elsewhere in this same file (e.g. the NaN-fallback at line 752 does log a `WARN`-adjacent context via callers). F4: even when the main flavour reduction *did* honour `precomputed_weights`, the subsequent NaN/inf fallback recomputes an **unweighted** arithmetic mean and splices it in per-cell -- rows with a NaN member value quietly get the unweighted mean instead of a weighted one. Both match the checklist's "does a weighted fit's downstream ... step stay weighted, or silently drop back to unweighted?" bug class. Suggested fix: log a one-line WARN in the `TypeError` branch of F3; for F4, either weight the fallback mean too (`np.average(stacked, axis=0, weights=weights_arr)` when available) or fall back per-row only where `stacked` itself is finite for the weighted computation.

### F5 -- `combine_float_predictions` default flavour contradicts its own module's documented production default

The module docstring (float_aggregation.py, lines 12-19) is explicit that the bench data does **not** support making `"robust"` the default and that "the production resolver keeps `flavour='mean'`"; the real production caller (`_resolve_float_ensemble_flavour`) does exactly that. But `combine_float_predictions(stacked, *, flavour: str = "robust", ...)` itself defaults to `"robust"`. Any caller that reaches this function directly without going through `_resolve_float_ensemble_flavour` (tests, notebooks, a future second production caller) silently gets the opt-in, unvalidated-as-default behaviour. Suggested fix: change the function's own default to `"mean"` to match its documented contract, or add an inline comment explaining why the function-level default intentionally diverges from the production resolver's default.

### F6 -- Mojibake comment

A single comment line in `process_method.py` contains `РІР‚вЂќ`, a classic UTF-8-decoded-as-cp1251-then-re-encoded corruption of an em dash (`—`). Purely cosmetic (doesn't affect runtime), but worth a one-line fix given this repo's explicit Windows-encoding conventions.

### F7 -- `additive_interaction_diagnostic`'s ratio gate is broader than its own stated rationale

`ratio = additive_score / full_score if full_score > 0 else nan` guards against dividing by an uninformative `full_score`, but the docstring's own justification ("undefined ... when the full model itself has no signal to compare against") describes `full_score == 0`, not `full_score < 0`. A negative-but-real metric value (e.g. a fold with a bad R^2 that is nonetheless not "no signal") silently produces `nan` and thus `recommend_interaction_engineering=False`, even in cases where the caller likely *does* want a recommendation. Suggested fix: only treat `full_score == 0` (or use an epsilon band) as undefined, and let a genuinely negative `full_score` still compute a (very negative) `ratio`.

### F8 -- Shared mutable default argument

`ensembling_methods=SIMPLE_ENSEMBLING_METHODS` as a keyword default aliases the module-level list. No current code path mutates it (`score_flavours.py`'s `filter_sign_sensitive_flavours`/`collapse_to_single_flavour_if_identical` both rebind to fresh lists), so this is not live today, but it's a landmine for the next edit. Suggested fix: default to `None` and do `ensembling_methods = list(ensembling_methods) if ensembling_methods is not None else list(SIMPLE_ENSEMBLING_METHODS)` at the top of `score_ensemble`.

## Proposals

| ID | Category | File:Line | Summary |
|----|----------|-----------|---------|
| PR1 | test coverage | `src/mlframe/models/_optimization_search.py` | Add a regression test constructing `MBHOptimizer` with integer-dtype `known_evaluations`, duplicate `known_candidates`, `direction=Minimize`, and `dedup_known_evaluations=True` (default), asserting the surrogate is NOT fit on `INT64_MIN` -- would have caught F1. |
| PR2 | test coverage | `src/mlframe/models/_optimization_search.py` | Add a regression test asserting `MBHOptimizer(model_name="bogus", ...)` raises `ValueError` at construction time, matching the pattern already used for the other Wave-31 checks -- would have caught F2. |
| PR3 | architecture | `src/mlframe/models/ensembling/base.py` (981 LOC) | This file sits right at this repo's own ~900-1000 LOC monolith-split threshold (already the largest file in the cluster). It currently mixes leaf helpers, the numba probe, the `StreamingAccumulator`/Welford implementation, `combine_probs`, and the diversity-correlation helpers. The Welford/`StreamingAccumulator` block (lines ~243-378, ~135 LOC, largely self-contained and only used by the not-yet-wired streaming big-frame path per its own comments) is a natural first candidate to carve into a sibling (e.g. `_ensembling_streaming.py`) before the next feature addition pushes this file over the line. |
| PR4 | ML best practice / observability | `src/mlframe/models/ensembling/base.py:700-704`, `:752-763` | Beyond the F3/F4 fixes, consider stamping a `res["_weighted_fallback_used"]` diagnostic (mirroring the existing `res["_gate_bypassed"]`/`res["_diagnostic_mae_blowout"]` sentinel-key convention in `score_gate.py`) whenever a weighted reduction silently degrades to unweighted, so operators can audit how often this fires in production suites. |
| PR5 | perf / dev tooling | `src/mlframe/models/ensembling/predict.py:44-107` | The module-level `_gate_cache`/`_gate_cache_order` (keyed on `id()` of the member-prediction arrays) is documented as safe under `loky` process-pool parallelism (each worker is a separate process). If a future caller ever switches `score_ensemble`'s `backend=` to a threading backend, this module-global dict would need a lock -- worth a one-line assertion or comment pinning the "loky only" assumption so a future backend change doesn't silently introduce a race. |
| PR6 | docs | `src/mlframe/models/tuning.py` | `CatboostParamsOptimizer`/`ParamsOptimizer` is a substantial (~420 LOC), fully-documented public API with no mention in any top-level README/docs file found in this scope -- worth a one-line pointer from the package README given its complexity (CatBoost CTR-config generation, DB-backed surrogate-model trial suggestion). |

## Coverage notes

Every file in scope was opened and read in full; none were skipped or partially covered. Two behavioural claims in the findings above (the `np.full(..., dtype=int64)` cast behaviour behind F1, and the numpy `weights=` support behind F3's guard) were verified empirically via a throwaway read-only `python -c` invocation in this session rather than assumed from documentation, per this session's no-hand-wave requirement -- no repository state was modified to do so. I did not execute the actual `MBHOptimizer`/`score_ensemble` code paths themselves (that would require constructing a live CatBoost/ExtraTrees fit and is outside the read-only audit charter), so F1/F2 are traced via direct code-path reasoning plus the standalone numpy verification, not an end-to-end repro; they are reported as CONFIRMED given the numpy-cast behaviour is deterministic and the code path is unconditional (no guard exists that would prevent an integer-dtype `known_evaluations`/an unrecognized `model_name` from reaching the buggy line).
