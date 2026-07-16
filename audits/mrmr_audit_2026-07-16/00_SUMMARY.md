# MRMR module audit — 2026-07-16

Scope: `src/mlframe/feature_selection/filters/mrmr/` (all files) plus satellite modules
(fe/, info_theory/, permutation.py, evaluation.py, _stability_cluster.py) reached from it.

9 parallel audit agents, each covering an independent angle. Full raw output of each agent
is preserved in its own file in this directory:

1. [01_correctness_bugs.md](01_correctness_bugs.md) — logic/statistical/state bugs
2. [02_performance.md](02_performance.md) — redundant work, reflection overhead
3. [03_code_quality_design.md](03_code_quality_design.md) — mypy gaps, module size, API smells
4. [04_test_coverage.md](04_test_coverage.md) — coverage depth across core/caching/regression/fe/biz_val
5. [05_concurrency_and_statistics.md](05_concurrency_and_statistics.md) — thread-safety + MI/redundancy math
6. [06_docs_backlog_drift.md](06_docs_backlog_drift.md) — MRMR_RESEARCH.md / backlog vs actual code
7. [07_memory_scalability.md](07_memory_scalability.md) — large n/p memory behavior
8. [08_sklearn_joblib_compat.md](08_sklearn_joblib_compat.md) — get_params/clone/pickle/joblib
9. [09_error_messages_ux.md](09_error_messages_ux.md) — warning/error message quality

## Top findings (P0 — real bugs)

1. **`_mrmr_class.py:3095-3114`** — broad `except Exception` around stability-selection silently
   falls back to classic MRMR on ANY error (including a mistyped `stability_selection_method`,
   which has no separate validation). See 01, 05.
2. **`_mrmr_class.py:2978-3002`** (+ `_cmi_cuda.py:726`, `permutation.py:44`) — GPU circuit breakers
   are unprotected process-global booleans, unconditionally re-armed at the top of every `fit()`.
   Concurrent fits can un-poison a dead CUDA context for another in-flight thread. See 05.
3. **`_mrmr_class.py:2805-2806` vs `2849`** — `n_jobs=-1` / `parallel_kwargs=None` are resolved to
   concrete values BEFORE `store_params_in_object` snapshots them → `get_params()` never reports
   the sentinel; `clone()`/cross-machine joblib freeze the driver's core count forever. See 08.
4. **`evaluation.py:875,911`** — cmi-permutation/CPT null-test seed depends only on candidate index,
   not on `random_seed`/round/conditioning set → correlated null draws across rounds, and the
   `random_seed=` knob has zero effect on this component (defeats stability audits). See 05.
5. **`_mrmr_class_fit_helpers.py:321-362`** — multi-output (`union`/`intersect`) fit never
   populates `degenerate_columns_`/`provenance_`/`fe_provenance_` — `AttributeError` on these
   documented public attributes only in multi-output mode. See 01.
6. **`_stability_cluster.py:89,100`** — `cluster`/`complementary_pairs` stability selection builds a
   dense n×p copy + p×p correlation matrix bypassing `sis_screen_threshold` entirely — at p=50k,
   ~20GB for the matrix alone. See 07.
7. **`_mrmr_class.py:3364-3371`** — `mi_correction='chao_shen'` accepted as valid but silently
   degrades to plug-in MI, visible only via `logger.warning` (not `warnings.warn`). Docs
   (`MRMR_RESEARCH.md`) claim it's "DONE". See 06, 09.

## P1 — systemic design/UX issues

- Inconsistent warning channel: identical "requested feature will be ignored" situations use
  `warnings.warn(UserWarning)` in some places and `logger.warning`/`logger.info` in others —
  the latter is invisible without configured logging. See 09.
- `fit()` is ~730 LOC, `__init__` signature is ~2600 LOC (~300 params) — needs decomposition /
  config-object grouping (precedent: `CatFEConfig`). See 03.
- `inspect.signature(__init__)` reflection repeated 2-3x per fit across different helpers —
  cache once. See 02.
- `__setstate__` builds a full throwaway `MRMR()` instance just to source ctor defaults —
  expensive under frequent unpickle (joblib workers), and leaks worker-local `psutil.cpu_count()`
  into restored state. See 02, 08.
- mypy gaps: implicit-Optional (`groups: ... = None`), `cv: object | int | None`, mixin attrs
  typed `Any` instead of concrete types. See 03.
- `n_workers` vs `n_jobs`, `random_seed` vs `random_state` — self-acknowledged naming confusion
  (the `__repr__` override exists specifically to work around it). See 03.

## P2 — test coverage gaps

- No test constructs a multi-output (2D) y at all — not even a correctness-of-rejection test.
- `groups=` is tested only for its warn/raise gating, never for an actual leakage-prevention effect.
- GPU/CPU parity for scorers (jmim/cmim/cmi) only runs under `@pytest.mark.gpu`, which per project
  history does not execute in the default CPU-only full-suite run.
- `get_support()` never tested element-wise against a hand-computed boolean mask.
- `fast_search` profiles have no dedicated test.
- Best-in-class example found: `test_jmim_cache_parity.py` (behavioral, cache-hit/miss parity gate).

## P3 — documentation drift

`docs/MRMR_RESEARCH.md` claims Chao-Shen correction, KSG estimator, and SU-normalization-by-default
are all "DONE" — none are true in the current code. Backlog also references a parameter name
(`mrmr_redundancy_algo`) that doesn't match the actual knob (`redundancy_aggregator`).

## Suggested order of work

1. Fix P0 #1, #2, #3, #4 — these either mask real errors or break reproducibility/parallelism.
2. Fix #5, #6 — concrete, narrow failure scenarios (multi-output, large-p stability selection).
3. Fix #7 + unify warning channels — one pass, separate PR.
4. Add the missing tests (multi-output, groups-leakage, get_support element-wise).
5. Sync `MRMR_RESEARCH.md` with actual code — small separate PR.
6. Refactor `fit()`/`__init__` decomposition — separate task, non-blocking.
