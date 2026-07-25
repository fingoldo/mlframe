# HERMITE_WAVELET — audit (2026-07-25)

Scope: the Hermite/orthogonal-polynomial pair-FE package `hermite_fe/` (`__init__.py`, `_hermite_basis_eval.py`,
`_hermite_oracle.py`, `_hermite_prewarp.py`, `_hermite_prewarp_gpu_resident.py`, `_hermite_robust.py`), the
plug-in MI kernel/dispatcher sibling `_hermite_fe_mi.py`, the CMA/Optuna/random-batch optimiser pair
(`_hermite_fe_optimise.py`, `_hermite_fe_optimise_pair.py`), and the Haar-wavelet basis FE pair
(`_wavelet_basis_fe.py` + its device-born batched twin `_wavelet_basis_fe_batched.py`). These implement the
leak-safe closed-form basis-discovery slice of the FE catalog: fit consumes `y` only to select a transform;
the transform replays as a pure function of `X` at `transform()` time. All files verified against the current
source; mypy is clean across the whole cluster.

## Prior-finding verification (`mrmr_audit_2026-07-22/orth_basis_a.md`, `orth_basis_b.md`)

| Prior ID | Status now | Evidence |
|----------|-----------|----------|
| ORTH_BASIS_A-1 (P1, seed=0→1 collision in `_run_random_batch_search`) | **FIXED** | `_hermite_fe_optimise.py:510` now `np.random.default_rng(seed)`; fix note at :505. (The two remaining `seed if seed > 0 else 1` at :408/:699 are the pycma `"seed"` option, where 0 = "random seed" — intentional, not the same bug.) |
| ORTH_BASIS_A-2 (P1, bare `except` on `es.ask()`/`es.tell()`) | **FIXED** | `_hermite_fe_optimise.py:418-422, 465-467, 724-726` all now `logger.warning(...)` before break. |
| ORTH_BASIS_A-3 (P2, docstring example crashes with `n_jobs=`) | **OPEN** | See HERMITE-3. |
| ORTH_BASIS_A-5 (P2, stale "not wired" batched-wavelet docstring) | **FIXED** | `_wavelet_basis_fe_batched.py:3-10` now documents it is wired and default-ON. |
| ORTH_BASIS_A-6 (P2, false KTC-routing docstring on `plugin_mi_classif_dispatch`) | **FIXED** | `_hermite_fe_mi.py:403-408, 435-438` corrected (now describes the GROUND-TRUTH override). |
| ORTH_BASIS_A-4 (P2, LOC drift) | Partially: `_hermite_fe_optimise.py` now 970 LOC (was 957), still under 1k; not re-flagged. |

## Findings

| ID | Severity | Category | File:Line | Summary | Repro / failure scenario |
|----|----------|----------|-----------|---------|--------------------------|
| HERMITE-1 | P2 | fragility / import-order | `_hermite_fe_mi.py:40` | The documented `hermite_fe ↔ _hermite_fe_mi` cycle is benign ONLY if `hermite_fe` (the parent) is imported first. Importing `_hermite_fe_mi` as the first touch raises `ImportError`, and `_numba_polynom_optimizer.py:65` holds a top-level `from ._hermite_fe_mi import …` that is only safe because its own line :62 imports `.hermite_fe` first. Any future sibling that top-level-imports `_hermite_fe_mi` before the parent re-instates the crash. | Confirmed by execution: `python -c "import mlframe.feature_selection.filters._hermite_fe_mi"` → `ImportError: cannot import name '_ensure_cuda_kernels' from partially initialized module … _hermite_fe_mi` (raised inside `hermite_fe/__init__.py:810`). The parent-first order still holds today for every real caller, so latent, not live. |
| HERMITE-2 | P3 | house-convention (comment style) | 72 lines across the cluster, e.g. `_hermite_fe_mi.py:212` ("GPU-saturation Task #2"), `:388`/`:395` ("FIX1"), `:403`/`:435` ("ORTH_BASIS_A-6 fix"); `_hermite_prewarp.py:184` ("FIX4 (2026-06-28)"), `:452-454` ("X_EDGE_CASES_BEST_PRACTICES-5 fix … MI_GREEDY_RECIPES-1, ORTH_BASIS_A-1, GPU_INFRA_D-3"); `_hermite_fe_optimise.py:419/466/505/593/725/959` ("ORTH_BASIS_A-1/A-2 fix", "Wave 58"); `_hermite_oracle.py:22` ("Wave 23 P2"), `:86` ("Layer-103"); `_hermite_robust.py:14,253` ("backlog idea #17"); `_hermite_fe_optimise_pair.py:91,498,623` ("backlog idea #20", "GPU_INFRA_D-1 fix"). | CLAUDE.md bans finding-IDs / date-stamps / Wave-N / Layer-N / FIX-N / "Task #" / "backlog idea #" markers in comments (belongs in git history / PR). The prior repo-wide cleanup missed these. Cosmetic; no behaviour impact. |
| HERMITE-3 | P3 | docs (example crashes) | `hermite_fe/__init__.py:28` | Module-docstring usage example `optimise_hermite_pair(x_a=…, x_b=…, y=…, n_trials=200, max_degree=4, n_jobs=1)` passes `n_jobs=`, which the function does not accept and has no `**kwargs` for. | Confirmed via `inspect.signature(optimise_hermite_pair)`: no `n_jobs` param, no VAR_KEYWORD. Copy-pasting the shipped example raises `TypeError: … unexpected keyword argument 'n_jobs'`. (Prior ORTH_BASIS_A-3, still open.) |
| HERMITE-4 | P3 | dead-store | `_hermite_fe_mi.py:446` | `_n, _k = X_cols.shape` in `plugin_mi_classif_batch_dispatch` — neither `_n` nor `_k` is read anywhere in the function body (446-479). | Pure dead store; remove or fold into the njit call. No behaviour impact. |
| HERMITE-5 | P3 | docs (stale) | `hermite_fe/_hermite_prewarp.py:456` | `_ksg_mi_1d` docstring asserts it "is currently dead code (re-exported but never called from src/ or tests/)". It IS still dead in `src/` (only re-exported at `hermite_fe/__init__.py:799`), but `tests/feature_selection/regression/test_regression_mrmr_audit_2026_07_22.py:3618` now calls it, so the "or tests/" clause is false. | Cosmetic; the fix (threaded `random_state` param) is correct. Update the docstring to drop "or tests/". |

## Non-findings / confirmed-clean angles

- **Circular-import back-binding (@njit name resolution).** `_hermite_fe_mi.py:40` top-level `from .hermite_fe import _quantile_bin_njit` eagerly binds the global the three `@njit` kernels (`_plugin_mi_classif_njit:66`, `_plugin_mi_regression_njit:102-103`, `_plugin_mi_from_binned_njit`) reference as a bare global — verified bound before any kernel first-calls (parent defines it at `__init__.py:76`, re-imports the sibling at :810). No numba `__getattr__`-triggered NameError path. The load-order caveat is HERMITE-1.
- **`resident_raw_baseline_mi` first-touch shield.** Lives outside this cluster (`_resident_raw_mi.py`, `_unified_fe_gate.py`, `_conditional_gate_fe.py`, `_orthogonal_univariate_fe/`), so not owned here; its direct `_hermite_fe_mi` imports are all lazy/in-body and hit the parent-already-loaded state, so they never trigger the HERMITE-1 first-touch crash.
- **Rank-1 ALS prewarp leak-safety.** `fit_pair_prewarp_als`/`fit_operand_prewarp` (`_hermite_prewarp.py`) consume `y` only for the OLS/ALS coefficient solve; the emitted spec (`{basis, degree, coef, preprocess}`) is closed-form in `x`, and `apply_operand_prewarp:406` replays with no `y` reference. Coeffs frozen at fit. Verified.
- **GPU-resident prewarp parity.** `_hermite_prewarp_gpu_resident.py` `_build_basis_matrix_gpu` mirrors the host `_build_basis_*` recurrences (same seed columns, same column order) in f64; `_als_sweep_gpu` mirrors the CPU init + `g_norm`/`f_norm` stabilisers + iteration count + `solve(AtA,Atb)`→`lstsq` fallback. Device error tuple in `warm_start_als_seed:178-205` correctly narrows to genuine device/linalg faults (LinAlgError, CUDARuntimeError, OOM, cuSOLVER/cuBLAS RuntimeError subclasses) so a twin logic bug propagates instead of silently degrading. Continuous-`y` handling is consistent CPU↔GPU (both center `yc = y - mean`, same no-variance guard before upload).
- **Wavelet CPU↔batched↔device-born parity.** `_select_wavelet_legs` (CPU), `select_wavelet_legs_batched` (host-stack), and `_select_wavelet_legs_batched_device` all: split rows on the same `idx%3` mask, gate MIN_HALF_ROWS over the FULL leg, use cardinality-3 `_dense_leg_codes` (`leg+1`; absent value → empty bin → 0 MI, selection-equivalent to the CPU `np.unique` ternary partition), compute the same MAD floor, and sort survivors by train MI. `_bin_y_codes` identical across paths. No divergence found.
- **High-degree basis-tail overflow.** `_build_basis_matrix_gpu:69-76` deliberately keeps the fast-growing Laguerre/Hermite recurrence in f64 (documented ~10% f32 cancellation error) while only the standardised input column/target ride the relaxed f32 upload; the CPU basis-eval kernels (`_hermite_basis_eval.py`) run f64. Preprocessors z-score / min-max / shift bound the argument before eval. No unguarded overflow path found.
- **mypy.** Clean (`Success: no issues found in 6 source files` + 4 more, shared cache).
- **Security.** No eval/exec/subprocess/pickle.load/yaml.load/SQL/HTTP/UI surface anywhere in the cluster.

## Proposals (perf / refactor / test — not bugs)

1. **Break the HERMITE-1 latent cycle defensively.** Move `_hermite_fe_mi.py:40`'s `from .hermite_fe import _quantile_bin_njit` to a lazy in-body import inside each `@njit`-free wrapper OR make the three `@njit` kernels take `_quantile_bin_njit` via a module-level closure bound after the parent finishes; add a one-line smoke test that imports `_hermite_fe_mi` FIRST (fresh interpreter) to lock the fix and catch any future top-level-import sibling. Bench: import-time only, negligible.
2. **Zero-coverage completeness for the resident/oracle twins.** `_hermite_prewarp_gpu_resident.py` (2 test refs, GPU-gated) and `_hermite_oracle.py`'s ParamOracle CPU-backend path (`benchmark_polyeval_cpu_backends`, oracle enabled) have thin/GPU-only coverage; add a CPU-only oracle round-trip test (record→recommend→assert faster backend picked) so the `MLFRAME_POLYEVAL_ORACLE=1` path is exercised without a GPU.
3. **Doctest the `hermite_fe` module usage example** (closes HERMITE-3 permanently): a trivial import-and-signature check on the documented call shape would catch this drift class (recurs across the cluster).
