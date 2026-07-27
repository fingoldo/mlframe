# orth_basis_a — Hermite/orthogonal-polynomial pair FE, wavelet/hinge/integer-lattice/periodic univariate FE

This cluster is the "closed-form basis discovery" slice of the FE catalog: the `hermite_fe/` subpackage
(probabilist's-Hermite/Legendre/Chebyshev/Laguerre pair-FE via CMA-ES/Optuna search over `f(x_a)*g(x_b)`-style
combinations, plus its ALS warm-start, robust axis/warp-fit layer, and CPU/GPU polyeval kernels), the CMA/Optuna/
random-batch/numba/cupy optimiser dispatch (`_hermite_fe_optimise.py` + `_hermite_fe_optimise_pair.py`), the
production driver that wires hermite pair-FE into `MRMR.fit` (`polynom_pair_fe.py`), and four independent
univariate/pairwise operator families: Haar wavelet multiresolution basis FE (`_wavelet_basis_fe.py` +
its GPU-batched twin), slope-change hinge/piecewise-linear basis FE (`_hinge_basis_fe.py`), integer-lattice
gcd/lcm/bitwise pairwise FE (`_integer_lattice_fe.py`, prod, + `_integer_lattice_fe_proto.py`, experimental),
modular/periodic decomposition FE (`_periodic_fe.py`), and two small prototype/infra files
(`_extra_basis_fe_proto.py`, `_fourier_detect_cap.py`). All families follow the same leak-safe contract: fit
consumes `y` only to select a closed-form transform, the transform itself (recipe) replays as a pure function of
`X` at `transform()` time. Confirmed no DB/network/UI surface exists anywhere in this cluster (angle 9 N/A), and
no `eval`/`exec`/`subprocess`/`os.system`/`pickle.load`/`yaml.load` anywhere (angle 7 clean).

Cross-referenced against `audits/mrmr_audit_2026-07-20/c7a_orth_basis_fe.md` (this cluster's direct predecessor
report — note its file list is actually the *sibling* `_orthogonal_univariate_fe/` directory, a different
cluster not assigned here, except for `_wavelet_basis_fe.py`/`_wavelet_basis_fe_batched.py`/`_hinge_basis_fe.py`/
`polynom_pair_fe.py` which do overlap), `c4_gpu_infra.md`, `c8_usability_wrappers.md`, `gpu_residency.md`,
`edge_cases.md`, and the concurrent `mrmr_audit_2026-07-22/gpu_infra_d.md` (audits `_cupy_polynom_optimizer.py` /
`_numba_polynom_optimizer.py`, dispatch targets called *from* `_hermite_fe_optimise_pair.py` but not themselves
in this cluster).

## Findings

| ID | Severity | Category | File:Line | Summary | Prior-audit status |
|----|----------|----------|-----------|---------|---------------------|
| ORTH_BASIS_A-1 | P1 | bug / reproducibility | `_hermite_fe_optimise.py:499` (`_run_random_batch_search`) | `rng = np.random.default_rng(seed if seed > 0 else 1)` silently substitutes `1` for `seed<=0` — verified empirically (`default_rng(0)` and `default_rng(1)` draw byte-identical streams here) — while the SAME call's sibling RNGs (`_hermite_fe_optimise_pair.py:306` multi-fidelity subsample, `:789` noise-floor null) both correctly keep `seed=0` as `0` (`seed if seed > 0 else 0`). Reachable via the public `optimizer="random_batch"` kwarg AND via the DEFAULT `MRMR(fe_optimizer="cupy_kernel")` path, which silently falls back to `_run_random_batch_search` whenever `import cupy` fails (any CPU-only host) — see `_hermite_fe_optimise_pair.py:623-630`. A direct caller of `optimise_hermite_pair(seed=0)` / `composition.validate_pair_fe_cv(seed=0)` (both public, `seed` a plain user kwarg) gets a silently different, non-requested RNG stream, and two sibling RNGs *within the same call* disagree on how `seed=0` is treated. | NEW — same bug class as the concurrent report's GPU_INFRA_D-3 (`_numba_polynom_optimizer.py` / `_cupy_polynom_optimizer.py`, files outside this cluster) but a distinct instance/location; not raised by the 2026-07-20 audit for this file. |
| ORTH_BASIS_A-2 | P1 | error-handling / design | `_hermite_fe_optimise.py:415-419` (`_run_cma_search_batch`, `es.ask()`/`es.tell()`) and `:702-740` (`_run_cma_search`, same pattern) | `try: solutions = es.ask() except Exception: break` and (batch variant) `try: es.tell(...) except Exception: break` swallow ANY cma-library fault with a bare `except Exception` and **zero logging**, silently truncating the CMA generation loop early. Every other exception handler in this same file/module logs via `logger.debug`/`logger.warning` on suppression; these two do not, and the outer call-site wrapper (`_hermite_fe_optimise_pair.py:654-656`, which does `logger.warning(...)`) never fires because the exception is already caught *inside* `_run_cma_search`/`_run_cma_search_batch`, before it can propagate out. A real cma-library regression (e.g. a shape/API change after a `cma` package upgrade) would silently degrade hermite pair-FE recall fit-over-fit with no diagnostic trace anywhere in logs. Violates the repo's documented "no silent except-Exception swallowing without logging" convention. | NEW. |
| ORTH_BASIS_A-3 | P2 | docs | `hermite_fe/__init__.py:28` (module docstring usage example) | The module docstring's own usage example — `optimise_hermite_pair(x_a=col_a, x_b=col_b, y=target, n_trials=200, max_degree=4, n_jobs=1)` — raises `TypeError: optimise_hermite_pair() got an unexpected keyword argument 'n_jobs'` when run verbatim (confirmed by execution); the function has no `n_jobs` parameter and no `**kwargs` catch-all anywhere in its signature. | NEW. |
| ORTH_BASIS_A-4 | P2 | architecture / module-size | `_hermite_fe_optimise.py` (957 LOC), `_wavelet_basis_fe.py` (920 LOC) | Both exceed the repo's ~800-900 LOC guideline (957 and 920 lines respectively). `hermite_fe/__init__.py` (812 LOC) and `_hermite_fe_optimise_pair.py` (816 LOC) sit right at the edge of the same guideline. None are pathological, but all four are candidates for the next split before further feature growth. | NEW (LOC drift since the 2026-07-20 split that created `_hermite_fe_optimise_pair.py` — that split itself is documented in-file as motivated by the same 1k-LOC threshold, and `_hermite_fe_optimise.py` has since regrown past 900). |
| ORTH_BASIS_A-5 | P2 | docs | `_wavelet_basis_fe_batched.py:3-5` (module docstring) | States the module "is imported by NOTHING in the production path yet" and exists only for parallel validation "once `test_wavelet_batched_mi_parity` pins selection-equivalence it gets wired". In fact `_wavelet_basis_fe.py:565` already imports and calls `select_wavelet_legs_batched` from this module whenever `_binnedmi_gpu_enabled()` is true, and per the 2026-07-20 `gpu_residency.md` report the underlying `MLFRAME_FE_GPU_DEVICE_BORN_WAVELET` flag this routes through is DEFAULT ON in production. The "not wired yet" docstring claim is stale and could mislead a future reader into skipping this file when auditing the live GPU path. | NEW (the wiring predates this audit but the docstring was never updated after it landed). |
| ORTH_BASIS_A-6 | P2 | docs | `_hermite_fe_mi.py:400-407, 430-437` (`plugin_mi_classif_dispatch` / `plugin_mi_classif_batch_dispatch`) | Both docstrings state the function "Routes to ... via the kernel tuning cache (per-host measurement-backed)". The actual bodies never call into `kernel_tuning_cache` at all — every code path (`forced==""`, cuda available, GPU not globally disabled) falls through to an explicit "GROUND-TRUTH OVERRIDE" comment and always returns the njit backend, justified inline by real end-to-end measurements. The override itself is a sound, well-evidenced engineering decision; the docstring's claim about the mechanism is simply false and would mislead a maintainer into thinking a KTC-driven auto-tune is happening here (it is not — that's what the CUDA-vs-CPU crossover for a *different* kernel, `polyeval`, actually does in `_hermite_oracle.py`). | NEW. |
| ORTH_BASIS_A-7 | P2 | bug (data-integrity, narrow) | `polynom_pair_fe.py:562` (`run_polynom_pair_fe`'s per-pair injection loop) | `_new_data_cols`/`_new_col_names`/`_new_col_nbins` are appended BEFORE the `X[...] = _t_vals` / `X.with_columns(...)` assignment (lines 566-569) and before `engineered_features`/`hermite_features_list`/`engineered_recipes` are updated (lines 570-585); the surrounding `except Exception` at line 593 only logs and continues, it does not undo the earlier list appends. If the `X` assignment itself raises after the list append has already run, the unconditional `np.concatenate` at the loop's end (line 602) still bakes that column into `data`/`cols`/`nbins` with no matching `X` column and no recipe — `MRMR.transform()` would then either `KeyError` on the missing recipe or silently misalign column indices. | Still open — first reported in `c7a_orth_basis_fe.md` (P2); `git log` on this file shows no commit reordering the append/assignment since 2026-07-20 (latest touches are lint/memmap/pre-coercion fixes only). |
| ORTH_BASIS_A-8 | P2 | efficiency / statistics (prototype, not wired to prod) | `_integer_lattice_fe_proto.py:49-55, 108` (`_perm_null_hi`, called from `scan_integer_lattice_pairs`) | `rng=np.random.default_rng(rng_seed)` is constructed FRESH inside the innermost loop over every `(pair, op)` candidate, so every single candidate in a scan is tested against the IDENTICAL 12 permutation draws of `y` (same shuffle indices every time, since the generator restarts at the same seed each call). Wasteful (re-instantiates a `Generator` object per candidate instead of threading one through) and reduces the effective independence of the per-candidate null estimate. The PRODUCTION sibling `_integer_lattice_fe.py:196-204`'s `_perm_null_hi` has the identical per-call-`default_rng(seed)` shape, so this isn't unique to the prototype, but severity is capped low here because the module docstring explicitly marks it "EXPERIMENTAL prototype... NOT wired into prod". | NEW; not flagged for either file by the 2026-07-20 audit. |
| ORTH_BASIS_A-9 | (cross-ref, no new severity) | test_gap — CLOSED | `_hinge_basis_fe.py` / `_hinge_detect_gpu_resident.py` (outside this cluster; detector consumed by `_hinge_basis_fe.py:293-305`) | The 2026-07-20 audit's P1 findings #2/#3 in `c7a_orth_basis_fe.md` ("no direct CPU-vs-GPU parity test for the hinge detector exists" / "no quantitative regression pin for the subsample-cap claim") are now CLOSED: `tests/feature_selection/filters/test_hinge_detect_subsample.py::test_hinge_gpu_subsampled_matches_cpu_full_n_above_cap` explicitly cites "mrmr_audit_2026-07-20 B-16" and asserts CPU full-n vs GPU-subsampled tau agreement (tol 0.15) at n=500k, above the default 250k cap. The underlying algorithmic divergence itself (GPU thins to <=250k rows, CPU always scans full n) remains an accepted, now-tested design tradeoff — not a live bug. | FIXED since 2026-07-20 (test added). |
| ORTH_BASIS_A-10 | (cross-ref, no new severity) | test_gap — mostly CLOSED | `_wavelet_basis_fe.py` / `_wavelet_basis_fe_batched.py` | The 2026-07-20 `gpu_residency.md` P1 finding ("`test_wavelet_batched_mi_parity` is cited in the docstring but does not exist anywhere") is now mostly resolved: `tests/feature_selection/fe/basis/test_wavelet_batched_mi_parity.py` exists and pins `select_wavelet_legs_batched` (CPU) parity plus a fused-kernel identity check. No test function is literally named `test_wavelet_batched_mi_parity` (the closest is `test_select_wavelet_legs_batched_same_admitted`), so the in-code docstring citations (`_wavelet_basis_fe.py:562`, `_wavelet_basis_fe_batched.py:6,73`) are still nominally imprecise, but the actual coverage gap is closed. | Mostly FIXED since 2026-07-20 (test file added; docstring's exact function-name citation is stale but harmless). |

## Test coverage summary (angle 6)

Coverage for this cluster is broad and generally good: `optimise_hermite_pair` / the hermite prewarp / robust-fit /
basis-eval kernels have dedicated unit, biz_value, e2e, GPU-parity, and perf-regression tests (`tests/feature_selection/fe/basis/test_hermite_*`,
`tests/feature_selection/biz_val/test_biz_val_filters_hermite_fe.py`, `tests/feature_selection/gpu/test_hermite_*`,
`tests/feature_selection/gpu/test_prewarp_als_device_born.py`, `tests/feature_selection/gpu/test_gpu_basis_column_parity.py`).
Wavelet, hinge, integer-lattice, and modular/periodic FE each have their own dedicated basis/biz_val/provenance test
files. No zero-coverage or smoke-only (`assert result is not None`) files were found in this cluster's production
code. The two experimental prototypes (`_extra_basis_fe_proto.py`, `_integer_lattice_fe_proto.py`) have no tests,
which is expected and appropriate for unwired prototypes explicitly marked "NOT wired into prod" — flagging this
only for completeness, not as a gap needing action.

## Proposals

### Thread one shared `np.random.Generator` through the per-pair permutation-null scan instead of reseeding per candidate
Both `_integer_lattice_fe.py::_perm_null_hi` and its prototype twin construct `np.random.default_rng(seed)` fresh on
every call inside a loop over candidates. Building the `Generator` once per `cheap_integer_lattice_scan`/
`scan_integer_lattice_pairs` call and passing it in (still deterministic given the same top-level `seed`, but genuinely
independent draws per candidate rather than the identical 12 shuffles reused every time) would be a pure efficiency win
with no behavioural downside for the many-candidate-pair case, and would make the per-candidate null estimate a
slightly more honest independent sample.

### Add an explicit `n_jobs`-free usage example test that doctests the `hermite_fe` module docstring
Since ORTH_BASIS_A-3 shows the shipped usage example crashes verbatim, a cheap regression guard is a doctest (or a
plain unit test importing and eval-checking the documented call shape) that keeps the top-of-file docstring's example
honest going forward — this class of drift (docstring says X, code accepts Y) recurs across this cluster (also
ORTH_BASIS_A-5, ORTH_BASIS_A-6) and a single mechanical doctest-style check would catch all three cheaply.

### A tent/triangular (C0-continuous) sibling to the Haar wavelet basis
Restating the 2026-07-20 proposal from `c7a_orth_basis_fe.md` verbatim since it is still unaddressed and remains a
genuine, non-duplicate gap in this exact cluster: the wavelet family only ships the discontinuous Haar step; a
continuous dyadic "tent" hat-function multiresolution basis would capture a smooth localized bump without the
staircase artefacts Haar produces, reusing the same held-out scale-selection + incremental-MI admission machinery.
