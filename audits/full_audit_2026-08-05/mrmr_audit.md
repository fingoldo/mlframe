# MRMR audit — 2026-08-06

Scope: the MRMR (feature selection) module proper, in the `wt_full_audit_2026_08_05` worktree —
`src/mlframe/feature_selection/filters/mrmr/` (class facade: `_mrmr_class.py`, `_mrmr_class_config.py`,
`_mrmr_class_fit_helpers.py`, `_mrmr_class_transform.py`, `_mrmr_config_dataclasses.py`,
`_mrmr_setstate_defaults.py`), `src/mlframe/feature_selection/filters/_mrmr_fit_impl/` (`_fit_impl_core.py`
— the ~10,100-line `_fit_impl` body, `_finalise.py`, `_helpers.py`, `_eng_dedup_batch_corr.py`,
`_fe_stage_temporal_agg.py`), and MRMR's own diagnostic/support siblings: `_mrmr_degenerate.py`,
`_mrmr_validate_transform.py`, `_mrmr_partial_fit.py`, `_mrmr_fingerprints.py` (the process-wide fit-cache
identity/content-hash layer), `_mrmr_artifacts.py`, `_mrmr_explain.py`, `_mrmr_stability_report.py`,
`_mrmr_tree_rescue.py`, `_mrmr_sis_screen.py`/`_mrmr_sis_apply.py`, `_mrmr_passthrough.py`,
`_mrmr_fe_provenance.py`, `_relaxmrmr_3d.py`. Corresponding tests under `tests/feature_selection/mrmr/**`
were sampled for coverage gaps against what's read here.

Excluded per the task's scope instructions: `screen_predictors`/`evaluate_candidate`/`compute_relevance_score`
(`evaluation.py` and friends) — the generic Fleuret-style greedy MI/redundancy engine MRMR calls into, but
which is shared infrastructure also used by other FS filters, not MRMR's own code; the ~150+ individual FE
generator families (`_hermite_prewarp`, `_wavelet_basis_fe`, `_hinge_basis_fe`, `_pairwise_modular_fe`, etc.)
under `feature_selection/filters/` — these are FE building blocks MRMR orchestrates, not MRMR itself, and are
already covered by the main 36-cluster audit's per-family perf/correctness passes documented extensively in
`CLAUDE.md`; Boruta/RFECV/ShapProxied and other non-MRMR selectors.

**Context**: this module already went through a dedicated audit (`audits/mrmr_audit_2026-07-25/`, referenced
by the master tracker) plus continuous single-issue fix/perf commits visible in `git log` for every file read
here (concurrency locking, cache-key symmetry, degenerate-column diagnostics, transform-time column-drift
detection, sample_weight/groups wiring, `fit_cache`/byte-cap eviction). The code is unusually
well-documented — nearly every non-obvious branch carries a "pre-fix this used to X" or "confirmed live"
comment — and several patterns that read as suspicious on first pass (e.g. `groups` appearing to be silently
dropped inside `_fit_impl`) turned out to be intentional, already-documented design (group-aware MI is wired
via a thread-local set by the `fit()` wrapper *before* `_fit_impl` runs; the `groups` array's only remaining
job inside `_fit_impl` is fit-cache-key disambiguation). Given that maturity, no severe (P0/P1-class)
correctness or leakage bugs were found in the files read; the findings below are narrower gaps. Areas
confirmed clean by direct reading (not padding, just noting no separate row was warranted): the fit-cache
lookup/store locking (`_MRMR_FIT_CACHE_LOCK` — symmetric lock scope on both the read+replay side and the
write+LRU/byte-cap-eviction side, `setdefault`-not-overwrite to make concurrent-arrival deterministic);
`audit_degenerate_columns`'s all_nan/constant/duplicate/collinear scan (purely diagnostic, does not affect
selection, correctly precedence-ordered); the transform-time engineered-recipe replay's topological
multi-pass resolution and its column-drift `RuntimeError`s; `get_feature_names_out`'s sklearn column-drift
contract and its cached-by-identity engineered-name list.

| ID | cluster | file:lines | description | disposition |
|----|---------|-----------|-------------|-------------|
| MRMR-1 | mrmr | `src/mlframe/feature_selection/filters/_mrmr_validate_transform.py:266-276` | `_validate_inputs`'s object-dtype +/-inf guard (`np.frompyfunc(lambda v: isinstance(v, float) and np.isinf(v), ...)`) only catches values that are instances of the builtin `float` (which covers Python floats and `np.float64`, since `numpy.float64` subclasses `float`), but NOT `np.float32`/`np.float16`/`np.longdouble` infinities stored in an object-dtype column (e.g. a ragged read, a mixed-precision upstream pipeline, or `df['c'] = df['c'].astype(object)` after an earlier `np.float32` computation). Such a value silently passes this guard — the exact "undefined bin on inf" failure mode the guard's own error message and the module docstring say it exists to prevent — and only the float64/native-float path (`select_dtypes(include=["floating"])`) is dtype-complete; the object-column fallback is not. Low likelihood in practice (float32 rarely ends up boxed in an object column) but a real, narrow gap in a guard whose whole job is catching exactly this. | TODO |
| MRMR-2 | mrmr | `src/mlframe/feature_selection/filters/_mrmr_partial_fit.py:83-92` (`_to_series`) | `MRMR.partial_fit` unconditionally rejects a multi-column `y_new` (`ValueError: y as DataFrame must be single-column`) with no ndarray-multicolumn path either (`np.asarray(y).ravel()` flattens any 2-D array into 1-D instead of raising, which is arguably worse — it silently reinterprets a genuine `(n, k)` multilabel/multi-target `y_new` as `n*k` single-target rows). `MRMR.fit` itself supports multilabel/multi-output `y` (see `_fit_multioutput` dispatch in `_mrmr_class.py`), so this is an undocumented capability regression specific to the incremental/streaming API: a caller doing multilabel MRMR via `fit()` who then switches to `partial_fit()` for streaming updates gets either a confusing single-column-only `ValueError` (DataFrame `y_new`) or a silent shape-corrupting `.ravel()` (ndarray `y_new`) instead of a working streaming multilabel path or an explicit "not supported" error. | TODO |
| MRMR-3 | mrmr | `src/mlframe/feature_selection/filters/_mrmr_partial_fit.py:91` | Direct consequence of MRMR-2 on the ndarray branch: `_to_series(y)` does `np.asarray(y).ravel()` with no shape check at all, so a genuine 2-D `y_new` ndarray (multilabel/multi-output) is flattened into a `(n*k,)` series and concatenated against an `(n,)`-row `X_new` buffer with no error — `partial_fit` would proceed on badly misaligned rows/labels rather than failing loudly. A one-line `if arr.ndim > 1: raise ValueError(...)` guard (matching the DataFrame branch's explicit rejection) is missing here. | TODO |
| MRMR-4 | mrmr | `tests/feature_selection/mrmr/**` (coverage gap re: MRMR-1..3) | No test found exercising (a) an object-dtype column holding a non-Python-`float` infinity (e.g. `np.float32('inf')` boxed into an `object` array) against `_validate_inputs`'s inf guard, or (b) `MRMR.partial_fit` called with a genuinely 2-D/multilabel `y_new` (DataFrame or ndarray) to pin either a working streaming-multilabel path or an explicit rejection. `test_mrmr_partial_fit_dtype_mismatch.py` and the `partial_fit`-adjacent tests under `tests/training/feature_selection/` cover dtype/column-alignment edge cases but not the multi-column-`y` case specifically. | TODO |
| MRMR-5 | mrmr | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fit_impl_core.py:73-86` (`_fit_impl`, ~10,100 total lines) | Architecture/maintainability note, not a functional bug: `_fit_impl` is one single ~10,000-line function, an order of magnitude past the project's own "carve before ~800-900 LOC" convention (`CLAUDE.md`, "New code goes in focused submodules from the start"). The file's own header comment explicitly claims an LOC-budget exemption ("one giant function cannot be split without distorting the fit control flow"), which is a defensible position for a tightly-coupled greedy-selection state machine — but at this size it is a real audit-surface and onboarding cost (this audit itself could only sample sections of it rather than verify every branch), and the exemption is self-granted rather than reflecting an actual technical impossibility (the file already successfully extracts small nested closures like `_fe_budget_ok`/`_eng_dedup_prefer`; a staged pipeline of well-named private helper functions passing an explicit fit-state object, rather than 10k lines of closures over `_fit_impl`'s locals, is a plausible refactor path, just a large one). Flagging for awareness rather than proposing an immediate rewrite, given the correctness risk of restructuring a function this size without an exhaustive behavioral-equivalence harness. | TODO |

| MRMR-6 | mrmr | `filters/_mrmr_validate_transform.py:114-121` (`_validate_string_params`) | Full-file re-read: the generic string-param validator loop does `if _val is None: continue` before checking allow-list membership. Correct for params whose allow-list legitimately contains `None` as a sentinel (`nbins_strategy`, `redundancy_aggregator`), but for params whose allow-list does **not** include `None` — `quantization_method` (`_VALID_QUANTIZATION_METHODS = ("quantile", "uniform")`), `mrmr_relevance_algo`, `mrmr_redundancy_algo`, `nan_strategy`, `fe_unary_preset`, `fe_binary_preset`, `cluster_aggregate_mode`, `mi_correction`, `stability_selection_method`, `dcd_distance`, `dcd_swap_method`, `additional_rfecv_selection_rule` — an explicit `MRMR(quantization_method=None)` silently bypasses the fail-fast `ValueError` the function's docstring promises ("Raise ValueError on bad constructor strings"). The bad value then reaches `_mrmr_fit_impl/_fit_impl_core.py:6022`'s `method=str(self.quantization_method)`, turning `None` into the literal string `"None"` — not a recognised discretisation method — so the failure (or silent misbehaviour) surfaces far from the actual cause with no message pointing back to the constructor param. Reproducible directly: `MRMR(quantization_method=None).fit(X, y)` skips the intended clean-error path. | TODO |
| MRMR-7 | mrmr | `filters/mrmr/_mrmr_class_config.py:258-270` (`_effective_n_jobs`) | `psutil.cpu_count(logical=False)` can return `None` on hosts (containers/some VMs) where physical-core detection fails — documented `psutil` behaviour. `_effective_n_jobs` does `if n_jobs == -1: return int(psutil.cpu_count(logical=False))` with no `None` guard, so on such a host every default-`n_jobs=-1` fit raises `TypeError: int() argument ... not 'NoneType'` instead of degrading gracefully. Every other host-probe in the MRMR tree (e.g. `_mrmr_sis_screen.py:_free_ram_bytes`) wraps its psutil call in try/except with a safe fallback; this one does not. | TODO |
| MRMR-8 | mrmr | `filters/_mrmr_sis_screen.py:257-271` (`sis_screen`, y-encoding branch chain) | The 3-way `if/elif` chain encoding `y` before the marginal-MI pass (nominal-string → factorize; low-cardinality-relative-to-`n_rows//20` integer → factorize; float/high-cardinality → quantile-bin) has no `else`. A small-`n_rows` integer target whose cardinality is moderate (falls between the `n_rows//20` factorize threshold and the `max(nbins,2)` quantile threshold) skips all three branches and reaches `_mi_classif_batch` unencoded. The downstream kernel (`_plugin_mi_classif_njit`) tolerates non-dense labels via a `y_max - y_min + 1` span rather than requiring 0-based codes, so this is not silently wrong for ordinary small label sets — but a large-magnitude/sparse integer target landing in this gap allocates its MI histogram sized by the raw span rather than the true class count, a latent memory/perf cliff on an otherwise-cheap O(p·n) screen. No test exercises this specific branch combination. | TODO |

**Summary**: 8 findings, all P2/P3-tier (no P0/P1 correctness or leakage bugs surfaced in the files read).
MRMR-1 and MRMR-6 are the two with real (if narrow) production-correctness relevance: MRMR-1 is a dtype gap
in an +/-inf guard; MRMR-6 is a fail-fast contract violation (`_validate_string_params` silently accepts
`None` for params whose allow-list excludes it, then the bad value propagates deep into the fit as the
literal string `"None"`). MRMR-2/3 are a genuine, user-facing API gap between `fit`'s and `partial_fit`'s
target-shape contracts, with a silent data-corruption mode (the `.ravel()`) rather than a clean failure on
the ndarray side. MRMR-4 documents the corresponding test gaps (extended by MRMR-8's SIS-screen gap).
MRMR-7 is a real crash-on-specific-hardware gap with no observed occurrence in this environment but a known
trigger condition. MRMR-5 is a standing architecture observation on an already self-acknowledged exemption,
included because 10k lines in one function is a large enough number to be worth a written note even where
the codebase has preemptively defended the decision.
