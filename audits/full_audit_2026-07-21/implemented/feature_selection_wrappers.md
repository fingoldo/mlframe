# feature_selection/wrappers (RFECV etc., non-MRMR wrapper-style selection) -- mlframe audit

## Scope

Every file was opened and read in full this session (none skipped/truncated).

- `src/mlframe/feature_selection/wrappers/__init__.py`
- `src/mlframe/feature_selection/wrappers/_enums.py`
- `src/mlframe/feature_selection/wrappers/_helpers.py`
- `src/mlframe/feature_selection/wrappers/_helpers_importance.py`
- `src/mlframe/feature_selection/wrappers/_helpers_importance_agg.py`
- `src/mlframe/feature_selection/wrappers/_knockoffs.py`
- `src/mlframe/feature_selection/wrappers/_noise_floor.py`
- `src/mlframe/feature_selection/wrappers/_univariate_ht.py`
- `src/mlframe/feature_selection/wrappers/_auto_tune.py`
- `src/mlframe/feature_selection/wrappers/_benchmarks/__init__.py`
- `src/mlframe/feature_selection/wrappers/_benchmarks/bench_auto_rule_noise_fp.py`
- `src/mlframe/feature_selection/wrappers/_benchmarks/bench_cpx33_kendall.py`
- `src/mlframe/feature_selection/wrappers/_benchmarks/bench_cpx34_helpers_importance.py`
- `src/mlframe/feature_selection/wrappers/_benchmarks/bench_dichotomic_adaptive_step.py`
- `src/mlframe/feature_selection/wrappers/_benchmarks/bench_noise_floor_plateau.py`
- `src/mlframe/feature_selection/wrappers/rfecv/__init__.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_checkpoint.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_configs.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_cv_setup.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_diagnostics.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_finalize.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_fit.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_fit_fold.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_fit_init.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_fit_outer_loop.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_fit_setup.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_group_time_series_split.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_mbh_optimizer.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_multioutput.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_must_include.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_nan_policy.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_sffs.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_stability_select.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_validate.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_benchmarks/__init__.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_benchmarks/bench_cpx31_outer_loop.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_benchmarks/bench_cpx32_hash_streaming.py`
- `src/mlframe/feature_selection/wrappers/rfecv/_benchmarks/bench_fit_sig_cache.py`

Total files reviewed: 38. Total LOC reviewed: 9475 (per `wc -l` sum over the file list above; `rfecv/_benchmarks/__init__.py` and `_benchmarks/__init__.py` are 0-1 line stubs included in that sum).

I additionally cross-referenced the (in-scope-adjacent, not itself audited) `tests/feature_selection/wrappers/**` directory to check test-coverage claims for several findings below (grep + two targeted `Read`s), and ran one read-only Python introspection snippet comparing `SearchConfig`/`FIConfig`/`RobustnessConfig` pydantic-model defaults against `RFECV.__init__`'s flat-kwarg defaults (finding W4). No source file was edited; the introspection only imported and read attributes of already-installed classes.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| W1 | P1 | silent-failure / sample-weight | `rfecv/_stability_select.py:77-268` | `stability_selection=True` completely ignores `self._fit_sample_weight_`; bootstrap fits and FI scoring are always unweighted even when the caller passed `sample_weight=`. |
| W2 | P1 | silent-failure / ML-correctness | `_helpers_importance_agg.py:240-274` | `aggregate_importances_dispatched` (used whenever `importance_agg="dispatched"`, the RFECV default) silently drops `run_weights` for the `tree` and `linear` branches -- `fi_decay_rate` has zero effect on the default aggregation path. |
| W3 | P1 | reproducibility / hardcoded seed | `rfecv/_fit_fold.py:321-332`, `rfecv/_stability_select.py:176-181`, `_helpers_importance.py:349,362` | `get_feature_importances(..., random_state: int = 0)` is never passed a fold-/bootstrap-varying seed from either call site, so every permutation-based FI computation across every CV fold, every outer iteration, and every stability bootstrap uses the identical RNG seed. |
| W4 | P1 | config-drift / auto_tune correctness | `rfecv/_configs.py:44-46,92`, `_auto_tune.py:222-231` | `FIConfig.votes_aggregation_method` defaults to `None` while `RFECV.__init__`'s flat default is `VotesAggregation.Borda`, violating the "field defaults MUST match" invariant `_fit.py`'s `auto_tune=True` logic depends on; the promised bottom-of-file validator does not exist anywhere. Confirmed via introspection (only field that mismatches across all three configs). |
| W5 | P2 | sample-weight consistency | `rfecv/_fit_fold.py:386-397`, `mlframe/estimators/baselines.py:26-56` | The N=0 dummy-baseline score (`get_best_dummy_score`) is always computed WITHOUT `sample_weight`, while the real per-N fold scores ARE weighted (`_fold_test_sw`) when the caller supplies weights -- an inconsistent baseline feeds the "beats-dummy" early-stop check and `cv_results_`. |
| W6 | P2 | sample-weight consistency / docs | `rfecv/_sffs.py:22-118` | The `swap_top_k` SFFS pass's `cross_val_score` calls never forward `sample_weight`; the existing docstring only discloses that `fit_params` / `val_cv` / early-stopping are skipped, not sample weights. |
| W7 | P2 | dead code | `_univariate_ht.py:146-165,193-259,288-322` | `_rank_with_ties`, `_mann_whitney_u_z`, and `_kruskal_wallis_h` are fully implemented but never called anywhere in production or benchmarks (only the combined-pass `_v2` variants are used). |
| W8 | P2 | docs/API accuracy | `_enums.py:11-12`, `_helpers.py:444-462` | `OptimumSearch.ScipyLocal`/`ScipyGlobal` are documented as "Brent" / "direct, diff evol, shgo" but their implementations are thin aliases that delegate straight to the dichotomic suggester and never call scipy. |
| W9 | P2 | usability / silent-by-default warnings | `rfecv/_validate.py:47-55,58-67,70-77,301-323` | The `p>=5000` no-cap, `max_runtime_mins<1s` units-confusion, `cv>=n_samples` (effective LOO), and high-cardinality misconfiguration warnings are all gated behind `getattr(self, "verbose", 0)` (default 0) and never fire under default settings, unlike the leakage-scan warning in the same file which is unconditional. |

### W1 -- stability_selection silently drops sample_weight

`RFECV.fit(X, y, sample_weight=w)` validates `w` in `_init_fit_state` (shape/finite/non-negative checks, raises on mismatch) and stores it on `self._fit_sample_weight_` *before* branching into `self._fit_stability_selection(...)` when `stability_selection=True` (`rfecv/_fit.py:280-281`). `_fit_stability_selection` (in `_stability_select.py`) never reads `self._fit_sample_weight_`, never accepts a `sample_weight` parameter, and its per-bootstrap `est_clone.fit(X_sub, y_sub)` / `get_feature_importances(...)` calls carry no weight information at all. A caller who validated their weighted-fit setup on the default (MBH) path, then flips `stability_selection=True` for the small-n/high-p regime the docstring recommends it for, silently loses their weighting with no warning -- the validated-but-discarded weights create false confidence. Fix direction: thread `self._fit_sample_weight_` (sliced by the bootstrap `idx`) into the estimator `.fit()` call (guarded the same way `_fit_accepts_sample_weight` guards it on the main path), or raise/warn explicitly when both `stability_selection=True` and `sample_weight` are supplied.

### W2 -- `importance_agg="dispatched"` ignores `fi_decay_rate` for tree/linear families

`_helpers.get_next_features_subset` computes `_run_weights` from `fi_decay_rate` and forwards it to `aggregate_importances_dispatched(..., run_weights=_run_weights)` whenever `importance_agg == "dispatched"` (the RFECV **default**, flipped on in the ctor: `importance_agg: str = "dispatched"`) and the estimator family is tree or linear (i.e. RandomForest/LightGBM/CatBoost/XGBoost/LogisticRegression/Ridge/... -- the overwhelming majority of real usage). Inside `aggregate_importances_dispatched`, `run_weights` is only consumed by the `else` branch that falls back to the legacy `get_actual_features_ranking`; `aggregate_tree(feature_importances, k_cv=k_cv)` and `aggregate_linear(signed_importances)` do not accept or use `run_weights` at all. A user who sets `fi_decay_rate=0.05` (the value the ctor docstring itself recommends: "Recommended 0.02-0.1 for long runs (>=30 iters)") on a tree- or linear-family estimator gets *zero* effect from the knob -- silently. I confirmed the existing `TestFiDecay.test_decay_weights_shift_ranking_toward_recent_runs` test (`tests/feature_selection/wrappers/rfecv/test_wrappers_rfecv_fi_semantics.py:67-96`) only calls `get_actual_features_ranking` directly, so it cannot catch this dispatched-path regression. Fix direction: thread `run_weights` into `aggregate_tree`/`aggregate_linear` (weighted mean/std per feature) or explicitly warn/no-op-log when `fi_decay_rate>0` is combined with `importance_agg="dispatched"`.

### W3 -- permutation-based FI reuses the identical RNG seed across every fold and bootstrap

`_fit_outer_loop.py` generates a fresh, deterministic-but-distinct `_fold_seed` per fold (`int(self._rng.integers(...))`) and threads it into `_eval_fold_body` where it *is* used for `frac`-subsampling (`local_rng = np.random.default_rng(fold_seed)`, `_fit_fold.py:118-120`). However the same function's call to `get_feature_importances(...)` (`_fit_fold.py:321-332`) never passes `random_state=`, so `get_feature_importances`'s hardcoded default `random_state: int = 0` (`_helpers_importance.py:362`) is used on every single fold, of every outer RFECV iteration -- and `_conditional_permutation_importance`/`permutation_importance` receive that same fixed seed every time. The stability-selection bootstrap path has the identical gap (`_stability_select.py:176-181`). This affects the accuracy-preferred **default** resolution too: `get_feature_importances` auto-resolves `importance_getter='auto'` to `'permutation'` under the cell-budget cap (`_helpers_importance.py:389-396`), so ordinary small/medium RFECV runs are silently affected. Practically this means the class docstring's stated mechanism -- "use CV to calculate fold FI ... to mitigate noise" -- is weakened: the permutation-noise component of each fold's FI estimate is drawn from the same pseudo-random sequence every time rather than being independently sampled, so cross-fold voting averages less independent noise than it appears to. Reproducibility (same input -> same output) is unaffected; statistical power/robustness of the fold-voting ensemble is what's degraded, silently. Fix direction: derive a per-fold/per-bootstrap child seed from `self._rng` (mirroring the existing `_fold_seed` pattern) and pass it as `random_state=` into every `get_feature_importances` call site.

### W4 -- `FIConfig.votes_aggregation_method` default mismatches `RFECV.__init__`, breaking the auto_tune override-detection invariant

`rfecv/_configs.py`'s module docstring states: "Field defaults MUST match RFECV.__init__ flat defaults; the validator at the bottom asserts this invariant" -- no such validator exists anywhere in the file (I read it in full) or in the test suite (`tests/test_meta/test_subconfig_wiring_parity.py` covers `mlframe.training.configs`, not RFECV's grouped configs; no `fi_decay_rate`/`votes_aggregation_method`-keyed cross-check test exists under `tests/feature_selection/wrappers/`). This is not merely a stale comment: `_fit.py`'s `auto_tune=True` path (`_fit.py:206-235`) decides whether to apply an auto-tuned suggestion for field `k` via `if getattr(self, k) == getattr(SearchConfig(), k): setattr(self, k, suggested_value)` -- i.e. it treats "current value equals the config class's own default" as a proxy for "the user never explicitly set this flat kwarg." I verified via a direct Python introspection (importing `RFECV`, `SearchConfig`, `FIConfig`, `RobustnessConfig` and diffing every shared field name) that exactly one field currently violates the invariant: `FIConfig.votes_aggregation_method` defaults to `None`, but `RFECV.__init__(votes_aggregation_method: VotesAggregation = VotesAggregation.Borda, ...)`. A user who never touches `votes_aggregation_method` has `self.votes_aggregation_method == VotesAggregation.Borda`, which never equals `FIConfig().votes_aggregation_method == None` -- so the auto-tuner would treat every un-overridden user as having explicitly overridden this field, silently refusing to ever apply an auto-tuned suggestion for it. Today's `suggest_configs` doesn't yet populate `votes_aggregation_method`, so the live blast radius is currently zero, but the safety net that's supposed to prevent this exact class of drift (and would fire the moment `suggest_configs` is extended, per the module's own stated Wave-7 roadmap) does not exist. Fix direction: either change `FIConfig.votes_aggregation_method`'s default to `VotesAggregation.Borda` (matching the ctor) or implement the promised validator (e.g. a `test_meta` test enumerating `SearchConfig`/`FIConfig`/`RobustnessConfig` fields against `inspect.signature(RFECV.__init__)` defaults).

### W5 -- unweighted dummy baseline vs weighted real-N scores

`_eval_fold_body` scores the dummy (N=0 anchor) via `get_best_dummy_score(estimator=estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scoring=scoring)` (`_fit_fold.py:395`), and `mlframe.estimators.baselines.get_best_dummy_score` has no `sample_weight` parameter at all -- it always calls `scoring(model, X_test, y_test)` unweighted. Meanwhile the SAME fold's real-N score is computed with `scoring(_fitted, X_test, y_test, sample_weight=_fold_test_sw)` when the user supplied weights (`_fit_fold.py:307-313`). Under non-uniform sample weights this makes the dummy-vs-real comparison apples-to-oranges: the early-exit check `if final_score < state.nofeatures_score: ... "dummy baseline already beats the first explored subset"` (`_fit_outer_loop.py:357-367`) and the `cv_results_["nfeatures"]==[0, ...]` diagnostic row are both computed against a baseline that doesn't reflect the same weighting the user asked the rest of the fit to honour. Fix direction: add an optional `sample_weight=` to `get_best_dummy_score` and forward `_fold_test_sw` the same way the real-N scoring path does.

### W6 -- SFFS swap pass drops sample_weight (undocumented)

`_sffs_swap_pass`'s `cross_val_score(clone(estimator), trial_X, y, cv=cv, scoring=scoring, n_jobs=1)` (`_sffs.py:84-87`) never forwards `sample_weight`, even though `_finalize.py`'s call-site comment only discloses that the swap pass skips `fit_params` / `val_cv` / early stopping -- sample weights are a distinct code path (`self._fit_sample_weight_`) from `fit_params` and are not mentioned. A user relying on `swap_top_k>0` with weighted data gets an unweighted final-mile refinement without being told. Fix direction: either thread `sample_weight` through (via `cross_val_score`'s `fit_params={"sample_weight": ...}` / `params=`) or extend the existing disclosure comment to explicitly name sample_weight alongside fit_params/val_cv/ES.

### W7 -- dead code in `_univariate_ht.py`

`_rank_with_ties`, `_mann_whitney_u_z`, and `_kruskal_wallis_h` are complete, `@njit`-compiled, correctly-documented implementations, but grepping the whole cluster (production + `_benchmarks/`) shows they are called from nowhere -- every call site uses the combined-single-sort `_v2` variants (`_mann_whitney_u_z_v2`, `_kruskal_wallis_h_v2`, `_rank_and_tiesum`) instead. They add ~110 lines of numba-compiled surface area (and numba compile-time cost on first import) that is neither tested nor exercised. Fix direction: delete them, or if intentionally kept as an "old vs new" reference for a future A/B (matching this codebase's `_suggest_dichotomic(step='auto')`-style rejected-attempt convention), add a one-line comment saying so and a benchmark/test that actually calls them so they can't silently bit-rot.

### W8 -- `OptimumSearch.ScipyLocal`/`ScipyGlobal` docstrings no longer match behaviour

`_enums.py` documents `ScipyLocal = "ScipyLocal"  # Brent` and `ScipyGlobal = "ScipyGlobal"  # direct, diff evol, shgo`. Per `_helpers.py:444-462`'s own docstrings (an intentional, well-explained Wave-2 simplification: the argmax of a piecewise-linear interpolant always lands on an already-evaluated breakpoint, so a real scipy optimizer call was redundant with dichotomic search), `_suggest_scipy_local`/`_suggest_scipy_global` are now thin aliases of `_suggest_dichotomic` and never import or call scipy. This is flagged as a docs-accuracy gap, not a behavioural bug -- the alias is a deliberate, well-reasoned simplification -- but the enum-level comment a caller sees first (e.g. via `help(OptimumSearch)` or IDE hover) still promises Brent / DIRECT / differential-evolution / SHGO, which is actively misleading about what search strategy actually runs. Fix direction: update the `_enums.py` inline comments to reference the dichotomic-alias behaviour (or link to the `_helpers.py` docstring).

### W9 -- misconfiguration warnings silently suppressed by default (`verbose=0`)

Four distinct misconfiguration checks in `_sanitize_X_inputs` -- p>=5000 with no `max_nfeatures` cap (unbounded search time), `max_runtime_mins` under 1 second (likely a minutes/seconds unit-confusion), `cv >= n_samples` (degenerates to LeaveOneOut, many metrics NaN), and numeric columns with cardinality > 0.5*n (ID/hash-like, biases tree FI and breaks knockoffs' Gaussian assumption) -- are all gated on `getattr(self, "verbose", 0)` truthy, and `RFECV.__init__`'s own default is `verbose: Union[bool, int] = 0`. So under the OUT-OF-THE-BOX default configuration none of these operator-facing safety warnings ever fire, even though they exist specifically to catch common misuse. This is inconsistent with the leakage-correlation scan two paragraphs below in the same file, which logs unconditionally (`logger.warning(_msg)` with no verbose gate, `_validate.py:384`) precisely because leakage is dangerous enough to always surface. Fix direction: drop the `verbose` gate on these four checks (they are one-time O(1)-ish per-fit diagnostics, not per-iteration noise, so the performance rationale that legitimately gates other `verbose`-only logging in this file doesn't apply here).

## Proposals

| ID | Category | File | Summary |
|----|----------|------|---------|
| P1 | test-coverage | `tests/feature_selection/wrappers/rfecv/` | Add a `stability_selection=True` + non-uniform `sample_weight` regression test (would need W1 fixed first, or should assert-and-document the current gap). |
| P2 | test-coverage | `tests/feature_selection/wrappers/rfecv/test_wrappers_rfecv_fi_semantics.py` | Extend the F8 `fi_decay_rate` test to exercise `importance_agg="dispatched"` end-to-end (fit a tree-family RFECV with `fi_decay_rate>0` and assert the ranking differs from `fi_decay_rate=0`), which would have caught W2. |
| P3 | test-coverage / meta-test | `tests/test_meta/` | Add the validator `rfecv/_configs.py`'s own docstring promises: enumerate `SearchConfig`/`FIConfig`/`RobustnessConfig` fields and assert each default equals `inspect.signature(RFECV.__init__)`'s matching parameter default. Prevents recurrence of W4-style drift for any future knob, not just the one found here. |
| P4 | perf/architecture | `_helpers_importance.py`, `rfecv/_fit_fold.py` | Thread the already-generated per-fold `_fold_seed` (and an analogous per-bootstrap seed in `_stability_select.py`) into `get_feature_importances(random_state=...)` so permutation-based FI gets independent randomness per fold/bootstrap while staying fully deterministic from `self.random_state` (fixes W3). |
| P5 | ML-best-practice | `mlframe/estimators/baselines.py`, `rfecv/_fit_fold.py` | Add optional `sample_weight=` to `get_best_dummy_score` and forward the fold's test weights, so the N=0 anchor is directly comparable to the weighted real-N scores (fixes W5). |
| P6 | code-quality | `_univariate_ht.py` | Remove (or explicitly repurpose with a comment + test) the unused `_rank_with_ties` / `_mann_whitney_u_z` / `_kruskal_wallis_h` functions (W7). |
| P7 | perf/architecture | `rfecv/_fit_outer_loop.py`, `rfecv/_fit_fold.py` | The `0 not in evaluated_scores_mean` dummy-baseline gate is checked once per fold inside the concurrently-dispatched `_eval_fold_body`; on the very first outer iteration every fold computes the (potentially expensive, `get_best_dummy_score` tries 2-4 DummyClassifier/Regressor strategies) dummy independently before any of them can observe the others' results, which is correct but means the dummy is computed `n_splits` times redundantly instead of once. Not a correctness bug (each fold legitimately needs its own dummy on its own X_test/y_test), but worth a one-line comment clarifying this is intentional per-fold dummy scoring, not an accidental N-times repeat of the same computation, since a future reader may "fix" it into a shared single dummy score across folds (which would be wrong -- the dummy needs the SAME per-fold X_test as the real score to be a valid same-fold baseline). |
| P8 | docs | `_enums.py` | Update `OptimumSearch.ScipyLocal`/`ScipyGlobal` inline comments to describe the current dichotomic-alias behaviour instead of the historical Brent/DIRECT/differential-evolution/SHGO implementation (W8). |

## Coverage notes

- I did not execute `pytest` against `tests/feature_selection/wrappers/**` (read-only audit constraint); test-coverage claims above (W1/W2 regression-test gaps) are based on `Grep`/`Read` of the specific test files most likely to cover the behaviour, not a full-suite run. It's possible another test file I didn't open covers one of these combinations; I checked the most on-point candidates (`test_rfecv_sample_weight_unit.py`, `test_rfecv_fold_sample_weight_cat.py` for W1/W5/W6; `test_wrappers_rfecv_fi_semantics.py`, `test_importance_agg_dispatch.py`'s name for W2) but did not open every one of the ~74 files under `tests/feature_selection/wrappers/`.
- `mlframe.estimators.baselines.get_best_dummy_score` (referenced in W5) lives outside my assigned `src/mlframe/feature_selection/wrappers/**` scope; I read only its signature/body for context on the call site inside my scope, per the audit instructions for symbols imported from outside the assigned cluster. I have not audited the rest of `mlframe/estimators/baselines.py`.
- The `MBHOptimizer` class (`mlframe.models.optimization`) that `_mbh_optimizer.py` constructs is outside my scope; I read only enough of its call signature to assess `_build_mbh_optimizer`'s own logic, not the optimizer's internals.
- `Leaderboard` (`mlframe.votenrank`), consumed by `get_actual_features_ranking` in `_helpers_importance.py`, is likewise outside scope and was not audited beyond its call signature.
- I did not run the repo's own `test_no_file_over_1k_loc.py`-style architecture gate, but manually confirmed via the file list that every file in this cluster is under the ~900-1000 LOC convention (`rfecv/__init__.py` at 921 lines and `_helpers_importance.py` at 899 lines are the largest, both already carved into siblings and just under/at the soft threshold -- not flagged as a new finding since they're pre-existing and at the documented boundary, not over it).
