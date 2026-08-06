# mlframe full audit 2026-08-05 — meta-test proposals (raw, pre-dedup)

Per-finding meta-test ideas plus each cluster-level architectural proposal, collected verbatim from the 36 auditors. To be deduped/synthesized into a final centralized-test plan (pyutilz-first) after the fix wave.

## Per-finding meta-test ideas

| Finding | Idea |
|---------|------|
| FE_ROOT_B-1 | Call knn_aggregate(coords, coords, labels, k=k) with query is ref and no group_ids; assert no row's own label contributes to its own neighbour aggregate. Generalize to any kNN/radius aggregator whose docstring claims query-is-ref leak-safety. |
| FE_TRANSFORMER_A-1 | Generic custom-objective gradient checker: for every (preds, train_data)->(grad,hess) LightGBM/XGBoost objective closure in the repo, assert returned grad/hess match finite-difference/sympy derivatives of the claimed loss to ~1e-4. |
| FS_WRAPPERS-1 | Fuzz RFECV.fit with mixed-dtype X (object/category column with NaN) across estimator families (linear, sklearn RandomForest, CatBoost) under nan_in_X_policy='impute' and assert no crash. |
| METRICS-1 | AST/CFG def-before-use checker: flag any local read on a path where no preceding assignment dominates it, across all multi-branch functions. |
| METRICS-2 | Parametrized sweep calling every exported metric function with empty arrays; assert NaN or a documented ValueError, never an undocumented exception. |
| METRICS-3 | Property test comparing fast_concordance_index against a brute-force independent C-index reference on data with deliberately duplicated prediction values, not just tie-free noise. |
| PREPROCESSING-1 | AST/grep scanner flagging cumsum(x**2) - cumsum(x)**2/n -shaped expressions repo-wide; a property test asserting expanding/rolling variance matches pd.Series.expanding().var() within tolerance across offset/scale sweeps (offset up to 1e9, scale down to 1e-1). |
| PREPROCESSING-2 | Synthetic column where identity is genuinely optimal: assert the leaky pre-split CV score for StandardScaler/RankGauss does not exceed an honest per-fold-refit score by more than noise; scanner flagging a .fit/.fit_transform call outside a CV fold loop whose output is later indexed by both train_idx and test_idx of that loop. |
| PREPROCESSING-5 | Fuzz test feeding analyse_and_clean_features 2-valued object/category columns across a matrix of Python value types (str, int, float, bool, Decimal, Timestamp, tuple, custom object) asserting no UnboundLocalError. |
| TRAINING_CORE_B-2 | Parity test: train a composite-target suite, save to disk, and assert predict_from_models and predict_mlframe_models_suite produce matching, correctly-y-scaled predictions on the same frame. Generalized: a sibling-predict-entry-point drift scanner diffing which shared safety helpers each public predict function calls vs. reimplements inline. |
| TRAINING_NEURAL-1 | Scanner: flag any _fit_common-style method receiving an is_partial_fit flag that calls a *_fit(...) helper unconditionally while a sibling helper in the same function IS gated on that flag. |
| TRAINING_NEURAL-2 | Fit RecurrentClassifierWrapper/RecurrentRegressorWrapper with a pandas.Categorical-dtype cat_features column with an unseen value in eval_set/predict; grep scanner for .map(...code_maps/mapping...) sites lacking a preceding .astype(object). |
| TRAINING_PIPELINE_MISC-1 | Property test: call attach_new_columns(df, new_cols) with a pandas df carrying a shuffled/non-default index and new_cols built with a fresh RangeIndex; assert output equals row-order-based expectation, not label-based. Generalize as an AST/grep scanner flagging pd.DataFrame(..., index=range(...)) / .reset_index(drop=True) results later passed into a .join()/pd.concat(axis=1) against a differently-indexed frame. |
| X_SECURITY_ROBUSTNESS-1 | For every class that scrubs secrets inside __repr__, assert __str__ is aliased/overridden identically; or a property test constructing a canary-secret instance and asserting the canary is absent from repr(x), str(x), f"{x}", and "%s" % x. |
| X_SECURITY_ROBUSTNESS-2 | AST-grep every function calling joblib.load/dill.load/pickle.load and assert the load is unreachable unless a shared trusted-root validator ran unconditionally on that code path (not inside an `if trusted_root is not None:` guard). |
| CALIBRATION-1 | For every group-bearing fit_*/apply_* pair, build an object-dtype group array containing None/np.nan and assert pd.isna(canonical) == pd.isna(original), and that the fitted table never contains a "nan"/"None" key. |
| CALIBRATION-2 | Call pick_best_calibrator(..., selection="same_oof") with oof_y encoded as {-1,+1} vs {0,1} on the same underlying data and assert the reported ece_ci matches between encodings. |
| COMPETITION_EVALUATION-1 | Grep every docstring phrase 'materialized to a list'/'consumed eagerly'/'without exhausting' and assert the corresponding parameter is wrapped in list(...)/tuple(...) before the first iteration in the function body. |
| COMPETITION_EVALUATION-2 | AST scanner: flag any single function parameter passed both to pandas.Series.map() and to np.asarray()/used as a numeric array in different branches of the same function. |
| COMPETITION_EVALUATION-3 | Property test: shuffle original row order relative to time_col, run with auto_remediate=True, and assert each reported range's width stays within a small constant factor of the true per-fold row count. |
| COMPETITION_EVALUATION-4 | Differential property test: compare every custom rank-correlation helper against scipy.stats.spearmanr on a heavily-tied (<=5 distinct values) fixture, asserting divergence stays under a fixed tolerance. |
| FE_ROOT_A-1 | Property test: for every flag combination and n in {0,1,2,3}, assert len(compute_numerical_aggregates_numba(arr, **kw)) == len(get_basic_feature_names(**kw)) |
| FE_ROOT_A-2 | Parametrized test: for every op, sweep seg_len from min_periods to window_K-1 and assert no exception / correct output length |
| FE_ROOT_A-3 | Fuzz test injecting a NaN/None into entity_col on a row with a finite value_col; assert no crash and documented handling |
| FE_ROOT_A-4 | Unit test: call each Bayesian entry point with each positivity-constrained kwarg set to 0.0 and -1.0; assert a clean ValueError, not an internal exception or silent NaN |
| FE_ROOT_A-5 | Perf regression test comparing add_anchor_extrapolation_features wall-time at n=200k against a sibling njit-accelerated anchor function on equivalent data |
| FE_ROOT_B-2 | Scanner: flag function pairs whose docstrings assert 'same definition/formula as X' but whose std/var calls use different explicit ddof values. |
| FE_ROOT_B-3 | Grep for 'streams/chunks to bound memory' docstring claims and confirm the implementation actually flushes/yields per chunk rather than accumulating everything before returning. |
| FE_ROOT_B-4 | Inject one duplicated-coordinate (rank-deficient) neighbourhood into an otherwise well-conditioned batch; assert only that row's output is NaN. |
| FE_ROOT_B-5 | Request 2 future windows where only 1 hits the data boundary with targets_creation_fcn set; assert the row is skipped like the no-targets_creation_fcn path. |
| FE_ROOT_B-6 | Entity with a huge historical cumulative sum and a small windowed delta; assert the windowed sum/mean matches a direct (non-subtractive) computation to tight tolerance. |
| FE_TRANSFORMER_A-2 | Scan for fit-then-predict-on-same-array patterns inside functions whose docstring mentions residual/hardness/disagreement/band; cross-reference against sibling files in the same directory that already solved it via OOF. |
| FE_TRANSFORMER_A-3 | AST/grep scanner flagging the literal [:, None, :] - ... [None, :, :] pattern anywhere under feature_engineering/, requiring use of squared_dists or an explicit size-bounded justification. |
| FE_TRANSFORMER_B-1 | Compare Performer y_estimate against brute-force softmax-exact attention on a small synthetic dataset across n_features; generic scanner for per-row max-stabilization applied to an array later reduced via sum/matmul across the same row axis. |
| FE_TRANSFORMER_B-2 | Assert m_focal's fitted structure/objective differs from m_lgb's beyond random_state for task='regression'; scanner for flavor-flag guards with no matching branch for the other task value. |
| FS_BENCHMARKS_A-1 | AST checker: for every `_*.py` module imported by a sibling module in the same package, verify every `module.attr` access resolves to a name bound at module scope in the imported file, not nested inside `if __name__ == "__main__":` or any other conditional. |
| FS_BENCHMARKS_B-1 | AST/grep scanner: flag np.asarray(<pandas-obj>) / .values / .to_numpy() without copy=True immediately followed by an in-place index-assignment on the same local with no intervening .copy(); behaviourally, call the timed function twice and assert the source object's content hash is unchanged. |
| FS_BENCHMARKS_C-1 | Grep-based scanner: flag any module-level string literal passed to open(..., "a"/"w") whose containing function has no reachable os.makedirs/Path.mkdir call. |
| FS_BORUTA_ROOT-1 | Scanner: find attributes initialized via np.zeros(...) then grown by repeated np.vstack (the 'history_*' pattern); flag any consumer of the resulting DataFrame/array that does not slice off row 0 while a sibling consumer of the same attribute does. |
| FS_WRAPPERS-2 | Parametrized fuzz test: same minority-class-size violation across y dtypes {int64, float64, bool}; assert the identical clear ValueError fires for all three. |
| FS_WRAPPERS-3 | Micro-benchmark CI gate asserting select_features_fdr on p=5000 synthetic W completes under ~0.5s. |
| METRICS-4 | AST scanner flagging any function parameter that never appears as a Name/Load node in the function body — catches dead parameters in public metric APIs. |
| MODELS-1 | Run a full suggest/submit cycle with a float search_space (np.linspace) and assert every submitted candidate is excluded from later suggestions and dtype is preserved. |
| MODELS-2 | Fuzz test asserting len(members)==len(tags) for every returned tuple across the score.py gate pipeline, with a synthetic member forced to fail the K>2 quality gate. |
| MODELS-3 | Sample N=2000 candidates via generate_valid_candidates on a real CatboostParamsOptimizer and assert boosting_type=='Ordered' and posterior_sampling=True each survive check_rules at a nonzero rate. |
| MODELS-9 | Call the function with y as a pandas.Series built from a shuffled/non-contiguous index and assert results match the plain-ndarray call bit-for-bit. |
| PREPROCESSING-3 | Regression test with 3 unevenly-spaced groups asserting the interpolated fill is closer to the near neighbor (by order_col value) than the positional midpoint. |
| PREPROCESSING-4 | Test asserting train_encoded's correlation with y_train on a purely-random (label-independent) categorical column stays near the smoothing-implied floor, not inflated by self-leakage; grep scanner for <series>.groupby(<same-frame-column>) mapped straight back onto that same frame without a fold split. |
| PREPROCESSING-6 | Regression test asserting apply_gaussian_power_transform raises or clearly flags (not silently passes through raw values) when apply-time data violates the Box-Cox positivity precondition the fit-time data satisfied. |
| REPORTING_A-1 | AST/grep scanner flagging `\bis\s+(object\|str\|int\|float\|bool\|dict\|list\|tuple)\b` outside isinstance(...) calls. |
| REPORTING_A-2 | Property test with random importance vectors containing injected NaNs; assert returned top-k never includes a NaN-importance feature when enough non-NaN features exist. Same unguarded pattern recurs 3x more in sibling diagnostics_dispatch.py (out of scope, not fixed here). |
| REPORTING_A-3 | Render via a 2-format/2-backend plot_outputs DSL then call build_combined_html_report; assert every <img src> path in the produced HTML exists on disk. Existing tests only use single-backend 'matplotlib[png]', confirmed via grep of test_diagnostics_dispatch_followup.py. |
| REPORTING_A-4 | Scanner: for every def render_*_diagnostic(/def render_*_from_suite( in reporting/, grep tests/ for a call to that exact function name; flag zero-hit functions. |
| REPORTING_B-1 | AST/grep scanner: flag any np.digitize(X, edges) call where X's defining expression is not first passed through np.isfinite-based filtering in the same function. Pair with a property test injecting one NaN into a binning input and asserting it is excluded, not folded into the extreme bin. |
| TRAINING_COMPOSITE_CORE_A-1 | Fuzz the helper with a strict-subset train_idx plus a synthetic strongly-correlated column and assert it is actually selected; generic scanner for call sites passing a full frame alongside an index-sliced array to the same helper without also slicing the frame. |
| TRAINING_COMPOSITE_CORE_A-2 | Parametrize composite estimator biz-value tests over pandas AND polars fixtures with a non-numeric column, asserting no crash and matching predictions; grep training/composite/** for hasattr(X,'iloc'/'loc') used as sole dispatch without a get_column/is_polars_df branch. |
| TRAINING_COMPOSITE_CORE_A-2b | Same as -2's meta-test; also assert _concat_feature's polars branch is ever actually reachable given the upstream row-subset already downgraded the frame. |
| TRAINING_COMPOSITE_CORE_B-1 | Benchmark predict_quantile wall-time at fixed n_query/n_trees across n_train in {1k,10k,100k,1M} and assert near-linear (not quadratic) scaling; generalize as a scanner flagging predict-path kernels whose innermost loop bound is the training-set length rather than a leaf/bucket size. |
| TRAINING_COMPOSITE_CORE_B-2 | Mock estimator whose fit() accepts sample_weight but raises TypeError for an unrelated reason; assert _fit_final propagates rather than silently retrying unweighted. Repo-wide grep for `except TypeError:` guarding a `.fit(` retry without a signature check. |
| TRAINING_COMPOSITE_ENSEMBLE_ESTIMATOR_TRANSFORMS-1 | For every ADDITIVE_BASE_TRANSFORMS member, predict() and predict_quantile(alpha=0.5) on the same out-of-fit-range rows and assert both stay inside the soft-shrink-bounded envelope; generically, flag any predict_quantile-shaped sibling of a predict function that skips a guard the sibling calls. |
| TRAINING_CORE_A-1 | Property test: build a random correlation-pair graph (nodes=columns, edges=pairs above threshold, including a 3+-clique case), run the drop-selection function, assert every connected component retains >=1 undropped node. Generalizable to any 'drop one side of each correlated pair' helper in the codebase. |
| TRAINING_CORE_B-1 | Integration test asserting metadata['model_schemas'] contains an entry for the first (model, weight) combo when skip_identity_equivalent_pre_pipelines fires; generalized as an early-exit-completeness AST/property scanner (break/return must not skip bookkeeping the normal path performs). |
| TRAINING_LOOSE_A-1 | Scanner: flag any raise ValueError(f"...{arr[i]!r}...") (array indexing inside an f-string passed to raise) sitting inside a try whose except clause does not also explicitly catch IndexError/KeyError -- the message-construction code can itself fail and get silently reclassified as 'no error' by a broad sibling except Exception. |
| TRAINING_LOOSE_B-1 | Grep/AST scanner: flag any except Exception block wrapping a call into a *_njit*/*_numba* module that contains no logger.* call in its body. |
| TRAINING_LOOSE_B-2 | Force id-collision by del + gc.collect() between building two structurally-different indexed_subgroups dicts and assert the cache does not return the first result for the second. |
| TRAINING_LOOSE_C-1 | Fuzz group_ids down to 1 distinct group for every cross_val_predict-based OOF helper and assert a documented sentinel return instead of a raised exception. |
| TRAINING_LOOSE_C-2 | Build create_linear_model('elasticnet', LinearModelConfig(l1_ratio=0.9), use_regression=False) and assert get_params()['l1_ratio'] == 0.9. |
| TRAINING_LOOSE_C-3 | Fit the shim twice with the same X but two different init_score vectors and assert the second booster matches a from-scratch fit on the second init_score. |
| TRAINING_LOOSE_C-4 | Fit with eval_sample_weight=[w], then refit the same instance with the same eval X and eval_sample_weight=None; assert the reused Dataset's weights are all-ones, not the stale w. |
| TRAINING_LOOSE_C-5 | Stamp oof_probs on a model entry with oof_target=None and a differently-valued test_target; assert _calibration_block returns status=skipped rather than a fabricated ok verdict. |
| TRAINING_NEURAL-3 | Attach an LR scheduler to MuonAdamWHybrid, step it, assert the inner _muon/_adamw optimizers' actual lr changed; generalize as a rule that any Optimizer wrapping >1 inner optimizer must expose param_groups as a live property. |
| VOTENRANK-1 | Property test forcing every Shapley value <= 0 (mock score_fn to a constant) and assert ensemble_pred is not all-zero. Generalize: any function with a documented 'falls back to X' branch must be tested to actually return X's own values, not a zero/clipped-derived substitute. |
| X_ARCHITECTURE_API_CONSISTENCY-1 | Regex-extract every 'from mlframe... import ...' line from package docstrings repo-wide and exec() each, failing on ImportError. |
| X_ARCHITECTURE_API_CONSISTENCY-2 | AST scanner grouping BaseEstimator classes by directory/family, flagging fit() signatures missing sample_weight when >50% of siblings have it and no docstring opt-out marker exists. |
| X_CICD_DEPENDENCIES-1 | Grep every .github/workflows/*.yml for `git clone`/`pip install .*@ git+` targeting a first-party sibling repo and assert each is followed by a 40-char commit SHA checkout, not a bare branch clone. |
| X_CICD_DEPENDENCIES-2 | Meta-test parsing the pinned ruff==X.Y.Z in pyproject.toml's dev extra and every rev: under ruff-pre-commit in .pre-commit-config.yaml, asserting equality. |
| X_CICD_DEPENDENCIES-3 | Feed a real Dependabot grouped-PR title sample through the extracted bash regex and assert it correctly classifies as eligible or explicitly-deferred, not silently skipped. |
| X_CICD_DEPENDENCIES-4 | Meta-test parsing every mypy hook's --cache-dir in .pre-commit-config.yaml and asserting none resolve outside the current checkout (no ../ prefix). |
| X_CICD_DEPENDENCIES-5 | Run `pre-commit run --all-files` in a clean env with only the documented CONTRIBUTING.md steps applied and PY_CI_SHARED_DIR unset; assert it does not fail with a config-resolution error. |
| X_ML_CORRECTNESS_META-1 | Scanner: flag any getattr(x, 'oof_*', None) fallback to a differently-named *_target/*_probs attribute without an explicit row-count/index-identity check between the two sources. |
| X_TEST_SUITE_ARCHITECTURE-1 | AST scanner flagging any try block with a top-level assert whose matching except handler catches Exception/BaseException (or bare except) without re-raising -- the assertion's failure path is provably unreachable. |
| X_TEST_SUITE_ARCHITECTURE-2 | AST scanner flagging test_* functions whose only assertion(s) are nested inside an `if hasattr(x, ...):` block with no else -- same vacuous-guard shape as the already-fixed stopfile-callback F7 finding from the 2026-07-21 cycle, recurring in a new subsystem. |
| CALIBRATION-3 | Call with floor=np.ones(k-1) and floor=np.ones(k+1) and assert a clear ValueError, not IndexError. |
| CALIBRATION-4 | Monkeypatch build_reliability_overlay_spec to raise, call pick_best_calibrator(..., emit_plot=True), assert it still returns a normal result with plot_path=None. |
| CALIBRATION-5 | Bit-identity test comparing the vectorized draw against the current per-row loop across several (n, n_bootstrap, seed) combinations, plus a wall-time bench at n_bootstrap=1000. |
| COMPETITION_EVALUATION-5 | Parametrized test calling every public rank_*/*_scan function in evaluation/ with its list-typed input set to [] and asserting a clean empty-DataFrame return, never a KeyError. |
| COMPETITION_EVALUATION-6 | Same meta-test as COMPETITION_EVALUATION-5, applied to subpopulation_ratio_drift_check with zero-row train_df/test_df. |
| COMPETITION_EVALUATION-7 | Same meta-test as COMPETITION_EVALUATION-5, applied to rank_subpopulation_drift_severity with subgroup_cols=[]. |
| COMPETITION_EVALUATION-8 | Given 4 independent occurrences, add a code_audit scanner rule flagging pd.DataFrame(rows) immediately followed by .sort_values(...) without an intervening 'if not rows:' guard. |
| COMPETITION_EVALUATION-9 | Differential test comparing bootstrap_auc_brier_ll_ece_batch and bootstrap_metrics on a rare-positive, small-n, unstratified fixture chosen to produce a high single-class-resample rate; assert both emit a comparable failure-rate warning. |
| CORE_INFRA_MISC-1 | AST/grep scanner flagging arr[:-x] slices where x is a parameter with no proven-positive guard earlier in the function. |
| CORE_INFRA_MISC-2 | Call every *_available()/*_supported() capability-probe function in the codebase with a deliberately-unfitted estimator and assert it returns bool rather than raising. |
| CORE_INFRA_MISC-3 | Fuzz every public function taking an integer k/window/neighbours parameter with {-1, 0} and assert a clear ValueError, never silent NaN/Inf output. |
| CORE_INFRA_MISC-4 | Regression test calling visualize_prediction_vs_truth with a length-1 samples tuple and asserting no exception. |
| FE_ROOT_A-6 | Construct a synthetic new_df with higher average log-likelihood than train and assert the diagnostic can still fire for a large shift |
| FE_ROOT_A-7 | Simulate ImportError/AttributeError on pyutilz.performance.kernel_tuning.registry.kernel_tuner and assert `import mlframe.feature_engineering._recursion_autotune` does not raise |
| FE_ROOT_B-10 | Series with one NaN far past K; assert fractional-diff output for affected rows is NaN (once fixed) or is explicitly documented as zero-substituted. |
| FE_ROOT_B-7 | Call compute_numaggs(np.array([1.0]), return_float32=True) and assert the result is an np.float32 ndarray, matching normal-length input. |
| FE_ROOT_B-8 | Call with n_grid=1 and assert ValueError instead of silent NaN/garbage output. |
| FE_ROOT_B-9 | Perf bench: compact-kernel NW at large n with a narrow bandwidth should show a measured speedup once windowed. |
| FE_TRANSFORMER_A-4 | Fuzz every compute_*_features function with n_train sized just below its largest internal k/n_neighbors parameter and assert graceful degradation, not a raise. |
| FE_TRANSFORMER_A-5 | Call every compute_*_features function accepting a tuple-shaped parameter with a non-default-length tuple and assert output column count/names track the actual parameter, not the default. |
| FE_TRANSFORMER_A-6 | Grep module docstrings for 'OOF' and cross-check the described code path actually contains a KFold/cross_val_predict call. |
| FE_TRANSFORMER_B-3 | Scanner: grep every NearestNeighbors(...).fit(A) followed by .kneighbors(A) on the same array; each must explicitly handle/drop the self-match or document why not. |
| FE_TRANSFORMER_B-4 | Property test with max_bins_per_feat forced above 100 asserting no two distinct (bin0,bin1) pairs collide; scanner for manual col_a*<const>+col_b radix encodings not derived from actual column cardinality. |
| FE_TRANSFORMER_B-5 | AST/token near-duplicate detector across _fit_*baseline* helpers flagging >85% body-similarity clusters of 3+; diff their except/fallback branches specifically for log-level and fallback-value drift. |
| FS_BENCHMARKS_A-2 | Grep/AST scanner: flag any np.random.{randint,rand,randn,choice,shuffle,permutation} call in a file that elsewhere also constructs a seeded np.random.default_rng(...) Generator -- mixing legacy global RNG with a seeded Generator in one reproducibility-sensitive script is the bug pattern. |
| FS_BENCHMARKS_B-2 | Grep for any()/all() applied directly to dict.values() in a function that elsewhere filters the same dict with `if v is not None`; flag as a truthy-vs-None mismatch. |
| FS_BENCHMARKS_C-2 | For every kernel_name registered via the new kernel_tuner(...) registry, grep for any other function calling KernelTuningCache.update("<same name>", ...) directly; flag dual writers to one cache key. |
| FS_BENCHMARKS_C-3 | Grep scanner for sys.path.insert/append calls with an absolute drive-lettered or home-directory path literal. |
| FS_BORUTA_ROOT-2 | Property test: for functions indexing a precomputed step/grid array from a nonzero literal start offset, fuzz tiny input sizes (n=0,1,2,3) and assert no IndexError and no deviation from the documented starting fraction. |
| FS_BORUTA_ROOT-3 | Grep-based scanner for '/ len(<expr>)' not preceded by a truthiness/length guard in the same function; fuzz each such function with an empty collection as the divisor source. |
| FS_WRAPPERS-4 | Call every top-level RFECV mode flag (stability_selection=True, multioutput_strategy=..., default) with estimator=None, estimators=None and assert the identical ValueError message fires in every mode. |
| FS_WRAPPERS-5 | Regression test: call .transform() twice on the SAME caller-owned polars DataFrame and assert byte-identical results both times, to catch a future polars version where self_destruct becomes destructive here. |
| FS_WRAPPERS-6 | Pass a model whose .score() raises on the permuted single-column X and assert _conditional_permutation_importance returns a NaN entry instead of propagating the exception. |
| METRICS-5 | Unit test asserting macro avg does NOT deflate when a declared class is absent, pinning the documented (and actual) contract. |
| METRICS-6 | CI check compiling every @njit(parallel=True) function under warnings-as-errors and asserting no NumbaPerformanceWarning fires on a representative call. |
| METRICS-7 | Lint rule: any public optimal_*/fit_*/*_cutpoints/*_threshold function that both consumes y_true and returns a fitted value used to score new data must contain 'holdout' in its docstring. |
| MODELS-15 | Extend the existing test_f5_package_has_curated_all meta-test to assert every name in each curated __all__ resolves to an object whose __module__ starts with 'mlframe'. |
| MODELS-4 | Call check_rules(params, allow_if_values_or={(): [{'x': 1}]}) and assert it does not raise NameError. |
| MODELS-5 | AST/grep unused-parameter scanner over public function signatures flagging parameters never referenced as a Name node in the body. |
| MODELS-6 | Build a DMatrix with a sentinel that does not round-trip exactly through float32, run the objective, and assert cells are still correctly masked or a clear error is raised. |
| MODELS-7 | Build an array with one exact 0.0 cell and a separate 1e-150 cell; assert combine_probs(flavour='qube') doesn't lose precision on the tiny-positive cell. |
| MODELS-8 | Call the function with an explicit kwargs dict, then assert the dict is unchanged after the call (compare to a pre-call snapshot). |
| PREPROCESSING-10 | Test calling clusterize(show_plot=True) asserting a Figure object (or saved file) is actually produced/returned, not silently discarded. |
| PREPROCESSING-7 | Regression test asserting apply_outlier_policy(train,...) and apply_outlier_policy(test,...) called independently produce different cap bounds when train/test distributions differ. |
| PREPROCESSING-8 | Test asserting every public fit_X/apply_X pair defined in a preprocessing/*.py module's own __all__ is also reachable from mlframe.preprocessing (package-level __all__ superset check). |
| PREPROCESSING-9 | Unit test calling _get_nunique with a 3-element skip_vals on a float array asserting either correct behavior or a clear ValueError, not a silently wrong count. |
| REPORTING_A-5 | Grep every render_*_diagnostic/render_*_from_suite for a seed param, then grep its body for seed= passed to the composer call; flag accepted-but-unused seed params. |
| REPORTING_A-6 | Grep-based duplicate-constant scanner: flag any module-level NAME=<literal> whose exact (name, literal) pair also occurs in a sibling file within the same package with no import relationship between them. |
| REPORTING_A-7 | Call render_multi_target_panels with target_type='multiclass_classification' and a 2-column probs (below the >=3 classes guard); assert a warning is logged, not just that the return is None. |
| REPORTING_B-2 | Parametrized test calling render_split_error_diagnostics with n > DIAG_ROW_CAP and timestamps supplied; assert the returned worst_k_table has a non-null timestamp column. Generalize to every optional kwarg (ids, feature_importances) threaded through a subsample branch in this dispatcher. |
| REPORTING_B-3 | Generic AST checker for `A or B if C else D` (no parens) anywhere in the repo. Plus a cross-backend parity test comparing both renderers' resolved bar width across bin_centers length in {0,1,2,N} x bin_width in {None, explicit}. |
| REPORTING_B-4 | Grep scanner flagging any nested double for-loop under reporting/ whose body calls auto_text_color( -- require auto_text_colors_batch instead. |
| REPORTING_B-5 | Property test: force the ImportError fallback path, feed a matrix where exactly one strongly-discriminating column has a single NaN, assert its surrogate importance is not zero. |
| TRAINING_COMPOSITE_CORE_A-3 | Parametrized test forcing the no-pandas branch (monkeypatch _HAVE_PANDAS=False) and asserting NaN labels map to -1 identically to the pandas path. |
| TRAINING_COMPOSITE_CORE_A-4 | Perturb only validation-fold values and assert the reported per-fold cv_gain is unaffected under a properly fold-local imputation; generic scanner for a summary statistic computed outside a CV split loop but later indexed by [tr]/[va] inside it. |
| TRAINING_COMPOSITE_CORE_A-5 | A memory-bound regression test (tracemalloc) on a large synthetic n * y_grid combination; generic scanner for triple-broadcast [..., None]/[None, ...] arrays whose 3 axes are all caller-controlled sizes with no documented bound. |
| TRAINING_COMPOSITE_CORE_B-3 | Fit CompositeOrRawStacker on n=0 and n=1 rows and assert a clear, actionable error rather than sklearn's internal KFold traceback; generalize as a fuzz sweep of every composite estimator at n in {0,1}. |
| TRAINING_COMPOSITE_CORE_B-4 | Bench _compute_offsets against a vectorized groupby baseline at n=1M rows and assert the gap stays within a small documented multiplier; doc-lint scanner flagging 'vectorised'/'one pass' docstring claims over a function body containing a bare for-loop over .tolist(). |
| TRAINING_COMPOSITE_CORE_B-5 | Bench calibrate_venn_abers wall-time vs n_cal at {1k,5k,20k,50k} with continuous scores and assert sub-quadratic scaling, or a regression test pinning a documented g-threshold warning. |
| TRAINING_COMPOSITE_CORE_B-7 | Unit test calling melt_to_long_gbm_features with NaN-containing X and asserting no unexpected NaN in '_count' (or context columns); repo-wide scanner for .groupby([...]).transform( sites without an explicit dropna= where the grouper key can be NaN. |
| TRAINING_COMPOSITE_DISCOVERY-1 | Fuzz any `size_guard = n // k` pattern feeding np.percentile/np.median/np.quantile with n in {0,1,2} and an all-failing inner computation; flag guards that admit size==0. |
| TRAINING_COMPOSITE_DISCOVERY-2 | Grep for module-level mutable caches where a Lock/RLock guards some access sites but not others in the same file; flag any unlocked .get()/[...] read on a cache with a sibling locked write. |
| TRAINING_COMPOSITE_DISCOVERY-3 | Reachability scanner: for every non-__init__, non-test module in a package, flag it if every cross-reference to its public symbols resolves only to __init__.py re-exports and tests/. |
| TRAINING_COMPOSITE_ENSEMBLE_ESTIMATOR_TRANSFORMS-2 | For every pair of registry entries whose forward/inverse function objects are identity-equal, assert identical membership in every transform-name-keyed capability/guard set. |
| TRAINING_CORE_A-3 | Perf regression test: multi-target suite (K>=3 targets sharing feature columns) with cb/lgb; count-patch the Pool/Dataset constructor and assert call count is O(1) in K, not O(K). |
| TRAINING_CORE_B-3 | AST-based orphan-symbol scanner over src/ flagging any module-level function/class with zero call sites outside its own definition file and no test import -- a strong signal of abandoned refactors. |
| TRAINING_FEATURE_HANDLING_TARGETS-1 | AST scanner: for any f/f_batch sibling pair, flag when the batch variant's keyword-only param set is a strict subset of the singular variant's. |
| TRAINING_FEATURE_HANDLING_TARGETS-2 | Static call-graph scanner: flag any call site that doesn't forward an in-scope sample_weight variable to a callee whose signature declares one. |
| TRAINING_FEATURE_HANDLING_TARGETS-3 | Parametrized test asserting ax.get_ylim() actually contains the data range for a regression TemporalAuditResult with large target_rate values. |
| TRAINING_FEATURE_HANDLING_TARGETS-5 | Cross-consistency test: for every is_X/get_X registry pair, assert every literal in is_X's alias set resolves via get_X without an unknown-model fallback. |
| TRAINING_LOOSE_A-2 | Doc-consistency lint: grep docstrings claiming 'all three splits' / 'train/val/test union' semantics and cross-check the function body actually touches all three named split variables the way described. |
| TRAINING_LOOSE_A-3 | Concurrency regression test: two SimpleNamespace bundles sharing one nested fitted-model object, saved from two threads simultaneously with an injected delay in the strip step; assert the shared object's attributes are correctly restored after both calls and both files deserialize with non-null attributes. |
| TRAINING_LOOSE_B-3 | Test with n_test in the millions under a bounded peak-RSS assertion, or a static check flagging unchunked outer-product broadcasts in this module. |
| TRAINING_LOOSE_B-4 | Monkeypatch the neural import to raise on 1st and 2nd calls, call _get_neural_components twice, and assert the import was only attempted once. |
| TRAINING_LOOSE_C-10 | Call _probe_available_memory_bytes(cuda_available=False) then cuda_available=True with different monkeypatched probe values and assert the second call returns the GPU value, not the cached CPU value. |
| TRAINING_LOOSE_C-6 | Call compute_feature_distribution_drift(..., feature_names=<subset>) and assert every key in categorical_psi['per_feature'] is a member of the subset. |
| TRAINING_LOOSE_C-7 | Pass a binary target split that is 30% NaN and assert the summary surfaces the missing count rather than silently folding NaN into the negative bucket. |
| TRAINING_LOOSE_C-8 | Call post_calibrate_model with two different calib_set_size values on the multi-output path and assert either the outputs differ or a DeprecationWarning fires. |
| TRAINING_LOOSE_C-9 | Call iter_extra_metrics with empty y_true/probs_NK/preds_NK and assert no crash occurs. |
| TRAINING_NEURAL-4 | Build field_groups covering only a subset of columns and assert fit() raises rather than silently training with reduced feature coverage. |
| TRAINING_NEURAL-5 | Assert build_keras_mlp(num_layers=k) produces exactly k hidden Dense layers of the configured width. |
| TRAINING_NEURAL-6 | Fit RecurrentClassifierWrapper with an un-named string column present and use_learnable_cat_embeddings=True; assert it factorizes instead of raising ValueError. |
| VOTENRANK-14 | Call get_tracker_table against a synthetic dirpath containing a name like 'my_model_cola_dev_0' (task with an embedded underscore) and assert a clear, named error or correct parsing rather than a bare unpacking ValueError. |
| VOTENRANK-2 | AST/grep meta-test flagging any expression recomputed inside a loop bounded by an n_iterations/n_permutations-style parameter that is identical to an expression already computed once in the caller and not passed through. |
| VOTENRANK-3 | Call shapley_model_values/shapley_blend with an integer-coded 3-class y and no explicit score_fn; assert it raises rather than returning a garbage score. Meta-test: grep default score_fn/metric_fn implementations branching on len(np.unique(y))==2 with no explicit else-raise for higher cardinality. |
| VOTENRANK-4 | Construct a query row with mean_dist far past the underflow threshold for the given similarity_scale and assert region_similarity_weights(...).sum(axis=1) is still 1.0. |
| VOTENRANK-5 | Include a constant-prediction model in oof_preds with a below-best individual score; assert it appears in (or is loudly, explicitly skipped from) diversity_ablation_report's output. |
| VOTENRANK-6 | Pass one oof_preds entry with an extra trailing axis (n,1) alongside otherwise-valid (n,) arrays and assert a clear raise. Generalize as a scanner rule across constrained_weight_blend/geometric_weight_blend/dual_optimizer_weight_blend/adversarial_stochastic_blend, none of which shape-validate their stacked prediction arrays either. |
| VOTENRANK-7 | Build a Leaderboard with exactly 1 model row and 2+ tasks, call minimax_election(), assert it returns [the_one_model]. Extend the existing F14 hand-computable election sweep to n_models=1 across every *_election method. |
| X_ARCHITECTURE_API_CONSISTENCY-3 | Scan BaseEstimator __init__ params for RNG-like names (seed/rng/random_seed) differing from random_state when 3+ siblings in the same family already use random_state. |
| X_ARCHITECTURE_API_CONSISTENCY-4 | Grep n_estimators on classes without a fitted estimators_/estimator_ list and cross-check docstring for 'epoch' language instead of 'tree'/'base learner'. |
| X_CICD_DEPENDENCIES-6 | Scan scripts/CI for pytest-xdist worker-count formulas and assert the divisor matches the documented policy value. |
| X_CICD_DEPENDENCIES-7 | Meta-test asserting at most one blocking hook per (tool, target-scope) pair in .pre-commit-config.yaml. |
| X_ML_CORRECTNESS_META-2 | Property test: for any module exposing multiple named method= encoder/estimator variants, assert distinctly-named methods diverge on at least one non-trivial synthetic input (target_mean vs target_james_stein currently do not). |
| X_OSS_HYGIENE_PACKAGING-1 | Extract the sklearn-version matrix from sklearn-matrix-ci.yml and assert every 'scikit-learn A through B' prose claim in README/CONTRIBUTING matches it exactly. |
| X_OSS_HYGIENE_PACKAGING-2 | Parse CHANGELOG.md's link-reference footer for every releases/tag/vX.Y.Z and compare/vA...vB URL and verify via `gh api repos/<owner>/<repo>/tags` that each referenced tag exists remotely. |
| X_OSS_HYGIENE_PACKAGING-3 | List every src/mlframe/<name>/__init__.py top-level subpackage and assert README.md's Modules table has a matching row (or an explicit documented exclusion). |
| X_OSS_HYGIENE_PACKAGING-4 | Flag any doc paragraph asserting a negation ('still unfixed', 'no protection against X') for a mechanism that a '~~claim~~ -- shipped' marker elsewhere in the SAME file already marks resolved. |
| X_OSS_HYGIENE_PACKAGING-5 | Generalized stale-path scanner: verify every backtick-quoted path/to/file.py reference in docs resolves to a real file in the repo. |
| X_OSS_HYGIENE_PACKAGING-6 | Same contradiction scanner as X_OSS_HYGIENE_PACKAGING-4, applied specifically to 'Out-of-scope'/'Not yet done' list items vs the rest of the same doc. |
| X_OSS_HYGIENE_PACKAGING-7 | Extract 'logging level for `X`' strings from docs and verify some logging.getLogger(__name__) call site's resolved module path equals or is a parent of X. |
| X_OSS_HYGIENE_PACKAGING-8 | Execute (or importlib-resolve) every `python -m <module>` / `python <path>.py` snippet found in docs and fail the check on ModuleNotFoundError / missing file. |
| X_SECURITY_ROBUSTNESS-3 | A duplication scanner flagging near-identical os.path.commonpath([abs_root, ...]) blocks appearing in more than one file, run specifically on files that also call joblib.load/dill.load/pickle.load. |
| X_SECURITY_ROBUSTNESS-4 | Raise a canary-credential-bearing exception from a mocked mlflow.start_run, let it propagate out of get_or_create_mlflow_run, and assert the canary is absent from the propagated exception's str(), not just from the log record. |
| X_SECURITY_ROBUSTNESS-5 | A generic fuzz/property test applied to every fit-boundary transformer in preprocessing/: fit on an all-NaN column and a zero-row DataFrame, assert either a clean fit or a ValueError/TypeError naming the column, never a raw IndexError/KeyError. |
| X_TEST_SUITE_ARCHITECTURE-3 | Same AST scanner as X_TEST_SUITE_ARCHITECTURE-1, extended to also flag except bodies calling pytest.skip(...) around an assert, not just except: pass. |
| X_TEST_SUITE_ARCHITECTURE-4 | Same AST scanner as X_TEST_SUITE_ARCHITECTURE-1/-3. |
| X_TEST_SUITE_ARCHITECTURE-5 | A soft test_meta ratchet counting local constant/categorical-column fixture-construction sites vs. mlframe.testing.parametric imports, alerting if the ratio doesn't improve release over release. |
| X_TEST_SUITE_ARCHITECTURE-6 | A generic test_meta sensor asserting every def main(argv...) found under src/mlframe/**/__init__.py or cli*.py has >=1 real call site under tests/, catching this CLI-entry-point-shipped-untested class generically. |
| CALIBRATION-10 | Construct fold thresholds with a negative mean and large spread; assert is_stable is False. |
| CALIBRATION-11 | N/A - documentation-only; if the Dykstra claim is meant literally, a test comparing output against the true weighted-least-squares solution under non-uniform weights would fail and should not be added without fixing the algorithm. |
| CALIBRATION-12 | Grep-based scanner flagging comments matching Wave \d+\|iter\d{3,}\|P[0-3]-\d+ in source files, centralized in the code_audit scanner suite. |
| CALIBRATION-13 | Grep-based scanner flagging " -- " inside docstrings/comments, run as a pre-commit hygiene check limited to touched files. |
| CALIBRATION-6 | AST scanner flagging self.<name> = ... assignments in a method with no matching class-level annotation anywhere in the class body. |
| CALIBRATION-7 | Grep-based scanner flagging module-level variables whose only justification is enabling a source-inspection test. |
| CALIBRATION-8 | AST scanner flagging def f(param: dict = <module-level-name>) where the default Name references a mutable module-level literal. |
| CALIBRATION-9 | Call estimate_calibration_quality_binned(y, p, nbins=0) and nbins=-1, assert ValueError not ZeroDivisionError. |
| COMPETITION_EVALUATION-10 | Add --doctest-modules (or xdoctest) to the OSS-hygiene lint pass so a malformed doctest fails CI instead of bit-rotting. |
| COMPETITION_EVALUATION-11 | Grep-based hygiene rule flagging '^\s*#\s*(print\|logger\.\w+)\(' lines outside clearly-marked docstring example blocks. |
| CORE_INFRA_MISC-10 | Same fuzz/property meta-test as CORE_INFRA_MISC-3: sweep window/size-like integer params through {-1, 0} and assert a clear error, never silent NaN. |
| CORE_INFRA_MISC-11 | N/A (pure redundant-computation nit); low value to build a dedicated scanner for this specific pattern. |
| CORE_INFRA_MISC-5 | Simulate the NVRTC-probe-failure path and assert len(mlframe.gpu_disable_errors) == 1. |
| CORE_INFRA_MISC-6 | Static dead-code scanner (vulture or a small AST pass) flagging unreachable code after an unconditionally-returning loop body. |
| CORE_INFRA_MISC-7 | Grep-based hygiene scanner flagging any docstring that is exactly/starts with TODO/FIXME while the symbol has a matching test file by naming convention. |
| CORE_INFRA_MISC-8 | Construct a y with exactly classes.size members per rare class so the true 0.25-fraction fold is too small to stratify; call with test_size=None and assert graceful degradation rather than a raised error from inside train_test_split. |
| CORE_INFRA_MISC-9 | Monkeypatch a polars DataFrame's to_pandas() to raise, call to_pandas_or_array, and assert a WARNING (not just DEBUG) is logged. |
| FE_ROOT_A-10 | N/A (naming-only); could be caught by a linter flagging 'nonzero'-named variables built from isnan/notna rather than != 0 |
| FE_ROOT_A-8 | Grep-based CI lint forbidding 'Wave [0-9]+' / bare date-stamp patterns in source comments outside the CLAUDE.md-sanctioned bench-attempt-rejected note format |
| FE_ROOT_A-9 | Heuristic checker flagging comments referencing 'branch below'/'the following if' whose next statement is not an if/match |
| FE_ROOT_B-11 | Repo-wide scanner for module-private functions with zero call sites. |
| FE_ROOT_B-12 | N/A (naming-only). |
| FE_ROOT_B-13 | N/A (micro-optimization, no behavior change). |
| FE_ROOT_B-14 | Call with shift=-3 and assert either a raised error or an explicitly documented no-op. |
| FE_ROOT_B-15 | Repo-wide grep for hardcoded personal absolute paths (e.g. C:/Users/Admin) outside this one file. |
| FE_ROOT_B-16 | N/A (doc hygiene). |
| FE_ROOT_B-17 | N/A (benchmark-script-only, not production-facing). |
| FE_ROOT_B-18 | A fallback tier introducing a column also present in right_value_cols; assert a clear ValueError instead of an opaque construction failure. |
| FE_ROOT_B-19 | Call knn_within_bucket_aggregate with agg_fns=('min',) and confirm behavior matches the documented contract. |
| FE_ROOT_B-20 | Synthetic dataset where >90% own-group density exceeds the k*4 cap; assert a warning is logged rather than silent NaN-padding. |
| FE_TRANSFORMER_A-10 | Static scanner: for every compute_*_features(..., standardize=True, ...) signature in transformer/, verify the parameter name is referenced in the function body. |
| FE_TRANSFORMER_A-11 | Dead-code scanner over every _*_shared.py consolidation module: grep the repo for real call sites (not comments) and flag zero-caller functions. |
| FE_TRANSFORMER_A-12 | Static always-true/always-false guard linter flagging a condition duplicating an already-established invariant from an earlier return. |
| FE_TRANSFORMER_A-13 | Grep for numpy-module attribute assignment inside function bodies (not module top-level) across the package. |
| FE_TRANSFORMER_A-14 | None needed beyond docstring-consistency review. |
| FE_TRANSFORMER_A-7 | Regex-extract '= N feature' docstring claims and assert the sum of the itemized preceding terms equals N (and equals the code's n_features constant). |
| FE_TRANSFORMER_A-8 | Same as FE_TRANSFORMER_A-7. |
| FE_TRANSFORMER_A-9 | Assert every module docstring's itemized feature groups have a corresponding column-name pattern actually produced by _make_df. |
| FE_TRANSFORMER_B-6 | Feed a heavily-tied/discrete regression target and assert no stratum's output is silently (0.0, 0.0) for every row; scanner for np.quantile(y, linspace(...)) calls not followed by strictly-increasing-edge protection. |
| FE_TRANSFORMER_B-7 | Scanner comparing single-class guards across sibling branches of the same function/module for asymmetric extreme-case coverage. |
| FE_TRANSFORMER_B-8 | Edge-case test with a 1-row training fold asserting a clear error or documented non-extreme sentinel, not a silent huge value; scanner for negative-index-capable slices derived from min(k, n-1) computations with no lower bound on n. |
| FS_BENCHMARKS_A-3 | Grep/AST rule: flag any dict literal where every key's value textually equals its key, used only via `.get(x, x)` -- such self-mapping dicts are always removable dead code. |
| FS_BENCHMARKS_B-3 | Ruff B018 / unused-function lint swept across _benchmarks/; an AST check for top-level functions defined but never referenced. |
| FS_BENCHMARKS_B-4 | Enable ruff B018 (useless-expression) across _benchmarks/. |
| FS_BENCHMARKS_B-5 | Same ruff B018 sweep as FS_BENCHMARKS_B-4. |
| FS_BENCHMARKS_B-6 | Same ruff B018 sweep as FS_BENCHMARKS_B-4. |
| FS_BENCHMARKS_B-7 | Same ruff B018 sweep as FS_BENCHMARKS_B-4. |
| FS_BENCHMARKS_B-8 | Grep _benchmarks/**/*.py for np\.random\.RandomState\( and flag any hit outside an explicitly-justified legacy-API context. |
| FS_BENCHMARKS_B-9 | Grep for sys.path.insert inside fs_quality/ (a package whose other members already import cleanly) and flag the odd one out. |
| FS_BENCHMARKS_C-10 | AST-similarity scanner across sibling files in one package: hash small function bodies and flag near-duplicate same-named functions defined in multiple files. |
| FS_BENCHMARKS_C-4 | Regex scanner over first-party comments for #.*\bWAVE\s*\d+\b (case-insensitive), wired into the code_audit comment-hygiene checker. |
| FS_BENCHMARKS_C-5 | Docstring/signature cross-checker: regex-extract numeric 'default=' claims from docstrings and diff against inspect.signature defaults of the same-named parameter. |
| FS_BENCHMARKS_C-6 | Doc-vs-CLI checker: extract --flag tokens from a script's 'Run::' docstring block and diff against parser.add_argument calls in the same file. |
| FS_BENCHMARKS_C-7 | Doc-vs-filename checker: verify any .py token in a 'run this script' docstring block matches os.path.basename(__file__). |
| FS_BENCHMARKS_C-8 | Diff the docstring's enumerated subcommand names against sub.add_parser(...) calls in the same file; flag any subparser missing from the docstring. |
| FS_BENCHMARKS_C-9 | Grep scanner: flag id(...) used as a dict/set key anywhere in the codebase for manual object-lifetime review. |
| FS_BORUTA_ROOT-4 | AST checker: flag a for-loop immediately followed by a conditional block referencing the loop variable and a pre-loop-initialized comparison variable that is updated nowhere else in the loop body. |
| FS_BORUTA_ROOT-5 | Regression test: call TentativeRoughFix() on a fixture with a nonempty tentative set and assert len(bs.tentative) == 0 afterward. |
| FS_BORUTA_ROOT-6 | Fuzz pass: call every public FS entrypoint taking a Sequence of folds/candidates with an empty sequence and assert a clear ValueError naming the parameter, not a raw library exception. |
| FS_WRAPPERS-7 | Parametrized test over the 4 combinations of (X_estimator is None/not None) x (col_pos is None/not None) with string features_indices; assert a clear error for the unsupported combination. |
| FS_WRAPPERS-8 | Concurrency probe: instrument set_params with a call counter/lock-contention check and assert it is called once per outer iteration rather than once per fold per thread. |
| FS_WRAPPERS-9 | Unit test with swap_top_k exceeding the number of available drop-candidates; assert the logged/returned attempted-pair count matches the actual loop iteration count. |
| METRICS-10 | Grep sweep asserting every metric that silently skips rows on a documented undefined-at-0 condition also emits a matching warnings.warn call. |
| METRICS-11 | Unit test with an out-of-range label asserting accuracy and weighted_averages use a consistent, documented denominator. |
| METRICS-12 | Repo-wide grep gate for em-dash/en-dash code points in tracked .py files, advisory pre-commit/CI check. |
| METRICS-13 | none beyond human/LLM comment review. |
| METRICS-14 | Direct unit test comparing the multiclass log-loss branch of compute_all_metrics against sklearn.metrics.log_loss on random multiclass data. |
| METRICS-8 | none beyond doc-linkage review. |
| METRICS-9 | Property test feeding an out-of-range label to every ordinal/multiclass metric and asserting a raise or a documented ignore contract. |
| MODELS-10 | Monkeypatch generate_valid_candidates to return [] and assert create_ctr_params raises a clear error instead of StopIteration. |
| MODELS-11 | AST scanner flagging every except Exception/bare except whose only side effect is logger.debug(...), cross-referenced against sibling handlers in the same file that log at WARNING. |
| MODELS-12 | Push the same array objects through _compute_outlier_gate twice, mutating values (not identity) between calls; assert the second call reflects the new content. |
| MODELS-13 | Call with search_space=[1,2,2,3] and known_candidates=[2]; assert the distance array is sane at both index-1 and index-2. |
| MODELS-14 | Call create_ctr_params(params={'loss_function': 'QueryCrossEntropy'}) and assert the CrossEntropy-specific skip does not fire. |
| PREPROCESSING-11 | Grep-based hygiene scanner for docstring references to bare alphanumeric codes (F\d+, #\d+, etc.) not defined anywhere in the same file/module, repo-wide. |
| PREPROCESSING-12 | Grep/AST scanner flagging single-element parenthesized literals passed to a parameter whose name/usage implies a tuple/sequence (e.g. skip_vals=(X) without a trailing comma) across the repo's test/bench scripts. |
| REPORTING_A-10 | Grep/AST scanner over comments and docstrings for characters with ord(c) > 127 outside data-carrying string literals; regression-gate on newly introduced non-ASCII. |
| REPORTING_A-11 | Property test: metrics dict with two nested sub-dicts sharing a key with different float values; assert the documented (not accidental) precedence rule holds. |
| REPORTING_A-8 | mypy --no-implicit-optional in CI; or an AST check for a bare-typed parameter later compared via Is/IsNot None in the function body. |
| REPORTING_A-9 | mypy --disallow-untyped-defs scoped to reporting/ as a CI gate. |
| REPORTING_B-6 | Doc-linter: AST-extract numeric literals in a function body and cross-check against numeric literals quoted near comparison words in its docstring; flag any docstring number with no matching body literal. |
| REPORTING_B-7 | Standard dead-code/simplification lint (ruff SIM rules / vulture-style check) catching a ternary whose both branches are provably equal. |
| REPORTING_B-8 | Property test feeding a datetime64 timestamps array containing one NaT and asserting that row is excluded from the standardized-residual series. |
| REPORTING_B-9 | Property test calling _is_binary_score with an all-NaN y_true and asserting it returns False; generalize into a scanner flagging np.all/np.any over a boolean-filtered subset with no explicit .size guard on the filtered array. |
| TRAINING_COMPOSITE_CORE_A-6 | Grep-based scanner across training/composite/** for more than one distinct pd.factorize/np.unique(return_inverse=True) construction outside the shared helper module. |
| TRAINING_COMPOSITE_CORE_A-7 | Parametrized test calling n_features on a plain 1-D ndarray fixture, asserting it returns 1. |
| TRAINING_COMPOSITE_CORE_A-8 | Test calling set_params(nonexistent_kwarg_xyz=1) on a class bound to sklearn_set_params and assert it raises, mirroring sklearn.base.BaseEstimator.set_params. |
| TRAINING_COMPOSITE_CORE_B-6 | AST scanner flagging LGBMRegressor(/XGBRegressor(/RandomForestRegressor( construction sites in mlframe.training.composite.* with no random_state= keyword. |
| TRAINING_COMPOSITE_CORE_B-8 | Repo-wide grep for `except Exception as ...:` immediately followed by `raise ImportError(`; regression test simulating a transient non-ImportError failure during the guarded import and asserting it is NOT relabeled as 'not installed'. |
| TRAINING_COMPOSITE_DISCOVERY-4 | Same reachability scanner as TRAINING_COMPOSITE_DISCOVERY-3; additionally require every intentionally-standalone module to carry an explicit 'not auto-integrated' docstring marker so the scanner can auto-distinguish intentional from orphaned. |
| TRAINING_COMPOSITE_DISCOVERY-5 | Grep for identically-named module-level globals declared in 2+ sibling files of a carved-out module family; flag any declaration never referenced again in the same file via an AST unused-module-global check. |
| TRAINING_COMPOSITE_ENSEMBLE_ESTIMATOR_TRANSFORMS-3 | AST-detect any registry transform whose inverse body is a pure linear combination of base/t_hat (no nonlinear calls) and flag it if absent from ADDITIVE_BASE_TRANSFORMS. |
| TRAINING_COMPOSITE_ENSEMBLE_ESTIMATOR_TRANSFORMS-4 | AST scanner: flag any module-level constant whose name collides with a same-named constant already imported/defined in a sibling module of the same package, where the local copy has zero references in its own file. |
| TRAINING_CORE_A-2 | Textual doc-consistency checker: flag docstrings containing 'normali[sz]e ... sum to 1' near a weight/probability return whose function body has no matching '/ .sum()'-shaped normalization statement. |
| TRAINING_CORE_B-4 | Regex-based code_audit scanner flagging comment lines matching Wave/iter\d+/CODE-[A-Z0-9-]+/date-stamp patterns, run report-only until the cleanup pass lands. |
| TRAINING_FEATURE_HANDLING_TARGETS-4 | Repo-wide AST grep for truthy checks on numeric params named max_*/min_*/n_*/*_count where 0 is a valid value. |
| TRAINING_FEATURE_HANDLING_TARGETS-6 | Repo-wide AST scan for module-level UPPER_CASE constants assigned once and never read elsewhere in the file/package. |
| TRAINING_FEATURE_HANDLING_TARGETS-7 | Property test asserting the two functions produce identical np.argsort orderings for the same synthetic per-model ranks within one query group. |
| TRAINING_LOOSE_A-4 | Scanner: flag functions receiving a dict parameter that later do param[key]=value / param.setdefault(...) without a preceding copy of that same parameter. |
| TRAINING_LOOSE_A-5 | Property test: for functions with a documented 'at least 1 X' invariant on one axis of a 2-D array, fuzz both axes independently at size 0 and assert ValueError (not silent NaN) in every degenerate case. |
| TRAINING_LOOSE_A-6 | Scanner: for functions with a n_*: int loop-count parameter feeding a for _ in range(n_*) loop whose results are later reduced, check for an explicit >=1 guard near the top. |
| TRAINING_LOOSE_A-7 | Lint rule: flag any module where the same simple top-level binding expression appears more than once at module scope with no intervening use. |
| TRAINING_LOOSE_A-8 | Scanner: flag any ternary of the exact shape 'X if X is not None else None' (or 'X if X else X') anywhere in the codebase -- always a no-op by construction. |
| TRAINING_LOOSE_B-5 | Property test asserting rows with genuinely different rounded feature vectors are never merged, or document and accept the theoretical gap. |
| TRAINING_LOOSE_B-6 | Scan **/_profile_*.py and **/bench_*.py harnesses for cProfile.Profile() usage with no corresponding dump_stats call in the same file. |
| TRAINING_LOOSE_B-7 | A code_audit-style regex/AST scanner flagging date/Wave/iter/Fix-ID patterns in non-test source comments, reported as an aggregate count per file for a tractable repo-wide sweep. |
| TRAINING_LOOSE_C-11 | A repo-wide grep-based linter flagging file:line comment references that no longer resolve to the named symbol's actual definition site. |
| TRAINING_LOOSE_C-12 | A docstring-vs-signature cross-check linter flagging parameter names mentioned in a docstring that aren't in any function signature in the same module. |
| TRAINING_LOOSE_C-13 | Same docstring-vs-code cross-check linter as C-12, extended to numeric literals referenced in docstrings vs the constants actually used. |
| TRAINING_LOOSE_C-14 | Same docstring-vs-code cross-check linter as C-12/C-13. |
| TRAINING_LOOSE_C-15 | Construct SuiteArtefactCache(bytes_limit=0), seed an on-disk entry, call evict_lru(), and assert total_bytes() == 0. |
| TRAINING_NEURAL-7 | Repo-wide scanner: any class exported in a package __all__ with zero production (non-test) construction call sites is a dead/orphaned-API candidate. |
| TRAINING_NEURAL-8 | Doc-consistency grep for cross-file 'see X's docstring for Y' claims with no matching string in X. |
| TRAINING_PIPELINE_MISC-2 | Unit test constructing CBIterationMetricsCallback directly with target_type='learning_to_rank' against a mock ranker (predict present, predict_proba absent) and asserting iteration_metrics_ gets populated. Generalize as a grep scanner for `"<word>" in some_string_var` used as a type dispatch key, cross-checked against the variable's declared Literal/enum domain. |
| TRAINING_PIPELINE_MISC-3 | Microbenchmark comparing list-comprehension vs vectorized form at n=10k/1M asserting bit-identical output plus speedup; generalize as a grep scanner for np.array([... for ... in range(...)]) feeding a downstream numeric array. |
| TRAINING_PIPELINE_MISC-4 | CI lint step: regex for non-ASCII characters scoped to # comment text (excluding string literals/docstring examples) flags accidental foreign-language/encoding leaks. |
| VOTENRANK-10 | N/A (naming-only; manual per-function name/semantics review). |
| VOTENRANK-11 | Call confidence_gated_blend with per_sample_gate_calibration=True reusing the exact blended array as its own calibration set and assert a warning is logged. |
| VOTENRANK-12 | Micro-benchmark regression timing spearman_exp with nan_number in the low thousands before/after a vectorized rewrite; assert not slower and bit-identical NaN placement. |
| VOTENRANK-13 | Call compute_iia against a 2-row table and assert it raises a clear error instead of silently returning (0.0, 0.0, [0]*num_repetitions). |
| VOTENRANK-8 | Repo-wide regex/AST CI gate matching #\s*(F\|WAVE\|FINDING)[-_ ]?\d+\b across every .py file, catching this pattern anywhere it recurs. |
| VOTENRANK-9 | Grep for def _?bootstrap_\w+ whose docstring contains 'not a (true \|real )?bootstrap' -- a self-contradicting name/docstring pair, generically detectable. |
| X_ARCHITECTURE_API_CONSISTENCY-5 | Meta-test asserting hasattr(module, '__all__') across all mlframe subpackage __init__.py files that follow the 'curate explicitly' convention. |
| X_ARCHITECTURE_API_CONSISTENCY-6 | Grep-check that a new BaseEstimator class either has as many self.<param>= lines as __init__ params, or calls store_params_in_object; flag classes matching neither. |
| X_ARCHITECTURE_API_CONSISTENCY-7 | AST scan collecting declared verbose parameter types per subpackage; flag subpackages mixing bool and int among their own public functions. |
| X_ARCHITECTURE_API_CONSISTENCY-8 | N/A - documented design choice; consider a lint rule warning on any new class exceeding ~40 constructor parameters. |
| X_CICD_DEPENDENCIES-8 | None generically automatable; manual stale-comment sweep. |
| X_OSS_HYGIENE_PACKAGING-10 | Extract every 7-10 hex-char backtick-quoted token from docs/*.md and verify via `git cat-file -e` that it resolves to a real commit. |
| X_OSS_HYGIENE_PACKAGING-11 | Same stale-path scanner as X_OSS_HYGIENE_PACKAGING-5, extended to check cited line numbers are <= the current file's line count as a cheap drift signal. |
| X_OSS_HYGIENE_PACKAGING-12 | Same stale-path scanner as X_OSS_HYGIENE_PACKAGING-5, applied to tests/**/*.py references specifically. |
| X_OSS_HYGIENE_PACKAGING-13 | Run the stale-path scanner over source-file docstrings/comments that feed auto-generated docs too, not just the generated markdown, so staleness is caught at the source. |
| X_OSS_HYGIENE_PACKAGING-14 | AST-detect any function computing skew/kurtosis via the raw-power-sum binomial-expansion identity instead of two-pass centered-moment accumulation, as a pyutilz code_audit scanner. |
| X_OSS_HYGIENE_PACKAGING-9 | Parse mkdocs.yml's nav: + not_in_nav: into one covered-path set and assert every *.md/*.ipynb under docs_dir is covered by it. |
| X_SECURITY_ROBUSTNESS-6 | Unit test instantiating LocalDiskBackend directly and calling .write("../../evil", b"x") / .write("..\\evil", b"x"), asserting it raises rather than writing outside root. |
| X_SECURITY_ROBUSTNESS-7 | Call embed_website_to_mlflow(url="http://x", fname="../../evil") and assert it raises instead of writing outside the CWD. |
| X_SECURITY_ROBUSTNESS-8 | None needed; a ruff/flake8 redefined-while-unused style rule already catches this class generically. |
| X_TEST_SUITE_ARCHITECTURE-7 | N/A -- already tracked by the existing whitelist/debt-list mechanism in test_no_inspect_getsource.py; flagged only because this file's whole purpose is regression-proofing this cluster's own findings. |

## Cluster-level meta-test ideas

### fs_benchmarks_a
- Add a package-level smoke test that imports every .py module directly under src/mlframe/feature_selection/_benchmarks/ (non-recursive) and, for shared '_'-prefixed helper modules specifically, asserts every attribute referenced via `module.attr` by a sibling bench_*.py file (found via a simple AST scan of `Attribute` nodes rooted at an `Import`) actually exists on the imported module after plain `import module` -- this would have caught FS_BENCHMARKS_A-1 without ever running the benchmark itself.
- Add a repo-wide grep-based CI lint rule (not just this cluster) that flags any module-level `np.random.{randint,rand,randn,choice,shuffle,permutation}` call co-located in a file that also constructs a seeded `np.random.default_rng(...)`/`np.random.RandomState(seed)` instance -- this is the general shape of FS_BENCHMARKS_A-2 and is cheap to check statically anywhere reproducibility matters (benchmarks, synthetic-data generators, bootstrap/permutation code).

### fs_benchmarks_b
- Repo-wide ruff B018 (useless-expression) enablement, including _benchmarks/, to catch the dead-expression-statement class found 4x in this cluster.
- A generic aliasing checker: for any function that times/repeats a callable against a closured mutable object (DataFrame/ndarray), assert the object's content hash is unchanged after repeated invocation.
- A generic any()/all()-on-dict.values() checker that flags the pattern whenever the same dict is elsewhere filtered with an explicit `is not None` check, since that signals truthy-vs-None confusion.
- A convention-consistency grep (RandomState vs default_rng, sys.path-hack vs package import) run per-package so a lone stylistic outlier among otherwise-uniform siblings gets surfaced automatically.

### fs_benchmarks_c
- Directory-creation-before-write scanner: flag any open(<literal path>, 'a'|'w') at module/function scope with no reachable os.makedirs/Path.mkdir call in the same function.
- Absolute-dev-path-in-sys.path scanner: grep for sys.path.insert/append with a drive-lettered (Windows) or home-directory (Unix) absolute path literal anywhere in the repo.
- Wave/phase/date comment-metadata scanner: regex '#.*\b(WAVE|Wave)\s*\d+\b' over all first-party comments, feeding the existing code_audit comment-hygiene checker.
- Docstring-vs-code drift checker (three sub-checks in one AST+regex pass): (a) numeric 'default=X' claims in a docstring vs the real inspect.signature default of the named parameter; (b) --flag tokens in a 'Run::' usage block vs actual argparse.add_argument calls; (c) any .py filename token in 'run this script' docstring prose vs the real __file__ basename.
- Cross-writer cache-key collision scanner: for every kernel_name/cache-key string registered via the new pyutilz kernel_tuner registry, grep for any other call site writing to KernelTuningCache.update()/cache.update() with the same literal string; flag dual writers as a shadowing hazard.
- Duplicate-helper-across-siblings scanner: within each _benchmarks subpackage, hash small (3-15 line) function bodies and flag near-duplicate same-named functions defined in more than one sibling file — catches incomplete 'consolidated into a shared module' refactors before they drift further.

### fs_boruta_root
- Scanner for np.zeros(...)-initialized 'history' accumulators grown via vstack: verify every read-site treats row 0 consistently (either all strip it or none do).
- Fuzz every public feature_selection function with degenerate inputs (n_rows in {0,1,2}, empty candidate/fold sequences, zero feature columns) and assert a clear ValueError rather than a raw numpy/library exception or IndexError/ZeroDivisionError.
- AST pass over every 'for' loop for trailing sibling blocks that reference the loop variable or loop-scoped accumulators after the loop body ends, as a signal of accidentally-dedented logic.
- Regression test template: after any '*RoughFix'/'*Resolve'/'*Finalize' method that moves items out of a pending/tentative collection, assert the pending collection is empty or otherwise updated to reflect the resolved state.

### fs_wrappers
- Generic AST/grep scanner: flag any function that selects columns via a dtype-agnostic predicate (e.g. `.isna().any()` over ALL columns) but then processes them with a numeric-only operation (`.to_numpy(dtype=float, ...)`, `.astype(float)`) with no is_numeric_dtype/is_bool_dtype guard in between -- this exact 'impute-crashes-on-strings' shape recurs anywhere a 'graceful NaN handling' contract is claimed.
- Generic property test: for every public fit()-like entrypoint with multiple dispatch modes (flags/strategies that early-return to a different code path), assert that every mode enforces the SAME required-parameter validation the default path enforces -- catches validation checks placed after a mode-specific early-return instead of before it.
- Generic grep scanner: find every dtype.kind-based branch gate (e.g. `dtype.kind in "iu"`) that feeds a correctness-relevant validation or classification decision, and cross-check it against sibling classification helpers in the same package that already handle the float-typed-integer-label case, flagging any gate that doesn't.
- Generic perf-doc-compliance checker: for every function whose docstring or a caller's warning message makes a big-O or wall-time claim ('sub-second', 'O(n log n)', etc.), add a benchmark assertion pinning that claim at the documented scale, and flag any nested nested-loop-over-dict-values pattern that would contradict it.
- Generic grep scanner: find every `to_pandas(..., self_destruct=True, ...)` (or any other explicitly-destructive/in-place kwarg) call site and require a matching ownership/ 'is this our own copy' gate in scope, flagging any call site that omits it while a sibling call site in the same module/package has one.

### preprocessing
- Repo-wide AST scanner for sum(x**2) - sum(x)**2/n (or cumsum-based) variance/std-shaped expressions outside a documented numerically-stable two-pass implementation; flag for review against a stable reference on large-offset synthetic data.
- Repo-wide scanner for '.fit'/'.fit_transform' calls whose output array is later indexed by both a CV fold's train_idx and test_idx -- a strong signal of preprocessing statistics leaking across the fold boundary.
- Repo-wide fuzz harness feeding every fit_*/apply_* pair and every 'analyse'/'clean' entry point in preprocessing/ a matrix of degenerate 2-valued columns across Python value types (str, numeric, bool, Decimal, Timestamp, tuple) to catch UnboundLocalError-class exhaustiveness bugs in type-dispatch branches.
- Package __all__ parity check: every fit_X/apply_X pair defined in a submodule's own __all__ must also be reachable from the parent package's __all__.
- Grep for single-element parenthesized tuple literals missing a trailing comma (e.g. `=(0.0)` where a sequence is expected) across test/bench scripts repo-wide -- a recurring silent-typo class.

### reporting_b
- AST/grep scanner for np.digitize/np.searchsorted calls lacking a prior isfinite filter on the same variable within the enclosing function, applied repo-wide (not just reporting/) -- this bug class recurred 3 times in this one cluster alone.
- A cross-backend bit-identity test suite for every chart builder's matplotlib vs plotly renderer output (bar widths, bin counts, per-cell text colors) run as a fixture matrix -- the width bug shows the two backends can silently diverge despite the package's own 'renders identically on either backend' design contract.
- A doc/code numeric-literal consistency checker across reporting/: grep every docstring number near a comparison word and cross-check it against the function's actual body literals.
- A generic 'subsample path silently drops an optional kwarg' regression-test generator for every diagnostics_dispatch entry point with a row cap, parametrized over every optional array argument (timestamps, ids, sample weights).

### training_composite_discovery
- Concurrency-safety linter: any module-level mutable cache guarded by a Lock on some access sites must be guarded on all access sites in the same file (catches TRAINING_COMPOSITE_DISCOVERY-2, reusable across pyutilz's other caches).
- Size-guard-admits-zero fuzz check: for any `x.size >= n // k` pattern feeding a reduction undefined on empty input, fuzz n in {0,1,2,k} with an all-failing inner computation and assert no exception escapes.
- Public-but-unreachable module reachability scanner across the whole mlframe package: flag any non-test, non-__init__ module with zero call-syntax references to its public functions outside its own file and its dedicated test file (would have caught both TRAINING_COMPOSITE_DISCOVERY-3 and -4 in one pass).

### training_loose_c
- A generic OOF-computation fuzz harness that sweeps degenerate group_ids (0, 1, 2 distinct groups) against every cross_val_predict-based helper in the training package and asserts a documented sentinel return instead of a raised exception.
- A cache-consistency property test applicable to every *_shim.py Dataset/DMatrix reuse cache: for each optional per-fit input (label, weight, init_score/base_margin), fit twice with the same X but two different values of that input and assert the second fit's booster matches a from-scratch fit on the second input -- catches 'cache hit forgot to reset auxiliary state X' as a class.
- A docstring/code cross-checker (AST-based) that flags any parameter name mentioned in a function or module docstring that does not appear in that function's signature (or, for module docstrings, any function signature in the module) -- would have caught the loss_recommendation.py / quantile_wrapper.py / mlp_runtime_defaults.py stale-docstring findings mechanically.
- A config-field dead-code checker: for every Pydantic config field, grep whether it is read anywhere outside its own class/tests; flag fields that are only ever assigned/documented but never consumed (would catch evaluation.py's calib_set_size).

### training_pipeline_misc
- Generic scanner: flag any pd.DataFrame(..., index=range(n)) or .reset_index(drop=True) result later consumed by a helper doing df.join(...)/pd.concat([...], axis=1) against a frame whose index provenance differs — the exact bug class behind TRAINING_PIPELINE_MISC-1, and likely recurring anywhere a 'sibling composite-FE module' pattern is copy-pasted across a package.
- Generic scanner: string-substring membership checks (`"x" in some_var`) used to dispatch on a typed/enum-like value should be flagged when the containing module also imports or references a Literal/enum listing more values than the substring set covers (TRAINING_PIPELINE_MISC-2 class).

### x_oss_hygiene_packaging
- Generalized stale-doc-reference scanner: extract every backtick-quoted `path/to/file.py[:N[-M]]` token from docs/**/*.md and README/CONTRIBUTING/CHANGELOG, verify the file exists in the repo (or is an external URL) and that any cited line number is <= the file's current line count; run over source-file docstrings/comments too since auto-generated docs (composite_config_reference.md) inherit their staleness from there.
- Commit-SHA existence checker: extract every backtick-quoted 7-10 hex-char token from docs/CHANGELOG and verify via `git cat-file -e <sha>` (local) or `gh api repos/<owner>/<repo>/tags`+commit lookup (remote, for release-tag links) that it resolves; flag orphaned short-hashes left behind by history rewrites.
- Doc self-contradiction scanner: for every `~~claim~~ — shipped` / `✅ done <claim>` marker in a doc, flag any other paragraph in the same file asserting the negation ("still unfixed", "No protection against X", "TODO", an "Out-of-scope"/"NOT started" list entry) referencing the same named mechanism/flag/class.
- Executable-command doc checker: extract every fenced ```` `python -m <module>` ```` / ```` `python <path>.py` ```` snippet from docs and actually attempt the import/file-existence check (or a dry run), failing the check on ModuleNotFoundError / missing path.
- mkdocs nav-completeness checker: parse mkdocs.yml's `nav:` + `not_in_nav:` into one covered-path set and assert every `*.md`/`*.ipynb` under `docs_dir` is covered (or on an explicit, reasoned exclude list) so no page silently falls outside the stated navigation policy.
- Package-inventory-vs-docs checker: list every `src/mlframe/<name>/__init__.py` top-level subpackage (excluding `_benchmarks`/dunder/private dirs) and assert README.md's Modules table has a matching row for each, so a new subpackage can't silently ship undocumented.
- sklearn/Python-version-claim consistency checker: extract the CI matrix's actual tested version list from `.github/workflows/*matrix*.yml` and assert every prose "tested/supported on A through B" claim in README/CONTRIBUTING matches it exactly.
- Logger-name-in-docs checker: extract every `` `pkg.path.name` `` string preceded by "logger"/"logging level for" in docs and verify some `logging.getLogger(__name__)` call site's resolved module path equals or is a parent of that string.
- Numerical-stability regression scanner (candidate for pyutilz code_audit): AST-detect any function computing skew/kurtosis via the raw-power-sum binomial-expansion identity (sum(x**k) combined algebraically) rather than a two-pass centered-moment accumulation, to catch the exact catastrophic-cancellation bug class documented three times in this repo's history before a fourth instance ships.

### x_test_suite_architecture
- Generic AST scanner: any `try` block containing a top-level `assert` whose matching `except` handler catches `Exception`/`BaseException` (or bare `except:`) without re-raising -- the assertion's failure path is provably unreachable, whether the handler body is `pass` or a misleadingly-worded `pytest.skip(...)`. Found 4 live instances this session (X_TEST_SUITE_ARCHITECTURE-1/-3/-4 plus one already-adequate false positive triaged out); generalizes cleanly into a repo-wide `pyutilz.code_audit` scanner.
- Generic AST scanner: `test_*` functions whose only executed assertion(s) are nested inside an `if hasattr(x, "attr"):` block with no `else` -- the test can execute zero assertions and still pass. Found 2 live instances (X_TEST_SUITE_ARCHITECTURE-2) recurring the same shape as the already-fixed F7 stopfile-callback finding from the prior audit cycle; a dedicated `test_meta/test_no_vacuous_hasattr_guarded_assert.py` (mirroring `test_no_inspect_getsource.py`'s whitelist-ratchet design) would catch both existing debt and prevent new instances.
