# Orthogonal scoring/routing/meta-selection FE

10 findings, 5 proposals.

## Findings

### [P1] bug -- src/mlframe/feature_selection/filters/_orthogonal_ksg_mi_fe.py:358

**The 2026-06-12 'BUG2 FIX' (freeze fit-time basis-preprocess params so transform() replays byte-exactly) was applied to the canonical Layer-21 hybrid_orth_mi_fe_with_recipes and to _orthogonal_triplet_fe.py / _orthogonal_quadruplet_fe.py / _orthogonal_cluster_basis_fe.py / _orthogonal_diff_basis_fe.py, but was never propagated to the rest of the scorer-zoo *_with_recipes builders in this cluster, which still call build_orth_univariate_recipe()/EngineeredRecipe(...) without preprocess_params.**

MRMR.fit() with fe_hybrid_orth_ksg_enable=True (or copula/dcor/hsic/cmim/jmim/tc/routing/adaptive_arity/adaptive_degree/bootstrap/elasticnet/lasso/auto_scorer/ensemble/meta) picks a winning engineered column; the recipe is built WITHOUT preprocess_params, so MRMR.transform() on a row-sliced or distributionally-shifted test/inference frame REFITS the z-score mean/std (or min-max lo/hi) from the transform-time rows instead of replaying the frozen fit-time axis. The docstring for the fix (engineered_recipes/_orth_basis_recipes.py:95-103) documents the measured effect: a ~1e-3 z-score drift that the downstream quantiser turns into a |delta|=1 MI-bin drift on a nested basis column -- i.e. silently different (wrong) feature values at inference time on any row-sliced test set, exactly the bug class the fix targeted. Confirmed missing (grep for 'preprocess_params' returns 0 hits) in: _orth_auto_scorer_fe.py:579, _orthogonal_adaptive_arity_fe.py:684 (plus its arity-2/3/4 EngineeredRecipe/build_orth_quadruplet_cross_recipe calls at 702-710/723-731/745-752, none of which pass preprocess_params_i/j/k/l either), _orthogonal_adaptive_degree_fe.py:364, _orthogonal_bootstrap_mi_fe.py:433, _orthogonal_cmim_fe.py:618, _orthogonal_copula_mi_fe.py:459, _orthogonal_dcor_fe.py:466, _orthogonal_elasticnet_fe.py:344, _orthogonal_hsic_fe.py:651, _orthogonal_jmim_fe.py:519, _orthogonal_ksg_mi_fe.py:358, _orthogonal_lasso_fe.py:437, _orthogonal_routing_fe.py:553, _orthogonal_scorer_auto_fe.py:657, _orthogonal_three_gate_mi_fe.py:704, _orthogonal_total_correlation_fe.py:631. Every one of these functions' own docstring claims 'Recipes are byte-identical to Layer 21' -- a claim that is no longer true post-2026-06-12 since Layer 21's own with_recipes function got the fix and these siblings did not. No replay-parity test exists for any of these 16 call sites (test_orth_cluster_basis_replay_parity.py and test_orth_triplet_quad_replay.py exist only for the already-fixed families; test_fe_replay_provenance_exactness.py has zero references to ksg/copula/dcor/hsic/cmim/jmim/adaptive_arity/adaptive_degree/bootstrap/elasticnet/lasso/routing/three_gate/total_correlation/ensemble).

### [P1] bug -- src/mlframe/feature_selection/filters/_orthogonal_triplet_fe.py:269

**A second, separately-documented label-truncation bug fix ('plain .astype(int64) truncation merges distinct labels and destroys continuous-y signal -- everything in [0,1) collapses to class 0', fixed via the _coerce_y_int64/_coerce_y_classif densify-via-np.unique(return_inverse=True) helper in _orthogonal_cmim_fe.py, _orthogonal_jmim_fe.py, _orthogonal_three_gate_mi_fe.py, _orthogonal_bootstrap_mi_fe.py, _orthogonal_adaptive_arity_fe.py, _orthogonal_adaptive_degree_fe.py, _orthogonal_total_correlation_fe.py) was never propagated to 5 sibling files in the same cluster, which still coerce y via unconditional truncating `.astype(np.int64)`.**

y_arr = np.asarray(y).astype(np.int64) if not np.issubdtype(np.asarray(y).dtype, np.integer) else np.asarray(y, dtype=np.int64) truncates (floors) any non-integer-dtype y with NO cardinality/round-value check, unlike the fixed sibling files' _coerce_y_int64 which densifies via np.unique. If y is a float-encoded classification target whose values are not exact integers (e.g. class codes that drifted through a float32 round-trip, e.g. 2.9999998, or normalized/rescaled labels in [0,1)), distinct classes silently collapse into the same MI/CMI bin, corrupting the engineered-column ranking without any error or warning. Confirmed sites: _orthogonal_triplet_fe.py:269 and :447, _orthogonal_quadruplet_fe.py:284 and :436, _orthogonal_cluster_basis_fe.py:413, _orthogonal_diff_basis_fe.py:307, _orthogonal_routing_fe.py:233.

### [P1] test_gap -- src/mlframe/feature_selection/filters/_orthogonal_ksg_mi_fe.py

**No replay/parity test exists validating that MRMR.transform() reproduces MRMR.fit()'s engineered column values for any of the 16 scorer-zoo *_with_recipes functions identified in the preprocess_params finding above -- the exact test class (test_orth_cluster_basis_replay_parity.py, test_orth_triplet_quad_replay.py) that would have caught the missing-preprocess_params regression exists only for the 4 families that already got the fix.**

A row-sliced-test-vs-full-fit-frame replay-drift regression in ksg/copula/dcor/hsic/cmim/jmim/tc/routing/adaptive_arity/adaptive_degree/bootstrap/elasticnet/lasso/scorer_auto/ensemble/three_gate/meta_scorer would pass the full existing test suite silently, since none of it asserts fit-vs-transform value equality for these families specifically (confirmed via grep: test_fe_replay_provenance_exactness.py has zero references to any of these module/function names).

### [P2] bug -- src/mlframe/feature_selection/filters/_orthogonal_meta_scorer_fe.py:208

**fingerprint_signal()'s per-column Pearson/Spearman correlation probes use bare `except Exception:` (falling back to NaN), which is BROADER than the module's own declared _NUMERIC_ERRORS tuple and contradicts the module docstring's explicit design invariant that 'a genuine programming error (AttributeError, KeyError, NameError, ...) must propagate, not be silently coerced to 0.0 -- that would misroute the scorer by faking absence of signal.'**

If pandas' Series.corr() (or a malformed/mistyped column) raises an unexpected non-numeric exception (e.g. AttributeError from a broken column proxy, or a KeyError from an index-misaligned y_series), the inner `except Exception: r = float("nan")` (line 208-209) and the two dcor_proxy inner excepts (`r_sp`/`r_sym`, lines ~277-284) silently swallow it and feed a NaN into the fingerprint instead of propagating -- exactly the outcome the module's own top-of-file comment says must never happen, and it can silently misroute predict_best_scorer() to the wrong scorer with no diagnostic trace.

### [P2] bug -- src/mlframe/feature_selection/filters/_orthogonal_adaptive_arity_fe.py:412

**_adaptive_arity_mi_resident_block's GPU device-born fallback uses `except Exception: return None` with NO logging at all, unlike its sibling raw_and_product_mi_resident (_gpu_resident_cross_basis.py:262-264) which logs the failure at debug level before falling back to host.**

Correctness is preserved (the caller falls back to the exact host _mi_classif_batch path), but any genuine bug introduced in build_leg_product_matrix_gpu / _resident_mi for the adaptive-arity family (Layer 78) is completely invisible -- not even at DEBUG level -- making a GPU-path regression silently degrade to slower host compute forever with zero breadcrumb to investigate, inconsistent with the sibling triplet/quadruplet scorers' logged fallback.

### [P2] bug -- src/mlframe/feature_selection/filters/_oracle_scorer_select.py:124

**_quality_objective's `except Exception: q = float("nan")` silently swallows any error from unpacking the bake-off closure's `(scorer_name, quality)` output tuple with no logging, so a bug in the benchmark closure silently records NaN quality into the persistent Param-Oracle store instead of surfacing.**

If OracleScorerSelector.benchmark_all_scorers' internal quality-objective closure ever returns a malformed/wrong-shaped output (e.g. a 3-tuple after a future refactor, or an exception object), `_scorer, q = output` raises, is silently caught, and `quality=NaN` gets appended to the on-disk oracle store via oracle.record(). Because ParamOracle's `maximize="quality"` selection presumably treats/sorts NaN unpredictably (NaN comparisons are always False), this could either silently exclude that scorer from ever being recommended or (depending on the sort implementation) corrupt ranking -- with no log line anywhere pointing at the root cause.

### [P2] bug -- src/mlframe/feature_selection/filters/_meta_fe_recommender.py:164

**_fe_structure's per-column cardinality/monotonicity/integrality probes (`card = int(np.unique(vals).size)` at line 163-165, `_is_integral` at 233-236, `_is_monotone` at 242-246) use bare `except Exception:` with no logging, silently degrading to a fallback value on ANY exception including genuine programming errors.**

A bug in np.unique/np.mod/np.diff on a pathological column (e.g. an object-dtype column that slipped past the kind-check, or a memory error) is silently absorbed into a plausible-looking fallback (card = n_finite, False, False) with zero diagnostic trace, so the FE-flag recommendation for that dataset could be systematically wrong (e.g. missing a numeric_decompose/modular recommendation) and there would be no log evidence to debug it from. Docstring explicitly commits to 'Never raises' but doesn't distinguish expected data-shape issues from genuine bugs the way the sibling _orthogonal_meta_scorer_fe.py's _NUMERIC_ERRORS convention (imperfectly) attempts to.

### [P2] gpu_residency -- src/mlframe/feature_selection/filters/_orthogonal_ksg_mi_fe.py

**Design finding (no bug): of the 25 files in this cluster, GPU-resident scoring exists ONLY for the cross-basis product families (_orthogonal_triplet_fe.py, _orthogonal_quadruplet_fe.py, _orthogonal_adaptive_arity_fe.py) via _gpu_resident_cross_basis.py. The rest of the scorer zoo -- KSG, copula, dCor, HSIC, CMIM, JMIM, total-correlation, three-gate OOF, cluster-basis, diff-basis, bootstrap-MI, elasticnet, lasso, adaptive-degree, conditional-routing, meta-scorer/auto-scorer/ensemble/oracle dispatch -- has zero cupy/GPU-resident code path and always runs on host, even under MLFRAME_FE_GPU_STRICT.**

_(no issues found in this cluster for this angle)_

### [P2] bug -- src/mlframe/feature_selection/filters/_oracle_scorer_select.py:100

**_cached_read_rows's staleness key is (store_path, os.path.getmtime(path)); on filesystems/environments with coarse (1-second) mtime resolution, two writes to the oracle store within the same wall-clock second would collide on the same cache key and the second write's rows would not invalidate the cache until a later write bumps the mtime.**

A rapid burst of OracleScorerSelector.observe_scorer()/benchmark_all_scorers() calls within the same second on a coarse-mtime filesystem could leave recommend_scorer() serving stale cached rows (missing the most recent observation) for up to _ROWS_CACHE_MAX=8 distinct (path, mtime) entries' worth of subsequent calls, until a write lands in a different mtime second. Windows NTFS has fine (100ns) resolution so this is unlikely to bite on the primary dev platform, but the code has no defence if run on a coarser filesystem (e.g. some network/container filesystems truncate to 1s).

### [P2] test_gap -- src/mlframe/feature_selection/filters/_meta_fe_recommender.py

**MetaFERecommender's learned-vs-cold-start branching (recommend()'s `_has_confident_history` gate, and the `learned.get(f, rules.get(f, False))` per-flag fallback-merge when the oracle only has PARTIAL history for some flags) has no test exercising the partial-coverage merge path specifically.**

test_meta_fe_recommender.py (biz_val) likely covers the pure cold-start and pure fully-learned cases; a bug in the per-flag merge (`{f: bool(learned.get(f, rules.get(f, False))) for f in ALL_FE_MASTER_FLAGS}` at line 409) that only manifests when the oracle has recorded SOME but not all of the 10 master flags for a fingerprint bucket would not be caught by an all-or-nothing test.

## Proposals

### (coverage_gap) Add replay-parity tests for the 16 scorer-zoo *_with_recipes functions missing preprocess_params freezing

Model after test_orth_cluster_basis_replay_parity.py / test_orth_triplet_quad_replay.py: fit each of hybrid_orth_mi_{ksg,copula,dcor,hsic,cmim,jmim,tc,bootstrap,elasticnet,lasso,adaptive_arity,adaptive_degree,conditional_routing,auto_scorer,ensemble,three_gate,meta}_fe_with_recipes on a synthetic frame, take a random row-slice of the SAME fit frame, replay each recipe via apply_recipe(), and assert byte-equality against the values the fit-time generator produced for those rows. This single test class would both catch the current preprocess_params gap and act as a regression gate for future scorer-zoo additions.

### (coverage_gap) Fix the missing preprocess_params threading itself (root cause, not just add tests)

Each of the 16 affected *_with_recipes functions already parses (basis, degree) back out of the appended column name; add the same 3-line pattern the fixed siblings use: call _evaluate_basis_column(x, chosen_basis, chosen_degree, return_params=True) once per appended column and pass preprocess_params=_pp into build_orth_univariate_recipe (and preprocess_params_i/j/k/l into the pair/triplet/quadruplet builders in _orthogonal_adaptive_arity_fe.py's arity-2/3/4 branches). Per CLAUDE.md 'Enable corrective mechanisms by default', this should be a single pass across all 16 sites rather than gated behind an opt-in flag.

### (coverage_gap) Fix the missing _coerce_y_int64 densify pattern in the 5 remaining truncating call sites

Replace the raw `y_arr = np.asarray(y).astype(np.int64) if not np.issubdtype(...) else ...` pattern in _orthogonal_triplet_fe.py (2 sites), _orthogonal_quadruplet_fe.py (2 sites), _orthogonal_cluster_basis_fe.py, _orthogonal_diff_basis_fe.py, and _orthogonal_routing_fe.py with the shared _coerce_y_int64/_coerce_y_classif densify-via-np.unique helper already used by cmim/jmim/three_gate/bootstrap/adaptive_arity/adaptive_degree/total_correlation, ideally hoisted to one shared helper in a common module (e.g. _orthogonal_univariate_fe) so future FE families import the fixed version instead of copy-pasting the unfixed pattern again.

### (fe_idea) GPU-resident scoring for the scorer zoo beyond cross-basis families

The KSG/copula/dCor/HSIC/CMIM/JMIM/TC scorers currently always run on host even under MLFRAME_FE_GPU_STRICT. Given these are dispatched per-column in the auto-scorer bake-off (_orth_auto_scorer_fe.py) and the ensemble rank-fusion path (_orthogonal_scorer_auto_fe.py), which run ALL scorers on every engineered column, a GPU-resident plug-in-MI-style kernel for at least the batchable ones (plug_in, copula already partially batched via _copula_mi_batch) would reduce the H2D/compute cost of Layer 68/69's O(n_scorers * n_cols) bake-off, mirroring the pattern already proven for cross-basis families.

### (edge_case) Gate the truncating-y and unlogged-except patterns with a repo-wide lint rule

Given this same 'fix landed in some files, not propagated to siblings' pattern occurred TWICE independently in this cluster (preprocess_params freeze, y-truncation densify), consider a lightweight custom check (e.g. a grep-based CI gate) that flags any `build_orth_univariate_recipe(` call without a `preprocess_params` kwarg in the same call, and any `.astype(np.int64)` on a freshly-asarray'd y without going through a shared _coerce_y_* helper -- would have caught both regressions at review time instead of via manual multi-file audit.
