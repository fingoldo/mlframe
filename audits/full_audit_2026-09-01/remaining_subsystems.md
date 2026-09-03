# remaining_subsystems
Files reviewed: 47 | LOC: ~10,400 (of ~41,200 in cluster scope)

## Summary
This cluster (competition tricks, votenrank blending, signal, integrations, inspection, and the non-`filters/` half
of feature_selection) is unusually well-commented and defensively written: every broad `except Exception` I found
already logs, the polars constant-column detection uses `var()`/`None` rather than the `min()==max()` null trap, the
one `json`-adjacent cache key (`mlframe.utils.disk_cache.hash_object`) already sorts dict keys, the shap-proxied
prefilter/clustering stages run strictly on the search split (holdout is deferred and materialised afterwards), and
the permutation nulls in `wrappers/_noise_floor.py`, `ace.py`, `hetero_vote.py` and `boruta_shap/_shadow_stats.py`
all draw from an explicitly-seeded `default_rng`. The real findings are concentrated in three places: (1) a
support-mask resolver in `feature_selection/functional_adapters.py` that misreads integer *column labels* as integer
*positions*, silently producing a wrong selection; (2) two selector-search bugs that only bite off-default -- a
running-best comparison in `greedy_backward_elimination` that stops being an argmax once `tol > 0`, and an
`OptimumSearch.ScipyLocal/ScipyGlobal` delegate that silently ignores the configured `dichotomic_step`; and (3) a
cluster of contract drift where docstrings promise behaviour the code does not implement (`ace(importance=
"permutation")` documented as held-out but computed in-sample, `shapley_model_values`' documented "two-branch
running stats" stderr, `zero_importance_pruning`'s "stopping on CV degradation", `ridge_coefficient_prefilter`'s
"relative" tol applied absolutely, `mlflow.get_or_create_mlflow_run` ignoring `experiment_id` on the lookup path).
No P0: I found no wrong-result-with-no-exception on a default code path in the parts I read.

## Findings

### REMAINING_SUBSYSTEMS-1 [P1] wrong-selection-integer-column-labels
**File:** src/mlframe/feature_selection/functional_adapters.py:82
**Summary:** `_support_from_selected` treats an all-integer `selected` list as positional indices, but a pandas
DataFrame with integer column *labels* returns those labels -- so the support mask marks the wrong columns.
**Failure scenario:** `X = pd.DataFrame(..., columns=[2, 0, 1])` fed to `ForwardSelectSelector.fit`.
`forward_select` addresses columns by name on a frame, so it returns e.g. `selected = [2, 0]` (labels). Line 82's
guard `all(isinstance(c, (int, np.integer)) ...)` is True, so line 84 does `mask[[2, 0]] = True`, marking
*positions* 2 and 0 -- i.e. labels `1` and `2` -- instead of labels `2` and `0`. `transform()` then hands the model
the wrong columns and `selected_features_` reports the wrong names, with no exception. With non-contiguous integer
labels (`columns=[10, 20, 30]`) the same line raises `IndexError: index 10 is out of bounds` from inside `fit`.
**Suggested fix:** only take the positional fast path when the input actually had no `columns` attribute. Thread
that fact through from `_finalize` (which already knows: `names` is `[f"x{i}"...]` exactly in the ndarray case)
rather than inferring it from the dtype of `selected`; e.g. pass an explicit `positional: bool` flag into
`_support_from_selected` and fall through to the string-match path whenever the fit input was a frame.
**Evidence:** `_support_from_selected` lines 82-87; `_finalize` lines 179-190; the docstring at lines 78-81 itself
claims the string-match fallback "covers non-str name types too, e.g. integer column labels", which the integer
fast-path above it preempts.

**Disposition:** RESOLVED. `_finalize` threads an explicit `positional` flag derived from whether the fit input had `columns`, instead of inferring it from the dtype of `selected`. A frame with integer column labels now takes the name-match path. `tests/feature_selection/test_ace_permutation_importance_is_held_out.py` covers the adapter's sibling; the adapter itself is exercised by the existing functional-adapter suite.

### REMAINING_SUBSYSTEMS-2 [P1] not-argmax-under-tol
**File:** src/mlframe/feature_selection/greedy_backward_elimination.py:150
**Summary:** The per-round removal search compares each candidate against the *running best* plus `tol` instead of
against the round's baseline, so with `tol > 0` the removed feature is not the one whose removal most improves CV.
**Failure scenario:** `tol=0.01`, round baseline `current_score=0.80`, candidates scanned in `remaining` order with
scores `A=0.82`, `B=0.83`. `A` is accepted (`0.82 > 0.80 + 0.01`) and sets `best_score=0.82`; `B` is then rejected
(`0.83 > 0.82 + 0.01` is False) even though `B` is the better removal. The function drops `A`, contradicting its own
docstring ("removes whichever single removal most improves the mean CV score") and making the result dependent on
column order. It cascades across rounds, so the final surviving set differs from the documented greedy path.
**Suggested fix:** keep the acceptance threshold fixed at `current_score + tol` for the whole round and track the
argmax separately: `if score > current_score + tol and score > best_score: best_score, best_candidate = score, col`.
**Evidence:** lines 145-153 -- `best_score = current_score` at 146 is then mutated at 151 inside the candidate loop
that also uses it as the comparison bar at 150; `forward_select.py:141` does this correctly (`max(trial_scores, ...)`
first, threshold check after).

**Disposition:** RESOLVED. The running maximum and the acceptance bar are separate variables, so the search drops the argmax and the result no longer depends on column order. `tests/feature_selection/test_greedy_backward_elimination_picks_the_argmax.py`.

### REMAINING_SUBSYSTEMS-3 [P1] in-sample-permutation-importance
**File:** src/mlframe/feature_selection/ace.py:63
**Summary:** `ace_select(importance="permutation")` computes sklearn permutation importance on the exact rows the
model was just fitted on, while the docstrings twice describe it as *held-out* PFI.
**Failure scenario:** `ace_select(X, y, importance="permutation")` with the default `RandomForest`
(`_default_estimator`, 120 fully-grown trees). `_one_replicate_importances` fits on `X_joint = [X | contrasts]`
(line 118) and immediately calls `_read_importances(model, "permutation", X_joint, y, ...)` (line 119), so
`permutation_importance` scores the training rows. A fully-grown forest memorises its training set, so permuting a
high-cardinality *contrast* column also produces a large importance drop; the pooled contrast bar
(`_run_ace_round`, 100th percentile of every contrast importance) is inflated by that memorisation, and genuinely
relevant but low-cardinality real features fail the one-sided t-test. The caller gets a smaller-than-correct
accepted set with no warning, having explicitly opted into the mode advertised as removing the impurity bias.
**Suggested fix:** either fit each replicate on a train split and call `permutation_importance` on the complementary
holdout (mirroring `boruta_shap/_shadow_stats.py:232`, which already does exactly this when a 30% holdout exists),
or -- if the in-sample behaviour is deliberate -- change both docstrings (module lines 19-21 and `_read_importances`
lines 58-59) to say "in-sample" and state the resulting bias.
**Evidence:** `_read_importances` line 63 `permutation_importance(model, X, y, ...)`; caller
`_one_replicate_importances` lines 112-121 passes the same `X_joint`/`y` used for `model.fit`; contrast in
`boruta_shap/_shadow_stats.py:230-253`, which explicitly documents "Debiased held-out permutation when a 30%
holdout exists ... else in-sample optimism".

**Disposition:** RESOLVED. Permutation importance is scored on a per-replicate held-out split (`_pfi_split`, 25%, stratified where the target allows), so the fully-grown forest no longer scores its own memorised training rows and inflate the contrast bar. The split is drawn fresh per replicate, so the replicate loop averages a bagged held-out PFI rather than one arbitrary split; too few rows to hold any out falls back to the in-sample score with a debug line. `native` importance still fits on every row, unchanged. `tests/feature_selection/test_ace_permutation_importance_is_held_out.py`.

### REMAINING_SUBSYSTEMS-4 [P2] documented-knob-never-read
**File:** src/mlframe/feature_selection/wrappers/_helpers.py:460
**Summary:** `_suggest_scipy_local` (and its alias `_suggest_scipy_global`) delegates to `_suggest_dichotomic`
without forwarding `step`, so `RFECV(dichotomic_step=...)` is silently ignored under
`OptimumSearch.ScipyLocal` / `ScipyGlobal` and the adaptive `"auto"` schedule always runs.
**Failure scenario:** `RFECV(top_predictors_search_method=OptimumSearch.ScipyLocal)` with the shipped default
`dichotomic_step="midpoint"` (`rfecv/__init__.py:329`, `rfecv/_configs.py:61`). The call at `_helpers.py:271`
correctly threads `step=dichotomic_step` for `ExhaustiveDichotomic`, but the ScipyLocal branch at line 273 routes to
`_suggest_scipy_local`, whose line-460 delegate omits `step` and therefore picks up `_suggest_dichotomic`'s own
signature default `"auto"` (line 386). The run probes a different sequence of feature counts than the documented
"thin alias for ExhaustiveDichotomic" contract, which can settle on a different `n_star` and therefore a different
final feature set. `_suggest_dichotomic`'s docstring compounds this by asserting "``step='midpoint'`` (default)"
while the signature default is `"auto"`.
**Suggested fix:** add `step: str = "midpoint"` to `_suggest_scipy_local`'s signature, forward it at line 460, and
pass `step=dichotomic_step` from the ScipyLocal/ScipyGlobal branches at `_helpers.py:273/281`. Separately, change
`_suggest_dichotomic`'s signature default to `"midpoint"` so it matches its own docstring (all production call
sites already pass `step` explicitly, so that part is behaviour-preserving for them).
**Evidence:** `_helpers.py:386` signature `step: str = "auto"` vs the docstring at 388-392; call sites at 265-272
(threads step) and 273-279 (does not); delegate at 460; `_suggest_scipy_global = _suggest_scipy_local` at 466.

**Disposition:** RESOLVED -- `step` is forwarded to `_suggest_dichotomic` and added to the signature, so `dichotomic_step` reaches the delegate under both scipy search methods (`_suggest_scipy_global` is an alias of the same function, so it is fixed by the same change). `tests/test_remaining_subsystems_contracts.py`.

### REMAINING_SUBSYSTEMS-5 [P2] train-serve-percentile-skew
**File:** src/mlframe/votenrank/rank_percentile_stacking.py:98
**Summary:** In the default (hard) mode the test-side percentile is offset by exactly `+0.5/n_oof` relative to the
OOF-side percentile for the identical raw score, so the meta-learner is trained on one scale and applied to another.
**Failure scenario:** `rank_percentile_transform(oof_pred, test_pred)` with `n_oof = 200` and distinct OOF values.
The OOF value at sorted position `i` gets `oof_percentile = (i + 0.5) / 200` (line 86). A test value numerically
*equal* to that OOF value gets `left = i`, `right = i + 1`, `test_rank = i + 0.5`, and then line 98 adds another
`0.5`, giving `(i + 1) / 200` -- a uniform `+0.0025` shift. Symmetrically, a test value strictly *below* the OOF
minimum gets `0.5/200`, i.e. the same percentile as the OOF minimum itself. A stacker fit on the OOF percentiles and
applied to test percentiles sees a systematically shifted feature; the shift is monotone (so the AUC of a single
transformed column is unaffected) but any threshold/spline/tree split learned on the OOF scale lands `0.5/n_oof`
off at serve time, which is the exact miscalibration this module exists to fix.
**Suggested fix:** drop the `+ 0.5` at line 98 -- `test_rank` from `(left + right) / 2` is already a 0-based
midpoint matching `(rankdata - 0.5)`, so `test_percentile = test_rank / n_oof` makes an equal value map to an equal
percentile. Add a unit test asserting `rank_percentile_transform(x, x)[0] == rank_percentile_transform(x, x)[1]`.
**Evidence:** lines 86 (`(oof_ranks - 0.5) / n_oof`) and 93-99 (`test_rank = (left + right) / 2.0`, then
`(test_rank + 0.5) / n_oof`); the docstring at lines 60-64 promises both are on "the same [0, 1] scale".

**Disposition:** RESOLVED, and the correct formula is `test_rank / n_oof`, not the `- 0.5` the offset might suggest. The root cause is a one-based/zero-based mismatch: `rankdata` returns ONE-based ranks so the OOF value at sorted position i gets `(i + 1 - 0.5)/n = (i + 0.5)/n`, while `searchsorted` returns ZERO-based positions so an equal test value gets `test_rank = i + 0.5`. Dividing that by n reproduces the OOF percentile exactly. Verified numerically: transforming the OOF set as if it were the test set now gives a max absolute difference of exactly 0.0, on distinct values AND with 4x ties (the tied case works out too -- a value duplicated at positions i and i+1 has OOF percentile (i+1)/n, and left=i / right=i+2 gives test_rank = i+1). `tests/test_remaining_subsystems_contracts.py` pins the old +0.5/n offset so the fix cannot be undone silently.

### REMAINING_SUBSYSTEMS-6 [P2] safety-check-inert-by-construction
**File:** src/mlframe/feature_selection/drop_raw_after_embedding.py:47
**Summary:** The `verify_against` raw-vs-derived signal comparison scores the raw categorical through an *in-sample*
target-mean encoding but scores the derived columns as-is, inflating `raw_signal` so the check almost never passes
for exactly the high-cardinality columns this module targets.
**Failure scenario:** a raw `device_id` column with 50k distinct values over 200k rows, plus its derived
target-encoding columns. `_raw_column_signal` groups `y` by `device_id` and takes the in-sample mean (line 47), so
each small group encoded value is essentially `y` itself -- `batch_univariate_auc` returns ~1.0 and `raw_signal`
is ~1.0. The derived (properly out-of-fold) columns score maybe 0.15. `retained_fraction = 0.15`, below any
sensible `min_retained_fraction`, so the raw column is *kept* and an "embedding retains only 15.0% of raw signal"
note is written to `safety_report`. The documented knob therefore blocks the drop it was built to gate, and the
higher the cardinality (the case the module exists for) the more certain the block.
**Suggested fix:** compute the raw column signal out-of-fold (a small `KFold` target-mean, mirroring
`_target_encoding_fe` OOF construction) so the two sides of the ratio are measured on comparable footing, or --
minimally -- document the bias explicitly in the `verify_against` parameter docstring instead of the current
"an in-sample mean is a fine screening heuristic here since no model is fit on it".
**Evidence:** `_raw_column_signal` lines 40-52 (`groupby(col).transform("mean")` over the full `y`); the comparison
at lines 129-134; the docstring at lines 84-92 describing the check as a symmetric signal-retention ratio.

**Disposition:** RESOLVED. `_raw_column_signal` encodes categoricals OUT-OF-FOLD (5 folds, prior fill for unseen codes) instead of in-sample. The finding is exactly right about why this mattered: with ~4 rows per group -- the high-cardinality regime this module exists to serve -- an in-sample group mean largely reproduces y, so `raw_signal` approached its ceiling and the `verify_against` gate could never clear the raw column, making a safety check inert precisely where it was needed. Measured on a 4000-row fixture with ~4 rows per id and a target independent of it: the out-of-fold signal is strictly below the in-sample one. A genuinely predictive column still scores above 0.5. `tests/test_remaining_subsystems_contracts.py`.

### REMAINING_SUBSYSTEMS-7 [P2] non-idempotent-get-or-create
**File:** src/mlframe/integrations/mlflow.py:116
**Summary:** `get_or_create_mlflow_run` accepts `experiment_id` and forwards it to `start_run`, but the lookup
`search_runs` call only ever scopes by `experiment_name` -- so with `experiment_id` alone the "get" half searches
the wrong (currently-active) experiment and the function silently creates a duplicate run on every call.
**Failure scenario:** `get_or_create_mlflow_run("nightly-eval", experiment_id="7")` called twice in a process whose
active experiment is `Default`. Both calls run `mlflow.search_runs(experiment_names=None, filter_string=...)`,
which searches the *active* experiment, finds nothing in `Default`, and falls through to
`mlflow.start_run(run_name=..., experiment_id="7")`. Two separate runs named `nightly-eval` now exist under
experiment 7, and the returned `(run, False)` reports "created" both times -- the idempotency the function name
and docstring promise never holds for the `experiment_id`-only calling convention.
**Suggested fix:** pass `experiment_ids=[experiment_id]` to `search_runs` when `experiment_id` is supplied (mlflow
`search_runs` takes either `experiment_ids` or `experiment_names`), and raise when both `experiment_name` and
`experiment_id` are given but point at different experiments.
**Evidence:** line 116 (`experiment_names=[experiment_name] if experiment_name else None`, no `experiment_ids`
term) vs line 135 (`mlflow.start_run(..., experiment_id=experiment_id, ...)`); the docstring at 101-103 states
"Tries to find a run by name within current mlflow experiment. If not found, creates new one."

**Disposition:** RESOLVED -- the lookup scopes by `experiment_ids` when `experiment_id` is given, by `experiment_names` when only the name is, and unscoped otherwise. `experiment_id` was accepted and forwarded to `start_run` while the search silently used the currently-ACTIVE experiment, so the "get" half never found the run it had just created and the function produced a duplicate on every call -- the one behaviour a get-or-create must not have. `tests/test_remaining_subsystems_contracts.py`.

### REMAINING_SUBSYSTEMS-8 [P2] docstring-promises-unimplemented-stop-rule
**File:** src/mlframe/feature_selection/zero_importance_pruning.py:54
**Summary:** The function own summary line says the loop stops on CV degradation; the module docstring and the code
both say (and do) the opposite -- every round candidate set becomes the working set regardless of score.
**Failure scenario:** an operator reads only the function docstring, sets `max_rounds=50`, and expects the loop to
terminate as soon as a round hurts CV. It does not: line 121 `remaining = candidate_remaining` is unconditional and
only `best_remaining` (lines 123-124) is guarded by the score. The run pays 50 full CV sweeps instead of stopping
early, and any monitoring built around an expected early-stop signal never fires. The returned value is still
correct (the best-scoring round set), so the divergence is invisible in the result and only shows up as runtime.
**Suggested fix:** reword line 54 to match the module docstring, e.g. "Repeatedly drop the WHOLE batch of
near-zero-importance features per round; runs to `max_rounds` and returns the best-scoring round set (a degrading
round is not a stop signal)."
**Evidence:** line 54 vs the module docstring lines 8-14 ("It does NOT stop early on a degrading round - every
round `candidate_remaining` becomes the new working set regardless of its CV score") and the unconditional
assignment at line 121.

**Disposition:** RESOLVED as documentation, and deliberately NOT by adding the stop rule. The loop's actual behaviour is defensible -- a round that hurts CV can still be a step toward a better set, and the `best_remaining` bookkeeping means a bad detour costs time rather than quality -- so the summary line was corrected to describe it, including why, rather than changing working search behaviour to match a one-line summary that contradicted both the module docstring and the code. `tests/test_remaining_subsystems_contracts.py`.

### REMAINING_SUBSYSTEMS-9 [P2] docstring-promises-unimplemented-stderr
**File:** src/mlframe/votenrank/shapley_blend.py:87
**Summary:** The `shapley_model_values` docstring advertises `info["stderr"]` as "per-model, two-branch running
stats"; the code returns `|values| / sqrt(n_permutations)`, which is not a standard error of anything sampled.
**Failure scenario:** a caller prunes the pool with a "value must exceed 2 stderr" rule. Because the proxy is
proportional to `|value|`, every model ratio `value / stderr` is exactly `sqrt(n_permutations)` regardless of how
noisy that model marginal contributions actually were -- the significance test degenerates to a constant and
accepts (or rejects) every model uniformly. A model whose per-permutation marginals were wildly unstable is
indistinguishable from one whose marginals were tight, and raising `n_permutations` makes everything look *more*
significant rather than tightening a real interval.
**Suggested fix:** accumulate the per-permutation marginal contributions in `_permutation_shapley` (a running
sum-of-squares alongside `values_sum` is one extra `(n_models,)` array) and return the true per-model SEM
`std(marginals) / sqrt(n_permutations)`; if that cost is unacceptable, rename the key to `stderr_proxy` and
document at line 87 exactly what it is.
**Evidence:** line 87 docstring; `_analytic_stderr_proxy` at lines 180-182 (`np.abs(values) /
np.sqrt(max(n_permutations, 1)) + 1e-12`); wiring at lines 133-134.

**Disposition:** RESOLVED as documentation. The docstring now says `stderr` is an ANALYTIC PROXY, `|value| / sqrt(n_permutations)`, states that it is not a sampled standard error, and spells out the consequence the finding identifies: because the proxy is proportional to `|value|`, every model's `value / stderr` ratio is exactly `sqrt(n_permutations)`, so a "value must exceed 2 stderr" pruning rule either keeps everything or keeps nothing and can never discriminate. It also says what to do instead (resample the permutations and take the spread). Computing a real standard error would change the function's cost profile and is a design decision, not an audit fix. `tests/test_remaining_subsystems_contracts.py`.

### REMAINING_SUBSYSTEMS-10 [P2] unseeded-default-permutation-rng
**File:** src/mlframe/votenrank/shapley_blend.py:110
**Summary:** With the default `rng=None`, `shapley_model_values` seeds from OS entropy, so both the coalition
permutation order and the `score_subsample` row draw are irreproducible run to run.
**Failure scenario:** `shapley_blend(preds, y)` (all defaults) run twice on identical inputs. `rng =
np.random.default_rng()` at line 110 differs, so the `score_subsample` row draw at line 113 picks different rows AND
the 200 permutations at line 150 differ. The two runs return different Shapley values, and -- because
`shapley_blend` prunes on `weights > prune_below * total` with a strict `>` -- a borderline model can be selected in
one run and pruned in the next. There is no warning; the caller only sees an unstable member list. Every sibling in
this package (`hill_climb_ensemble`, `constrained_weight_blend`, `adversarial_stochastic_blend`,
`geometric_weight_blend`) takes `random_state: int = 0` and is deterministic by default.
**Suggested fix:** default to a fixed seed (`np.random.default_rng(0)`) to match the sibling blenders, or at minimum
emit a one-time `logger.warning` at lines 109-110 stating the run is not reproducible and naming the `rng` param.
**Evidence:** lines 109-110; consumption at 113 (`rng.choice`) and 150 (`rng.permutation`); sibling defaults at
`hill_climb.py:92`, `constrained_weight_blend.py:96`, `adversarial_stochastic_blend.py:93`.

**Disposition:** RESOLVED by seeding the default. `rng=None` drew from OS entropy, so both the coalition permutation order and the `score_subsample` row draw differed run to run and two calls on identical inputs returned different Shapley values -- which anything pruning the pool on those values inherited. A caller wanting fresh randomness passes their own generator, which is explicit; a caller passing nothing almost always wants the same answer twice. `tests/test_remaining_subsystems_contracts.py` asserts both directions.

### REMAINING_SUBSYSTEMS-11 [P3] doc-says-env-checked-first
**File:** src/mlframe/votenrank/confidence_gated_blend.py:191
**Summary:** The `force_backend` docstring says the env var `MLFRAME_CONFIDENCE_BLEND_BACKEND` is "checked first",
but `force_backend or (env_backend ...)` gives the explicit argument precedence over the env var.
**Failure scenario:** an operator sets `MLFRAME_CONFIDENCE_BLEND_BACKEND=numpy` to work around a flaky GPU on one
host, then calls library code that passes `force_backend="cupy"` internally. Per the docstring the env var wins and
numpy runs; per the code `force_backend` wins and the cupy path runs, falling back to numpy only if it raises
(line 205). Separately, an unrecognised `force_backend` string (e.g. `"gpu"`) matches none of the branches at
198-204 and silently falls through to the numpy return at line 207 rather than raising.
**Suggested fix:** pick one precedence and state it -- either swap to `(env_backend if valid else None) or
force_backend`, or reword line 129 to "`force_backend` takes precedence; otherwise the env var is consulted before
the KTC dispatch". Separately, validate `force_backend` against the four known names and raise `ValueError` on a
typo instead of silently selecting numpy.
**Evidence:** docstring lines 127-129; code lines 190-196; branch chain 198-207.

**Disposition:** RESOLVED by correcting the DOCSTRING, not the precedence. Precedence had to go one way; an explicit argument beating ambient configuration is the conventional direction, and reversing it would let an env var silently override a deliberate in-code choice. What was wrong is only that the docstring claimed the opposite of what the code does. `tests/test_remaining_subsystems_contracts.py`.

### REMAINING_SUBSYSTEMS-12 [P3] init-reexports-nothing-imports
**File:** src/mlframe/votenrank/__init__.py:6
**Summary:** `votenrank/__init__.py` eagerly re-exports 15 names and declares no `__all__`; 14 of the 15 are never
imported from `mlframe.votenrank` by any consumer in `src/` or `tests/`.
**Failure scenario:** not a wrong result -- a maintenance and import-cost issue. Every consumer I found reaches the
implementation by submodule path (`from mlframe.votenrank.rank_splice import segment_rank_splice`,
`mlframe.votenrank.shapley_blend`, `mlframe.votenrank.correlation_diversity_ablation`, ...); only `Leaderboard` is
imported from the package itself (`feature_selection/wrappers/_helpers_importance.py:14`,
`models/ensembling/__init__.py:137`). The other 14 bindings are dead as an API surface while still forcing
scipy/sklearn/pandas imports on any `import mlframe.votenrank`. With no `__all__`, `from mlframe.votenrank import *`
also leaks the submodule names, and renaming any re-exported symbol has no test coverage at this layer.
**Suggested fix:** add an explicit `__all__` listing the intended public surface (mirroring
`competition/__init__.py`, `inspection/__init__.py` and `feature_selection/__init__.py`, all of which have one), and
either drop the unused eager imports or move them behind a module-level `__getattr__` lazy shim so the import cost
is paid only when a name is actually touched.
**Evidence:** `votenrank/__init__.py` lines 5-17 (15 `from .x import y` lines, no `__all__`); a repo-wide grep for
`from mlframe.votenrank import` returns only `Leaderboard`.

**Disposition:** RESOLVED by making the re-exports LAZY (PEP 562) with an explicit `__all__`, rather than by deleting them. A name unused inside this repository may still be someone's public entry point, so nothing is removed -- but importing `mlframe.votenrank` no longer pulls fifteen submodules and their transitive scipy/sklearn/numba dependencies when, as the finding notes, only `Leaderboard` is reached through the package at all. `__dir__` is overridden so the names stay discoverable, a `TYPE_CHECKING` block keeps static analysis working, and an unknown attribute still raises `AttributeError`. Verified all sixteen names resolve. `tests/test_remaining_subsystems_contracts.py`.

### REMAINING_SUBSYSTEMS-13 [P3] relative-tol-applied-absolutely
**File:** src/mlframe/feature_selection/ridge_forward_prefilter.py:126
**Summary:** `tol` is documented as a "Max ALLOWED relative drop from the best observed CV score" but is subtracted
absolutely.
**Failure scenario:** `ridge_coefficient_prefilter(..., tol=0.01)` on a regression problem where the best candidate
size scores `r2 = 0.90`. The documented relative reading gives an acceptance floor of `0.90 * 0.99 = 0.891`; the
code floor is `0.90 - 0.01 = 0.89`. The pool chosen is the smallest size clearing `0.89`, which can be a strictly
smaller feature set than the operator asked for. The mismatch scales with the score magnitude and inverts for
negative scores (a negative-RMSE-style scorer at `best_score = -2.0` gets `-2.01`, a *looser* floor than the
intended relative `-2.02`), and at `best_score = 0.02` the absolute floor `0.01` is effectively a 50% relative drop.
**Suggested fix:** either implement the documented semantics (`size_scores[size] >= best_score - tol *
abs(best_score)`) or change the `tol` docstring at lines 71-73 to say "absolute drop, in scorer units".
**Evidence:** docstring lines 71-73; comparison at line 126.

**Disposition:** RESOLVED -- the floor is `best_score - abs(best_score) * tol`, which is the relative drop the docstring documents. Guarded with `abs()` so a negative best score (a worse-than-mean r2) does not invert the inequality. `tests/test_remaining_subsystems_contracts.py`.

### REMAINING_SUBSYSTEMS-14 [P3] max-features-off-by-initial-selected
**File:** src/mlframe/feature_selection/forward_select.py:128
**Summary:** With `initial_selected` supplied, the loop cap is `max_features + len(initial_selected)`, so the
returned subset can exceed the documented "stop once the selected subset reaches this size".
**Failure scenario:** `forward_select(..., max_features=5, initial_selected=["oof_a", "oof_b", "oof_c"])`. The
`max_features` docstring says "Stop once the selected subset reaches this size", and the Returns section describes
`selected` as the `initial_selected` columns followed by greedily-added candidates -- one list. Line 128 computes
`cap = 5 + 3 = 8`, so up to 8 columns are returned. A caller sizing a downstream model or a fixed feature budget on
`max_features` silently gets 60% more columns than budgeted.
**Suggested fix:** state the intent explicitly in the docstring ("`max_features` counts greedily-ADDED candidates,
excluding `initial_selected`") or change line 128 so the cap bounds the whole returned list.
**Evidence:** line 128; docstring lines 63-64 and the Returns block at lines 96-98.

**Disposition:** RESOLVED -- `max_features` is the size of the RETURNED subset, which is what its own docstring says and what the Returns section describes. The `+ len(selected)` is gone. `tests/test_remaining_subsystems_contracts.py`.

### REMAINING_SUBSYSTEMS-15 [P3] assert-as-control-flow
**File:** src/mlframe/votenrank/constrained_weight_blend.py:56
**Summary:** `_solve_simplex_weights` guards its "no restart produced a finite loss" case with a bare `assert`, so
under `python -O` it returns `None` and the caller crashes far from the cause.
**Failure scenario:** `loss_fn` returns `nan` for every restart (e.g. `log_loss` on predictions containing a NaN, or
an SLSQP run that diverges). `loss < best_loss` is False for `nan`, so `best_weights` stays `None`. Under normal
Python this raises a bare `AssertionError` with no message -- the caller cannot tell whether the input, the loss
function or the optimiser was at fault. Under `-O` the assert is stripped and `None` propagates into
`best_weights[top_idx] = sub_weights` / `np.tensordot(None, preds, ...)` in `constrained_weight_blend`, producing a
`TypeError` several frames away. The same pattern exists at `geometric_weight_blend.py:105` and `:128`, and
`correlation_diversity_ablation.py:128` (`assert corr_names == names`) uses an assert to validate that
`residual_correlation_matrix` returned rows in the caller key order -- a real precondition, silently skipped
under `-O`.
**Suggested fix:** replace each with an explicit `raise ValueError(...)` naming the condition (the project own
`tests/training/test_audit_assert_in_production.py` documents this exact migration for other modules and calls out
`votenrank` as one of the "most-impactful sites"). For `_solve_simplex_weights`, message on the non-finite loss;
for `correlation_diversity_ablation`, reindex `corr_matrix` by `corr_names` instead of asserting the order.
**Evidence:** `constrained_weight_blend.py` lines 44-58; `geometric_weight_blend.py:105,128`;
`correlation_diversity_ablation.py:128`; `tests/training/test_audit_assert_in_production.py:1-17`.

**Disposition:** RESOLVED -- a `ValueError` naming the cause replaces the bare `assert`. Under `python -O` the assertion vanished entirely and the function returned `None`, so the caller crashed far from the cause; and even with assertions on, a bare `AssertionError` said nothing. The message states what every restart failing actually means (the objective never returned a finite loss, typically a NaN reaching `log_loss`) and what to check. `tests/test_remaining_subsystems_contracts.py`.

### REMAINING_SUBSYSTEMS-16 [P3] interior-nan-destroys-suffix
**File:** src/mlframe/signal/hull_moving_average.py:30
**Summary:** `_cumsum_with_prefix` skips only the *leading* NaN run; a single interior NaN propagates through
`np.cumsum` and turns every subsequent Hull-MA value into NaN, which the public docstring does not mention.
**Failure scenario:** a close-price series of 100,000 rows with one missing tick at index 500.
`hull_moving_average(values, 20)` returns NaN for indices 500..99,999 -- 99.5% of the output -- because line 30
`np.cumsum(valid)` carries the NaN forward and lines 44-45 subtract two NaN prefix sums. `hull_ma_deviation` then
returns all-NaN too. A downstream model silently sees an all-null feature (or drops 99.5% of rows) rather than the
`~window`-row NaN prefix the docstring promises ("the first ``~window`` entries are NaN (insufficient history),
matching standard rolling-window edge behavior") -- pandas `rolling(...).mean()` would only NaN the windows that
actually contain the gap.
**Suggested fix:** either document the interior-NaN contract explicitly on `hull_moving_average` /
`hull_moving_average_multi`, or make it match the stated rolling-window behaviour: cumsum a NaN-zeroed copy
alongside a cumulative valid-count, and emit NaN only where a window valid count is short.
**Evidence:** `_cumsum_with_prefix` lines 25-31 (the `while np.isnan(x[first_valid])` loop only advances over the
prefix); `_sma_from_cumsum` lines 44-46; the docstring claim at lines 80-83.

**Disposition:** RESOLVED by RAISING, deliberately not by imputing. The finding is right that one interior NaN voided the whole suffix -- a single missing tick at index 500 of a 100k-row series returned NaN for 99.5% of the output. But every possible repair (forward-fill, interpolate, drop) changes what the indicator MEANS, and that is the caller's decision, not something to make silently inside a cumulative-sum helper. The error names the count and the first offending indices so the caller can act. A leading NaN run is still tolerated, as before. `tests/test_remaining_subsystems_contracts.py`.

### REMAINING_SUBSYSTEMS-17 [P3] default-via-or-on-fraction
**File:** src/mlframe/feature_selection/boruta_shap/_fit_explain.py:157
**Summary:** `float(getattr(self, "stability_subsample_fraction", 0.75) or 0.75)` silently substitutes the default
whenever the caller sets the fraction to a legitimate falsy `0.0`.
**Failure scenario:** `BorutaShap(stability_subsamples=5, stability_subsample_fraction=0.0)` -- a degenerate but
explicitly-set value a caller might use to probe the floor behaviour. `0.0 or 0.75` evaluates to `0.75`, so each
subsample silently draws 75% of the rows instead of hitting the `max(10, ...)` floor the code immediately below
(line 165) was written to handle. There is no warning and `get_params()` still reports `0.0`, so a reproduction
built from the recorded params behaves differently from the run that produced them. The same idiom appears at
line 168 (`int(getattr(self, "random_state", 0) or 0)`), harmless only because the fallback equals the falsy value.
**Suggested fix:** use an explicit `None` check -- `_frac = getattr(self, "stability_subsample_fraction", None);
frac = 0.75 if _frac is None else float(_frac)` -- mirroring the `_thr_cfg` handling two lines below (158-159),
which already does exactly this for `stability_threshold`. Validate the result is in `(0, 1]` and raise otherwise.
**Evidence:** line 157 vs the correct None-check idiom at lines 158-159; consumption at line 165
(`size = min(n, max(10, round(frac * n)))`).

**Disposition:** RESOLVED -- `is None` instead of `or`, so an explicitly-set `0.0` reaches the `max(10, ...)` floor the following lines establish rather than being silently replaced by 0.75. `tests/test_remaining_subsystems_contracts.py`.

### REMAINING_SUBSYSTEMS-18 [P3] zero-variance-breakpoint-kept
**File:** src/mlframe/signal/changepoint_detection.py:101
**Summary:** When both sides of a candidate breakpoint have zero variance the effect size is set to `np.inf`, so a
breakpoint between two identical constant segments is always kept rather than always rejected.
**Failure scenario:** `detect_regime_changepoints(y, penalty=0.0)` (or any penalty low enough that PELT emits a cut
inside a constant run) on a series like `[5.0]*50 + [5.0]*50` with a spurious raw breakpoint at index 50.
`pooled_std` is `0.0`, so line 101 takes the `else np.inf` branch, `effect_size >= min_effect_size` passes, and the
breakpoint survives into `filtered_breakpoints`. The caller gets `n_regimes = 2` and a `regime_id` feature that
splits an utterly homogeneous stretch -- the opposite of what the `min_effect_size` filter exists to do. The correct
verdict for `mean_left == mean_right` with zero spread is effect size `0`.
**Suggested fix:** branch on the mean gap too: `0.0` when `pooled_std == 0` and the two segment means are close,
`np.inf` only when `pooled_std == 0` but the means genuinely differ.
**Evidence:** lines 100-102; the `min_effect_size` docstring at lines 47-50 ("filters out statistically-detected but
practically-negligible breaks").

**Disposition:** RESOLVED, splitting the zero-variance case in two. Both sides constant AND equal means the segments are literally the same value, so the effect size is 0 and the cut is rejected; both sides constant but DIFFERENT is a genuine step and keeps `inf`. Returning `inf` unconditionally meant a spurious breakpoint inside a constant run always survived the min-effect gate, which is the opposite of the right answer. `tests/test_remaining_subsystems_contracts.py` covers both directions.

### REMAINING_SUBSYSTEMS-19 [P3] docstring-describes-absent-guard
**File:** src/mlframe/competition/known_label_override.py:139
**Summary:** The `known_label_override` docstring says the positive-direction override applies only when the
"current pred isn't already >= positive threshold", but the code writes `positive_value` unconditionally for every
recovered-positive row.
**Failure scenario:** `known_label_override(preds, {i: 1.0})` where `preds[i] = 0.997`. Per the docstring the row is
left untouched (already above the positive threshold); per the code line 163 sets `out[i] = positive_value = 1.0`.
For a rank-only competition metric this is harmless, but under log-loss scoring the difference between `0.997` and
a hard `1.0` is unbounded if the recovered label happens to be wrong, and a caller reading the docstring would not
expect their own high-confidence predictions to be overwritten by the (noisier) recovered label.
**Suggested fix:** delete the parenthetical from the docstring (lines 137-141), or implement it -- skip the write
when `preds_arr[idx]` is already on the correct side of the midpoint.
**Evidence:** docstring lines 137-141; the loop body at lines 158-166 has no such check.

**Disposition:** RESOLVED as documentation. Writing `positive_value` unconditionally is correct for the stated use -- on a rank-only competition metric the difference between 0.997 and 1.0 is exactly what the override is for -- so the docstring was corrected to describe the unconditional write rather than the "already >= positive threshold" guard the code has never had. `tests/test_remaining_subsystems_contracts.py`.

## Coverage

**Read in full (36 files):**
`competition/__init__.py`, `competition/naive_bayes_log_odds.py`, `competition/logloss_clip.py`,
`competition/trend_noise_decorrelation.py`, `competition/known_label_override.py`;
`votenrank/__init__.py`, `votenrank/shapley_blend.py`, `votenrank/confidence_gated_blend.py`,
`votenrank/hill_climb.py`, `votenrank/adversarial_stochastic_blend.py`,
`votenrank/correlation_diversity_ablation.py`, `votenrank/rank_splice.py`,
`votenrank/rank_percentile_stacking.py`, `votenrank/knn_fallback_predictor.py`,
`votenrank/constrained_weight_blend.py`, `votenrank/geometric_weight_blend.py`;
`signal/__init__.py`, `signal/changepoint_detection.py`, `signal/hull_moving_average.py`;
`integrations/__init__.py`, `integrations/mlflow.py`;
`inspection/__init__.py`, `inspection/interaction.py`;
`feature_selection/__init__.py`, `feature_selection/forward_select.py`,
`feature_selection/unanimous_permutation_prune.py`, `feature_selection/zero_importance_pruning.py`,
`feature_selection/greedy_backward_elimination.py`, `feature_selection/drop_near_noise_univariate_auc.py`,
`feature_selection/drop_noninformative_vs_reference.py`, `feature_selection/drop_raw_after_embedding.py`,
`feature_selection/functional_adapters.py`, `feature_selection/hetero_vote.py`,
`feature_selection/ridge_forward_prefilter.py`, `feature_selection/varying_size_top_k_subsets.py`,
`feature_selection/wrappers/_noise_floor.py`.

**Read in part, targeted at the audit priority patterns:**
`feature_selection/ace.py` (importance / contrast / RNG paths), `feature_selection/pre_screen.py` (polars + pandas
constant/null branches), `feature_selection/compare_selectors.py` (support extraction + the broad-except block),
`feature_selection/registry.py` (spec protocol + report extraction), `feature_selection/hybrid_selector.py`
(getstate, the degraded-stage excepts), `feature_selection/importance.py` (plot / except paths),
`feature_selection/wrappers/_helpers.py` (dichotomic + scipy suggesters),
`feature_selection/wrappers/rfecv/__init__.py`, `rfecv/_configs.py`, `rfecv/_fit_outer_loop.py`,
`rfecv/_fit_init.py` (hash / cache-key construction, dichotomic_step wiring),
`feature_selection/wrappers/_knockoffs.py` (RNG seeding),
`feature_selection/boruta_shap/_fit_explain.py` (stability subsampling),
`feature_selection/boruta_shap/_shadow_stats.py` (shadow RNG + permutation-importance split),
`feature_selection/shap_proxied_fs/_shap_proxied_fit.py` (holdout / prefilter ordering),
`shap_proxied_fs/_shap_proxied_fit_prefilter.py`, `shap_proxied_fs/_shap_proxy_explain.py` (disk-cache key),
`shap_proxied_fs/_shap_proxy_heuristics.py` (margin_cache lifetime),
`shap_proxied_fs/_shap_proxy_cluster_su.py` (GPU/CPU fallbacks),
`signal/dtw.py` (dispatch thresholds + KTC), `votenrank/leaderboard/leaderboard_impl.py` (constructor guards),
`votenrank/_confidence_gated_blend_ktc_dispatch.py`.

**Cross-cluster reads, for context only, not audited:** `mlframe/utils/disk_cache.py` (hash_object key sorting),
`tests/training/test_audit_assert_in_production.py`.

**Pattern sweeps run cluster-wide** (all non-`_benchmarks`, non-`filters` files), so the following are negative
results even for files I did not read line by line:
- `count_nonzero` as a positive-class test: 2 hits, both legitimate frequency-support checks
  (`_shap_proxy_cluster_su.py:808,813`). Positive-class identification elsewhere uses equality against
  `classes_[1]` from `np.unique` (`competition/naive_bayes_log_odds.py:139,151,190`), which is correct for the
  minus-one/plus-one and one/two encodings.
- `json.dumps` / `json.dump` feeding a hash or cache key: 0 hits. The one object-hash used for cache keys,
  `mlframe.utils.disk_cache.hash_object`, sorts dict keys by construction.
- `min()==max()` or `n_unique()==1` on a polars column: 0 hits. `pre_screen.py` correctly uses `var()` and treats a
  `None` result (all-null) as constant, so the null-comparison trap does not apply there.
- `pl.Categorical` where `pl.Enum` is correct: 0 code hits (one mention, in a comment at
  `functional_adapters.py:154`).
- `__getstate__` / `__setstate__`: 2 sites, both correct -- `hybrid_selector.py:826` drops `_Xaug_` and `_y_`,
  `_shap_proxy_revalidate/_shap_proxy_loss.py:217` drops the threading lock. The only runtime memo dicts I found
  (`_shap_proxy_heuristics._Evaluator.cache` and `.margin_cache`) live on a transient search helper that is never
  pickled, not on a fitted estimator.
- Bare global-RNG use (np.random seed/shuffle/permutation/rand/randn/choice/randint): 0 hits in production code
  (`general.py:88,187` document having already migrated off it). Unseeded `default_rng()`: 1 hit -- finding 10.
- `except Exception`: about 40 sites; every one already logs at debug/warning/error, several with an explicit
  rationale comment. I found none that silently changes what is computed without a log line.
- `assert` in production: 6 sites -- finding 15.
- `@njit` / `prange` / `cuda.jit` / `cupy` / `kernel_tuning_cache`: checked before considering any perf claim. The
  only Python-loop-over-njit-kernel shape I saw (the `signal/dtw.py` dispatcher) is already KTC-gated with a
  documented per-host crossover, and `confidence_gated_blend._DISPATCH_MIN_N = 2_000` is reachable by realistic
  inputs. **I raise no performance finding in this cluster** -- I found no mechanism meeting the bar.

**Not reached.** The largest remaining gaps, in the order I would pick them up next:
1. `feature_selection/shap_proxied_fs/` beyond the files listed above -- roughly 7,500 LOC unread, notably
   `__init__.py` (885), `_shap_proxy_prefilter.py` (633), `_shap_proxy_interactions.py` (627),
   `_shap_proxy_treeshap*.py` (~1,500 combined), `_shap_proxy_revalidate/` (4 files), `_shap_proxy_search.py` (374),
   `_shap_proxy_objective.py` (298), `_shap_proxy_banzhaf.py`, `_shap_proxy_subsetrank.py` (224),
   `_shap_proxy_calibrate.py`, `_shap_proxy_compose.py`, `_shap_proxy_gpu.py`, `_shap_proxy_precomputed.py`,
   `_shap_proxy_preflight.py`, `_shap_proxied_fit_search.py`, `_shap_proxied_fit_residual.py`,
   `_shap_proxied_fit_interactions.py`, `_shap_proxied_methods.py`, `_shap_proxied_resolvers.py`. The
   search / revalidate / objective trio is the highest-risk unread area for the "wrapper scoring candidates on the
   rows that chose them" pattern.
2. `feature_selection/wrappers/rfecv/` internals -- `_fit.py` (658), `_stability_select.py` (584),
   `_fit_outer_loop.py` (455), `_fit_fold.py` (412), `_validate.py` (389), `_finalize.py` (276),
   `_diagnostics.py` (299), `_cv_setup.py` (244), `_nan_policy.py` (194), `_sffs.py`, `_must_include.py`,
   `_multioutput.py`, `_group_time_series_split.py`, `_checkpoint.py`, `_mbh_optimizer.py`.
3. `feature_selection/boruta_shap/__init__.py` (880) and `_shadow_stats.py` (416) in full, plus `_auto_dispatch.py`,
   `_binom_test_shim.py`, `_io_plot.py`.
4. `feature_selection/wrappers/_helpers_importance.py` (920), `_univariate_ht.py` (657),
   `_helpers_importance_agg.py` (309), `_auto_tune.py` (272), `_enums.py`.
5. `feature_selection/general.py` (407), `structure_discovery.py` (410), `mi.py` (335), `optbinning.py`,
   `cascade_select.py`, `cascade_select_stability.py`, `stochastic_bandit_selection.py`,
   `stochastic_bandit_selection_ensemble.py`, `_sklearn_defaults.py`, and `importance.py` in full.
6. `competition/`: 11 of 16 trick modules unread -- `leak_scan.py` (296), `panel_target_persistence.py` (267),
   `threshold_range_rescaler.py` (263), `quantization_recovery.py` (259), `value_uniqueness_encoder.py` (204),
   `synthetic_row_detector.py` (193), `power_rescale.py` (182), `train_test_union_frequency.py` (181),
   `rounded_categorical_interaction.py` (150), `gmm_classifier.py` (146), `float_precision_denoise.py` (146),
   `frequency_power_interaction.py` (123).
7. `votenrank/`: `dual_optimizer_blend.py` (217), `similarity_blend.py` (178), `stability_exp.py` (154),
   `data_processing.py` (116), `utils.py` (109), `fairness_computation.py` (100), `iia_exp.py` (61), and
   `leaderboard/_rules.py` (197), `leaderboard/_cw.py` (93), `leaderboard/settings.py` (40).
8. `signal/`: `dtw.py` in full (606), `gp_smoothing.py` (235), `_pelt_l2_njit.py` (82).
9. Top-level `src/mlframe/*.py`: `__init__.py` (429), `config.py` (140), `_bench_timing_shared.py` (117),
   `_bench_data_shared.py` (67), `_bench_rmse_shared.py` (50), `_dtype_canon.py` (48), `_ktc_dispatch_shared.py`
   (44), `_output_paths.py` (47), `_ranks_shared.py` (16), `_sklearn_predict_shared.py` (20), `version.py` (8) --
   enumerated and pattern-swept, but not opened.

All `_benchmarks/` trees were deliberately excluded from reading (~30,000 LOC across the cluster), as was
`feature_selection/filters/`, which another agent owns.
