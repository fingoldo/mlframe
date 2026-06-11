# Composite Targets audit - dimension: Cross-target ensemble / OOF / stacking (2026-06-10)

Scope: src/mlframe/training/composite/ensemble/{__init__,_cross_target,stacking,feature_stacking}.py
plus call sites in training/core/_phase_composite_post_xt_ensemble/__init__.py, core/_phase_helpers.py,
core/predict.py, composite/discovery/_stacked.py and tests exercising the scope.
Known-fixed items (C-Low-1 empty-shape, C-P1-3 kfold+time WARN, ENS-Low-7 import hoist, C-P2-2 id()-key
warning, SOLVER-COPY, NO-REFIT determinism) re-verified in current code and NOT re-reported.

---

## N1 - P1 | bug | sample_weight misaligned with fit rows after the group-aware eval carve

File: src/mlframe/training/composite/ensemble/__init__.py:431-434 (also 463-465, 703, 868-871, 902-904; root at 285)

_carve_inner_eval_split has two return shapes: the tail carve returns fit rows as a PREFIX of the input
(X.iloc[:cut], y[:cut], lines 288-291), while the group-aware branch returns fit rows as a boolean-mask
SCATTER (_slice_rows(X, fit_mask), y[fit_mask], line 285). Every one of the five call sites then slices the
per-row weights as a prefix: _sw_fit_c = _sw_train_valid[:len(_t_fit_c)] (431-434), sample_weight[:len(_y_fit_r)]
(463-465), _sw_stack[:len(_y_fit)] (703), _sw_stack_valid[:len(_t_fit_c)] (868-871), _sw_stack[:len(_y_fit_r)]
(902-904). When group_ids AND sample_weight are both supplied (both are threaded by the suite,
_phase_composite_post_xt_ensemble/__init__.py:488,492), the first k weights of the UNSPLIT ordering are applied
to a SCATTERED set of k fit rows - every inner OOF refit trains with weights attached to the wrong rows.
Silent: lengths match, no error, no log. Wrong OOF RMSEs -> wrong ensemble weights on every weighted+grouped run.

Fix: make _carve_inner_eval_split accept sample_weight and return the carved sw_fit (or return the fit-row
indices/mask as a 5th element) and use it at all five call sites. Regression test with non-uniform weights +
group carve asserting the fitted estimator received sw[fit_mask] (recording stub estimator).

---

## N2 - P1 | bug | kfold OOF branch never carves an eval set for composite components -> ES boosters dropped

File: src/mlframe/training/composite/ensemble/__init__.py:670

In the kfold>1 branch the composite-component path calls
_maybe_pass_sample_weight(inner_clone, X_stack_valid, t_stack, _sw_stack_valid) with NO eval_set (line 670),
while the raw-component path right below carves one (698-707), and BOTH paths carve in the single-split
(863-875, 897-908) and external-holdout (422-438, 454-470) branches. A composite inner clone carrying
early_stopping callbacks raises "For early stopping, at least one dataset and eval metric is required" ->
caught by the per-component except (713-719) -> component silently excluded. This is exactly the prod
pathology documented in _maybe_pass_sample_weight's own docstring (194-199). Since the suite DEFAULT is
oof_holdout_source="kfold" (_phase_composite_post_xt_ensemble/__init__.py:336), every ES-configured composite
component is dropped from the OOF surface in the default configuration.

Fix: mirror the single-split composite path: carve (X_fit, t_fit, X_ev, t_ev) from (X_stack_valid, t_stack)
with the fold-subset group ids and pass eval_set=. Add a kfold test with a composite spec + early-stopping
inner asserting the component survives.

---

## N3 - P1 | leak | outer OOF split is never group-aware: same-group rows span refit-train and holdout

File: src/mlframe/training/composite/ensemble/__init__.py:618-621 (kfold), 791-794 (random single split)

group_ids is accepted (line 532) but used ONLY for the inner early-stop eval carve (691-697, 818-823). The
outer split that defines the honest holdout itself is a plain KFold(shuffle=True) (618-621) or
rng.permutation(n_train) (791-794). On grouped data (users/sessions/wells - the exact scenario the
_carve_inner_eval_split docstring 244-251 cites as a prod incident for the INNER split), rows of the same
group land in both refit-train and holdout -> within-group memorisation inflates every component's OOF quality
-> NNLS/linear-stack weights and the dummy-floor gate (_phase_composite_post_xt_ensemble/__init__.py:526+)
are computed on a leaked surface. The code defends the cheap inner ES split against group leakage while
leaving the outer split - the one the weights are fit on - fully group-blind.

Fix: when group_ids is not None, use GroupKFold (kfold path) and a group-level permutation carve (single-split
path: permute unique groups, accumulate until n_holdout, mirroring _carve_inner_eval_split's own group logic).

---

## N4 - P1 | leak | explicit but non-monotone time_ordering silently downgrades to a random shuffle

File: src/mlframe/training/composite/ensemble/__init__.py:777-783 (helper at 160-180)

use_time_split flips on only when the caller-supplied time_ordering is already monotone non-decreasing;
otherwise the explicit time signal is SILENTLY discarded and the split becomes a random shuffle (only the
positive case logs, 780-783). Realistic triggers: panel/grouped frames sorted by (group, time) - timestamps
non-monotone over the whole frame; any NaT in a timestamp column (-> NaN -> False at line 178). Result:
future rows leak into refit-train for AR-strong targets, OOF RMSE goes optimistic, ensemble weights and
dummy-floor gate are biased - precisely the failure mode the trailing-slice path was built to prevent. The
suite threads ctx.timestamps here (_phase_composite_post_xt_ensemble/__init__.py:314-320), so reachable in
production. tests/training/test_composite_oof_time_aware_logged.py pins only monotone / no-signal cases.

Fix: when time_ordering is supplied but non-monotone, np.argsort(time_ordering, kind="stable") and take the
trailing holdout_frac of the SORTED order as holdout (past-train / future-holdout regardless of row order);
at minimum WARN that the explicit time signal was ignored. Pin both behaviours.

---

## N5 - P1 | perf | composite_predictions_as_feature deep-copies the whole pandas frame

File: src/mlframe/training/composite/ensemble/feature_stacking.py:73-74

out = df.copy(); out[column_name] = preds - df.copy() defaults to deep=True, a full duplication of every
column. CLAUDE.md "Memory / RAM constraints (CRITICAL)" explicitly bans df.copy() on hot paths because frames
reach 100+ GB (OOM observed in 2026-04-22 prod logs). This helper is the documented building block for
composite x FE-pipeline stacking on the training frame (module header, 16-22) - exactly the hot path the rule
targets. The polars branch (line 71) is fine (with_columns shares column buffers); only pandas pays the copy.

Fix: size-gate per repo convention: under ~2 GB (df.memory_usage(deep=False).sum()) keep the copy
(API-friendly); above it, add the column in place and document the mutation contract (or take an
inplace: bool knob; suite calls inplace=True + try/finally removal - the CLAUDE.md mutate-and-restore pattern).

---

## N6 - P2 | leak | OOF refit reuses transform fitted_params estimated on the FULL train (holdout included)

File: src/mlframe/training/composite/ensemble/__init__.py:659-660 (kfold), 849-851 (single split)

Both internal-split branches build fold-train T via transform.forward(y_stack[valid], base_stack[valid],
spec["fitted_params"]) and re-wrap prediction with transform_fitted_params=spec["fitted_params"] - but those
params (e.g. linear_residual alpha/beta OLS coefficients) were fit during discovery on the FULL train,
including the rows now held out. Holdout target values influence the parameters used to predict that same
holdout -> systematic optimism in composite components' OOF RMSE relative to raw components, tilting ensemble
weights toward composite members. The Transform registry exposes a cheap fit callable
(composite/transforms/__init__.py:124, OLS-grade cost) so per-fold refit is essentially free. The
external-holdout branch (397-400) is NOT affected (params' fit support disjoint from prediction frame).

Fix: in kfold and single-split branches call transform.fit(y_stack[valid], base_stack[valid]) and use the
fold-fitted params for both forward and from_fitted_inner; keep spec["fitted_params"] as fallback on refit failure.

---

## N7 - P2 | bug | OOF memo cache key omits holdout_frac, time-split mode, external-holdout identity and group_ids

File: src/mlframe/training/composite/ensemble/__init__.py:598

_full_key = (cache_key, int(kfold), int(random_state)). Two calls with the same caller cache_key (documented as
summarising "(component, X, y, sw) identity", line 304) but different holdout_frac (0.1 vs 0.2), different
time_ordering (None vs monotone - flips random<->trailing slice), with vs without external_holdout_X, or
different group_ids return the FIRST call's matrix for the second call's request. The external-holdout path
shares the same _full_key (511-512), so an external-holdout result can be served to a later train-tail request
and vice versa.

Fix: extend the key with round(holdout_frac, 6), a time-mode flag, an external-holdout flag/len, and a cheap
groups fingerprint (len + sha1 of first/last 64 ids) - keeps the RAM rule intact.

---

## N8 - P2 | bug | kfold>1 silently wins over the external holdout, contradicting the documented precedence

File: src/mlframe/training/composite/ensemble/__init__.py:616 (vs docstring 547-549, external check at 752)

Docstring: "External holdout (preferred when external_holdout_X is supplied)". Code: the kfold>1 branch returns
at 616-745 BEFORE the external-holdout check at 752 - a caller passing both gets K-fold and the external frame
is ignored without any log. The suite never sets both (_kfold_for_oof=1 under external_val, caller 344-366),
but the function is public API (6+ test files + benchmarks call it directly) and the doc/code contradiction
will bite external callers.

Fix: hoist the external-holdout check above the kfold branch (matching the docstring), or WARN + document the
actual precedence; pin with a test.

---

## N9 - P2 | docs/dead-code | external_holdout_base_per_spec accepted, documented as required - never read

File: src/mlframe/training/composite/ensemble/__init__.py:359 (doc at 553-557; pass-through at 764-766)

_compute_oof_with_external_holdout declares external_holdout_base_per_spec but the body never references it
(verified by grep: signature + pass-through only). The public docstring states "Caller is responsible for
providing the parallel base columns via external_holdout_base_per_spec for composite components" - the suite
does real work building _base_val_per_spec (_phase_composite_post_xt_ensemble/__init__.py:300) to satisfy a
dead contract. The parameter exists for a real failure mode: when a shim pre-pipeline drops the base column
from X_holdout_t, wrapped.predict fails and the component is silently dropped - the supplied base columns were
meant to be the fallback.

Fix: wire it (per-spec external base arrays as override when the column is missing from X_holdout_t) - converts
silent component drops into successful predictions; or delete param + doc sentence + caller-side construction.

---

## N10 - P2 | bug | _maybe_pass_sample_weight silently retries without sample_weight/eval_set on broad ValueError

File: src/mlframe/training/composite/ensemble/__init__.py:224-232

The combined fit(X, y, sample_weight=, eval_set=) call is wrapped in except (TypeError, ValueError): pass, then
retried with sample_weight alone, then plain. (a) Silent contract downgrade: if the weighted call fails for a
DATA reason (sklearn signals nearly all data problems as ValueError, e.g. a length-mismatched sw from caller
bug N1) while the plain call succeeds, the model trains UNWEIGHTED and nothing is logged - canonical
silent-error-swallowing. (b) A genuinely failing fit may run 3 full fit attempts before the error surfaces.

Fix: validate len(sw)==len(y) up front (raise on mismatch); WARN naming dropped kwargs on every fallback hop;
keep the broad catch only around signature probing; narrow the fit-retry trigger to TypeError.

---

## N11 - P2 | bug | _transform_via swallows pre-pipeline transform failures -> OOF silently evaluates a different pipeline than deployed

File: src/mlframe/training/composite/ensemble/__init__.py:53-63

except Exception: return X with zero logging. If a fitted shim pre_pipeline.transform raises on the
stack/holdout slices, the inner clone is refit on RAW X while the deployed PrePipelinePredictShim predicts
through the transformed space. When raw columns are numerically compatible the refit succeeds, predictions are
finite, and the component's OOF RMSE measures a model the suite will never deploy - weights assigned to the
wrong object. If only ONE of the two _transform_via calls fails (stack vs holdout: 378-379 / 644-645 /
830-831), train and predict spaces silently diverge.

Fix: WARN with component name + exception on every fallback; treat one-side-only transform success as a
component failure (visible drop reason) instead of proceeding.

---

## N12 - P2 | bug | detect_gpu_in_use: XGBoost USE_CUDA build flag != GPU in use; LightGBM branch is a dead import

File: src/mlframe/training/composite/ensemble/__init__.py:111-131

(a) xgb.build_info()["USE_CUDA"] reports BUILD capability, and standard PyPI xgboost wheels ship CUDA-enabled -
on a CPU-only host with stock pip install, the function returns ["xgboost"] and the suite emits the "GPU
non-determinism amplified by K composite fits" warning for runs that never touch a GPU. (b) The LightGBM
branch (111-118) imports the library (cold import ~0.5 s) and then does nothing - detected is never appended.

Fix: AND the build flag with a physical-device probe (pyutilz.system.gpu_dispatch.is_cuda_available per repo
convention); delete the no-op lightgbm import (keep the explanatory comment).

---

## N13 - P2 | perf | env_signature cold-imports 9 libraries just to read versions - on the predict path

File: src/mlframe/training/composite/ensemble/__init__.py:150-156 (consumed at training/core/predict.py:106-110)

__import__(libname) fully imports each library; catboost/lightgbm/xgboost cold imports cost ~0.5-3 s each. The
drift check in predict.py calls env_signature() on model load, so a lightweight inference round-trip pays
seconds purely to read version strings - against the project's own ENS-Low-7 rationale (lines 18-22 of this
file: keep cold-import cost off predict).

Fix: importlib.metadata.version(dist) (no module import) with a name map (sklearn -> scikit-learn), catching
PackageNotFoundError -> None. Also record platform.python_version() + mlframe's own version (the interpreter
version matters for pickle drift at least as much as numpy's).

---

## N14 - P2 | perf/docs | linear/NNLS train-matrix stash is dead weight in production; three docstrings still describe the removed dropout-refit path

File: src/mlframe/training/composite/ensemble/_cross_target.py:257-259, 349-350 (creation); 525-536 (NO-REFIT predict); 623-641 (cap carry-over, contradictory comments); 644-655 (discard_train_matrix docstring)

predict follows the NO-REFIT policy (528-536) and never reads _linear_stack_train_preds/_train_y /
_nnls_stack_train_preds/_train_y; __getstate__ strips them from every pickle (677-691). Grep confirms zero
production readers - the only consumer left is a presence assert in
tests/training/test_ensembling_votenrank_votenrank.py:254-255. Yet the factories still stash them
(8*n*(K+1) bytes per ensemble held in RAM until pickling - ~720 MB at n=10M, K=8), cap_inference_components
carries + column-slices them (637-641) under a comment whose first half (623-625 "correct without re-slicing
here") contradicts its second half (633-639 slicing), and discard_train_matrix's docstring claims "the
dropout-refit path uses [these] at predict time" and ".refit / dropout-style methods will raise" - no such
method exists.

Fix: stop stashing by default (opt-in keep_train_matrix=False for diagnostics), update the three
docstring/comment sites to NO-REFIT reality, re-frame the votenrank test to the new contract.

---

## N15 - P2 | bug | stacking_aware_gate thresholds RAW NNLS weights though documented as a share of the unit budget

File: src/mlframe/training/composite/ensemble/stacking.py:191

survivors = [n for n, wv in raw_weights.items() if wv >= min_weight] - min_weight=0.05 is documented as "5% of
the unit weight budget" (155), but the comparison uses unnormalised NNLS coefficients. Weights only sum to ~1
when components are unbiased; with systematically scaled/biased components (e.g. shrunk booster predictions
where NNLS compensates with sum(w)=1.3) the effective threshold silently becomes 0.05/1.3 ~ 3.8% - gate
semantics drift with input calibration instead of staying a fixed budget share.

Fix: threshold on w / w.sum() (guard sum>0, else existing degenerate path). One-line change; the survivor
renormalisation block below (192-202) already exists.

---

## N16 - P2 | validation | residual_dedup_indices never checks len(oof_rmses) == K

File: src/mlframe/training/composite/ensemble/stacking.py:108-125

keep_pref = list(np.argsort(oof_rmses)) iterates len(oof_rmses) candidates while corr is KxK. A shorter rmse
vector silently omits columns [len..K-1] from BOTH kept and dropped - the function returns a keep-list that
silently discards valid members (the suite then drops those components,
_phase_composite_post_xt_ensemble/__init__.py:628-631). A longer vector raises IndexError deep in the loop.
All other public entry points in this package length-validate their parallel arrays.

Fix: raise ValueError on oof_rmses.shape[0] != K.

---

## N17 - P2 | bug | from_train_metrics mixes a train-scale baseline with OOF component RMSEs; notes/strategy mislabeled

File: src/mlframe/training/composite/ensemble/_cross_target.py:391-398 (notes at 467-479)

When component_oof_rmse is supplied but baseline_oof_rmse is None, a caller-passed baseline_train_rmse is kept
as the baseline and compared against OOF RMSEs (397-398 only overrides when the OOF baseline exists). Train
RMSE is systematically lower than OOF RMSE for the same predictor, so gains baseline-rmses shrink/go negative
across the board -> spurious "no component beats baseline -> single best" fallbacks or diluted weights against
an apples-to-oranges benchmark. Also: notes always label values baseline_train_rmse / component_train_rmses
(472-474) even when they hold OOF numbers, and strategy is always "oof_weighted" (471) even on the biased
train-RMSE fallback - both poison metadata forensics.

Fix: when OOF rmses are given and baseline_oof_rmse is None, ignore baseline_train_rmse with a WARN (fall to
max-rmse self-baseline) or require matching scale; record notes["rmse_source"]="oof"|"train" and name keys accordingly.

---

## N18 - P2 | bug | _carve_inner_eval_split falls back to a group-splitting tail carve when the group path cannot carve

File: src/mlframe/training/composite/ensemble/__init__.py:267, 281-291

When group_ids is supplied but uniq.size < 4 (267), or the first shuffled group alone overshoots n-100 rows
(cumulative condition at 281 fails), the function silently falls through to the group-blind tail carve -
splitting groups across fit/eval, exactly the within-group-leakage/under-stopping pathology its own docstring
(244-251) cites as a prod incident (+25% OOF RMSE degradation). 2-3 large groups is a realistic shape
(per-well panels).

Fix: when group_ids is present and the group carve cannot be honoured, return (X, y, None, None) (ES degrades
gracefully) or carve the smallest whole group as eval; never silently group-blind-split. Log the decision.

---

## N19 - P2 | usability/leak | composite_oof_predictions: splitter cannot receive groups/y; no per-fold sample_weight; sole prod call site never threads the time signal

File: src/mlframe/training/composite/ensemble/feature_stacking.py:133 (split), 151 (fit); call site composite/discovery/_stacked.py:95-99, 237

(a) kf.split(indices) passes neither y nor groups, so cv_splitter=GroupKFold(...) (the documented escape hatch)
raises "The 'groups' parameter should not be None" - group-honest OOF stack features are impossible through
this API. (b) fit_kwargs is forwarded verbatim per fold (151): fit_kwargs={"sample_weight": sw} gives a
full-length array against fold-subset rows - length mismatch or silent mis-weighting; no supported way to
weight fold fits. (c) The only production call site (_stacked.py:95-99, 237) passes neither time_aware=True nor
a cv_splitter, so pass-2 stacked OOF features on temporal targets are built with shuffled KFold - future
information leaks into the engineered _oof_* columns that pass 2 trains on, inflating stacked-spec selection.

Fix: add groups=None, sample_weight=None params; thread groups into kf.split(indices, groups=groups), slice
sample_weight[train_idx] per fold; in _stacked.py thread the discovery config's time signal + group ids.

---

## N20 - P2 | extension | OOF helper returns no holdout row indices -> sample-weighted stacking unreachable from the suite

File: src/mlframe/training/composite/ensemble/__init__.py:533 (return contract); call site _phase_composite_post_xt_ensemble/__init__.py:707-720

from_linear_stack / from_nnls_stack grew sample_weight support (_cross_target.py:139, 269) but the suite cannot
use it: compute_oof_holdout_predictions returns only (matrix, y_holdout, surviving_names) - holdout row indices
are discarded (kfold path additionally drops NaN rows, 737-741), so _sw_for_oof cannot be aligned to OOF rows.
The stack solvers therefore run unweighted on weighted-objective targets: the meta-model optimises a different
loss than the base models and the reported metrics.

Fix: return a 4th element holdout_idx (indices into the caller's train order; kfold: finite-row-filtered
natural-order indices). Thread sample_weight[holdout_idx] into the from_*_stack calls and optionally into the
per-component OOF-RMSE computation at the call site.

---

## N21 - P2 | extension | no time-respecting K-fold OOF; the default suite path deletes the time signal to get K folds

File: src/mlframe/training/composite/ensemble/__init__.py:609-615 (downgrade); caller _phase_composite_post_xt_ensemble/__init__.py:348-349 (_time_ordering = None)

kfold>1 + time signal downgrades to a single trailing slice (documented C-P1-3), and the suite's default
oof_holdout_source="kfold" explicitly nulls _time_ordering to keep K folds - so on temporal/AR targets the
default honest-OOF surface is SHUFFLED K-fold, training each fold on future rows: the same optimism the
external-holdout path was built to defend against (docstring 547-558). The standard fix is cheap:
expanding-window / forward-chaining folds (TimeSeriesSplit semantics, optionally with an embargo gap a la
de Prado purged CV) - each fold trains on the past, predicts the next block; first-block rows simply get no OOF
value (callers already tolerate NaN-row dropping, 737-741).

Fix: implement kfold>1 AND time-monotone as forward-chaining K folds (reuse the per-fold refit loop); flip the
suite to keep the time signal. biz_value test: on a synthetic AR(1) target, forward-chained OOF RMSE must be
honestly worse than shuffled-KFold OOF RMSE by a pinned margin.

---

## N22 - LOW | bug | inner eval carve seed hardcoded to 0 in 4 of 5 sites, ignoring the master seed

File: src/mlframe/training/composite/ensemble/__init__.py:424, 456, 864, 898 (vs 699 which threads int(random_state))

Four call sites pass random_state=0 to _carve_inner_eval_split; the kfold raw path threads the caller's seed.
Inconsistent, undocumented, contrary to the module's own derive_seeds philosophy (73-93). Changing the suite's
oof_random_state re-randomises the outer split but never the group-level eval carve - partially frozen
randomness that confuses seed-sweep variance estimates.

Fix: thread int(random_state) (or derive_seeds(random_state, ["inner_eval_carve"])) at all five sites.

---

## N23 - LOW | dead code | unused imports and dead locals across the package

Files/lines:
- __init__.py:8 math - unused in this module.
- __init__.py:9 warnings - unused.
- __init__.py:17 BaseEstimator, RegressorMixin - unused (class deliberately doesn't inherit them).
- __init__.py:23 ElasticNetCV - unused (ENS-Low-7 pre-warm rationale doesn't cover it).
- __init__.py:582 from sklearn.model_selection import train_test_split - imported in-function, never called.
- __init__.py:622 fold_frac - computed, never used.
- _cross_target.py:10 hashlib, :13 warnings, :21 ElasticNetCV - unused.
- stacking.py:121 order = list(np.argsort(oof_rmses)[::-1]) - dead variable (loop uses keep_pref only); the
  companion comment (120, "Process members worst-RMSE-first") describes a strategy the code doesn't use.

Fix: delete; for stacking.py:120-121 also fix the comment to match the best-first algorithm actually implemented.

---

## N24 - LOW | polish | residual_correlation_matrix emits RuntimeWarnings on constant columns; docstring overclaims the diagonal

File: src/mlframe/training/composite/ensemble/stacking.py:67 (vs guarded sibling at 118-119)

np.corrcoef on a zero-variance residual column emits RuntimeWarning: invalid value encountered in divide -
residual_dedup_indices wraps the identical call in np.errstate(invalid="ignore"); this function doesn't
(warning noise for near-perfect transforms with ~constant residuals). Docstring line 40 states "Diagonal is
1.0" - for a constant column the diagonal entry is NaN.

Fix: add the errstate guard; reword to "Diagonal is 1.0 for non-degenerate columns; NaN row/col indicates a
constant residual."

---

## N25 - LOW | usability | stacking.py degenerate fallbacks are completely silent (no logger in module)

File: src/mlframe/training/composite/ensemble/stacking.py:178-181, 185-189

Too-few-finite-rows and NNLS-failure paths return ALL names as survivors with uniform weights - the gate
silently turns itself off. The caller's INFO line only fires when survivors < total
(_phase_composite_post_xt_ensemble/__init__.py:667-685), so a permanently-degenerate gate is indistinguishable
from "gate ran and kept everything". The module has no logger at all.

Fix: add logger = logging.getLogger(__name__) and WARN in both fallbacks (reason + row counts).

---

## N26 - LOW | docs | stale from_linear_stack docstring (intercept folding, predict renormalisation); export_metadata omits is_convex

File: src/mlframe/training/composite/ensemble/_cross_target.py:144-157, 561-568

Docstring claims the intercept is "folded into the bias by absorbing it as an extra +b/n per component" and
that "predict re-normalises only the magnitudes" - both describe removed behaviour: the intercept is stored
separately and added (253, 540), and non-convex predict never renormalises (NO-REFIT block 525-550).
export_metadata (561-568) omits is_convex, so a metadata consumer cannot tell whether weights are a convex
combination or raw additive solver output (the intercept is only discoverable via notes["intercept"]).

Fix: rewrite the two docstring sentences; add "is_convex" (and top-level "intercept" for linear_stack) to
export_metadata.

---

## N27 - LOW | validation | group_ids length checks inconsistent: >= max(idx)+1 accepts misaligned arrays

File: src/mlframe/training/composite/ensemble/__init__.py:694, 821 (vs == checks at 266, 418)

The kfold and single-split paths accept any group_ids array at least as long as the max index
(_g_arr.shape[0] >= int(np.max(train_idx)) + 1), silently slicing a possibly longer, misaligned array (e.g. a
full-frame group vector passed where the train-subset one was expected - off-by-filtered-rows misalignment
ruins the group carve while looking valid). The external path and the carve itself use exact == checks.

Fix: standardise on _g_arr.shape[0] == n_train at all four sites; warn when the check fails (the silent None
fallback also hides genuine caller bugs).

---

## N28 - P2 | test-gap | no coverage for the highest-risk OOF parameter combinations

Files: tests/training/test_composite_oof_time_aware_logged.py; tests/training/test_ensembling_caching_future_fixes.py:310-339; tests/training/composite/test_composite_ensemble_oof_time_aware.py; tests/training/test_group_aware_eval_carve_and_xgb_dmatrix.py

Verified by grep: (a) no test calls compute_oof_holdout_predictions with group_ids= at all - nothing covers
group_ids+sample_weight together (would have caught N1) or the group-blind outer split (N3); (b) all kfold
tests use component_specs=[None] (raw components only) - the kfold composite branch (incl. missing eval carve,
N2) has zero coverage; (c) the non-monotone explicit time_ordering downgrade (N4) is unpinned (tests cover
only monotone / no-signal); (d) no test exercises two cache calls with same cache_key but different
holdout_frac (N7).

Fix: add four targeted regression tests alongside the fixes, each first verified to fail on pre-fix code.

---

## Summary counts

- P1: 5 (N1, N2, N3, N4, N5)
- P2: 16 (N6-N21, N28)
- LOW: 6 (N22-N27)
- No overlap with composite_discovery_audit_notes.md known-clean areas (Welford, transform round-trip, MI
  screening, tiny rerank) - re-verified as out of this dimension or still clean.
