# Composite Targets audit - dimension: CompositeTargetEstimator / sklearn compliance

Date: 2026-06-10. Scope: src/mlframe/training/composite/estimator/{_estimator,_predict,_update,_utils,__init__}.py,
src/mlframe/training/composite/post_shim.py, src/mlframe/training/composite/spec.py; context
tests/test_sklearn_compliance_composite.py. Cross-read for verification: transforms/{registry,simple,linear,__init__}.py,
streaming.py, core/_phase_composite_wrapping.py, core/_phase_train_one_target_body.py.

E1, E3, E8, E9 reproduced at runtime against current code (Python 3.14, pandas 2.3.3).
Known-fixed items from tests/composite_discovery_audit_notes.md were checked and are NOT re-reported.

---

## E1 - P0 / bug - multi-base domain-violation fallback crashes with broadcast ValueError

File: src/mlframe/training/composite/estimator/_predict.py:129 (context 123-133)

_predict_unclipped handles predict-time domain violations via
base_safe = np.where(domain_ok, base_arr, 1.0). For multi-base transforms
(linear_residual_multi, geometric_mean_residual, pairwise_interaction_residual)
base_arr is 2-D (n, K) while domain_ok (from _linear_residual_multi_domain,
transforms/linear.py:383-391) is 1-D (n,). NumPy broadcasting aligns trailing axes,
so (n,) vs (n, K) fails whenever K != n.

REPRODUCED: fitted CompositeTargetEstimator(transform_name=linear_residual_multi,
base_columns=(b1,b2)) with a NaN-tolerant inner, one base cell set to NaN, called
predict -> ValueError: operands could not be broadcast together with shapes (200,) (200,2) ().

This is exactly the realistic production scenario the fallback machinery exists for:
multi-base specs are auto-promoted by forward-stepwise discovery, the typical mlframe
inner (LightGBM / CatBoost / HistGB) accepts NaN natively, and a NaN feature row at
serving time is routine. Instead of the documented y_train_median fallback, predict
crashes. Zero test coverage on this path (the crash itself proves it).

Fix: mask = domain_ok[:, None] if base_arr.ndim == 2 else domain_ok;
base_safe = np.where(mask, base_arr, 1.0). Add a regression test (multi-base + NaN
base row -> fallback value, no crash), verified failing pre-fix.

---

## E2 - P1 / leak - per-model wrap builds y-clip envelope from FULL y (incl. test); promised end-of-target re-fit never executes

Files:
- src/mlframe/training/core/_phase_composite_wrapping.py:147-162 (per-model hook passes y_train=_y_full_arr = full y)
- src/mlframe/training/core/_phase_composite_wrapping.py:284-287 (end-of-target pass skips already-wrapped entries)
- contract: src/mlframe/training/composite/estimator/_estimator.py:239-246 (from_fitted_inner docstring: y_train = "Training-row y values")
- hook is live in the standard flow: core/_phase_train_one_target_body.py:894-905

emit_per_model_composite_y_scale_test wraps each freshly-fit composite inner via
from_fitted_inner(..., y_train=_y_full_arr) where _y_full_arr is the FULL target
(train + val + test rows). The in-code comment claims the end-of-target pass will
re-fit the envelope using the precise train slice - but _run_composite_target_wrapping
explicitly skips entries that are already a CompositeTargetEstimator (line 286
continue), so the re-fit never happens.

Consequences, all derived from full-y statistics inside from_fitted_inner
(_estimator.py:260-301): y_clip_low/high (Q001/Q999 envelope), y_train_median
(fallback constant) and t_clip_low/high (+/-10*std) all see test-row y values, which
then influence the clip bounds and fallback constant applied when producing TEST
predictions - honest-holdout contamination plus a violation of the documented
from_fitted_inner contract. Impact is bounded (clip envelopes, not the model itself)
but it systematically flatters test metrics whenever clips/fallbacks fire - which is
precisely the heavy-tail regime composites target.

Fix: in _run_composite_target_wrapping, when the entry is already wrapped, recompute
fitted_params_ y_clip / y_train_median / t_clip from _y_train_for_wrap (the precise
filtered_train_idx slice) instead of skipping; or pass the train slice to the
per-model hook. Regression test: wrapper produced by the full flow must carry an
envelope identical to one computed from the train slice only.

---

## E3 - P1 / bug - predict_quantile crashes for every requires_base=False (unary) transform

File: src/mlframe/training/composite/estimator/_predict.py:259-260

Unlike _predict_unclipped (which branches on transform.requires_base, lines 36-44),
predict_quantile unconditionally calls _resolve_base_columns() +
_extract_base_for_transform. For unary transforms (cbrt_y, log_y, yeo_johnson_y,
quantile_normal_y - all in the DEFAULT discovery transform list,
_composite_target_discovery_config.py:92) the documented configuration has no base
column, so _resolve_base_columns() returns an empty tuple and _extract_base_matrix
raises "ValueError: base_columns is empty; multi-base transforms require at least one
base column" - a misleading message for a unary transform.

REPRODUCED: CompositeTargetEstimator(transform_name=cbrt_y) + quantile-capable inner
-> fit OK, predict_quantile(X, 0.5) crashes as above.

Fix: mirror _predict_unclipped: when not transform.requires_base, skip extraction and
feed np.zeros_like(t_q) to the inverse (the unary adapter ignores base). Regression
test for a unary quantile round-trip.

---

## E4 - P1 / perf (RAM discipline) - fit() copies the entire X frame even when zero rows are dropped

File: src/mlframe/training/composite/estimator/_estimator.py:532 (helper at 651-666)

X_valid = self._subset_rows(X, valid) runs unconditionally. In the common case
(n_invalid == 0, all-True mask) this still materialises a full copy: pandas
X.loc[mask].reset_index(drop=True) copies every block, polars X.filter(...) gathers
into a new frame, ndarray boolean indexing always copies. On the 100+ GB frames this
wrapper is documented to ride (CLAUDE.md RAM rules; the module's own "NEVER
materialise" comments at _predict.py:74-79), this doubles peak RSS for nothing - the
exact failure mode the project bans.

Fix: X_valid = X if n_invalid == 0 else self._subset_rows(X, valid). Behaviour
identical (inner estimators ignore the pandas index; t_train / sample_weight are
ndarrays). Add a test asserting fit with an all-valid pandas frame passes the SAME
object to the inner (recording stub).

---

## E5 - P1 / bug - PrePipelinePredictShim._transform swallows ALL exceptions and silently feeds untransformed X to the inner

File: src/mlframe/training/composite/post_shim.py:44-48

The bare "except Exception: return X" means: when pre_pipeline.transform fails, the
shim silently feeds RAW X to the inner. The comment claims the inner will raise the
more descriptive error, but that only happens when raw X is incompatible with the
inner. The dangerous case is when it IS shape-compatible: SimpleImputer +
StandardScaler output has the same width as its input, so a linear/MLP tier inner
fitted on scaled features will happily predict on raw, unscaled X and return garbage
with NO error. Realistic trigger: sklearn raises on feature-name/order mismatch at
transform time (fit saw a frame, predict gets a reordered frame or ndarray) -> shim
falls back to raw X -> silently wrong component predictions inside the NNLS stack /
OOF refit. Same pattern duplicated in ensemble/__init__.py:48-56 (noted there; fix
both). This is the banned silent error-swallowing class.

Fix: stop swallowing - log at WARNING and re-raise; if a narrow pd/pl boundary case
must be tolerated, catch only that exception type and assert output-width equality
before using the fallback. Regression test: shim with a scaler fitted on named columns
+ predict on reordered columns must raise (or transform correctly), never silently
pass raw X through.

---

## E6 - P2 / bug - predict_quantile broken for grouped transforms (no groups kwarg, group column not dropped)

File: src/mlframe/training/composite/estimator/_predict.py:285-291 (inner call), 302/308/317 (inverse calls)

_predict_unclipped threads groups into transform.inverse (lines 64-72) and strips
group_column from X before the inner predict (lines 82-86). predict_quantile does
neither: (1) inner.predict_quantile(X, ...) receives X WITH the (typically string)
group column -> tree inners reject object dtype; (2) transform.inverse(t_q, base_arr,
params) for linear_residual_grouped raises ValueError "groups kwarg is required"
(transforms/linear.py:541-544). So quantile prediction on a grouped composite always
crashes. Grouped transforms are explicit-config (not auto-discovered), hence P2.

Fix: replicate the _predict_unclipped grouped plumbing (extract groups, build
inverse_kwargs, _drop_columns before the inner call). Related gap: from_fitted_inner
has no group_column parameter at all (_estimator.py:201-211), so a grouped spec cannot
be post-hoc wrapped; add the parameter while touching this area.

---

## E7 - P2 / bug - TypeError-based sample_weight fallback misattributes unrelated TypeErrors (3 sites)

Files (grep-all-instances per the class-level-pattern rule):
- src/mlframe/training/composite/estimator/_estimator.py:494-504 (transform.fit retry)
- src/mlframe/training/composite/estimator/_estimator.py:569-578 (inner estimator.fit retry)
- src/mlframe/training/composite/post_shim.py:52-58 (shim model.fit retry)

Pattern: try fit(..., sample_weight=w) except TypeError: fit(...). A TypeError raised
INSIDE a weight-accepting fit (bad dtype, bad kwarg deeper in the stack, user callback
bug) is misread as "does not accept sample_weight" and the call is silently retried
WITHOUT weights -> silently unweighted training (behaviour change, not crash), or a
double partial fit. The _estimator.py:573 log line then asserts the wrong diagnosis.

Fix: gate on signature instead of exception:
sklearn.utils.validation.has_fit_parameter(estimator, "sample_weight") for estimators;
inspect.signature(transform.fit).parameters for transforms (plain functions; cheap,
once per fit). Keep the except only as last resort, re-raising when the signature said
weights were accepted.

---

## E8 - P2 / bug - delegation properties return None when the fitted inner lacks the attr (hasattr trap)

File: src/mlframe/training/composite/estimator/_utils.py:26-35, 46-50 (binding at _estimator.py:699-703)

feature_importances_, coef_, intercept_, booster_ use getattr(inner, attr, None). On a
FITTED wrapper whose inner has no such attribute, hasattr(wrapper,
"feature_importances_") is True and the value is None.

REPRODUCED: fitted wrapper around a minimal inner -> hasattr == True, value == None.
Downstream importance-extraction code that duck-types
"if hasattr(m, 'feature_importances_'): m.feature_importances_.sum()" gets
AttributeError on None far from the cause - or treats the model as importance-capable
and writes None into reports. Same trap for booster_ (its docstring even claims it
raises NotFittedError but it returns None for a fitted non-LGB inner). get_booster
(lines 38-43) makes hasattr(wrapper, "get_booster") True for ANY inner, so XGB
duck-type detection always matches the wrapper.

Fix: post-fit, raise AttributeError("inner <cls> has no <attr>") when the inner lacks
the attribute (plain getattr without default) - hasattr then correctly returns False,
while NotFittedError (an AttributeError subclass) keeps pre-fit hasattr False too.
Keep n_features_in_ None-pre-fit convention as documented.

---

## E9 - P2 / bug - predict/predict_quantile raise RuntimeError instead of NotFittedError before fit

File: src/mlframe/training/composite/estimator/_predict.py:25-28, 244-247

sklearn convention (and the wrapper's own _utils._require_fitted; _update.py:40-43
mixes both styles) is sklearn.exceptions.NotFittedError. REPRODUCED: pre-fit predict
raises RuntimeError. Callers using "except NotFittedError" (sklearn tooling, suite
code) do not catch it; RuntimeError is not an AttributeError subclass so
hasattr-probing code sees it escape.

Fix: raise NotFittedError at both sites (subclasses ValueError + AttributeError,
preserving existing broad catches).

---

## E10 - P2 / bug - from_fitted_inner T-clip envelope is centered at ZERO; mis-centers non-residual (unary) transforms

File: src/mlframe/training/composite/estimator/_estimator.py:271-291

The post-hoc route reconstructs the T-scale clip as [-10*std(y), +10*std(y)] -
symmetric about 0. Sound for residual-family transforms (T centered near 0 by
construction) but wrong for the unary transforms in the default discovery list: log_y
has T = log(y+offset) centered at the log of the y level, cbrt_y at cbrt(median y),
yeo_johnson_y at the YJ mean. Whenever |center(T)| > 10*std(y) (near-constant targets
with a large offset, e.g. y ~ 1e5 +/- 1 -> T ~ 11.5, envelope +/-10), EVERY T_hat is
clipped to the envelope edge, the inverse collapses, and the y-clip then flattens all
predictions to a constant - silent accuracy destruction. Contrast with the .fit() path
(_estimator.py:609-623) which centers at median(T) with MAD and widens to observed
min/max of T_train.

Fix: for requires_base=False transforms, T_train is computable exactly inside
from_fitted_inner (transform.forward(y_train, zeros, transform_fitted_params)) - reuse
the same MAD-centered logic as .fit(). For bivariate transforms keep the proxy but
document its residual-centered assumption. Add a biz-value-style regression: log_y
wrap of a large-offset near-constant target must not clip in-distribution T_hat.

---

## E11 - P2 / bug - feature_names_in_ / n_features_in_ inconsistent for grouped transforms

Files: src/mlframe/training/composite/estimator/_estimator.py:636-640 + _utils.py:53-66

fit captures feature_names_in_ = list(X.columns) from the ORIGINAL X (including
group_column), but the inner is fitted on X_valid AFTER _drop_columns([group_column])
(lines 539-540). n_features_in_ delegates to the inner. Result for grouped transforms:
len(wrapper.feature_names_in_) == F while wrapper.n_features_in_ == F-1, breaking the
sklearn invariant n_features_in_ == len(feature_names_in_) that introspection and the
mlframe predict-side column resolution rely on. (Wrapper-perspective names SHOULD
include group_column - predict requires it - so it is n_features_in_ that is wrong at
the wrapper level.)

Fix: store the wrapper's own n_features_in_ at fit (len(feature_names_in_) /
X.shape[1]) instead of delegating; keep inner delegation only as fallback when the
wrapper attr is absent (from_fitted_inner route).

---

## E12 - P2 / perf - grouped transforms copy the whole pandas frame at fit AND every predict via X.drop()

Files: src/mlframe/training/composite/estimator/_estimator.py:539-540, 668-688; _predict.py:82-86

_drop_columns uses X.drop(columns=present). Under pandas 2.x without Copy-on-Write
(installed 2.3.3 default), drop materialises a full copy of the remaining columns - on
every predict call for grouped transforms, on frames that can be 100+ GB. Polars drop
is cheap (Arrow buffer reuse), so the cost is pandas-specific.

Fix: prefer a CoW-lazy column subset (X[remaining_cols] under CoW) or enable CoW; at
minimum hoist the surviving-column list and document the cost. Bench before/after per
the perf-measure-first rule.

---

## E13 - P2 / bug - update() refit leaves y_clip / t_clip / y_train_median stale under the very drift that fired

File: src/mlframe/training/composite/estimator/_update.py:71-74

When the Chow-style z-test detects drift, only alpha / beta are refit. The prediction
path still applies t_clip_* (MAD bounds of the PRE-drift T_train), y_clip_low/high
(pre-drift y envelope) and falls back to the pre-drift y_train_median. After a
level/scale regime change - precisely the event that triggers the refit - the
corrected predictions T_hat + alpha_new*base + beta_new can land systematically
outside the stale y-envelope and get clipped back toward the old regime, undoing the
correction the feature exists to provide.

Fix: on info["refit"], also refresh y_clip_low/high (envelope of the buffer y via
_y_train_clip_bounds), y_train_median, and the t-clip from buffer residuals under the
new (alpha, beta). Ship ON together with the refit (corrective-mechanism default-on
rule), logging the new bounds in the existing refit log line.

---

## E14 - P2 / usability - predict_quantile lacks domain-check/fallback, T-clip, and runtime-stats parity with predict

File: src/mlframe/training/composite/estimator/_predict.py:256-319

Gaps vs _predict_unclipped/predict:
1. No domain mask / fallback: a NaN base row passes the np.any(base_arr < 0) ratio
   guard (NaN < 0 is False) and yields silent NaN quantiles; ratio with base == 0
   yields quantile 0 silently (inverse t*base). predict would have substituted
   y_train_median / NaN per fallback_predict and counted the row.
2. No T-clip: heavy-tail quantile-head blow-ups go straight to the inverse (y-clip
   partially rescues, but the in-envelope wild middle documented at
   _estimator.py:598-608 passes through).
3. No runtime_stats_ update / callback: quantile traffic is invisible to the
   monitoring hook the point path feeds.

Fix: factor the domain-mask + fallback + clip + counters block out of
_predict_unclipped and reuse it per quantile column.

---

## E15 - P2 / bug - missing fit-time validation: unary y/X length mismatch and fallback_predict typo surface late and cryptically

File: src/mlframe/training/composite/estimator/_estimator.py:417-433 (length check only in the requires_base branch); _predict.py:135-143 (fallback_predict validated only when a violation occurs)

For requires_base=False, len(y) != len(X) is never checked; the mismatch surfaces as
pandas "IndexError: Boolean index has wrong length" inside _subset_rows (or a polars
filter panic) with no mention of the cause. And an invalid fallback_predict string is
only rejected on the first predict batch that contains a domain violation - possibly
weeks into production.

Fix: in fit, always validate len(y_arr) against the frame row count, and validate
fallback_predict in ("y_train_median", "nan") in fit (sklearn convention: validate in
fit, not init).

---

## E16 - LOW / docs - _extract_base claims structured-ndarray support it does not have; dead ndarray fallback in _extract_base_matrix

File: src/mlframe/training/composite/estimator/__init__.py:86-117 (esp. 114-117), 149-191 (esp. 189-191)

The _extract_base docstring and its TypeError message say "pass pandas / polars
DataFrame or a structured ndarray with named columns", but ANY ndarray (structured or
not) hits the TypeError. _extract_base_matrix's fallback loop ("preserves prior
behaviour for ndarray-with-names etc.") delegates per-column to _extract_base, which
raises - dead code for ndarrays. Misleading guidance when debugging a dropped-column /
wrong-X-type error.

Fix: either implement the structured-ndarray path (X[base_column].astype(float64) when
X.dtype.names contains it) - consistent with _subset_rows already supporting ndarray -
or correct the message/docstrings.

---

## E17 - LOW / usability - T-clip events are not in runtime_stats_ / callback; per-batch WARNING can spam

File: src/mlframe/training/composite/estimator/_predict.py:101-113, 179-206

predict counts y_clip_*_hits and domain_violation_rows cumulatively and ships them to
the monitoring callback, but t_low_hits/t_high_hits are computed, logged at WARNING
(every batch - a miscalibrated Huber inner emits one warning per predict call in
prod), and discarded. The runtime_stats_ docstring does not mention T-clip either.

Fix: add t_clip_low_hits/t_clip_high_hits to runtime_stats_ + the callback payload;
log the WARNING once (or every Nth), DEBUG thereafter.

---

## E18 - LOW / bug - CompositeSpec: frozen-but-unhashable, and __eq__ crashes when fitted_params holds ndarrays

File: src/mlframe/training/composite/spec.py:10-56

dataclass(frozen=True) generates __hash__ from fields, but fitted_params: dict is
unhashable -> hash(spec) raises TypeError (frozen implies hashable to most readers;
specs in sets/dict keys fail). Worse, the generated __eq__ compares the dicts; for
transforms whose params contain ndarrays (median_residual bin_edges/bin_medians,
monotonic_residual knots), spec_a == spec_b raises "ValueError: The truth value of an
array ... is ambiguous". Also Dict, Tuple imports unused.

Fix: fitted_params: dict = field(compare=False, hash=False) so the identity fields
(name/target_col/transform_name/base_column/extra_base_columns) define eq/hash; or set
eq=False. Drop unused imports.

---

## E19 - LOW / usability - PrePipelinePredictShim: estimator_ property defeats check_is_fitted; no predict_quantile delegation

File: src/mlframe/training/composite/post_shim.py:62-79

1. The estimator_ property exists (returning the unfitted inner) before fit, so
   sklearn.utils.validation.check_is_fitted(shim) passes on an unfitted shim.
2. The shim delegates only fit/predict. A quantile-capable member
   (CompositeTargetEstimator with a quantile inner) nested inside a shim loses
   predict_quantile, so predict_quantile_ensemble rejects it ("lacks
   predict_quantile") even though the underlying member supports it.

Fix: delegate predict_quantile(X, alpha) through _transform when
hasattr(self.model, "predict_quantile"); add __sklearn_is_fitted__ returning whether
the inner is fitted.

---

## E20 - LOW / docs - dead imports, shadowed logger import, undocumented pickling caveat

Files:
- src/mlframe/training/composite/estimator/__init__.py:7-15: warnings, deque,
  NotFittedError, BaseEstimator, RegressorMixin, clone, Callable, Dict, List all
  unused (verified by grep); logger defined (line 41) but never used.
- src/mlframe/training/composite/estimator/_estimator.py:17, 22, 24: math, deque,
  NotFittedError unused.
- _estimator.py:590-595: "import logging as _logging" inside fit duplicates the
  module-level logger.
- _estimator.py:120/175 docstring: runtime_stats_callback holding a lambda/closure
  makes the fitted wrapper unpicklable (it is an __init__ param so it cannot be
  excluded from state); docstring should warn to pass a module-level callable when
  the model will be persisted.

Fix: remove dead imports, use the module logger, add the pickling note.

---

## E21 - LOW / test-gap - sklearn-compliance matrix covers only diff + pandas + LinearRegression

File: tests/test_sklearn_compliance_composite.py:93-137

The CompositeTargetEstimator compliance class exercises one configuration:
transform_name="diff", pandas X, LinearRegression inner. Not covered: unary
(requires_base=False), multi-base (base_columns), grouped (group_column), polars X,
predict_quantile, and predict-time domain-violation fallback. E1, E3 and E6 are
crashes in exactly those uncovered rows. (Pickle round-trip exists elsewhere -
tests/training/test_composite.py:524 - so that is NOT a gap.)

Fix: parametrize _make() over (diff/pandas, cbrt_y/no-base,
linear_residual_multi/two-base, linear_residual_grouped/group-col, diff/polars) and
add a domain-violation-fallback row; assert clone/get_params/fit/predict per row.

---

## E22 - LOW / perf - update() boxes the whole buffer to Python floats and rebuilds two ndarrays per call

File: src/mlframe/training/composite/estimator/_update.py:54-64

self._buffer_y_.extend(y_arr.tolist()) boxes every value to PyFloat; then
np.asarray(self._buffer_y_) walks the full deque (default 10k elements) on EVERY
update call, even when the z-check will short-circuit on buffer_too_small. For a
streaming caller invoking update per micro-batch this is O(buffer) per call
(~0.5-1 ms at 10k) of avoidable overhead.

Fix: keep a preallocated float64 ring buffer (two ndarrays + head index) and pass
slice views into streaming_alpha_check_and_refit; or at least skip the ndarray
materialisation when buffer_n < online_refit_min_buffer_n.

---

## E23 - LOW / extension - update() rejects linear_residual_robust despite identical {alpha, beta} param shape

File: src/mlframe/training/composite/estimator/_update.py:36-39

The streaming refit is gated to transform_name in ("linear_residual",).
linear_residual_robust shares forward/inverse and the exact {"alpha","beta"}
fitted-param shape (transforms/registry.py:286-301), so the closed-form refit applies
verbatim (optionally using the robust fit on the buffer). Excluding it forces users of
the robust variant to forgo drift correction for no technical reason.

Fix: extend the allowlist to ("linear_residual", "linear_residual_robust"); use
_linear_residual_robust_fit on the buffer for the robust case.

---

## E24 - LOW / bug - predict_quantile_ensemble batched-call probe only catches TypeError/ValueError

File: src/mlframe/training/composite/estimator/__init__.py:305-312

The single-batched-call optimisation falls back to per-alpha scalar calls only on
(TypeError, ValueError). A member whose predict_quantile raises a library-specific
error on a sequence alpha (LightGBMError / CatBoostError / XGBoostError surfaced
through a CompositeTargetEstimator inner) crashes the whole ensemble instead of using
the documented scalar-alpha fallback path.

Fix: catch Exception for the PROBE call only (log at DEBUG which member/exception
triggered the fallback), keeping strict error propagation for the per-alpha calls that
constitute the real contract.

---

## Summary table

| id | sev | category | file:line |
|----|-----|----------|-----------|
| E1 | P0 | bug | estimator/_predict.py:129 |
| E2 | P1 | leak | core/_phase_composite_wrapping.py:155,286 |
| E3 | P1 | bug | estimator/_predict.py:259 |
| E4 | P1 | perf | estimator/_estimator.py:532 |
| E5 | P1 | bug | post_shim.py:46 |
| E6 | P2 | bug | estimator/_predict.py:302 |
| E7 | P2 | bug | estimator/_estimator.py:571 (+500, post_shim.py:56) |
| E8 | P2 | bug | estimator/_utils.py:27 |
| E9 | P2 | bug | estimator/_predict.py:26 |
| E10 | P2 | bug | estimator/_estimator.py:282 |
| E11 | P2 | bug | estimator/_estimator.py:638 |
| E12 | P2 | perf | estimator/_estimator.py:540 |
| E13 | P2 | bug | estimator/_update.py:73 |
| E14 | P2 | usability | estimator/_predict.py:256 |
| E15 | P2 | bug | estimator/_estimator.py:417 |
| E16 | LOW | docs | estimator/__init__.py:114 |
| E17 | LOW | usability | estimator/_predict.py:106 |
| E18 | LOW | bug | spec.py:10 |
| E19 | LOW | usability | post_shim.py:75 |
| E20 | LOW | docs | estimator/__init__.py:8 |
| E21 | LOW | test-gap | tests/test_sklearn_compliance_composite.py:104 |
| E22 | LOW | perf | estimator/_update.py:54 |
| E23 | LOW | extension | estimator/_update.py:36 |
| E24 | LOW | bug | estimator/__init__.py:311 |
