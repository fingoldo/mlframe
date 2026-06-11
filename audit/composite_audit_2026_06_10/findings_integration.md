# Composite Targets audit - Training-suite integration phases (2026-06-10)

Scope: _phase_composite_discovery.py, _phase_composite_wrapping.py, _phase_composite_post.py,
_phase_composite_post_summary.py, _phase_composite_post_lag_predict.py,
_phase_composite_post_xt_ensemble/ (both files), targets/_train_eval_select_target.py (composite parts),
composite hooks in core/_phase_train_one_target_body.py. All line numbers verified against current code.
Items in tests/composite_discovery_audit_notes.md (Welford, transform round-trip, MI LCB, tiny-rerank)
were re-checked only at the wiring level and are NOT re-reported.

---

## I1 - P1 / leak - per-model wrap uses FULL y (train+val+test) for clip envelope + fallback median; the promised end-of-target re-fit never happens

**File:** src/mlframe/training/core/_phase_composite_wrapping.py:147-162 (creation), :284-287 (skip that prevents the re-fit); hook call site core/_phase_train_one_target_body.py:894-905.

emit_per_model_composite_y_scale_test builds the production wrapper with
y_train=_y_full_arr where y_full is the ENTIRE target array (train+val+test rows).
CompositeTargetEstimator.from_fitted_inner derives y_train_median (the fallback
prediction for domain-violating rows) and y_clip_low/high (the post-inverse clip
envelope) from this array (composite/estimator/_estimator.py:239-262). The in-code
comment (lines 148-154) claims "The end-of-target pass will re-fit the envelope using
the precise train slice" - but _run_composite_target_wrapping line 286-287 does
"if isinstance(_inner, CompositeTargetEstimator): continue", so an entry wrapped by the
per-model hook is NEVER re-fit. Consequences:

1. Test/val y extremes widen the clip envelope used to compute the reported test
   metrics (test-information leakage into the predict path).
2. y_train_median fallback predictions on domain-violating rows embed test-y statistics.
3. The leaked envelope persists into the SAVED deployable artifact.

The hook runs unconditionally for every composite (model, weight) fit, so this is the
default behaviour, not an edge case.

**Fix:** the hook caller (_phase_train_one_target_body.py:894) has ctx; pass
y_full=np.asarray(_y_full)[ctx.filtered_train_idx] (ctx.filtered_train_idx is set at
_main_train_suite.py:617). Belt-and-braces: in _run_composite_target_wrapping,
instead of bare continue on already-wrapped entries, re-derive the y-stats
(median/clip bounds) from _y_train_for_wrap on the existing wrapper instance.
Add a regression test asserting the wrapper clip bound equals y_train.max() (not
y_full.max()) after a suite run whose test split contains the global maximum.

---

## I2 - P1 / bug - CT_ENSEMBLE + lag_predict never built for discovery-skipped targets in multi-target suites

**File:** src/mlframe/training/core/_phase_composite_post.py:195-217 (gate); src/mlframe/training/core/_phase_composite_discovery.py:235-261, 301-325, 541-547 (skip paths that export no specs entry).

The raw-only synthesis fires only when "not composite_specs_by_target_type" - i.e. when
the specs dict is GLOBALLY empty. The downstream ensemble loop (line 233) iterates
only targets that HAVE a composite_target_specs entry. Discovery writes a specs entry
(possibly an empty list) only for targets that reach _disc.export_specs(); the skip
paths - extreme-AR skip (discovery:252-261), auto-skip (310-316), row-align mismatch
(319-325), multilabel-unsupported (227-233), and fit exception (541-547) - continue
BEFORE the export. So in a suite with target A (discovery completed) and target B
(extreme-AR-skipped), B gets NO CT_ENSEMBLE, NO lag_predict injection, NO dummy-floor
gate, NO AR(1) failsafe - even though the extreme-AR skip log line at discovery:247
explicitly promises "lag_predict in CT_ENSEMBLE pool will carry the AR signal". The
promise only holds when B is the ONLY regression target (dict globally empty ->
synthesis fires). This silently re-creates the exact prod failure
always_build_ct_ensemble_for_raw was added for: shipping raw models without the
lag floor on the most pathological targets.

**Fix:** change the synthesis to per-target fill: for every regression target with
trained models that is missing from composite_specs_by_target_type[REGRESSION],
insert an empty [] spec list (do not gate on the dict being globally empty).
Alternatively have every discovery skip path write
metadata["composite_target_specs"][tt][tname] = [] alongside the failures entry.

---

## I3 - P1 / perf - train-RMSE proxy computed unconditionally (one full-train predict per component) then discarded under the default honest-OOF path

**File:** src/mlframe/training/core/_phase_composite_post_xt_ensemble/__init__.py:218-250.

_component_train_rmses is computed for EVERY component via
_get_train_pred(...) -> _comp.predict(filtered_train_df) (full filtered train frame).
With defaults (oof_holdout_frac=0.2, oof_holdout_source="kfold" -
_composite_target_discovery_config.py:805,826) the OOF block then REPLACES
_oof_rmses with honest K-fold OOF RMSEs (line 518), so the train predictions are
discarded. The wrap-pass prediction cache that could have amortised this is empty by
default (skip_wrap_pass_predict=True -> no cache writes), so every component pays a
full-train predict - seconds per booster, minutes per MLP on multi-million-row frames,
per original target. The train preds are only consumed when (a) OOF fails/disabled
(fallback _oof_rmses = _rmse_arr) or (b) the stacking fallback branch (lines 693-706)
runs because _oof_pred_matrix is None.

**Fix:** make the train-RMSE proxy lazy - compute it only after the OOF attempt fails
(or when _oof_frac <= 0). Both consumers are downstream of the OOF attempt, so the
move is mechanical.

---

## I4 - P1 / leak - default kfold OOF weighting surface silently discards time ordering: shuffled K-fold on time-ordered AR data biases stack weights and the dummy-floor gate

**File:** src/mlframe/training/core/_phase_composite_post_xt_ensemble/__init__.py:345-354.

When oof_holdout_source == "kfold" (the default), the code sets
_time_ordering = None unconditionally, even when ctx.timestamps were threaded
specifically to make the honest OOF time-aware (lines 313-319). Shuffled K-fold on
time-ordered, strongly autocorrelated data lets each fold refit see temporally
adjacent (future) rows of its holdout, so component OOF RMSEs are optimistic for
models that exploit AR structure. That biased surface then drives (a) NNLS/stack
weights, (b) the dummy-floor gate, (c) the AR(1) failsafe comparison, and (d) the
honest-OOF gate==deploy fallback - i.e. exactly the protections built for strong-AR
prod targets can pick an overfit stack over lag_predict. The comment frames the drop as
"K-fold is incompatible with time-aware semantics", which is true for shuffled K-fold,
but the conclusion (drop the time signal) is backwards: the time signal should win over
the K-fold preference.

**Fix:** when _time_ordering is available and monotone, prefer the time-aware path:
either auto-switch to the single trailing-slice holdout (kfold=1, keep time_ordering)
with a clear log line, or implement forward-chaining TimeSeriesSplit-style OOF in
compute_oof_holdout_predictions and route kfold+time there. At minimum log a WARNING
(not a silent drop) that timestamps were ignored.

---

## I5 - P2 / bug - stacking_aware_gate prunes a rebound local _pred_matrix but not _oof_pred_matrix; honest-OOF gate + AR(1) failsafe then die on a swallowed broadcast error

**File:** src/mlframe/training/core/_phase_composite_post_xt_ensemble/__init__.py:649-691 (gate prunes _pred_matrix, _oof_components, _oof_names, _oof_rmses), :757-832 (OOF validation gate uses _oof_pred_matrix).

When stacking_aware_gate_enabled=True (opt-in, default False) and the gate drops
components, "_pred_matrix = _pred_matrix[:, _keep_mask]" rebinds only the local;
_oof_pred_matrix keeps ALL columns. The OOF validation gate then computes
"_oof_pred_matrix * _w_full[None, :]" with _w_full of pruned length -> shape-mismatch
ValueError -> caught at line 828 and logged at INFO as "OOF gate check skipped;
ensemble retained". Net effect: enabling the stacking gate silently disables the
gate==deploy OOF check, the best-single fallback AND the AR(1) lag failsafe - the three
protections this function exists for. Contrast: the residual-dedup block (lines
604-636) correctly prunes _oof_pred_matrix in place.

**Fix:** in the stacking-gate branch also prune _oof_pred_matrix[:, _keep_mask]
(mirroring dedup), or run the OOF gate on _pred_matrix. Add a regression test:
gate fires -> AR1 failsafe still reachable.

---

## I6 - P2 / bug - per-spec base matrices keyed by base_column: collision when a multi-base spec shares its primary base with a single-base spec

**File:** src/mlframe/training/core/_phase_composite_post_xt_ensemble/__init__.py:263-300 (write at 298/300); lookup composite/ensemble/__init__.py:649 (base_train_full_per_spec.get(spec["base_column"])).

_base_full_per_spec / _base_val_per_spec are keyed by the spec PRIMARY base column.
A linear_residual_multi spec (auto-promoted via forward stepwise, discovery/_fit.py:739)
stores an (n, 1+K) matrix under the same key that a single-base spec on the same
primary column stores an (n,) vector. Whichever spec is processed last wins; the other
spec OOF refit then calls transform.forward with a wrong-width base ->
ValueError("base has N columns but fitted alphas has M entries") -> the component is
excluded from the OOF pool with a generic "kfold OOF refit failed" warning. The losing
component is silently absent from stack weights and gates.

**Fix:** key both dicts by spec["name"] (unique) and change the lookup in
compute_oof_holdout_predictions accordingly (one coordinated change, both sides in
the same commit).

---

## I7 - P2 / bug - MTR honest-OOF NNLS weighting is all-or-nothing and fails at DEBUG level only: a single component failure silently forfeits the benched ~9% win

**File:** src/mlframe/training/core/_phase_composite_post_xt_ensemble/_phase_composite_post_xt_mtr_oof.py:65-75 (per-component failure -> return None for the whole weighting), :84-86 (outer except -> DEBUG); caller swallow at __init__.py:133-139 (DEBUG).

If ANY component clone/fit/predict fails in ANY fold, compute_mtr_oof_nnls_weights
returns None and the ensemble falls back to equal-mean. The single-target path drops
only the failing component (the helper surviving_names contract); the MTR path should
do the same - drop the component, NNLS over the survivors (>=2). Additionally both
failure sites log at DEBUG, so a production run gives zero visible signal that the
benched honest-OOF NNLS (8/8 seeds, ~9% lower test RMSE per the module docstring)
silently degraded to equal-mean.

**Fix:** (1) per-component exclusion instead of global return None (mirror the
single-target surviving-set logic); (2) raise both log sites to WARNING with the
component name and reason.

---

## I8 - P2 / bug - suite-end verdict: y-scale branch has no test fallback and no fallback to model-list metrics, so composites can show "-" while metrics exist

**File:** src/mlframe/training/core/_phase_composite_post_summary.py:63-76 (y-scale branch) vs :94-118 (test fallback exists only in the else branch).

When composite_target_y_scale_metrics entries exist but carry no finite val metric
(val split absent / 0-row val / per-split predict failed - all tolerated upstream at
_phase_composite_wrapping.py:648-660), _best_val stays None and the target gets
best_model="-" in the verdict block. The else branch was explicitly given a TEST
fallback after a prod incident (comment lines 94-103), but the y-scale branch never
got the same treatment, nor does it fall back to the T-scale model-list metrics when
y-scale entries are all unusable. Same incident shape, different branch.

**Fix:** after the y-scale val pass, if _best_val is None, scan y-scale test metrics
(with the "(test fallback)" tag), then fall through to the model-list passes.

---

## I9 - P2 / perf - y-scale TEST chart + predict emitted twice per composite entry (per-model hook AND suite-end wrap pass)

**File:** src/mlframe/training/core/_phase_train_one_target_body.py:857-905 (immediate per-model emit); src/mlframe/training/core/_phase_composite_wrapping.py:345-395 (skip_predict=True chart re-emit), :496-508 (metric-block chart when skip_predict=False).

The per-model hook emits a test-split y-scale chart right after each composite fit
(one wrapper.predict(test_df_pd) per entry). At suite end,
_run_composite_target_wrapping emits the SAME chart again for every entry - in the
default skip_wrap_pass_predict=True branch (lines 349-395; the comment justifies it by
the 2026-05-27 bug report which the per-model hook has since addressed) and in the
full metric block alike (496-508). Cost: one extra full-test predict per entry
(~0.1s booster, ~5s MLP per the in-code estimate, x entries x composite targets), a
duplicated chart file write (same _yscale_{composite} path -> overwrite), and
duplicated report lines in the log.

**Fix:** stamp entry._yscale_chart_emitted = True in
emit_per_model_composite_y_scale_test and skip those entries in both wrap-pass chart
sites (keep the wrap-pass emit for entries the hook missed, e.g. read-only entry.model).

---

## I10 - P2 / bug - extreme-AR discovery skip applies the single picked-target diagnostic to ALL regression targets

**File:** src/mlframe/training/core/_phase_composite_discovery.py:193-200 (global report read), :235-261 (per-target gate that never checks the target name).

_lag1_ar comes from metadata["target_distribution_report"], which is computed for
ONE target (picked_target_name, see _main_train_suite_target_distribution.py:266-268).
The per-target discovery loop applies that single statistic to EVERY 1-D regression
target: if the picked target is extreme-AR, discovery is skipped for unrelated targets
that are not AR at all (lost composite candidates); if the picked target is benign, an
AR sibling is not protected. Same pattern exists in the MLP skip
(extreme_ar_skip_decision, _phase_train_one_target_mlp_helpers.py:192-229 - also
never compares against picked_target_name).

**Fix:** gate on _td_report.get("picked_target_name") == _tname_disc (cheap,
information already in the report); for other targets either skip the gate or compute
a per-target lag1 stat (the analyzer already has the machinery).

---

## I11 - P2 / bug - group-aware tiny-rerank gated on the analyzer RECOMMENDATION, not the actual split configuration

**File:** src/mlframe/training/core/_phase_composite_discovery.py:204-215 (_group_aware_recommended from knob_overrides.split_config.prefer_group_aware), :473-482 (rerank wiring).

The comment block (463-472) says the tiny-CV rerank must use GroupKFold "when the
production split is group-aware", but the proxy used is the target-distribution
analyzer recommendation prefer_group_aware. A user who explicitly configured a
group-aware splitter (group_ids supplied, splitter honours them) without the analyzer
recommending it gets a plain-KFold rerank - exactly the prod failure described in the
comment (random-KFold rerank promotes specs whose models fail the group-aware test).

**Fix:** OR the condition with the actual split configuration (the suite knows whether
the splitter was group-aware; thread that flag in), keeping the recommendation as a
fallback signal.

---

## I12 - P2 / bug - silent-swallow cluster in discovery: fit failure and row-mismatch not recorded in failures metadata; cache replay reconstruction failure silently empties specs

**File:** src/mlframe/training/core/_phase_composite_discovery.py:319-325 (row-align mismatch: log-only), :541-547 (fit exception: log-only), :444-457 (cache replay Spec reconstruction -> bare "except Exception: _cached_specs = []").

Three inconsistencies with the otherwise-disciplined failure bookkeeping:

1. Fit exception (541-547): logs a WARNING and continues, but writes NO
   composite_target_failures entry - downstream audits / the failures table cannot
   see that discovery crashed for this target (all other skip paths record a reason).
2. Row-align mismatch (319-325): same - log-only, no failures entry.
3. Cache replay (444-457): if CompositeSpec(**s) raises (e.g. a spec schema field
   added in a patch release - the config signature only folds major.minor versions,
   lines 107-110), _cached_specs silently becomes []:
   metadata["composite_target_specs"] claims specs exist while NO composite T columns
   are added to target_by_type, so the specs are never trained - a metadata/behaviour
   divergence with zero log signal (the bare except has no logging at all).

**Fix:** (1)+(2) append a failures entry with the exception text; (3) log at WARNING
and, on reconstruction failure, fall back to full re-discovery (treat as cache miss)
instead of replaying half a cache hit.

---

## I13 - P2 / perf - _build_disc_df_for_target physically copies the full filtered train frame per regression target on pandas

**File:** src/mlframe/training/core/_phase_composite_discovery.py:48-59.

pd.concat([filtered_train_df[cols_wo_target], target_series], axis=1) materialises a
new frame. On pandas without Copy-on-Write (2.x default), both the column-subset
selection and concat (default copy=True) copy the underlying blocks - a full
peak-RAM duplication of the (potentially 100+ GB per project convention) train frame,
once per regression target. The docstring claim that the memory cost equals "the new
column anyway" holds only under CoW-enabled pandas (where concat reuses blocks lazily).
Alternative reading: if the project pins pandas>=3 / CoW mode, the copy is lazy and
this is a non-issue - but nothing in this module asserts that, and the polars branch
(cheap with_columns) shows the intent was cheap injection.

**Fix:** use the project-prescribed mutate-and-restore pattern (inject the target
column, try/finally del) - immune to pandas mode; or extend CompositeTargetDiscovery.fit
to accept the target as a separate array (y= kwarg) so no frame mutation/copy is needed
at all. At minimum, gate on pd.options.mode.copy_on_write and warn.

---

## I14 - P2 / bug - lag_predict component excluded from the OOF pool whenever the lag column contains ANY NaN, defeating the failsafe in its primary regime

**File:** src/mlframe/training/core/_phase_composite_post_lag_predict.py:53-69 (verbatim column passthrough, NaN included); injection _phase_composite_post_xt_ensemble/__init__.py:164-195; exclusion trigger composite/ensemble/__init__.py:710-711 ("if not np.all(np.isfinite(preds)): raise ValueError" in the OOF helper).

Lag features routinely carry NaN at group starts / series heads. predict() returns
the column verbatim, so any holdout fold containing such a row makes the OOF helper
raise "non-finite holdout predictions" and drop lag_predict from the surviving set -
silently removing the AR(1) failsafe and the lag floor from the dummy-floor comparison
on EXACTLY the strong-AR grouped targets the component was built for (its purpose per
the class docstring). The drop surfaces only as a generic per-component WARNING in the
OOF helper.

**Fix:** make _LagPredictDeployableModel.predict impute non-finite lag values with a
fit-time-stored constant (e.g. train median of the lag column, stored in fit()), or
have the injection site wrap it with a median-imputing shim. This also fixes the same
fragility at deploy time (predict on a frame whose first rows have NaN lags).

---

## I15 - P2 / test-gap - Pack-G watchdog never runs for multi-base specs (linear_residual_multi): 1-D base passed where K columns required, exception swallowed at DEBUG

**File:** src/mlframe/training/core/_phase_composite_wrapping.py:546-559 (universal check builds _base_uni from the single base_column), :585-600 (additive check, same 1-D base; linear_residual_multi IS in _ADDITIVE_TRANSFORMS, line 540), swallowed at :579-584 and :642-647.

For specs with extra_base_columns (the auto-promoted linear_residual_multi), both
watchdog variants pass a 1-D base array to transform.inverse / transform.forward,
which raise ValueError("base has 1 columns but fitted alphas has K entries") - the
same failure mode the wrap step explicitly fixes at lines 289-301. The exceptions land
in "except ... logger.debug", so the watchdog silently provides ZERO coverage for
multi-base specs - the spec family most likely to have wrapper-math bugs (it already
needed two parity fixes in the wrap and OOF paths). No test pins watchdog behaviour on
a multi-base spec.

**Fix:** build the (n, 1+K) base matrix from extra_base_columns (mirror wrap-step
lines 298-301) in both watchdog blocks; add a unit test that corrupts a multi-base
wrapper and asserts the watchdog WARNING fires.

---

## I16 - LOW / usability - MTRESID detection runs a substring match over the combined "{target_name} {model_name} {cur_target_name}" string

**File:** src/mlframe/training/targets/_train_eval_select_target.py:117-125; call site core/_phase_train_one_target_model_setup.py:446-447.

is_composite_target_name(model_name) is applied to the concatenated display string,
so a user-supplied suite label or RAW target whose name contains a fragment like
"-diff-" / "-ratio-" (e.g. raw target "price-diff-1d") is stamped MTRESID instead of
MTTR. Reporting-only impact, but it flips the exact label that distinguishes T-scale
from y-scale rows in logs. Fix: pass cur_target_name as its own parameter and test
only it (the call site already has it separately).

---

## I17 - LOW / docs - chart header labels TEST-split mean/std as "MTTR/MTTS"; dead local _human_inner

**File:** src/mlframe/training/core/_phase_composite_wrapping.py:64-71.

_mttr/_mtts are computed on the y TEST slice but rendered as "MTTR/MTTS=" - the
suite-wide convention (see select_target) is that MTTR is the TRAIN-split mean.
Cross-reading charts against MTTR= log lines gives inconsistent numbers under drift.
Also _human_inner (line 64) is computed and never used. Fix: label as test mean/std,
delete the dead local.

---

## I18 - LOW / docs - recover_composite_y_scale_metrics return annotation is wrong

**File:** src/mlframe/training/core/_phase_composite_post.py:55-68.

Annotated "-> dict[int, np.ndarray]" but the delegate returns the wrap-pass cache keyed
by (id(wrapper), id(frame), shape) tuples (_phase_composite_wrapping.py:233,243).
Fix: dict[tuple, np.ndarray].

---

## I19 - LOW / usability - "wrapped %d model(s)" log overcounts; failed/skipped entries counted as wrapped

**File:** src/mlframe/training/core/_phase_composite_wrapping.py:326-330.

The log reports len(_entries) even when some entries had no predict, were already
wrapped, or failed from_fitted_inner (warning emitted, entry left in T-scale). An
operator grepping for the wrap confirmation can be told "wrapped 4 model(s)" when 1
remained T-scale. Fix: count actual wraps in the loop.

---

## I20 - LOW / extension - MTR CT_ENSEMBLE parity gaps with the single-target path

**File:** src/mlframe/training/core/_phase_composite_post_xt_ensemble/_post_xt_ensemble_mtr.py:267-288.

(a) Registered by APPENDING into models[tt][orig_tname] instead of a dedicated
_CT_ENSEMBLE__{name} key - downstream consumers that treat list entries as independent
fitted models (votenrank export, per-entry metric tables) see an ensemble containing
its own siblings; (b) no metrics attribute on the SimpleNamespace (the single-target
entry sets metrics={}; _entry_metric tolerates it only via the isinstance guard);
(c) no ensembles_chosen["cross_target"] stamp, so the predict-path replay parity that
the single-target path documents (__init__.py:866-871) has no MTR analogue.
Fix: mirror the single-target registration shape; stamp ensembles_chosen["cross_target"]
with the per-column strategy label.

---

## I21 - LOW / docs - MTRPerColumnEqualMeanEnsemble docstrings contradict the raw-NNLS-weights behaviour

**File:** src/mlframe/training/core/_phase_composite_post_xt_ensemble/_post_xt_ensemble_mtr.py:28-34 (class docstring: "Weights are normalised to sum to 1"), :107-112 (property doc) vs :162-174 (code deliberately keeps RAW weights with a rationale comment).

The code is right (normalising loses the optimum when components do not bracket y); the
two docstrings still describe the pre-fix convex behaviour. A maintainer trusting the
property doc would re-add normalisation and silently regress fits. Fix: update both
docstrings to the raw-weights contract.

---

## I22 - LOW / bug - filtered_train_idx=None turns _y_arr[filtered_train_idx] into a newaxis broadcast and a misleading warning

**File:** src/mlframe/training/core/_phase_composite_discovery.py:318-325 (cf. the None-aware handling 5 lines up at :279-281).

_y_arr[None] yields shape (1, n), so the guard fires with the confusing message
"y[1] vs filtered_train_df[N]" instead of stating the index is missing. The
BaselineDiagnostics precompute right above handles None explicitly, so the
inconsistency is local. Fix: explicit "if filtered_train_idx is None" skip with a
clear reason (and record a failures entry per I12).

---

## I23 - LOW / bug - unscoreable components get the MEDIAN train-RMSE imputed, granting them mid-pack weight in the fallback weighting

**File:** src/mlframe/training/core/_phase_composite_post_xt_ensemble/__init__.py:249-250.

"_rmse_arr[~_finite] = median(finite)" - a component whose predict failed on the
train slice is imputed with the median RMSE. On the fallback path (OOF failed),
from_train_metrics then gives this unverified component an average weight. Safer:
impute max(finite) (worst observed) or drop it from the pool. Only reachable on the
double-failure path (OOF failed AND component scoring failed), hence LOW.

---

## I24 - LOW / docs - dead imports and constants in _phase_composite_post.py after the Wave-100 carve-outs

**File:** src/mlframe/training/core/_phase_composite_post.py:13-30, 48.

SimpleNamespace, np, report_model_perf, get_transform, compute_oof_holdout_predictions,
PrePipelinePredictShim, _CrossEns, CompositeTargetEstimator, format_suite_end_summary,
_build_full_column_from_splits, _entry_metric, _fmt, _short_tag_fn, _strip,
_DEFAULT_OOF_RANDOM_STATE, _PROB_NORM_EPS are no longer referenced in this module (the
deliberate back-compat re-exports are the three noqa-F401-tagged ones).
_WATCHDOG_RELATIVE_THRESHOLD (line 48) is duplicated in the sibling by design ("kept in
sync", _phase_composite_wrapping.py:27-28) - fine; the unused imports add import weight
and mislead readers. Fix: delete dead names; keep the documented re-exports.

---

## I25 - LOW / extension - dummy-floor / prescreen / oof_weighted baseline assume the dummy primary metric is an RMSE

**File:** src/mlframe/training/core/_phase_composite_post_xt_ensemble/__init__.py:415-422, 539-554, 727-740.

All three read the strongest-dummy value at primary_metric and compare it against
component RMSEs. Today the regression primary metric is hard-coded val_RMSE
(baselines/_dummy_metrics_pick_plot.py:108) so units match; but the code reads the
metric name generically, and the baselines module already supports non-RMSE primaries
for other target types (val_pinball_mean, log_loss). If the regression primary ever
changes (e.g. to MAE for heavy-tail targets - a plausible future default), the gates
silently compare mixed units. Fix: guard the 3 sites with an endswith-RMSE check and
log when skipping.
