# Composite Targets audit — Tiny-CV rerank / stacked / stepwise / bayesian / auto-detect

Agent dimension: tiny-model CV fold construction, candidate comparability, multi-seed aggregation,
rerank escape conditions, stacked/residual discovery, forward stepwise multi-base, bayesian alpha
posterior, time/group/cat auto-detection.

Scope files (all read in full, current code as of 2026-06-10):
`src/mlframe/training/composite/discovery/{_screening_tiny,_tiny_rerank,_stacked,forward_stepwise,bayesian,auto_detect}.py`
plus cross-verified call sites: `discovery/_fit.py`, `core/_phase_composite_discovery.py`,
`core/_misc_helpers.py`, `composite/transforms/linear.py`, `composite/ensemble/feature_stacking.py`,
`training/_composite_target_discovery_config.py`, `training/_cv_aggregation.py`.

Known-fixed items from `tests/composite_discovery_audit_notes.md` (Welford, transform round-trip,
MI bootstrap LCB, multiseed fold-splitter reproducibility, ENS-P2-5 per-bin reuse, tie-break lexsort)
were re-checked and are NOT re-reported.

---

## A1 — P1 — perf — `_screening_tiny.py:144` + `_tiny_rerank.py:285-303`
**Default rerank parallelism leaves inner LightGBM `n_jobs=-1` uncapped → CPU oversubscription**

`_build_tiny_model` builds LGBMRegressor with `n_jobs=-1` (line 144). The `n_jobs=1` cap inside
`_one_fold` (lines 294-308 / 646-658) only fires when the FOLD-level `n_jobs > 1`
(`tiny_model_n_jobs`, default 1). But the per-SPEC rerank loop is parallel by default:
`tiny_rerank_n_jobs=0` (auto) → `_rerank_n_jobs = min(len(kept_specs), cpu_count)`
(`_tiny_rerank.py:287-297`). Result with defaults: N specs × all-core LGBM fits run concurrently
under the threading backend — e.g. 16 specs × 16 LGBM threads = 256 threads on a 16-core box.
This is exactly the 4-8x wallclock pathology the in-fold warning text itself describes, but at the
rerank level where no cap exists at all.

**Fix:** when `_rerank_n_jobs > 1`, build tiny models with `n_jobs=max(1, cpu_count // _rerank_n_jobs)`
(thread the effective outer parallelism into `_build_tiny_model` / `_tiny_cv_rmse_*`), mirroring the
existing fold-level cap. Add a perf regression check.

## A2 — P1 — bug — `_tiny_rerank.py:630-655` + `_composite_target_discovery_config.py:490-507`
**Wilcoxon gate is mathematically unpassable at the default `tiny_model_n_seed_repeats=3`**

The gate runs a one-sided paired Wilcoxon on per-seed RMSE diffs and rejects the spec when
`p > gate_alpha` (default 0.05). With n=3 paired diffs (no zeros), the minimum achievable one-sided
exact p-value is 1/2^3 = 0.125 > 0.05 — even a spec that wins on all 3 seeds is rejected. The code
gate `len(comp_per_seed) >= 3` (line 640) lets exactly this case through. Config default is
`tiny_model_n_seed_repeats=3`; the config comment recommends 5 ("for the test to have meaningful
power") but nothing enforces it — there is no validator linking `use_wilcoxon_gate` to
`n_seed_repeats`/`gate_alpha`. With n=5 only the perfect 5/5 outcome passes (p=1/32≈0.031).
Net effect: a user who flips `use_wilcoxon_gate=True` with defaults silently gets ZERO surviving
specs, indistinguishable from "no composite signal".

**Fix:** model_validator: if `use_wilcoxon_gate` and `0.5 ** n_seed_repeats > gate_alpha`, raise (or
warn + skip the Wilcoxon stage explicitly). Unit test pinning the minimum-power math.

## A3 — P1 — bug/perf — `_screening_tiny.py:432-434, 490-492` (with `622-629`)
**Multi-seed repeats are exact no-ops under TimeSeriesSplit / GroupKFold splits**

The multiseed wrappers vary only `random_state = base + s_idx*7919`. That seed reaches (a) the
KFold splitter and (b) the tiny model. When the spec's base is monotone (`time_aware=True` →
`TimeSeriesSplit`) or groups are present (`GroupKFold`), the splitter ignores `random_state`
entirely → identical folds every seed. The default families ("lightgbm", "linear") are then also
seed-invariant (LGBM with bagging/feature fraction 1.0 ignores its seed; Ridge is deterministic),
so all `n_seed_repeats=3` repeats produce bit-identical RMSEs. Consequences: (1) Phase B burns
3x the dominant tiny-CV wall time for zero information on every time-aware/grouped spec;
(2) the per-seed arrays fed to the Wilcoxon gate are pseudo-replicated — 3 identical diffs treated
as independent samples, manufacturing significance (or guaranteed rejection, see A2) from what is
effectively a single measurement.

**Fix:** short-circuit to 1 repeat when the effective splitter is deterministic and the family is
seed-invariant; have the Wilcoxon gate require shuffled-KFold per-seed variation (or fall back to
the threshold gate with a log line).

## A4 — P1 — bug — `_tiny_rerank.py:443-449, 468, 488`
**Raw-y baseline uses time-aware folds if ANY spec's base is monotone — non-monotone specs gated against a mismatched baseline**

`_any_base_monotone` is a single global flag: if at least one kept spec has a monotone base, the
raw baseline CV switches to `TimeSeriesSplit` for ALL comparisons. Specs with non-monotone bases
were scored with shuffled KFold (optimistic on temporal data, lower-variance), but the threshold
they must beat (`raw_baseline * tol`) is computed under TSS (typically higher RMSE: smaller
effective train sets, no shuffle). The gate is then systematically too lenient for the non-monotone
specs (and the cross-spec ranking mixes two fold regimes). Comment at lines 439-442 claims
"apples-to-apples" — it holds only when every base is monotone.

**Fix:** compute the raw baseline per fold-scheme (one shuffled-KFold baseline, one TSS baseline,
cached; at most 2x cost) and gate each spec against its matching scheme.

## A5 — P1 — leak — `_screening_tiny.py:604-609` + `_tiny_rerank.py:415-420`
**Composite tiny-CV scores only domain-valid rows; raw baseline and sibling specs score different row subsets**

`_tiny_cv_rmse_y_scale` drops rows failing `transform.domain_check` (lines 604-607) before the CV;
the raw baseline (`_tiny_cv_rmse_raw_y`) only drops non-finite y. With
`min_valid_domain_frac=0.7` a spec may legally be evaluated on as little as 70% of rows — and
domain-invalid rows (e.g. y<=0 / base<=0 for logratio) are typically the HARD regime. So:
(1) the raw-baseline gate compares composite RMSE on an easier subset vs raw RMSE on all rows;
(2) spec-vs-spec ranking (active in the DEFAULT config, where the gate is off) compares RMSEs over
different row populations with different y-variance. Production `CompositeTargetEstimator.predict`
must emit a prediction for every row (median fallback on domain-invalid), so deployed RMSE includes
those rows — the screening number does not; same mismatch class the wrapper-aware clip (R10b #4)
killed on the clip axis. Fold memberships also stop being identical to the raw run (different n),
quietly invalidating the "SAME folds" claim at `_tiny_rerank.py:415-418`.

**Fix:** keep fitting on valid rows, but SCORE each val fold on all its rows: emulate the production
fallback (train-fold y-median) for domain-invalid val rows, as the code already does for non-finite
inverse output (lines 686-693). That makes composite RMSE, raw RMSE and cross-spec RMSE
population-identical.

## A6 - P1 - bug - `_stacked.py:100,118-131` + `core/_misc_helpers.py:172-173` + `core/_phase_composite_discovery.py:596`
**`fit_stacked` pass-2 specs based on `_oof_*` columns are silently trained on all-NaN bases downstream**

Pass 2 adds OOF prediction columns `_oof_<spec.name>` to a LOCAL `df_aug` and discovers specs whose
`base_column` is such a column (the feature's headline purpose - residual-of-residual bases). Those
specs are merged into `self.specs_` and flow into suite training, where
`_build_full_column_from_splits` (`_misc_helpers.py:172`: missing column -> `continue`) silently
returns an all-NaN column for the `_oof_*` name. The transform then sees an all-NaN base -> every
row domain-invalid -> T imputed to a constant -> a garbage composite target is trained without any
error or warning. The pass-2 value is destroyed silently whenever `use_stacked_discovery=True`
(opt-in flag, wired at `_phase_composite_discovery.py:522`).

**Fix:** either (a) persist the OOF column recipe on the spec and have the suite rebuild `_oof_*`
columns via `composite_oof_predictions` before training, or (b) at minimum make
`_build_full_column_from_splits` WARN/raise on a fully-missing column and have `fit_stacked` drop
(with a log) pass-2 specs whose base is ephemeral. Regression test: stacked discovery end-to-end,
assert no spec trains on an all-NaN base.

## A7 - P1 - bug - `_stacked.py:299-311` + `core/_phase_composite_discovery.py:508-520, 585+`
**`fit_stacked_on_residual` pass-2 specs are auto-trained against RAW y with residual-fitted params**

The docstring says suite-side integration "is the follow-up step -- the current scaffolding returns
the specs for inspection", but the suite ALREADY wires the method (`_use_stacked_residual` branch,
`_phase_composite_discovery.py:508`) and then trains every spec in `_disc.specs_` (line 587+) by
applying `transform.forward(raw_y, base, fitted_params)` - with `fitted_params` that were fit on
`y_residual = y - pass1_pred`. The resulting target was never validated by any screening pass. The
`discovered_on_residual` marker set via `object.__setattr__` (line 305) is read NOWHERE in the
codebase (grep-verified), so downstream cannot route around it; it is also a non-field attribute on
a frozen dataclass (lost on `dataclasses.replace`). Pass-2 spec `target_col`/`name` also embed
`__y_residual__<target>`.

**Fix:** until residual-aware training exists, do NOT merge pass-2 specs into the `self.specs_`
consumed by the suite (return them on e.g. `residual_specs_`), or have the suite filter on the
marker - and make the marker a real `CompositeSpec` field. Integration test asserting pass-2
residual specs are not silently trained on raw y.

## A8 - P1 - perf - `_stacked.py:77-93`
**`fit_stacked` rebuilds the full train-slice `X_train` inside the per-spec loop**

`X_train = df.iloc[_train_idx_arr].reset_index(drop=True)` (pandas) or the polars mask+filter is
executed once per spec in `for spec in top_specs:` - up to `max_pass1_specs_to_stack=3` identical
materialisations of the train slice of a production-scale frame (multi-GB each on the 4Mx500
shape; OOM-grade on 100GB frames). The sibling `fit_stacked_on_residual` correctly hoists the
identical slice OUT of its loop (lines 209-226).

**Fix:** move the X_train construction above the `for spec in top_specs:` loop (verbatim move; the
loop body only needs `_factory` and `composite_oof_predictions`).

## A9 - P1 - bug - `forward_stepwise.py:83-133` + `transforms/linear.py:296-330` + `_fit.py:714-731`
**Forward stepwise has no finite-row masking: NaN-bearing lag bases (the flagship seed) disable or corrupt multi-base promotion**

`forward_stepwise_multi_base` never masks non-finite rows in `y` or the candidate arrays, and
`_linear_residual_multi_fit` does not either. Behaviour with NaN (lag features - the documented
canonical seed - routinely carry leading NaNs; `_auto_base_pool` arrays preserve them):
- K=1 (seed baseline `_cv_rmse([seed])`): `cond=1.0` short-circuit skips the guarded SVD, then
  `np.linalg.lstsq` on a NaN design raises `LinAlgError` -> propagates out -> caught by the blanket
  `except Exception` at `_fit.py:725` -> the WHOLE stepwise pass is skipped for that spec
  ("Keeping single-base spec" warning).
- K>=2 candidate trials: NaN poisons `col_norms` -> SVD raises inside the guarded block ->
  `cond=inf` -> `collinear_fallback` returns all-zero alphas + mean-y beta -> the candidate
  predicts a constant -> can never clear the marginal-gain gate. The candidate is silently
  unselectable even at 99.9% finite coverage.
Either way, multi-base auto-promotion (OPEN-1, benchmark-validated at geo-mean +83%) silently
no-ops exactly on autoregressive data. Phase A handles the same arrays via `domain_check` masking;
this path skipped that discipline.

**Fix:** inside `_cv_rmse_with_folds`, build a joint finite mask over y + the trial's base columns,
fit and score on masked rows; also mask before the final `_linear_residual_multi_fit` at
`_fit.py:738`. Regression test: seed base with 5 leading NaNs must still accept a genuinely
orthogonal second base.

## A10 - P2 - leak - `_screening_tiny.py:256-258, 618-620`
**Silent fallback from GroupKFold to shuffled KFold when the screening sample has < cv_folds groups**

`if int(np.unique(groups_clean).size) < cv_folds: groups_clean = None` reverts to shuffled KFold
with no log. That recreates exactly the prod incident the group-aware path was added for
(per-group memorisers promoted by random KFold; comment at `_tiny_rerank.py:114-123`) whenever the
20k screening sample contains e.g. 2 groups with `cv_folds=3` - silently. A 2-group GroupKFold
(n_splits=2) is strictly safer than a 3-fold shuffled KFold here.

**Fix:** use `GroupKFold(n_splits=min(cv_folds, n_groups))` when `n_groups >= 2`; only fall back to
KFold for n_groups < 2, and WARN when falling back.

## A11 - P2 - leak - `_screening_tiny.py:259-269, 621-629`
**Groups take precedence over `time_aware` with no warning - grouped time-series loses time ordering**

When both `groups` and `time_aware=True` are active, the splitter chain picks GroupKFold and the
time-aware request is silently dropped. GroupKFold folds mix past and future within every train
fold, so for panel data (group=entity, rows time-ordered) the rerank regains the future->past leak
that `time_aware` was added to remove. Also `_tiny_cv_rmse_y_scale` lacks the `cv_splitter` escape
hatch its raw sibling has (API asymmetry), so callers cannot inject a correct grouped-forward
splitter.

**Fix:** WARN once when both are requested; better, implement a grouped forward-chaining splitter
(val groups time-later than train groups) and add `cv_splitter` to `_tiny_cv_rmse_y_scale`.

## A12 - P2 - perf - `_screening_tiny.py:575, 741-754`
**`early_stop_threshold` (Pack #7, "saves 30-66% of fold-fit compute") has zero callers - the win is unrealised**

Grep over `src/mlframe` shows `early_stop_threshold` is referenced only inside `_screening_tiny.py`
itself. Neither `_tiny_model_rerank` nor any multiseed wrapper passes it, so the threshold is
always `inf` and the serial early-stop never fires. This is the `mi_direct_gpu_batched` failure
pattern (optimisation shipped but never wired). The natural wiring exists: the rerank knows
`raw_baseline * tol` and could abort folds of clearly-losing candidates. Note it also only applies
to the `n_jobs==1` serial branch.

**Fix:** wire `early_stop_threshold=threshold` (raw-baseline gate) into the rerank's
`_tiny_cv_rmse_y_scale_multiseed` calls when `require_beats_raw_baseline` is on; or remove the dead
parameter with a bench-attempt note. Add a test exercising the early-stop path.

## A13 - P2 - bug - `_screening_tiny.py:437-441, 495-499` + `_tiny_rerank.py:637-643`
**Wilcoxon per-seed arrays drop failed seeds without seed identity -> paired test can mis-pair**

The multiseed wrappers append only FINITE per-seed RMSEs (no seed index). If composite seed #2
fails while raw seed #3 fails, both arrays have length n-1 and pass the `len(comp)==len(raw)`
check at `_tiny_rerank.py:639`, but `diff = comp - raw` pairs seed 1-with-1, 3-with-2, ... - the
"paired" test runs on misaligned pairs. Fold failures are precisely the situations where one side
fails and the other does not.

**Fix:** return per-seed arrays as fixed-length (n_seed_repeats) with NaN for failed seeds; compute
diffs only on jointly-finite positions.

## A14 - P2 - bug - `_tiny_rerank.py:503-519`
**Per-bin raw baseline computed with different folds and seed protocol than the spec per-bin it gates against**

The regime-aware gate compares `spec_per_bin` (captured during the multiseed first pass, with
`time_aware=base_t_aware`, median across seeds) against `raw_per_bin` from a SINGLE-seed
`_tiny_cv_rmse_raw_y` call that does NOT pass `time_aware` (omitted at lines 503-519, while the
global raw baseline at 468/488 does pass it). On a monotone base the spec per-bin comes from TSS
folds and the raw per-bin from shuffled KFold (optimistically low) -> per-bin ratios biased high ->
spurious regime rejections. Default-off (`per_bin_n_bins=0`) hence P2, but when enabled it actively
mis-rejects exactly the lag-base specs it was built for.

**Fix:** pass per-base `time_aware` and use the same multiseed median protocol for the raw per-bin
as the first pass.

## A15 - P2 - bug - `_tiny_rerank.py:395-410`
**Per-bin fallback recompute omits `time_aware` and `groups`**

The ENS-P2-5 fallback path (recompute per-bin when missing from the first-pass cache) calls
`_tiny_cv_rmse_y_scale` without `time_aware` and without `groups`, so the recomputed per-bin
breakdown uses shuffled KFold even when the first pass used TSS/GroupKFold - the cached and
recomputed quantities are not the same estimator. Same defect class as A14, narrower trigger.

**Fix:** thread `time_aware=bool(_is_monotone_nondecreasing(base_screen))` and
`groups=_groups_screen` into the fallback call (both in scope).

## A16 - P2 - bug - `_tiny_rerank.py:207` + `ensemble/__init__.py:160-180`
**Time-aware auto-detection misses descending time-like bases and misfires on constant bases**

`base_t_aware = _is_monotone_nondecreasing(base_screen_local)`: (1) a monotone DECREASING time-like
base (countdown timers, depth-from-bottom well logs, reversed exports) returns False -> shuffled
KFold -> the future->past leak the detector exists to prevent; `detect_time_column_candidates`
(auto_detect.py:130-134) explicitly recognises descending time columns, so the two detectors
disagree. (2) A constant array passes (`diff >= 0` everywhere) -> TSS for no reason, wasting
training rows and de-aligning folds vs sibling specs.

**Fix:** treat weakly-monotone non-constant in EITHER direction as time-like (reverse row order for
descending before TSS) and exclude constant arrays.

## A17 - P2 - leak - `_stacked.py:95-99, 237-241` + `ensemble/feature_stacking.py:120-125`
**Stacked OOF predictions always use shuffled KFold - `time_aware` exists but is never passed**

`composite_oof_predictions` has `time_aware` / `cv_splitter` params, but both `fit_stacked` and
`fit_stacked_on_residual` call it with folds+seed only -> shuffled KFold on temporal data. The OOF
columns (pass-2 bases) and the residual target are then future-contaminated, inflating pass-2
discoveries; meanwhile the sibling tiny-rerank and forward-stepwise paths are time-aware (C-P2-11
flipped stepwise to time_aware=True by default for exactly this reason). Inconsistent leakage
discipline within the same feature.

**Fix:** detect time-awareness the same way the rerank does (monotone base / discovery time column)
or accept a `time_aware` kwarg on both fit_stacked* methods (default True like stepwise) and thread
it through.

## A18 - P2 - bug - `_stacked.py:229`
**`ranked[: max(1, len(ranked))]` is a no-op slice - residual variant has no pass-1 spec cap**

`fit_stacked` caps OOF computation at `max_pass1_specs_to_stack` (default 3);
`fit_stacked_on_residual` evaluates OOF for EVERY pass-1 spec - the dead expression
`max(1, len(ranked))` is a vestige of an intended cap parameter that was never added. Cost scales
linearly in pass-1 spec count, and "mean" aggregation over many weak specs dilutes the residual.

**Fix:** add `max_pass1_specs_to_aggregate: int = 3` mirroring the sibling, replace the no-op
slice, expose via config like `stacked_max_pass1_specs`.

## A19 - P2 - leak - `forward_stepwise.py:105-133` + `_fit.py:714-724`
**Forward stepwise CV is group-blind even when discovery is group-aware**

`forward_stepwise_multi_base` has no `groups` parameter and the `_fit.py` call site does not pass
the `_group_ids_for_rerank` that the tiny rerank uses. On grouped data the marginal-gain CV
(TimeSeriesSplit by default) mixes groups across train/val, so a candidate base that is a per-group
constant (group-mean proxy) shows phantom marginal gains - the same memoriser pathology the
group-aware rerank fix documented. TSS mitigates only if rows happen to be sorted by group.

**Fix:** add `groups` to `forward_stepwise_multi_base` (GroupKFold path mirroring
`_tiny_cv_rmse_raw_y`), pass the group ids at `_fit.py:714`.

## A20 - P2 - perf - `forward_stepwise.py:87-88, 114, 162`
**Per-call float64 pool copies and per-trial `column_stack` churn**

`candidates = {name: np.asarray(arr, dtype=np.float64)...}` re-materialises the WHOLE candidate
pool in float64 on every call - the `_fit.py` cache (`_pool_arrays_cache`) stores float32 arrays,
so K linear_residual specs sharing one pool still pay K full float64 conversions (~800 MB transient
each on the documented 4Mx25 shape). Inside the greedy loop, `np.column_stack` rebuilds the trial
matrix for every candidate in every round; a preallocated (n, max_k) buffer with column writes does
the same work with one allocation.

**Fix:** convert the pool to float64 once at the `_fit.py` cache level (or keep float32; the fit
already astype-promotes per call), and reuse a preallocated trial buffer in the candidate loop.

## A21 - P2 - bug - `bayesian.py:281-295`
**`bayesian_alpha_fit_bootstrap(n_bootstrap=0)` crashes in `np.percentile` on an empty array**

`alphas = np.empty(0)` -> loop skipped -> `np.percentile(alphas, 2.5)` raises on a zero-size array;
`np.mean([])` additionally emits a RuntimeWarning NaN. No validation on `n_bootstrap >= 1` (nor on
`ci_level` in (0,1), which silently produces wrong percentiles outside the range).

**Fix:** validate `n_bootstrap >= 1` and `0 < ci_level < 1` up front, raising ValueError with the
parameter name; tests for both.

## A22 - P2 - bug - `auto_detect.py:340-359`
**`detect_cat_columns` polars default candidates omit the int-as-cat heuristic the pandas branch has**

Pandas branch (352-359) includes integer columns with `nunique() <= max_unique`; the polars branch
(340-347) takes only `not _is_numeric_column(...)` - low-cardinality Int columns are silently never
candidates on polars frames. This is the exact asymmetry fixed for the sibling
`detect_group_column_candidates` by FUw3a1 (parity comment at lines 193-195;
`tests/training/test_regression_FUw3a1_composite_auto_detect_pl_pd_parity.py` covers ONLY the group
detector). Same frame, different flavour -> different candidate sets.

**Fix:** mirror the group-detector parity block (int dtypes + `n_unique() <= max_unique`) in the
polars branch of `detect_cat_columns`; extend the FUw3a1 parity test to `detect_cat_columns`.

## A23 - P2 - bug - `auto_detect.py:215-219, 352-359 (pandas) vs 191-207, 340-347 (polars)`
**Boolean columns: excluded by pandas branches, included by polars branches - and the docstring promises bools**

`pd.api.types.is_numeric_dtype(bool_series)` is True while `is_integer_dtype` is False, so bools
fail BOTH clauses of the pandas default-candidate filters in `detect_group_column_candidates` and
`detect_cat_columns`. In polars, Boolean is non-numeric -> bools are included. The
`detect_cat_columns` docstring explicitly says defaults are "all non-float columns (str / bool /
low-card int)" - wrong for pandas. (A bool column with min_unique=2 default in detect_cat IS a
legitimate binary categorical.)

**Fix:** add `or pd.api.types.is_bool_dtype(df[c])` to both pandas candidate filters; parity test
with a bool column on both flavours.

## A24 - P2 - bug - `auto_detect.py:399-403`
**`detect_cat_columns` score grows monotonically with cardinality, contradicting its own comment - ID-like columns rank first**

`info_bonus = n_unique / (log1p(n_unique) + 1)` is strictly increasing in n_unique, so
`score = coverage_top10 * info_bonus` REWARDS 500-1000-level columns over 10-100-level ones
whenever coverage does not fully collapse - the in-code comment claims the shape "rewards 10-100
categories more than 2 or 500". Concretely: a 500-level near-ID int (passes the
`min_samples_per_cat=20` gate on any >=10k-row frame with mild duplication) gets
score ~ 0.5*68.7 ~ 34 vs a clean 50-level categorical with full coverage at ~ 10.1. Classic
false-positive-on-int-ids risk for downstream FHC target encoding.

**Fix:** make the bonus unimodal (peak ~30-100 levels) or multiply by a sparsity penalty
`min(1, median_per_cat / (5*min_samples_per_cat))`; pin with a biz_value test where a 40-level true
categorical must outrank an 800-level pseudo-ID.

## A25 - P2 - bug - `auto_detect.py:124-134`
**Strict monotonicity (`diffs > 0`) rejects real timestamp columns with duplicate timestamps**

Numeric time-key detection requires STRICTLY increasing/decreasing values; any duplicate timestamp
(multiple events in one tick/second/batch - the norm in event data) fails `np.all(diffs > 0)` and
the column is silently not a candidate. The sibling `_is_monotone_nondecreasing` (ensemble) accepts
ties precisely because tied timestamps are normal. Datetime-dtype columns escape (always
candidates) but int/float epoch columns - the documented "timestamp-as-int" pattern - do not.

**Fix:** accept weak monotonicity (`diffs >= 0` with at least one `> 0`, i.e. non-constant), keep
direction reporting; score strictly-monotonic higher than tied if a tiebreak is wanted.

## A26 - P2 - bug - `auto_detect.py:340-359` (+ docstring 295-339)
**Datetime/temporal columns pass the `detect_cat_columns` candidate filter - target-encoding a date is a time-leak trap**

Datetime columns are neither float (pandas: `is_float_dtype` False, `is_numeric_dtype` False) nor
numeric (polars), so they land in the default candidate set of `detect_cat_columns`. A frame with
e.g. 100 distinct dates x >=20 rows each yields a high-scoring "categorical" whose FHC
target-mean/WoE encoding memorises per-period target means - out-of-time generalisation poison and
inconsistent with the time-vs-group separation the module's own time detector draws.

**Fix:** exclude temporal dtypes (datetime/date/timestamp/duration) from the default candidates of
`detect_cat_columns` (and `detect_group_column_candidates`); keep them reachable via explicit
`candidate_columns`.

## A27 - P2 - usability - `_tiny_rerank.py:218-271, 453-490`
**`cv_selector_mode` config is honoured by forward stepwise but silently ignored by the tiny rerank**

`_fit.py:682-685` reads `cv_selector_mode/alpha/confidence/quantile_level` from config and threads
them into `forward_stepwise_multi_base`. The rerank's `_tiny_cv_rmse_y_scale_multiseed` /
`_tiny_cv_rmse_raw_y_multiseed` calls pass none of them, so the underlying
`aggregate_fold_scores(mode=cv_selector_mode)` parameters (which both tiny-CV functions DO expose,
`_screening_tiny.py:219-222, 577-580`) stay at "mean". A user configuring `cv_selector_mode="t_lcb"`
gets penalty-aware selection in one phase and plain means in the adjacent phase with no indication.

**Fix:** thread the four `cv_selector_*` config fields into all four rerank call sites (composite
and raw, wilcoxon and plain), same getattr pattern as `_fit.py:682-685`.

## A28 - P2 - bug - `_screening_tiny.py:166-177`
**CatBoost tiny model lacks `allow_writing_files=False` - parallel fold fits race on `catboost_info/`**

`_build_tiny_model("catboost")` sets `verbose=False` only; CatBoost still writes
`catboost_info/learn_error.tsv` etc. to the CWD on every fit. The rerank runs many fits
concurrently (threading over specs and optionally folds) -> concurrent writers to the same
directory; on Windows this intermittently raises file-lock errors (which the blanket fold
`except Exception` then converts into silent NaN folds - fold count quietly drops), and it litters
the working directory.

**Fix:** add `allow_writing_files=False` (and a temp `train_dir` as belt-and-braces) to the
CatBoost kwargs in `_build_tiny_model`.

## A29 - LOW - bug - `forward_stepwise.py:110-112`
**Zero-base baseline sentinel `std(y)` is not CV-honest**

The empty-base baseline returns the full-sample `np.std(y)` while every candidate is scored by
out-of-fold RMSE of a fold-fitted model. The honest 0-base comparator is the CV-RMSE of the
train-fold-mean predictor (~ std but systematically higher, and much higher under TSS on trending
y). The mismatch shrinks the first base's relative gain on stationary y but can inflate it on
trending y under TSS. Only affects the `seed_bases=None` entry path. Fix: compute the 0-base
baseline through the same `kf.split` loop predicting `mean(y[train_idx])`.

## A30 - LOW - bug - `_screening_tiny.py:417-427` + `_tiny_rerank.py:241-247`
**`n_seed_repeats<=1` + degenerate-NaN + `use_wilcoxon` + per-bin: caller unpacks the per-seed array as the per-bin array**

When `_tiny_cv_rmse_y_scale` early-returns a bare NaN (e.g. `n < cv_folds*10`) the single-seed
wrapper returns `(nan, per_seed_arr)`; `_rerank_one_spec`'s wilcoxon branch reads `result[1]` as
`per_bin_first` -> an empty per-seed array is cached in `_per_bin_first_pass` (passes the
`is not None` check, blocking the fallback recompute) and the per-bin gate iterates over zero bins.
Benign today (the spec already has inf score) but the typing is wrong and fragile. Fix: normalise
the degenerate return to the full `(nan, full(n_bins, nan), per_seed)` shape.

## A31 - LOW - docs - `bayesian.py:50-105`
**Conjugate `bayesian_alpha_fit` lacks the "caller must filter to the valid domain" contract its bootstrap sibling documents**

NaN in `y`/`base` silently yields NaN coef / NaN posterior (or the degenerate-inf dict via the
LinAlgError path) with no warning; the bootstrap variant's docstring states the caller-filters
contract, the conjugate one does not, although they are documented as drop-in swaps. Fix: copy the
contract line (or add an explicit joint finite-mask - one line, removes the trap for both).

## A32 - LOW - bug - `auto_detect.py:205 vs 218`
**Polars `n_unique()` counts null as a distinct value; pandas `nunique()` excludes NaN - off-by-one parity at the `max_unique` boundary**

In the default-candidate gates of `detect_group_column_candidates` (and any parity fix for
`detect_cat_columns`), an int column with exactly `max_unique` non-null values plus nulls passes on
pandas and fails on polars. Fix: use `drop_nulls().n_unique()` in the polars gates.

## A33 - LOW - usability - `forward_stepwise.py:58, 87-88`
**Candidate array lengths are documented but never validated**

The docstring requires all candidate arrays to have `len(y_train)`; a mismatched array surfaces
later as an opaque `column_stack` ValueError. One-line validation loop naming the offending column.

## A34 - P2 - test-gap - `tests/training/test_residual_stacked_suite_integration.py` (+ `tests/training/composite/`)
**No end-to-end test covers stacked pass-2 specs through suite TRAINING, nor the Wilcoxon-gate power floor**

`test_residual_stacked_suite_integration.py` asserts the phase INVOKES `fit_stacked_on_residual`
and that specs come back; nothing asserts the merged pass-2 specs can be trained downstream with a
non-NaN base / correct target scale (would have caught A6/A7). No test exercises
`use_wilcoxon_gate=True` at default seeds (would have caught A2), and none pins that multiseed
under a monotone base actually produces seed-to-seed variation (A3). Suggested additions:
(1) biz_value: stacked discovery end-to-end, assert at least one pass-2 spec trains to a finite
non-constant composite target; (2) unit: wilcoxon gate with 3 seeds must raise/warn;
(3) unit: multiseed + monotone base returns >1 distinct per-seed value or short-circuits to 1
repeat.

---

### Verified-clean (checked, no finding)
- `bayesian_alpha_fit` conjugate NIG math: posterior scale s2*(XtX)^-1, t-marginal var factor
  v/(v-2), CI = t_q x scale (not std - correct), IG sampling via Gamma(v/2, scale=2/ssr) inverse,
  sigma2_posterior_mean = ssr/(v-2) - all correct; degenerate n<4 / singular-XtX branches sound.
- Bootstrap percentile units (2.5/97.5 for ci_level=0.95) correct; ddof=1 std correct.
- Rerank tie handling: stable np.lexsort((names, scores)) - deterministic (audit-notes item still
  good in current code).
- Forward-stepwise determinism: dict-insertion-order iteration + strict `<` best-update -> fully
  deterministic given the caller's pool order; `_fit.py` pool order is itself deterministic.
- Condition-number guard for multi-base lives in `_linear_residual_multi_fit` (centered-column SVD,
  fallback to mean-predictor) - correct placement; CV-RMSE then rejects the degenerate trial.
- Early-stop partial-sum bound (`sum > thr * cv_folds`) is conservative for every
  `aggregate_fold_scores` mode with direction="min", and `aggregate_fold_scores` falls back to mean
  for size-1 inputs - no edge crash (the issue is only that nothing calls it, A12).
- `_y_train_clip_bounds(y_clean[train_fold])` - train-fold-only fitting discipline respected in the
  y-scale CV (no val leakage into the clip).
- Module-level `_y_train_clip_bounds` import (race fix) present at `_screening_tiny.py:42`.
