# A3 — Ensembling / Stacking critique (read-only)

Scope: ensembling/stacking as wired into `train_mlframe_models_suite`. Two distinct ensemble families exist:

1. **Cross-target ensemble (CT_ENSEMBLE)** — `CompositeCrossTargetEnsemble` (NNLS / Ridge linear_stack / oof_weighted / mean) built per original target in `_phase_composite_post_xt_ensemble._build_cross_target_ensemble_for_target`, fed by `compute_oof_holdout_predictions`. This is the only *fitted meta-learner* path; almost all findings live here.
2. **Per-target simple ensemble** — `_finalize_per_target_ensembling` -> `mlframe.models.ensembling.score_ensemble`. Fixed combiners (arithmetic / harmonic / RRF / median / quad / geo), selected (not fitted) by an OOF-first ranking metric (`_ensemble_chooser`). No in-sample weight-fitting; discipline here is sound.

Verification was by reading source at file:line. Confirmed claims only; alternative readings flagged.

---

## P0

### A3-01 — Cross-target stack weights are derived on the VAL split (the ES detector), not an honest holdout
- **Severity:** P0
- **File:** `src/mlframe/training/core/_phase_composite_post_xt_ensemble.py:602-622`, `:724-745`; `src/mlframe/training/composite_ensemble.py:742-765`, `:338-365`
- **What's wrong:** The default OOF source is `oof_holdout_source="external_val"` (`_phase_composite_post_xt_ensemble.py:603-606`). When taken, `_ext_X = filtered_val_df` and `_ext_y = y[filtered_val_idx]` (`:610-622`), and these are passed as `external_holdout_X/_y` into `compute_oof_holdout_predictions` (`:740-742`). That routes to `_compute_oof_with_external_holdout` (`composite_ensemble.py:747-765`), which fits each component clone on the **full train** and predicts on the **val frame** (`composite_ensemble.py:359-364`, docstring line 360: "typically the suite's val split"). The resulting `(n_val, K)` prediction matrix and `y_val` are then handed to `from_nnls_stack` / `from_linear_stack` / `from_train_metrics` as `component_predictions` + `y_train` (`_phase_composite_post_xt_ensemble.py:935-956`), i.e. the stack weights AND the honest-OOF gate (`:978-1019`) are both computed on val. But val is the early-stopping detector for the very boosters that are components — it is biased optimistic, not an honest holdout. Per the project's own val/test/OOF terminology this is exactly the surface that must NOT be used for weight estimation or model selection.
- **Why it matters:** This is the textbook stacking leakage failure mode, just displaced one level: the components were early-stopped against val, so their val predictions are optimistically biased, and NNLS/Ridge will over-weight whichever component most overfit the val signal. The deployed ensemble's measured RMSE (the gate's number at `:979`) corresponds to a biased surface, so the gate can wave through an ensemble that loses on true holdout. Note the irony: the external_val path was introduced (docstring `composite_ensemble.py:542-551`) to fix a *train-tail distribution mismatch*, trading a correctness problem (biased weights) for a representativeness problem.
- **Concrete recommendation:** Do not derive stack weights on val. Either (a) use true K-fold OOF on train (the `kfold>1`, `time_ordering=None` path at `composite_ensemble.py:611-740` already produces honest `cross_val_predict`-style OOF) as the default weight surface, or (b) keep external_val only as the *gate/representativeness* split while deriving weights from train-K-fold OOF, or (c) carve a dedicated stacking holdout never touched by ES. At minimum, emit a loud WARN at the call site that external_val weighting double-dips the ES surface (mirroring the warning `from_train_metrics` already emits for train-RMSE at `composite_ensemble.py:435-439`).
- **Confidence:** High (wiring + docstrings confirm val is the surface; the only uncertainty is whether `filtered_val_df` in a given suite run was genuinely used for ES, but for boosters in this suite it is).

---

## P1

### A3-02 — `CompositeCrossTargetEnsemble` has no `__getstate__`; the stashed train-prediction matrix + train-y survive every pickle, and `discard_train_matrix` is never called
- **Severity:** P1
- **File:** `src/mlframe/training/_composite_cross_target_ensemble.py:281-283`, `:373-374`, `:721-743`; `src/mlframe/training/core/_phase_composite_post_xt_ensemble.py` (no `discard_train_matrix` call anywhere)
- **What's wrong:** `from_linear_stack` stashes `instance._linear_stack_train_preds = component_predictions[finite]` and `_linear_stack_train_y = y[finite]` (`:281-282`); `from_nnls_stack` stashes the analogous `_nnls_stack_train_preds` / `_nnls_stack_train_y` (`:373-374`). These are `(n_stack_rows, K)` float64 arrays. The class defines `discard_train_matrix()` to strip them before persistence (`:721-743`, its own docstring at `:728-732` says "Call this method right before persistence") — but a grep of the entire `training/` package finds **zero callers**: the suite never invokes it before saving. There is also no `__getstate__` on the class to exclude these attrs at pickle time (confirmed: `__getstate__` appears on `lgb_shim`, `xgb_shim`, `neural/base`, `_calibration_models`, but not on `CompositeCrossTargetEnsemble`). So every saved suite that built a `linear_stack`/`nnls_stack` CT ensemble serialises the full stacking matrix to disk and back into RAM on load.
- **Why it matters:** On the external_val path the matrix is `(n_val, K)` (small), but on the train-OOF / train-RMSE-proxy fallback paths (`:909-934`) the matrix is `(n_train, K)`. On this project's 100GB+ frames, `n_train` can be tens of millions of rows; `8 * n_train * (K+1)` bytes is gigabytes of dead weight per saved target, doubling load-time RAM at the worst moment. This is the exact "caching live training data into a persisted object" pattern the project memory warns about (`feedback_runtime_caches_break_pickle`), minus the un-picklable-object hazard (the arrays themselves pickle fine — it's a memory/size bug, not a pickle-crash bug).
- **Concrete recommendation:** Add `__getstate__`/`__setstate__` to `CompositeCrossTargetEnsemble` that drops `_*_train_preds`/`_*_train_y` (and re-inits them to None on load), OR call `discard_train_matrix()` in the finalize/save path before serialisation. If the dropout-refit-at-predict capability must survive a round-trip, gate retention on a size threshold (keep only when `n_stack_rows` is small). Pair with a regression test asserting the pickled blob size is bounded.
- **Confidence:** High.

### A3-03 — Time-aware split (`time_ordering`), per-row `sample_weight`, and `group_ids` for the OOF refit are ALWAYS `None` — dead code from an unbound `ctx`
- **Severity:** P1
- **File:** `src/mlframe/training/core/_phase_composite_post_xt_ensemble.py:585-593`, `:636-644`
- **What's wrong:** `_ctx_ts_full = getattr(ctx, "timestamps", None) if "ctx" in dir() else None` (`:586`), `_ctx_sw_dict = getattr(ctx, "sample_weights", None) if "ctx" in dir() else None` (`:593`), and `_ctx_groups = getattr(ctx, "group_ids", None) if "ctx" in dir() else None` (`:637`). `ctx` is **not a parameter** of `_build_cross_target_ensemble_for_target` and is not bound anywhere in the function (confirmed against the signature `:321-346`), so `"ctx" in dir()` is always False and all three resolve to `None` unconditionally. The function's own module docstring even admits this (`_phase_composite_post_xt_ensemble.py:5`: "Undefined `ctx` references ... evaluated to `None`"). Consequently: `time_ordering=None` (so the OOF holdout never uses the time-aware trailing-slice unless a base column happens to be monotone via the `_is_monotone_nondecreasing` auto-probe), `sample_weight=None` (weighted-loss / LTR suites get unweighted NNLS/Ridge weights), and `group_ids=None` (the group-aware eval carve in `_carve_inner_eval_split` is never exercised, so early-stopping inside the OOF refit can see same-group leakage — the precise pathology that `_carve_inner_eval_split`'s docstring at `composite_ensemble.py:243-249` says it exists to prevent).
- **Why it matters:** Three intended honesty mechanisms are silently inert on the real suite path. The most damaging is `group_ids=None`: the suite's whole motivation for the group-aware carve (avoid val_RMSE 10.64→13.34 distortion, per `composite_ensemble.py:246-249`) is defeated because the group ids never arrive. `sample_weight=None` means weighted suites estimate weights against the wrong loss. This is a wiring bug masquerading as a feature.
- **Concrete recommendation:** Pass the real `ctx` (or the explicit `timestamps` / `sample_weights` / `group_ids` arrays) into `_build_cross_target_ensemble_for_target` from `run_composite_post_processing` and replace the `if "ctx" in dir()` guards with direct use. Add a regression test that builds a CT ensemble on grouped data and asserts the group-aware carve path is taken (e.g. non-None `group_ids` reaches `_carve_inner_eval_split`).
- **Confidence:** High (statically certain: `ctx` is unbound in the frame).

### A3-04 — Per-batch refit at predict time uses the val-derived (biased) stashed matrix; dropout silently re-introduces the A3-01 leakage and is also non-deterministic across batches
- **Severity:** P1
- **File:** `src/mlframe/training/_composite_cross_target_ensemble.py:549-622`
- **What's wrong:** When any component fails to predict on a batch, the non-convex (`linear_stack`/`nnls_stack`) predict path refits Ridge/NNLS on the surviving columns of `_*_train_preds` (`:566-622`). That stashed matrix is whatever was used to build the ensemble — on the default path, the **val** predictions (A3-01). So a transient per-batch component dropout silently re-solves the stack on the biased val surface, and the refit weights differ from the originally-deployed weights. Because the refit is keyed/cached per surviving-subset (`:564`, `_refit_cache`), two inference batches with different dropout patterns get different effective models, and a batch where a component drops then recovers flips weights back and forth.
- **Why it matters:** Inference is no longer a pure function of the input row — it depends on which components happened to predict cleanly in the batch, and the refit inherits the val-bias. For online/streaming serving this is a correctness and reproducibility hazard. The convex strategies (`mean`/`oof_weighted`) handle dropout by simple renormalisation (`:629-636`) which is benign; the issue is specific to the fitted stacks.
- **Concrete recommendation:** (1) Fix A3-01 so the stashed matrix is honest OOF, not val. (2) Prefer a dropout policy that does not refit (e.g. zero-out the dropped column's contribution and renormalise the convex part, or fall back to `oof_weighted` weights) so predict stays a pure function of inputs; if refit must stay, document that it is non-deterministic across dropout patterns and pin the train matrix used.
- **Confidence:** High on the mechanism; Medium on real-world blast radius (depends how often components actually drop out at predict time).

### A3-05 — `_train_pred_cache` is keyed on `id(inner_model)` + `(id(frame), shape)`, vulnerable to CPython id recycling and shape-collision cross-contamination
- **Severity:** P1
- **File:** `src/mlframe/training/core/_phase_composite_post_xt_ensemble.py:485`, `:490-498`, `:919`, `:925-933`
- **What's wrong:** The train-prediction cache key is `(id(_inner_for_cache),) + (id(filtered_train_df), getattr(filtered_train_df, "shape", None))` (`:485`, `:490-498`). The code comment at `:484-485` claims "Frame-content key (id(frame)+shape) shields against id() recycling" — but `id()` + shape is **not** a content fingerprint: CPython recycles object ids after GC, and two different frames with identical shape collide on `(id, shape)` if the first was freed and the second reused the address. The companion `composite_ensemble.py` OOF cache docstring (`:303-312`, "C-P2-2: DO NOT include `id(train_X)`") explicitly warns this is wrong and recommends a content fingerprint — yet this sibling cache does exactly the id-based keying it warns against. Within a single `_build_cross_target_ensemble_for_target` call the frames are live so it's safe; the hazard is across calls/waves where a stale prediction (wrong model or wrong frame) can be served.
- **Why it matters:** A stale cache hit substitutes one component's predictions for another's in the stacking matrix, silently corrupting the weights and the gate decision with no error. It is a low-probability-per-call but high-impact-when-hit silent correctness bug, and it directly contradicts the project's own documented anti-pattern.
- **Concrete recommendation:** Key the cache on a content fingerprint of the frame (the project already has `pipeline_cache.fingerprint_df` / `_hash_frame`, referenced in `composite_ensemble.py:310-311`) combined with a stable per-model token, not `id()`. Or scope the cache strictly to a single build call (a local dict that cannot outlive the frames) so cross-call recycling is impossible.
- **Confidence:** Medium-High (mechanism is real and documented elsewhere in the same package; whether a collision has fired in prod is unverified).

---

## P2

### A3-06 — `_ens_pred = _ensemble.predict(filtered_train_df)` computed once per CT ensemble and never used — a wasted full-train K-component predict pass
- **Severity:** P2
- **File:** `src/mlframe/training/core/_phase_composite_post_xt_ensemble.py:962`
- **What's wrong:** Inside the OOF gate `try`, `_ens_pred = _ensemble.predict(filtered_train_df)` runs a full ensemble prediction over the entire train frame, but the result is never read again (verified: `_ens_pred` appears only at `:962`; the gate computes `_ens_holdout` from the cached `_oof_pred_matrix` at `:969-977`, and the later report uses a different `_ens_preds` per split at `:1117`). It is dead compute.
- **Why it matters:** `_ensemble.predict` runs every surviving component's `predict` over all train rows (`composite_ensemble.py:521-528`). On 100GB+ frames with K components this is a large, pointless pass on the ensembling hot path, repeated per target.
- **Concrete recommendation:** Delete line 962.
- **Confidence:** High.

### A3-07 — No diversity / redundancy control: near-duplicate components can collectively dominate the stack
- **Severity:** P2
- **File:** `src/mlframe/training/core/_phase_composite_post_xt_ensemble.py:411-471`, `:855-956`; `src/mlframe/training/_composite_cross_target_ensemble.py:476-503`
- **What's wrong:** Components are pooled as raw#i + per-spec composite entries + lag_predict (`:411-471`) with no decorrelation step before weighting. The optional `stacking_aware_gate` (`:866-908`) is **off by default** (`getattr(..., "stacking_aware_gate_enabled", False)`), and even when on it only drops below-min-weight members — it does not collapse two highly-correlated members. The `composite_stacking.residual_correlation_matrix` / `max_off_diagonal_correlation` diagnostic exists (`composite_stacking.py:27-81`) but is not wired into the CT path at all. `from_train_metrics` explicitly declines an independence-bound gate (`_composite_cross_target_ensemble.py:476-487`). So K boosters that are near-duplicates (same features, same target) can split a large combined weight, over-representing one signal.
- **Why it matters:** Stacking value comes from diversity; redundant members inflate variance of the weight estimate and can dominate. The risk is bounded here because NNLS/Ridge on correlated columns tends to concentrate weight (so duplicates don't double-count linearly), but on the convex `oof_weighted` path duplicates each get a full gain-share and do double-count.
- **Concrete recommendation:** Wire the existing `residual_correlation_matrix` diagnostic into the pool: drop or merge a member whose residual correlation with a higher-ranked member exceeds a threshold (~0.95) before weighting. At minimum flip `stacking_aware_gate_enabled` on by default per the project's "enable corrective mechanisms by default" rule, after a biz_value check.
- **Confidence:** Medium (the harm is real but partially self-limiting under L2/NNLS).

### A3-08 — `from_train_metrics` `oof_weighted` weights are gains over the WORST component when no baseline passed — fragile and undocumented at the call site
- **Severity:** P2
- **File:** `src/mlframe/training/_composite_cross_target_ensemble.py:443-503`; call site `core/_phase_composite_post_xt_ensemble.py:951-956`
- **What's wrong:** The `oof_weighted` strategy calls `from_train_metrics(component_oof_rmse=_oof_rmses.tolist(), baseline_oof_rmse=None)` (`:951-956`). With `baseline_oof_rmse=None`, the baseline defaults to `max(rmses)` (`:452`), so weights are `gain = max(rmse) - rmse`, normalised (`:464`, `:488`). The single worst component therefore always gets weight ~0 and the spread is driven entirely by the gap to the worst member — a non-robust scale. If two components tie for worst, both get ~0 weight even if they are individually decent. The dummy floor / lag_predict baseline that the suite already computes (`:790-803`) would be a far more meaningful `baseline_oof_rmse` and is NOT passed here.
- **Why it matters:** Weights depend on an arbitrary internal reference (worst member) rather than a meaningful benchmark, so adding/removing a bad component shifts all weights. The suite has a real baseline available and doesn't use it.
- **Concrete recommendation:** Pass the strongest-dummy / lag_predict OOF RMSE as `baseline_oof_rmse` into the `oof_weighted` call so weights are gain-over-naive (the convention the method was designed for, per `:387-401`).
- **Confidence:** High on the mechanism; Medium on impact (oof_weighted is only one of four strategies and not the default everywhere).

### A3-09 — `from_linear_stack` folds intercept into prediction but stores no provision to keep weights honest under the convex-vs-nonconvex contract on the gate path
- **Severity:** P2
- **File:** `src/mlframe/training/core/_phase_composite_post_xt_ensemble.py:964-977`; `src/mlframe/training/_composite_cross_target_ensemble.py:509-559`
- **What's wrong:** The OOF gate recomputes ensemble holdout preds inline rather than calling `predict`. For `linear_stack` it does `(_oof_pred_matrix * _w_full).sum(axis=1) + _intercept` (`:969-972`); for the non-linear-stack branch it **renormalises** the weights `_w_norm = _w_full / sum` before combining (`:974-977`). But `nnls_stack` is built with `is_convex=False` and deliberately NOT renormalised (`_composite_cross_target_ensemble.py:351-369`, comment: "Don't renormalise ... so the gate's measured RMSE corresponds to the deployed model"). The gate's `else` branch at `:974-977` renormalises **all** non-linear_stack strategies, including `nnls_stack`. So for an `nnls_stack` whose weights don't sum to 1, the gate evaluates a *renormalised* predictor that is NOT the one `predict` deploys (`predict` for nnls uses raw weights, `:622`). The gate's RMSE then doesn't correspond to the deployed model — the exact failure the nnls code comment says it avoids.
- **Why it matters:** The honest-OOF gate (the safety net that falls back to best-single) can pass/fail based on a different predictor than what ships, so a bad nnls_stack can slip the gate or a good one be wrongly rejected. Selection-altering, not just cosmetic.
- **Concrete recommendation:** In the gate, branch on `is_convex` (or strategy) exactly as `predict` does: use raw weights + intercept for `linear_stack`, raw weights (no renorm) for `nnls_stack`, renormalise only for `mean`/`oof_weighted`. Better: call `_ensemble.predict` on the held-out X (or reuse `_oof_pred_matrix` through the same combine code path `predict` uses) so the gate and deploy are guaranteed identical.
- **Confidence:** High (the renormalise-all `else` at `:974` provably mishandles nnls_stack).

### A3-10 — OOF pre-screen drops components on a LEAKY val_RMSE, compounding the A3-01 val dependence
- **Severity:** P2
- **File:** `src/mlframe/training/core/_phase_composite_post_xt_ensemble.py:646-722`
- **What's wrong:** Before the (already-val-based) OOF refit, a pre-screen predicts each component on `_ext_X` (val) with the already-trained model (no refit) and drops components whose `leaky_rmse / 1.5 > dummy_floor` (`:691-698`). The code is explicit that this RMSE "leaks ... through early-stopping signal" (`:653-658`). So the *component set* that even reaches weighting is filtered on the biased val surface, on top of A3-01 weighting on val. The 1.5 safety margin mitigates but does not remove the bias.
- **Why it matters:** A component that looks weak on the ES-biased val but is genuinely strong on true holdout can be pre-dropped and never get a weight. Compounds A3-01.
- **Concrete recommendation:** Once A3-01 is fixed to use honest OOF, run the pre-screen on the same honest surface (or keep it as a pure speed gate with an even more generous margin and a WARN). Document that the pre-screen is leaky in the dropped-components log line.
- **Confidence:** High that it uses val; Medium that it changes a final decision (the post-OOF dummy-floor gate re-runs honestly at `:777-853`, so pre-screen only affects which components are *eligible*).

---

## Low

### A3-11 — Module-level `_OOF_HOLDOUT_CACHE` is dead on the suite path (cache_key never passed)
- **Severity:** Low
- **File:** `src/mlframe/training/composite_ensemble.py:303-335`, `:588-597`; call site `core/_phase_composite_post_xt_ensemble.py:724-745`
- **What's wrong:** `compute_oof_holdout_predictions` is called from the suite without `cache_key=` (verified: no `cache_key=` in the call at `:724-745`), so `_full_key` stays `None` and the 16-entry LRU OOF cache is never consulted or populated. The cache + its careful LRU/eviction logic (`:321-335`) is dead code on this path.
- **Why it matters:** Not a correctness bug, but the careful "don't use id() in the key" design (`:303-312`) protects a cache nobody fills. Either wire a content-based `cache_key` (would save repeated OOF refits across waves) or note the cache as intentionally caller-managed only.
- **Concrete recommendation:** If repeated OOF refits across waves are a measured cost, pass a content-fingerprint `cache_key`. Otherwise document the cache as unused-by-suite to avoid future readers assuming it's active.
- **Confidence:** High.

### A3-12 — Auto time-split can silently flip the train-OOF holdout to a trailing slice when ANY base column is monotone
- **Severity:** Low
- **File:** `src/mlframe/training/composite_ensemble.py:778-788`, `:160-180`
- **What's wrong:** On the train-tail (non-external) OOF path with `time_ordering=None` (which is *always* the case given A3-03), the code probes every base column and switches to a trailing-slice holdout if any is monotone non-decreasing (`:781-788`). A base feature that happens to be monotone (a row counter, an accumulating sum, a sorted id) but is NOT a timestamp will silently change the holdout from random-shuffle to trailing-slice, altering the leakage characteristics and the resulting weights, with only an INFO log.
- **Why it matters:** Implicit, content-dependent behaviour change in the honest-holdout construction; surprising and hard to debug.
- **Concrete recommendation:** Only auto-switch on an explicitly designated time/order column, not on any monotone base. Given A3-03 means `time_ordering` never arrives anyway, the cleaner fix is to wire real timestamps (A3-03) and drop the base-column auto-probe, or guard it behind an opt-in flag.
- **Confidence:** Medium (depends on whether a monotone non-timestamp base occurs; on AR/lag targets a lagged-timestamp base is plausible, which is the intended case, but false positives are possible).

### A3-13 — `_carve_inner_eval_split` in the external-holdout path is hardcoded `random_state=0` and group-blind for composite components
- **Severity:** Low
- **File:** `src/mlframe/training/composite_ensemble.py:417-422`, `:449-454`, `:853-855`, `:886-888`
- **What's wrong:** In `_compute_oof_with_external_holdout` the eval-split carve for the raw branch passes `group_ids=group_ids` (`:453`) but the composite branch passes `_group_for_valid` derived only when shapes match (`:409-416`) and the single-split (`:853-855`, `:886-888`) carves use `random_state=0` with NO `group_ids` at all. So early-stopping inside those refits can see same-group leakage for composite components even when group ids are available — inconsistent with the raw branch's group-aware carve.
- **Why it matters:** Inconsistent honesty between raw and composite components in the same ensemble; composite OOF RMSE can be optimistic relative to raw, skewing weights toward composite members. Minor because A3-03 means group_ids is None today anyway, but this becomes a real bug the moment A3-03 is fixed.
- **Concrete recommendation:** Thread `group_ids` into the composite-branch carve in both `_compute_oof_with_external_holdout` and the single-split path, mirroring the raw branch.
- **Confidence:** Medium.

### A3-14 — MTR per-column NNLS ensemble fits weights on the val fold (in-sample-ish), acknowledged but shipped
- **Severity:** Low
- **File:** `src/mlframe/training/core/_phase_composite_post_xt_ensemble.py:372-409`, `:123-190`
- **What's wrong:** For MULTI_TARGET_REGRESSION the per-column NNLS ensemble (`MTRPerColumnEqualMeanEnsemble.fit`) is fit on `_fit_X = filtered_val_df` / `_fit_y_val` (`:379-408`). Its own docstring admits the held-out preds are "in-sample by default unless the caller passes a CV / OOF stack" (`:228-231`) and that val is "the closest stand-in for an honest-OOF set" (`:373-378`). Same val-as-weight-surface issue as A3-01 but for the MTR path, and the equal_mean fallback is benign.
- **Why it matters:** Same leakage class as A3-01, scoped to MTR; lower severity because MTR is a narrower path and equal_mean (no fit) is the common default.
- **Concrete recommendation:** Route honest train-K-fold OOF preds into `.fit()` instead of the val fold (the `.fit()` contract already supports any (X, y)); until then keep equal_mean as default for MTR.
- **Confidence:** High (explicitly documented in-source).

---

## Areas checked and found OK (no finding)

- **Per-target simple ensemble selection discipline** (`_ensemble_chooser.py:32-126`, `_phase_train_one_target_ensembling.py`): flavour ranking is OOF-first with explicit fallback to val then test, and a one-time WARN when test is used for selection (`_ensemble_chooser.py:110-117`). `score_ensemble` defaults `sort_metric` to `oof.*` and warns on val/test (`models/ensembling.py:130-152`). These are fixed combiners (no fitted meta-weights), so no in-sample weight leakage. Sound.
- **Convex-weight constraints** (`_composite_cross_target_ensemble.py:82-99`): `is_convex=True` enforces sum-to-1 + positivity with finite checks; `is_convex=False` for solver outputs (Ridge/NNLS) correctly skips renormalisation. Constraint handling at construction is sane (the gate-path inconsistency is A3-09, not here).
- **DiscoveryCache content-hash + size guard** (`composite_cache.py:182-426`, `:656-728`): keys are blake2b content fingerprints (row-order sensitive, dtype-aware, row-count folded in); a `MLFRAME_DISCOVERY_CACHE_MAX_BYTES` ceiling (default 1 GiB) refuses oversized entries to avoid loading a 100GB+ frame into RAM (`:670-699`); pickles only dataclass-derived dicts, atomic writes with fsync, sha256 sidecar verification. Collision-safe `_safe_key` (`:730-745`). This cache is well-built and does NOT risk caching live framework objects or huge frames. The pickle-breakage concern from the prompt does not apply to `DiscoveryCache` (it caches discovery specs, not Trainer/CUDA objects). The real pickle/memory hazard is A3-02 on the ensemble object, not in `composite_cache.py`.
- **Cross-target temporal leakage across targets:** the cross-target ensemble combines models for the SAME original target (raw + composite specs of that target + lag_predict), not predictions from other targets used as features — so there is no inter-target target leakage. The "cross-target" name refers to ensembling across the composite-target transforms of one target. No finding.
