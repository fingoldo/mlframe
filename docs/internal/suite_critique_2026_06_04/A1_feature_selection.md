# A1 — Feature Selection as wired into `train_mlframe_models_suite`

READ-ONLY critique. Scope: feature selection (FS) as the suite actually uses it.
Every claim below is anchored to a file:line that was read. Where a behavior is
defensible I say so.

## How FS is wired (verified data-flow)

1. **Suite-once unsupervised pre-screen** (`_phase_train_one_target_pre_screen.py`):
   default ON (`FeatureSelectionConfig.pre_screen_unsupervised=True`,
   `_feature_selection_config.py:94`). Drops variance≈0 / nulls>99% columns from the
   **train split only** (`compute_unsupervised_drops` reads `ctx.filtered_train_df` /
   `train_df_polars` / `train_df_pd` — all train-only, `pre_screen.py:11-14`,
   `_phase_train_one_target_pre_screen.py:91-97`), then re-applies the same drop set
   to val/test mirrors (`:111-120`). Latched once per suite via `ctx._pre_screen_done`.

2. **Per-target supervised selectors** (`_build_pre_pipelines` in
   `_setup_helpers_pre_pipelines.py`): MRMR (`use_mrmr_fs`), RFECV (`rfecv_models`),
   BorutaShap (`use_boruta_shap`), plus `custom_pre_pipelines`. All default OFF/empty
   (`_feature_selection_config.py:51-58`). ShapProxiedFS is registered
   (`registry.py:146`) but NOT auto-wired — opt-in only via `custom_pre_pipelines`.

3. The selector is carried as a `pre_pipeline` and fit inside the trainer via
   `_apply_pre_pipeline_transforms` (`_pipeline_helpers.py:534`): `fit_transform` on
   `train_df` + `train_target` (`:791-799`), val/test get `transform` only
   (`:776`, `:710-714`). `train_df`/`train_target` are the train split
   (`_trainer_train_and_evaluate.py:341-345,390-404`). **FS is fit train-only and
   sits OUTSIDE any model CV fold** (it's a once-per-(target,model-bucket) pre-fit).

## Leakage verdict (dimension 1): SAFE at the suite boundary

- Pre-screen: train-only by contract, verified (`pre_screen.py:66-189` never touches
  val/test; drop set computed on train, applied to all mirrors).
- MRMR / RFECV / ShapProxiedFS all receive train-only (X, y); val/test only ever see
  `.transform`. RFECV's internal CV is honest (it CV-scores subsets, `_rfecv_fit.py`),
  ShapProxiedFS's search/holdout split is internal to the train data
  (`shap_proxied_fs.py:1134-1137`), so its "honest holdout" is a within-train slice,
  never the suite's OOS/test.
- `groups` is sliced to `train_idx` before being threaded to the selector
  (`_trainer_train_and_evaluate.py:358-367`), and `sample_weight` likewise
  (`:373-388`).
- Provenance is stamped `source="train_only"` for pre-screen and rfecv
  (`_phase_train_one_target_pre_screen.py:101-108`, `_rfecv_fit.py:443-454`).

No target/MI/SHAP leakage from held-out data was found on the default or opt-in paths.
The findings below are correctness / defaults / efficiency / integration nuances, NOT
held-out-data leaks.

---

## Findings

### A1-01 — MRMR silently ignores `groups`: cross-group leakage in MI on panel/session data
- **Severity:** P1
- **File:** `src/mlframe/feature_selection/filters/mrmr.py:2435-2460`
- **What's wrong:** `MRMR.fit(groups=...)` is "ACCEPTED FOR API COMPAT BUT NOT
  CONSUMED" — MI is estimated per-row, ignoring group structure. With the default
  `strict_groups=False` it emits only a `UserWarning`. The suite DOES thread
  `groups` (sliced to train) into the selector fit
  (`_trainer_train_and_evaluate.py:354-403`), so on a group-aware suite (panel /
  user-session / sliding-window data) MRMR's relevance/redundancy MI is computed
  across rows that share a group. That over-states relevance for any feature that is
  constant-within-group / leaks the group identity, and the selected set is biased the
  same way the model's own group-aware CV is trying to prevent. This is an
  *estimation* leak (group structure ignored), not a train/test leak.
- **Why it matters:** The suite's whole point of `use_groups`/GroupKFold is to honor
  group boundaries. RFECV honors them (`_resolve_cv_and_val_cv(groups=...)`,
  `_rfecv_fit.py:287-295`); MRMR does not. A user who turns on grouped splitting plus
  `use_mrmr_fs=True` gets group-naive feature selection feeding a group-aware model,
  silently, behind a warning that is easy to miss in a suite log.
- **Recommendation:** Either (a) implement grouped MI (group-block permutation in the
  Fleuret/permutation kernels), or (b) make the suite default to `strict_groups=True`
  for MRMR when the split is group-aware (`split_config.use_groups`), so the
  contradiction fails loud instead of degrading silently. At minimum, surface the
  warning into `metadata` so it is visible post-hoc.
- **Confidence:** high

### A1-02 — By default the suite does NO supervised FS; only the unsupervised pre-screen runs
- **Severity:** Low (defaults-sanity / expectation)
- **File:** `src/mlframe/training/_feature_selection_config.py:51-58,94`
- **What's wrong:** `use_mrmr_fs=False`, `rfecv_models=None`, `use_boruta_shap=False`,
  `custom_pre_pipelines={}` — so out of the box the ONLY FS the suite performs is the
  variance/null unsupervised pre-screen. All the sophisticated supervised machinery
  (MRMR, RFECV, BorutaShap, ShapProxiedFS) is opt-in.
- **Why it matters:** This is a defensible design (trees do their own implicit
  selection, and supervised FS is expensive), but it means the bulk of the
  `feature_selection/` package is dark on the default path. A reader auditing "FS as
  used by the suite" should know the default is near-passthrough. Per the project's
  "fastest/most-accurate variant should be the default" principle, it is worth a
  deliberate decision (and a documented benchmark) on whether at least a cheap
  supervised filter should be default-on, rather than leaving FS effectively off.
- **Recommendation:** Document the default explicitly in the suite docstring
  (`feature_selection_config` param doc currently lists knobs but not that all
  supervised selectors are off by default). Optionally benchmark a cheap default
  (e.g. MRMR with a tight `max_runtime_mins`) on the wider test bed before deciding.
- **Confidence:** high

### A1-03 — MRMR default `random_seed=None` → non-reproducible selection across runs
- **Severity:** P2
- **File:** `src/mlframe/feature_selection/filters/mrmr.py:310-314`; suite default
  injection in `_setup_helpers_metadata.py:90-99` does NOT set a seed.
- **What's wrong:** The suite's `_default_mrmr_kwargs` sets `n_workers`, `verbose`,
  `fe_max_steps`, `max_runtime_mins` but NOT `random_seed`. MRMR's own default is
  `None`, which derives a process-stable-but-not-run-stable seed (`pid ^ id(self)`,
  per the ctor comment). The screening permutation tests, FE subsampling, and
  resample-for-sample-weight all consume RNG. So two suite runs on identical data can
  select different feature sets — and the suite has `split_config.random_seed`
  available (it is plumbed into the CatBoostEncoder, `_main_train_suite.py:488`) but
  is NOT forwarded to MRMR.
- **Why it matters:** Selection instability across runs undermines reproducibility,
  mlflow hash stability, and A/B comparison of FS effects. The suite already
  propagated `random_seed` to other components specifically to fix this class of bug
  (audit row FE-P2-5 at `:486-488`); MRMR was left out.
- **Recommendation:** In `_initialize_training_defaults` / `_default_mrmr_kwargs`,
  set `random_seed=getattr(split_config,"random_seed",None)` when the caller didn't
  pass one (only fills the gap, never overrides). Same for RFECV (`random_state`).
- **Confidence:** high

### A1-04 — `skip_retraining_on_same_shape` is a misnomer (keys on content, not shape) — verify it stays that way
- **Severity:** Low
- **File:** `src/mlframe/feature_selection/filters/_mrmr_fit_impl.py:246-252` (and the
  `_FIT_CACHE` content-key at `mrmr.py:185-192`)
- **What's wrong:** The parameter name `skip_retraining_on_same_shape` (default True,
  `mrmr.py:581`) reads as "replay if the shape matches", which would be a real
  correctness hazard (same-shape, different y → wrong replay). In fact the cache
  `signature = (X.shape, y.shape, _y_hash_for_sig, _x_hash_for_sig, _x_cols_sig)`
  includes content hashes of BOTH X and y, so a different y at the same shape does NOT
  hit the cache. Behavior is correct; the NAME is misleading and an empty-hash case
  falls through to full fit (`:250-252`).
- **Why it matters:** A future maintainer trusting the name could "optimize" the key
  back down to shape-only and reintroduce a silent y-leak across targets. The cross-
  target identity cache (`mrmr.py:2503-2529`, `mrmr_skip_when_prior_was_identity=True`
  default) is the related, riskier shortcut: it short-circuits FE on a *different* y
  when a prior fit on the same X returned identity. It is guarded by a y-fingerprint
  only when `mrmr_identity_cache_include_y=True` (default True), which is the safe
  setting — but the option to disable it exists.
- **Recommendation:** Rename to `skip_retraining_on_same_content` (keep an alias), or
  add a one-line invariant comment at the signature site pinning "MUST include y/X
  content hashes". Pin a regression test that a same-shape-different-y fit does not
  replay.
- **Confidence:** high

### A1-05 — Pre-screen failure is swallowed and latched, hiding schema-drift hazards
- **Severity:** P2
- **File:** `src/mlframe/training/core/_phase_train_one_target_pre_screen.py:142-146`
  (outer except) and `:121-133` (per-frame except)
- **What's wrong:** The whole pre-screen is wrapped in `except Exception` that sets
  `_pre_screen_done=True` and continues. The per-frame `apply_drops` path also catches
  per-frame and only WARN-logs (`:128-133`) — which the comment itself flags as a
  "schema drift hazard": if `apply_drops` succeeds on `train_df_pd` but raises on
  `val_df_pd`, train loses columns that val keeps, and downstream training hits an
  opaque "feature missing" error far from here. The outer latch means a transient
  failure permanently disables the pre-screen for the rest of the suite.
- **Why it matters:** The drops MUST be applied consistently across all mirrors or the
  schema diverges. A best-effort swallow turns a contract violation into a
  hard-to-trace downstream crash. This is the project's documented "silent error
  swallowing" anti-pattern.
- **Recommendation:** Make per-frame `apply_drops` failure fatal (or roll back the
  drop across all mirrors atomically) rather than WARN-and-continue — a partial drop
  is never acceptable. Keep the outer try but re-raise (or skip cleanly without
  partial application) instead of latching a half-applied state.
- **Confidence:** high

### A1-06 — Cross-target MRMR identity cache can skip FE for a genuinely different target
- **Severity:** P2
- **File:** `src/mlframe/feature_selection/filters/mrmr.py:633-636` (default True) and
  `:2503-2529`
- **What's wrong:** `mrmr_skip_when_prior_was_identity=True` (default) means: if a
  prior fit on the same X-fingerprint returned "identity" (all columns kept, no
  engineered features), a later fit with a DIFFERENT y short-circuits the entire FE
  pipeline and returns identity. The y-fingerprint guard
  (`mrmr_identity_cache_include_y=True`, default) keys the cache by a *sample* of y
  (`_mrmr_compute_y_fingerprint_sample`, ~1000-element blake2b), so distinct targets
  normally get distinct slots — but the rationale comment explicitly bets that
  "composite-target y values are highly correlated with raw y", which is a *quality*
  assumption, not a correctness guarantee. For a genuinely independent second target
  on the same X where FE *would* have helped, this defaults to silently returning
  raw-only selection.
- **Why it matters:** It trades correctness-for-the-second-target against an 88-min
  wall saving (the documented motivating case). On multi-target suites with weakly
  correlated targets, the second target can be under-selected without any signal in
  the logs beyond an info line.
- **Recommendation:** Keep the default but document the assumption in the suite-level
  config too (it's currently only in the MRMR ctor docstring). Consider gating the
  short-circuit on a measured y-correlation threshold rather than a blanket "prior was
  identity" rule. At minimum stamp the short-circuit into `metadata` for observability.
- **Confidence:** med

### A1-07 — FS re-fits per model-tier within a target (only partially hoisted/cached)
- **Severity:** P2 (efficiency)
- **File:** `src/mlframe/training/core/_phase_train_one_target_body.py:394-420`
  (per-model `clone(orig_pre_pipeline)` + `strategy.build_pipeline`) and the
  PipelineCache keyed on strategy content (`:441-455`)
- **What's wrong:** The pre_pipeline (selector) is `clone()`d fresh and re-wrapped per
  model strategy inside the per-target loop. FS is correctly hoisted ACROSS weight
  schemas (the weight loop reuses the fitted pipeline / `PipelineCache`, and
  `use_sample_weights_in_fs=False` keeps the FS cache valid across weights — verified
  `_feature_selection_config.py:74-81`). But across model TIERS (CB vs LGB vs Linear
  vs MLP) the selector is re-fit unless the content-keyed `PipelineCache`
  (`_pipeline_helpers.py:590-644`) or MRMR's process-wide `_FIT_CACHE`
  (`mrmr.py:185-192`) hits. The PipelineCache key folds
  `(requires_imputation, requires_scaling, requires_encoding)` so two tiers with
  matching preprocessing requirements DO share the slot — but the selector fit itself
  (MRMR/RFECV) is the expensive part and is keyed the same way, so tiers with
  differing preprocessing needs re-pay the full selection. The TODO at
  `_main_train_suite.py:674` ("create inner feature matrices once per featureset, not
  once per target") acknowledges the broader version of this.
- **Why it matters:** MRMR/RFECV are the most expensive ops in the suite (10s of
  minutes). Re-fitting the SAME selection on the SAME train X across model tiers is
  pure waste — the selected feature SET is target-and-data-dependent, NOT
  preprocessing-dependent. The caches mitigate but the key conflates "what the selector
  computes" (depends only on X,y) with "what preprocessing the downstream model needs".
- **Recommendation:** Cache the *selector's fitted support* keyed on
  `(target, selector_params_hash, train_X_content_hash)` — independent of model tier /
  preprocessing — and reuse it across tiers, applying the column subset before each
  tier's own preprocessing. MRMR's `_FIT_CACHE` already does most of this content-keyed
  work; the gap is that the suite re-runs `fit_transform` (which re-enters
  preprocessing) rather than reusing the support across tiers.
- **Confidence:** med

### A1-08 — RFECV default selection rule `auto` → `one_se_max` (recall-oriented, NOT parsimonious)
- **Severity:** Low (defaults-sanity)
- **File:** `src/mlframe/feature_selection/wrappers/_rfecv.py:253-255` and
  `_rfecv_stability_select.py:225-238`
- **What's wrong:** `n_features_selection_rule="auto"` resolves to `one_se_max` for
  ALL estimators — "LARGEST N within 1 SE of the best mean". The docstring itself
  warns this is "NOT parsimonious: on noise-robust learners (GBM/RF) it re-admits ~the
  entire pool". So the suite's default RFECV keeps a large feature set, which on
  tree downstreams largely defeats the parsimony purpose of running RFECV at all
  (the cluster-medoid pre-reduction at `registry.py:90-100` is the real reducer).
- **Why it matters:** A user adding `rfecv_models=["cb"]` expecting a compact feature
  set gets a near-full one unless they read the docstring and pass `one_se_min`. This
  is a deliberate, documented choice (recall over parsimony after `one_se_min`
  under-selected on plateaus), so it's defensible — but it is surprising and is the
  opposite of the parsimony most people expect from "RFE".
- **Recommendation:** Surface the resolved rule in the FS report / log so users see
  "RFECV kept 480/500 via one_se_max" and can opt into `one_se_min`. Consider whether
  the suite-level default for tree downstreams should be `one_se_min` given the medoid
  pre-reduction already protects recall.
- **Confidence:** high

### A1-09 — Default RFECV (and BorutaShap) wrap is cluster-medoid reduction with Pearson-only correlation
- **Severity:** Low
- **File:** `src/mlframe/feature_selection/registry.py:90-100,116-126`
- **What's wrong:** Both `_instantiate_rfecv` and `_instantiate_boruta_shap` default
  `cluster_reduce=True` and wrap the base selector in `GroupAwareMRMR(..., corr_method=
  "pearson", ...)`. Pearson captures only linear/monotone redundancy; two features
  that are redundant through a non-linear relationship (or XOR-style) won't cluster, so
  the medoid reduction can leave non-linear duplicates in and (less likely, with
  `expand=True`) the validation says signal-in-non-member is protected. The
  `min_reduction` guard makes it a no-op on near-uncorrelated data (good).
- **Why it matters:** MRMR's own redundancy machinery (DCD, conditional-MI) is
  non-linear-aware, but the RFECV/BorutaShap *pre-reduction* in front of them is
  linear-only. On data with non-linear collinearity the pre-reduction under-reduces,
  partially wasting the speedup it exists to provide. Not a correctness bug (expand=
  True keeps whole clusters) but a methodological mismatch with the rest of the package.
- **Recommendation:** Offer `corr_method="su"`/MI as an option for the medoid clustering
  (mirroring `cluster_correlated_features_su` used inside ShapProxiedFS,
  `shap_proxied_fs.py:1376-1407`), and benchmark whether it beats Pearson on the wider
  bed before flipping the default.
- **Confidence:** med

### A1-10 — `screen_predictors` and `MRMR.fit` have DIFFERENT defaults for the same knobs
- **Severity:** P2 (defaults-sanity / footgun)
- **File:** `src/mlframe/feature_selection/filters/_screen_predictors.py:65-66,72,98,108,114`
  vs `src/mlframe/feature_selection/filters/mrmr.py:319-321,339,365,588,298`
- **What's wrong:** The standalone `screen_predictors` function defaults diverge sharply
  from what `MRMR.fit` passes:
  - `full_npermutations`: screen default **1000**, MRMR ctor default **3**
    (`mrmr.py:320`).
  - `baseline_npermutations`: screen **100**, MRMR **2** (`mrmr.py:321`).
  - `fe_confirm_undersample_rows_per_cell`: screen **0.0** (strict), MRMR **5.0**
    (`mrmr.py:339`).
  - `max_consec_unconfirmed`: screen **30**, MRMR **10** (`mrmr.py:365`).
  - `fe_fallback_to_all`: screen **True**, MRMR **False** (`mrmr.py:588`).
  - `use_simple_mode`: screen **True**, MRMR **False** (`mrmr.py:298`).
  MRMR.fit overrides all of these when it calls into the screen, so the suite path is
  consistent — but anyone calling `screen_predictors` directly (tests, benches, the
  RFECV `prescreen="mrmr"` path at `_rfecv_fit.py:88-94` which constructs a fresh MRMR
  with `full_npermutations=3, cv=3`) gets a very different statistical regime.
- **Why it matters:** The `full_npermutations` 1000-vs-3 gap is a 300x compute and a
  totally different confidence gate. A maintainer reading `screen_predictors` defaults
  to understand "what the suite does" will be badly misled; the binding defaults live
  in the MRMR ctor. This is a latent correctness footgun for any direct caller.
- **Recommendation:** Make `screen_predictors`' defaults match the MRMR ctor (or
  document loudly at the top of `screen_predictors` that MRMR.fit overrides every
  statistical default, with the canonical values). Removing the divergence eliminates a
  class of "why does my direct screen behave differently" confusion.
- **Confidence:** high

### A1-11 — Pre-screen empty-`protected_columns` under group-aware split only WARNs
- **Severity:** Low
- **File:** `src/mlframe/training/core/_phase_train_one_target_pre_screen.py:86-90`
- **What's wrong:** When `split_config.use_groups=True` but the resolved
  `protected_columns` set is empty (group/ts column names couldn't be discovered from
  ctx / extractor / split_config), the code only WARN-logs and proceeds. A
  high-cardinality string group_id column "looks like near-all-unique strings" and
  could in principle be dropped by the null/variance filter (though string columns
  skip the variance branch — `pre_screen.py:102-103,164-165`). The defensive
  multi-source pull (`:73-85`) makes this unlikely, but the failure mode (dropping the
  group column, breaking GroupShuffleSplit downstream) is severe.
- **Why it matters:** If the group column ever IS dropped, the downstream grouped split
  fails far from the pre-screen with an opaque error. The warning is the only guard.
- **Recommendation:** When `use_groups=True` and `protected_columns` is empty, prefer
  to SKIP the pre-screen for that suite (or raise) rather than proceed with no
  protection — the cost of skipping is small vs. the cost of dropping the group key.
- **Confidence:** med

### A1-12 — FS retention is logged but not always stamped where downstream consumers expect it
- **Severity:** Low (integration / observability)
- **File:** `src/mlframe/training/_pipeline_helpers.py:800-812` (verbose-only log) and
  `_phase_train_one_target_schema.py:179,219-235` (report cache keyed on
  `kept_cols = train_df_transformed.columns`)
- **What's wrong:** The "FS selector retained N of M features" line is gated behind
  `verbose` (`:803`). The structured FS report (`_build_feature_selection_report`) is
  built from `train_df_transformed.columns` — the ACTUAL transformed frame — which is
  the right source (column names, not a separate index map, so no index-vs-name
  mismatch in the model fit path). That part is correct. The gap is that the one-glance
  retention summary is verbose-only, so in a quiet CI/prod run there is no quick record
  of how aggressively FS pruned.
- **Why it matters:** Minor — the structured report is in metadata regardless. But for
  the unsupervised pre-screen, the per-frame drop is logged at `verbose` too
  (`_phase_train_one_target_pre_screen.py:134-141`), so a non-verbose run leaves no
  trace of pre-screen activity except the provenance record's `n_dropped`.
- **Recommendation:** Emit a single non-verbose INFO line with kept/dropped counts per
  selector (pre-screen + supervised), or ensure the provenance `n_dropped` is always
  surfaced into `metadata["feature_selection"]` for non-verbose runs.
- **Confidence:** high

### A1-13 — `min_features_fallback=1` can hand a single-column matrix to the downstream model
- **Severity:** Low
- **File:** `src/mlframe/feature_selection/filters/mrmr.py:592-595`
- **What's wrong:** When screening selects zero features (all MI≈0), MRMR keeps the
  single highest-MI column (`min_features_fallback=1`, default) and sets
  `fallback_used_`. This avoids an empty-support crash (good), but means a downstream
  model can be trained on one essentially-noise feature with no loud signal beyond the
  `fallback_used_` attribute.
- **Why it matters:** On a target with no learnable signal, the suite trains a model on
  one near-random feature rather than failing clearly. The `fallback_used_` flag exists
  but isn't necessarily surfaced into the suite metadata / report.
- **Recommendation:** When `fallback_used_` is True, stamp a clear WARN + a
  `metadata` entry so operators know the FS found no signal and the model is on a
  fallback column (not a genuine selection).
- **Confidence:** med

---

## Dimensions explicitly checked and found OK

- **polars handling (dimension 5):** No wrapper-level whole-frame down-conversion on
  the FS hot path. On the polars fastpath the selector still runs but
  `skip_preprocessing=True` (`_setup_helpers.py:213-215`); MRMR materialises into an
  integer-quantised numpy `factors_data` matrix for the numba kernels rather than a
  pandas copy of the whole frame, and its resample path uses native polars `take`
  (`mrmr.py:2348-2356`). ShapProxiedFS calls `self._to_pandas(X)` at its own fit
  entry (`shap_proxied_fs.py:1100`) — but it is opt-in, not on the suite default path,
  and its docstring/registry note flags it as the SHAP-proxy path. No 100GB-frame
  down-conversion was found in the default or MRMR/RFECV paths.
- **Selection logic correctness (dimension 2):** MRMR relevance/redundancy uses
  conditional-MI (Fleuret) with cardinality-bias (Miller-Madow) correction at the
  gate and a Westfall-Young maxT null floor for wide pools
  (`_screen_predictors.py:436-538,707-799`) — methodologically sound and the leakage of
  the gate is internal (shuffled-y permutation null, not held-out data). RFECV
  CV-scores subsets honestly with groups threaded (`_rfecv_fit.py:287-410`). The SHAP
  proxy's `proxy_trust_guard` measures proxy-vs-honest rank fidelity on within-train
  anchors and warns below a calibrated floor (`_shap_proxy_revalidate.py:657-876`) —
  an honest self-check, not a leak.
- **Integration (dimension 6):** Selected columns flow to the model via the actual
  transformed frame's column names (`_phase_train_one_target_schema.py:219-221`); no
  column-name-vs-index mismatch in the fit path. Degenerate columns are handled by the
  pre-screen + `min_features_fallback` (see A1-13). The per-model `clone()` of the
  selector (`_phase_train_one_target_body.py:394-413`) correctly prevents fitted-state
  bleed across strategies, with a WARN fallback for non-cloneable custom pipelines.

## Severity rollup
- P1: A1-01
- P2: A1-03, A1-05, A1-06, A1-07, A1-10
- Low: A1-02, A1-04, A1-08, A1-09, A1-11, A1-12, A1-13
