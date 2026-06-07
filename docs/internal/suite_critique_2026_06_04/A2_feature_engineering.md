# A2 — Feature Engineering critique (as wired into `train_mlframe_models_suite`)

READ-ONLY audit, 2026-06-04. Scope: FE as actually consumed by the suite.
Every claim verified by reading source at the cited `file:line`.

## Executive summary

The "FE wired into the suite" surface is narrower and different from what the
project memory `project_mlframe_fe_transformer_shortlist` states. Verified facts:

- **The 103 `feature_engineering/transformer/*` functions (incl. the named
  "shortlist" cdist / local_lift / BGM / RFF / RSD-kNN) are NOT wired into
  `train_mlframe_models_suite` at all.** They are plain functions
  (`compute_*_features(...) -> pl.DataFrame`), referenced only from
  `tests/feature_engineering/transformer/*` and from inside the FE package.
  No `src/mlframe/training/**` module imports any of them. Even
  `test_biz_val_real_datasets.py` calls them directly and feeds the output to
  standalone boostings — it never routes through the suite. So the memory's
  "5 of 103 ship into the suite" is **inaccurate as stated**: the wiring is
  manual (user computes features, then passes the augmented frame as `df=`),
  not suite code. (Finding A2-01.)
- The FE the **suite genuinely applies** is three things:
  1. `apply_preprocessing_extensions` (PySR → TF-IDF → sklearn bridge:
     imputer/scaler/binarizer/KBins/PolynomialFeatures/RBFSampler-Nystroem/
     dim-reducer). `_pipeline_extensions.py`. RFF *is* present here as
     `nonlinear_features="RBFSampler"`.
  2. The category encoder in the model pre-pipeline — default
     `category_encoders.CatBoostEncoder` (a target encoder).
  3. Composite-**target** transforms (`_composite_transforms_nonlinear.py` +
     `composite_discovery.py` + `CompositeTargetEstimator`).
  Plus FS pre-pipelines (MRMR/RFECV/BorutaShap), out of scope here.

**Leakage discipline is generally sound** where it matters (composite discovery
fits transform params on `train_idx` only with a regression test;
CompositeTargetEstimator fits transform + inner on train rows; the sklearn
extensions pipe does `fit_transform(train)` then `transform(val/test)`; the
standalone transformers carry explicit per-fold OOF). No target-peeking or
full-dataset-fit leak was found in the suite-applied FE.

The findings below are mostly P2/Low: a misleading memory entry, a
half-implemented fastpath that forces a full polars→pandas down-conversion, a
non-threaded RNG seed, and several robustness/efficiency notes. One P1: the
docstring/contract for the extensions fastpath is wrong in a way that defeats
the project's no-eager-down-conversion rule on large polars frames.

---

## Findings

### A2-01 — Memory/docs claim "5 transformers wire into the suite"; none actually do
- **Severity:** P2
- **File:** `src/mlframe/feature_engineering/transformer/__init__.py:1-248`; corroborated by absence of any importer under `src/mlframe/training/**` (grep for `compute_local_lift_features|compute_class_distance_features|compute_rff_features|compute_bgmm*|compute_residual_stratified_distance_features` returns only `tests/**` and `feature_engineering/**`); `tests/feature_engineering/transformer/test_biz_val_real_datasets.py:48-90` imports them and runs standalone boostings, not the suite.
- **What's wrong:** Project memory `project_mlframe_fe_transformer_shortlist` and the docstring of `test_biz_val_class_distance_and_local_lift.py:1-7` assert that cdist/local_lift/BGM/RFF/RSD-kNN "ship into `train_mlframe_models_suite`". The suite never calls them. The only auto-applied kernel-feature path in the suite is `nonlinear_features="RBFSampler"` (an sklearn RFF), which is a *different* implementation from `transformer/random_features.compute_rff_features`.
- **Why it matters:** The audit brief and future maintainers are steered to look for wiring that does not exist, and the "subsumed by stacking" rationale is unverifiable because there is no suite path to subsume. Users who expect these features by configuring the suite get nothing; they must compute features manually and pass the augmented `df`.
- **Recommendation:** Either (a) correct the memory/test docstrings to say "research-only standalone functions; users add them to `df` before calling the suite (or wire via `custom_pre_pipelines` after writing a sklearn-estimator adapter)", or (b) if suite integration is intended, add a thin `BaseEstimator/TransformerMixin` adapter (Mode-A OOF on train, Mode-B on predict) and expose it via `custom_pre_pipelines` / a `FeatureEngineeringConfig`. Today neither exists.
- **Confidence:** High.

### A2-02 — `apply_preprocessing_extensions` down-converts polars→pandas even when zero stages are active
- **Severity:** P1
- **File:** `src/mlframe/training/_pipeline_extensions.py:359-421` (docstring 359-365; the `config is None` short-circuit at 370-371; `_to_pandas(train/val/test)` at 417-419 with no all-stages-inactive guard before it).
- **What's wrong:** The docstring promises "Fastpath: when `config` is None OR has zero active stages, returns inputs untouched with None pipeline." Only the `config is None` branch is implemented. A `PreprocessingExtensionsConfig()` with all-default (all-`None`) stages still reaches lines 417-419 and runs three full-frame `df.to_pandas(split_blocks=True, self_destruct=True)` conversions before `_build_extension_steps` returns `[]` and the function bails at line 689-699.
- **Why it matters:** Directly violates CLAUDE.md "Frame-type conversions are caller responsibility, NOT wrapper auto-magic" + the no-eager-down-conversion-on-large-frames rule. On a 100+ GB polars frame, an inert extensions config (easy to set inadvertently, e.g. a config object created with only `verbose_logging`/`random_seed`) triggers a full materialise-to-pandas of train+val+test = potential OOM, and `self_destruct=True` destroys the polars buffers in the process so there's no cheap rollback.
- **Recommendation:** Add an "any active stage?" predicate BEFORE `_to_pandas` (PySR enabled OR tfidf_columns OR scaler OR binarization_threshold OR kbins OR polynomial_degree OR nonlinear_features OR dim_reducer). If none active, `return train_df, val_df, test_df, None` immediately — fulfilling the documented fastpath. Cheap: it's the same disjunction already assembled in `_build_extension_steps:103-110` plus `pysr_enabled`/`tfidf_columns`.
- **Confidence:** High.

### A2-03 — RBFSampler / Nystroem / dim-reducer `random_state` is hardcoded to 42, not threaded from config
- **Severity:** P2
- **File:** `src/mlframe/training/pipeline.py:84` (`_build_extension_steps(config, n_features, random_state: int = 42)`), `:165` (`kw["random_state"] = random_state`), `:168` (`_build_dim_reducer(..., random_state)`); call site `src/mlframe/training/_pipeline_extensions.py:688` (`_build_extension_steps(config, n_features=n_features)` — no `random_state=` passed).
- **What's wrong:** `PreprocessingExtensionsConfig` carries a `random_seed` field (`_preprocessing_configs.py:366`) that is threaded into PySR (`_pipeline_extensions.py:221`) but NOT into `_build_extension_steps`. So the RFF projection matrix, Nystroem landmarks, PCA/ICA/random-projection seeds are always `42` regardless of `config.random_seed`. The same suite re-run with a different `random_seed` produces identical RFF/PCA features.
- **Why it matters:** Breaks the seed-reproducibility contract for the only auto-applied kernel-FE path (RFF). Defeats multi-seed stability assessment of the extension features and silently couples two runs that the user intended to differ. Same bug class flagged & fixed for CatBoostEncoder (FE-P2-5 / `_setup_helpers.py:271-287`); the extensions pipe was missed.
- **Recommendation:** Pass `random_state=int(getattr(config, "random_seed", 42))` from `_pipeline_extensions.py:688` into `_build_extension_steps`. One-line change; no behavior change when the caller leaves `random_seed=42`.
- **Confidence:** High.

### A2-04 — `_filter_to_numeric` mutates the caller's frame (bool→int8) in place; documented but still surprising on a shared frame
- **Severity:** Low
- **File:** `src/mlframe/training/_pipeline_extensions.py:318-347` (esp. 342-344 `_df[_c] = _df[_c].astype(int8)` and the polars `to_pandas(self_destruct=True)` at 330-334).
- **What's wrong:** To honor the no-copy rule, `_filter_to_numeric` promotes bool columns to int8 *in the caller's pandas frame* and, for polars input, self-destructs the polars buffers. The caller-frame mutation contract is documented in the docstring, but the function runs on `train`/`val`/`test` which at this point are the suite's working frames (already pandas copies from `_to_pandas`), so the blast radius is limited. The risk is if a future caller passes a frame it still needs in bool form afterwards.
- **Why it matters:** Subtle action-at-a-distance. Within the current suite flow it's safe (the frames are already the extensions-local pandas materialisations), so this is Low, but the in-place bool cast on `val`/`test` (lines 541-542) re-derives the dropped set independently per frame and could diverge if a bool column exists in train but is object in val (it would be dropped from one and kept-as-int8 in the other → column-set skew the downstream sklearn pipe then rejects).
- **Recommendation:** Compute the kept-numeric column set ONCE on `train` and apply the SAME column list to `val`/`test` (mirror the `_all_null_cols` pattern at 565-572 which already does cross-split alignment). Keeps the no-copy bool promotion but guarantees column parity.
- **Confidence:** Medium (parity skew is a latent edge, not observed).

### A2-05 — TF-IDF fit uses `.values` on a pandas Series, materialising a dense object array per text column
- **Severity:** Low
- **File:** `src/mlframe/training/_pipeline_extensions.py:518` (`train[col].fillna("").astype(str).values`), `:528` (same for val/test).
- **What's wrong:** `.fillna("").astype(str).values` builds a full N-length object ndarray for each text column before handing to `TfidfVectorizer`. For a wide multi-text-column frame on large N this is several transient object arrays. Not a leak; an allocation note.
- **Why it matters:** Minor peak-RAM bump per text column; object-dtype arrays are heavy. On the documented 1M-row sparse path this is dwarfed by the TF-IDF csr, so it's Low.
- **Recommendation:** `TfidfVectorizer` accepts any iterable of str; pass `train[col].fillna("").astype(str)` (the Series) directly, or `.to_numpy(dtype=object, na_value="")`. Negligible but free. Measure first per the perf rule; likely "no actionable speedup" but worth a note so it isn't re-flagged.
- **Confidence:** Medium.

### A2-06 — `compute_rff_features` standalone: bandwidth + RobustScaler are fit on the FULL input X (caller-fold responsibility, no in-function guard)
- **Severity:** P2 (research-only) / would be P1 if ever suite-wired
- **File:** `src/mlframe/feature_engineering/transformer/random_features.py:75-104` (RobustScaler fit on all of `X_arr`; `sigma_median_heuristic(X_std, ...)` on all rows).
- **What's wrong:** `compute_rff_features` has NO splitter/OOF mode — it standardises and estimates the median bandwidth on whatever X it is handed, then projects. Unlike `local_lift` / `class_distance` / `residual_stratified_distance` (which expose Mode A per-fold + Mode B), RFF is single-pass and unsupervised, so there's no *target* leak — but if a caller passes a concatenated train+val+test X, the scaler/bandwidth see val/test feature distributions. The docstring does not state the train-only-fit contract.
- **Why it matters:** Unsupervised feature leakage (distributional) is mild but real for distribution-shifted holdouts, and the function gives the caller no safe Mode-A path. The suite's RBFSampler path (pipeline.py) avoids this because the sklearn pipe fits on train only — so this finding is about the standalone function and any future suite wiring per A2-01.
- **Recommendation:** Document explicitly that the caller must fit on train rows and `transform` holdout via a returned (W, b, scaler) state, OR add a `splitter=`/`X_query=` Mode-A/Mode-B pair mirroring the sibling transformers. Today RFF is the odd one out: it cannot be used leakage-safely on a single combined frame.
- **Confidence:** High (verified the function has no fold mode).

### A2-07 — `local_lift` binary PR-AUC: `np.diff(recall, prepend=0.0)` double-counts the first step and the metric is non-standard
- **Severity:** Low
- **File:** `src/mlframe/feature_engineering/transformer/local_lift.py:59-68`.
- **What's wrong:** The "local PR_AUC" is a trapezoid-ish sum `(d_recall * precision).sum()` with `d_recall = np.diff(recall, prepend=0.0)`. With `prepend=0.0`, the first term uses `recall[0]-0` × `precision[0]`, which left-Riemann-sums rather than trapezoidal-averages adjacent precisions; the result is a monotone-but-biased proxy, not sklearn `average_precision_score`. It is consistent across train/val (so it's a valid *feature*), but the docstring calls it "trapezoidal PR_AUC", which it isn't.
- **Why it matters:** Purely a naming/interpretability issue — the column is still a useful monotone signal and is computed identically on every split (no leak, no train/predict skew). Low because it's research-only and the value is self-consistent.
- **Recommendation:** Rename the column/docstring to "local AP proxy" or switch to true trapezoidal `0.5*(precision[:-1]+precision[1:])` averaging if the trapezoidal claim is to be honored. Add a biz_value assertion on the proxy's rank-correlation with sklearn AP so a future "fix" doesn't silently change the feature.
- **Confidence:** Medium.

### A2-08 — Composite discovery / estimator leakage discipline: VERIFIED CLEAN (positive finding)
- **Severity:** Low (informational)
- **File:** `src/mlframe/training/_composite_discovery_fit.py:219-220, 351, 448` (`y_train = y_full[train_idx]`, `base_train = ...[train_idx]`, `transform.fit(y_train[valid], base_train[valid])`); `composite_discovery.py:109-125, 146-170` (`iter_transform` applies `forward` to all rows with train-fit params; documented + regression-tested `test_alpha_train_only_changes_with_train_idx`); `_composite_target_estimator.py:415-493` (domain_check → fit transform params on valid train rows → forward → fit inner on T).
- **What's wrong:** Nothing. Transform params (alpha/beta, bin medians/IQRs, PCHIP knots, EWMA anchors, frac-diff anchors) are estimated strictly on train rows; `forward` is then applied to all rows; predict-time inversion replays the persisted `fitted_params` via `from_fitted_inner`. No target peeking, no full-dataset fit.
- **Why it matters:** This is the suite's only target-aware FE, the prime leakage suspect — and it is disciplined. Worth pinning so a future refactor doesn't regress it.
- **Recommendation:** Keep the cited regression test; consider adding one that asserts `iter_transform` output on val/test rows is byte-identical whether or not those rows were present at `fit` time (idempotent-replay sensor).
- **Confidence:** High.

### A2-09 — `CompositeTargetEstimator` honors the no-down-conversion rule: VERIFIED CLEAN (positive finding)
- **Severity:** Low (informational)
- **File:** `src/mlframe/training/_composite_target_estimator.py:503-512` (comment + `_subset_rows` at 631-646 branch on polars/pandas/ndarray, never `to_pandas` on the whole frame); `_extract_base`/`_extract_base_matrix` pull narrow column ndarrays only.
- **What's wrong:** Nothing. The wrapper explicitly refuses to materialise polars frames and pushes the type decision to the suite boundary, exactly per CLAUDE.md. The historical `X.to_pandas()` regression (2026-05-10) is gone.
- **Recommendation:** None. Keep the regression sensor that pins this.
- **Confidence:** High.

### A2-10 — Predict-path extensions replay can silently return RAW columns on transform failure
- **Severity:** P2
- **File:** `src/mlframe/training/core/_predict_pre_pipeline.py:131-135` (`_apply_extensions_pipeline`: on any `ext_pipeline.transform(df)` exception, `logger.error(...)` then `return df` — the unmodified raw frame).
- **What's wrong:** When the persisted sklearn extensions pipeline raises at predict, the code logs an error and returns the input frame unchanged. The downstream model was fit on the EXTENSION columns (e.g. `pca0..pcaN`, `rff_*`, poly terms) and will then either crash on missing features or — for NaN-tolerant tree backends that ignore unexpected columns — produce nonsense from raw inputs. The log line says so, but predictions still flow.
- **Why it matters:** Silent-ish correctness failure at serve time. The error is logged but the function does not raise, so a serving harness that doesn't fail on a single ERROR log returns wrong predictions. This is the same "loud-fail beats silent skip" principle the TF-IDF branch right above it (`:78-84`) correctly enforces with a `raise KeyError`.
- **Recommendation:** Re-raise (or raise a dedicated `ExtensionsReplayError`) instead of `return df`, mirroring the TF-IDF missing-column branch. If a soft-fail mode is genuinely wanted, gate it behind an explicit flag, default hard-fail.
- **Confidence:** High.

### A2-11 — Extensions pipeline is fit once (not per-target): VERIFIED (positive — no redundant recompute)
- **Severity:** Low (informational)
- **File:** `src/mlframe/training/core/_phase_helpers_fit_pipeline.py:469-470` (single call in `_phase_fit_pipeline`, before the per-target loop in `_main_train_suite.py:671-676`).
- **What's wrong:** Nothing. `apply_preprocessing_extensions` runs once pre-loop; the fitted bundle is stamped into `metadata["extensions_pipeline"]` (`:546`) and reused. The pre-pipeline (encoder/imputer/scaler/FS) DOES re-fit per model, but that is cached via `_PRE_PIPELINE_CACHE` (`_pipeline_helpers.py:579-644`) which short-circuits structurally-identical pipelines across models in the same target. So no per-target FE recompute was found.
- **Why it matters:** Confirms the efficiency axis is already handled for the genuinely-wired FE; the only remaining recompute is per-target-type composite discovery, which is target-specific by definition.
- **Recommendation:** None.
- **Confidence:** High.

### A2-12 — `_get_pipeline_components` default target encoder is leakage-safe but undocumented as such at the call boundary
- **Severity:** Low
- **File:** `src/mlframe/training/core/_setup_helpers.py:285-287` (default `ce.CatBoostEncoder(random_state=_seed)`); wired as the `"ce"` step in `_strategies_base.py:502-504`; fit/transform via `_apply_pre_pipeline_transforms` (`_pipeline_helpers.py:719-799`, `fit_transform(train)` then `transform(val/test)`).
- **What's wrong:** Nothing functionally — CatBoostEncoder uses ordered target statistics (each row encoded from prior rows only), so it is leakage-resistant by construction and is fit on train-only then `transform`-applied to holdouts. But the default being a *target* encoder is a leakage-adjacent choice that isn't flagged anywhere near the suite entry point; a reviewer scanning for "is there target encoding in the default path?" must dig three modules deep to confirm it's the ordered (safe) variant rather than a naive mean encoder.
- **Why it matters:** Default-path target encoding is the single most common silent-leak source in tabular pipelines. It happens to be safe here, but the safety hinges on CatBoostEncoder's ordering — if a future maintainer swaps the default to `ce.TargetEncoder` / `ce.MEstimateEncoder` (naive smoothed mean) the suite would leak with no test catching it (the encoder is fit on train, so a train/test split wouldn't reveal in-fold optimism on val).
- **Recommendation:** Add a one-line comment at `_setup_helpers.py:285` noting the default MUST be an ordered/CV target encoder (not a naive mean encoder) for leakage safety, and add a biz_value/leakage sensor that fails if the default encoder produces train-fold optimism above a threshold (fit on train, measure val-AUC inflation vs an OrdinalEncoder baseline on a synthetic high-cardinality leak target).
- **Confidence:** High (CatBoostEncoder ordered-TS safety is correct; the gap is documentation + a guard sensor).

### A2-13 — Polynomial byte-cap auto-tune mutates `config` via `model_copy`/`copy` but the original config object identity is shared across folds/targets
- **Severity:** Low
- **File:** `src/mlframe/training/_pipeline_extensions.py:595-687` (dim_n_components clamp 595-615; polynomial byte auto-tune 616-687; both `config = config.model_copy(update=...)` / `copy.copy(config)`).
- **What's wrong:** The clamp/auto-tune rebinds the LOCAL `config` to a copy, so the caller's config object is not mutated (good). But because extensions run once pre-loop (A2-11), this is fine in the current flow. The Low note: the `copy.copy` fallback (603-605, 683-685) is a SHALLOW copy — if `PreprocessingExtensionsConfig` ever gains a mutable nested field (e.g. a dict of pysr_params), a shallow copy would share it. Today all auto-tuned fields are scalars so it's safe.
- **Why it matters:** Latent aliasing risk if the config grows mutable nested state; not a current bug.
- **Recommendation:** Prefer `config.model_copy(...)` exclusively (pydantic deep-copies on `model_copy`); drop the `copy.copy` fallback or make it `copy.deepcopy`. Negligible cost (runs once).
- **Confidence:** Medium.

### A2-14 — `class_distance` / `local_lift` standalone: `RobustScaler` refit per fold is correct, but Mode B fits slices on full y_train including query rows when query ⊆ train
- **Severity:** Low (research-only)
- **File:** `src/mlframe/feature_engineering/transformer/class_distance.py:164-167` (Mode B: slices from full `y_train_f`, query `Xq`); `local_lift.py:132-139` (same shape).
- **What's wrong:** Mode B (`X_query` given) builds class/quantile slices from the full `y_train` and queries `X_query` against them. If a caller passes `X_query` rows that are also IN `X_train` (e.g. scoring train rows for diagnostics), each query row's nearest neighbor is itself (distance 0) and its own label leaks. This is the documented Mode-B contract (query rows are assumed disjoint holdout), so it's correct usage, but there's no self-exclusion guard and no assertion that query⊄train.
- **Why it matters:** Only bites if a user misuses Mode B to score training rows; the suite (per A2-01) doesn't call these so blast radius is the user's own harness. Low.
- **Recommendation:** Document the disjoint-query precondition in the public docstring (it's only in the leakage-discipline note, not the signature), or add an optional `exclude_self_ids=` mask for the score-train-rows diagnostic case.
- **Confidence:** Medium.

---

## Scope notes / alternative readings

- I treated "wired into the suite" as "called from `src/mlframe/training/**` during a `train_mlframe_models_suite` run". Under a looser reading — "exposed for users to add to `df` before the suite" — the memory's "5 shortlist" claim is defensible as a *recommendation*, not a wiring fact. A2-01 flags the literal-wiring gap; if the intended meaning is the looser one, A2-01 downgrades to a docs clarification.
- The genuinely-wired FE (extensions + composite target + cat encoder) is in good shape on leakage and the no-copy rule; the substantive actionable items are A2-02 (P1, eager down-conversion) and A2-10 (P2, silent raw-frame predict), then A2-03 (seed threading). The rest are Low/docs/robustness.
- I did not benchmark anything (READ-ONLY, static reading per brief). Perf-flavored notes (A2-05) are flagged as "measure first" per project policy and may resolve to "no actionable speedup".
