## MRMR module: error/warning/diagnostic-message audit

Scope: every `raise`, `warnings.warn`, `logger.warning/error/info` in the 8 mrmr/ files.

### `_mrmr_class.py`

**Line 2834-2839** — `warnings.warn("MRMR: both random_seed and random_state were set to different values; using random_seed=X and ignoring random_state=Y.", DeprecationWarning)`
Good: names both params, shows which wins. Minor: category is `DeprecationWarning`, but nothing is being deprecated here — it's a conflicting-value warning. `DeprecationWarning` is also silently filtered by default in most Python runtimes/interactive shells, so a genuinely actionable "you set two conflicting values" warning may never surface to the user. Should be `UserWarning`.

**Line 2841-2846** — `skip_retraining_on_same_shape` deprecation warning. Good message (explains rename + why). Same `DeprecationWarning`-visibility problem: real behavioral consequence (misnamed legacy param) delivered via a category many environments suppress by default.

**Line 3006-3010** — `raise NotImplementedError("MRMR.fit received groups but group_aware_mi=False and strict_groups=True. ...")`. Good, actionable, gives two remediation options. One nit: `NotImplementedError` is misleading — the feature (group-aware MI) *is* implemented, just not enabled; this is a configuration/validation error, not a missing-feature error. `ValueError` would better match user expectations (`except ValueError` handlers for bad config wouldn't catch this).

**Line 3014-3021** — groups-ignored `UserWarning`. Clear and actionable. Fine.

**Line 3032-3037** — polars LazyFrame auto-collect warning. Clear, actionable. Fine.

**Line 3047-3050** — Struct-column `ValueError`. Clear, names offending columns, tells user what to do. Fine.

**Line 3077-3081** — polars→pandas FE bridge failure: `f"MRMR.fit: polars->pandas FE bridge failed ({_pl_exc!r}); proceeding on the native path -- feature engineering may be skipped for this polars input."` Leaks the raw internal exception repr (`_pl_exc!r`) straight into a user-facing warning — could be an internal `KeyError`/`AttributeError` from `get_pandas_view_of_polars_df` with attribute names meaningless to an end user. Reasonable to keep the debug detail but should be logged at `logger.warning` with the repr, and the `UserWarning` shown to the user should stay high-level ("feature engineering was skipped for this polars input due to an internal error; see logs").

**Line 3107-3111** — stability-selection outer-loop exception: `f"MRMR stability_selection_method={_stab_method!r} outer-loop raised {type(_exc).__name__}: {_exc}. Falling back to classic fit."` Same pattern: raw exception text surfaced to the user via `UserWarning`. If the inner exception is e.g. a `KeyError` from deep in `cluster_stability_selection`, the user sees an obscure fragment with no idea what to actually do differently. Should say what the user can check (e.g. "verify `stability_n_bootstrap`/data have enough rows for clustering") in addition to the raw exception.

**Line 3126-3130** — NaN/Inf-in-y `ValueError`: gives counts and remediation. Good, and this is a case of validation happening at the right time (top of `fit`, not buried in `_fit_impl`). Fine.

**Line 3136** — `raise ValueError(f"multioutput_strategy must be None, 'joint', 'union', or 'intersect'; got {_mo_strategy!r}.")` Good — but note `'joint'` is listed as valid here yet the subsequent code (`_mo_strategy in ("union", "intersect")`) never handles `'joint'` — it just silently falls through to the single-target legacy path with no diagnostic that `'joint'` was accepted-but-a-no-op. This is a "validation says X is valid, runtime silently ignores X" gap — worth either implementing `'joint'` or removing it from the accepted set / warning when chosen.

**Line 3252-3256** — SIS front-gate failure: `f"MRMR SIS front gate raised {type(_sis_exc).__name__}: {_sis_exc}; falling back to the full-width MRMR path."` Same raw-exception-leak pattern as above — acceptable but terse; doesn't tell the user this is safe/expected vs. a bug to report.

**Line 3292-3295** — `logger.info("[MRMR] fe_auto=True -> enabled FE generators for this fit: %s", ...)`. This changes selection behavior (auto-enables FE generators) but is only surfaced via `logger.info` — invisible unless the caller has configured logging. Given `fe_auto=True` is a behavior-altering opt-in, this should also emit (or be paired with) a `UserWarning`/print-summary line, consistent with how `groups` / `stability_selection_method` failures use `warnings.warn`. As-is, a script user with default logging config gets no visibility into which generators got turned on — silent behavior change.

**Line 3297-3301** — fe_auto recommender exception: `UserWarning`, again with raw exception text leaked (`{type(_exc).__name__}: {_exc}`). Consistent at least with the module's general (if imperfect) style.

**Line 3197-3205, 3206-3210** — cross-target identity-cache hit/refuse messages: `logger.info`. This is a significant, potentially confusing behavior (skipping the ~minute FE pipeline and returning a cached selection for a *different* y) that is opt-in (`mrmr_skip_when_prior_was_identity=True`) but only logged at `INFO`. Given the consequence (silently returning a stale/proxy selection for a new target), this deserves at minimum a `UserWarning` on cache HIT (not just when refused) so a user without logging configured knows their fit did not actually examine `y`. Currently a user who forgot they set this flag would get a full, wrong selection with zero visible notification.

**Line 3325** — `raise ValueError(f"MRMR.mi_normalization must be 'none' or 'su'; got {_mi_norm!r}.")` Good.

**Line 3335-3336** — `redundancy_aggregator` invalid-value `ValueError`. Good, with an inline comment even noting it's specifically to avoid the typo-silently-falls-through problem — good practice, and correctly validated near top of `fit`.

**Line 3349** — `logger.info("[MRMR] redundancy_aggregator='auto' -> synergy detector: %s", ...)`. Same class of issue as the fe_auto message: a real behavior decision (whether JMIM redundancy is engaged) reported only via `logger.info`, invisible by default. This one at least stores `self._synergy_auto_decision_` for introspection, which mitigates it somewhat, but is inconsistent with the module's use of `warnings.warn` for behaviorally-similar decisions elsewhere (e.g. `fe_auto` failure path warns, success path doesn't warn or log-visibly).

**Line 3351-3355** — synergy detector exception: `UserWarning` with raw exception leaked; consistent with pattern above.

**Line 3371** — `logger.warning("[MRMR] mi_correction='chao_shen' is not yet wired into the relevance/null path and falls back to plug-in MI ('none'); no bias correction is applied this fit.")` This is important: user explicitly asked for a correction, and it is silently a no-op. Using `logger.warning` (invisible without logging configured) for something this consequential is inconsistent with e.g. line 3371's sibling issues (groups-ignored, fe_auto-failure) which correctly use `warnings.warn(UserWarning)`. A user calling `MRMR(mi_correction="chao_shen").fit(...)` in a plain script gets ZERO indication their correction was ignored. Should be `warnings.warn(UserWarning)` (or duplicate: both warn and log).

**Line 3382** — `logger.warning("[MRMR] group_aware_mi disabled this fit: non-uniform sample_weight resamples rows and would misalign groups; pass sample_weight=None or group_aware_mi=False.")` Same problem: an on-by-request feature (`group_aware_mi=True`) is silently disabled for the fit, message is actionable and clear, but only visible via `logger.warning`, not `warnings.warn`. Directly comparable to the `groups`-ignored case above (line 3014) which correctly uses `warnings.warn(UserWarning)` for what is functionally the identical situation ("you asked MRMR to use groups, it's not going to"). This is the clearest **consistency** violation in the file: two near-identical "groups won't be used" situations use two different notification channels, so one is guaranteed-visible and the other is invisible by default.

**Line 3384** — `logger.warning("[MRMR] group_aware_mi disabled: groups length %d != X rows %d.", ...)` Same issue — should be `warnings.warn`, and arguably should be a hard `ValueError` at fit-entry validation (a length mismatch between `groups` and `X` is almost certainly a caller bug, not a "gracefully degrade" situation) rather than "disable a feature and move on".

**Line 3541-3546** — identity-cache STORE — `logger.info`. Lower-stakes (informational only, doesn't change output), fine as INFO.

### `_mrmr_class_config.py`

**Line 248-249 / 251-255** — `_coerce_target_dtype`: both the "converted" (`logger.info`) and the more important "skipped downcast, keeping int64" (`logger.warning`) messages are gated behind `if self.verbose:`. Gating an actual behavioral notice behind `verbose` is inconsistent with the rest of the module (most `warnings.warn` calls fire unconditionally) and means: (a) by default (`verbose=0`), a user gets no notice their target dtype handling differs from expectation even at `logger.warning` level, compounding the invisibility problem since `logger.warning` is already easy to miss; (b) the message itself is fine/actionable (shows both ranges), the delivery is the problem. Recommend removing the `verbose` gate for the warning branch at minimum — a warning about skipping a memory optimization is cheap to always show and shouldn't require opting into verbosity.

**Line 379** — `logger.warning("recommend_enabled_fe: rule recommender failed: %s", _exc)` inside a classmethod introspection helper (`recommend_enabled_fe`). Reasonable — this is a lower-stakes diagnostic-only path (introspection API, not a real fit), `logger.warning` is defensible here since there's no `warnings.warn` precedent needed for a metadata-only helper. No action needed, but note the raw `_exc` interpolation again leaks internal exception text without guidance.

### `_mrmr_class_fit_helpers.py`

**Line 112** — `raise ValueError(f"unknown stability_selection_method={method!r}")` — Bare, doesn't list the valid options (`'classic'`, `'cluster'`, `'complementary_pairs'`), unlike most other validators in `_mrmr_class.py` (e.g. line 3136, 3335) which do enumerate valid choices. Inconsistent with the module's otherwise-good habit of listing legal values in the message. Should read: `f"unknown stability_selection_method={method!r}; expected one of 'classic', 'cluster', 'complementary_pairs'."` Also worth noting this is a **late validation** — `stability_selection_method` is set at construction time but only validated deep inside `_stability_outer_fit`, called from `fit()`, so a typo isn't caught until the user actually calls `fit()` (acceptable, since sklearn convention defers validation to `fit`, but there's no `__init__`/early check at all, so any misconfigured stability method surfaces only after doing a full fit's worth of work up to that point — actually check order: `_stability_outer_fit` is called early in `fit()` before heavy work, so this is fine in practice).

**Line 135, 138, 140, 143** — `sample_weight` validation (`_maybe_resample_for_sample_weight`): all four raises are clear, actionable, show the offending value/shape. Good quality, no issues.

**Line 244-247, 258-263** (`export_artifacts`) — both `ValueError`s are excellent: explain the precondition, why it's not met, and the exact remediation call (`MRMR(retain_artifacts=True, ...)`, or `MRMR._FIT_CACHE.clear()` + refit). Best-in-class examples in this module.

**Line 342** — `raise ValueError("MRMR multioutput: y has no output columns to fit.")` — Reasonably clear given how narrow the trigger is (empty multi-output `y`), though it doesn't explain *what a valid y shape looks like* or how the user got here (e.g. an all-NaN or single-row y that produced zero output columns after some upstream filtering). Minor: could add "check that y has at least one non-degenerate output column."

### `_mrmr_class_transform.py`

**Line 51** — `NotFittedError("This MRMR instance is not fitted yet. Call 'fit' before using 'get_feature_names_out'.")` Standard sklearn-convention message, fine, matches the ecosystem's expectations.

**Line 81** — `ValueError(f"input_features is not equal to feature_names_in_. Got {list(in_names)[:8]}, expected {list(saved)[:8]}.")` Good — shows actual vs expected (truncated to 8 for readability), consistent with sklearn's own column-drift error style. Fine.

**Line 86** — length-mismatch `ValueError`, clear and shows both lengths. Fine.

**Line 222** — `ValueError(f"transform_usability: which must be 'linear'|'universal'|'nonlinear', got {which!r}")` Clear, lists valid options. Fine.

**Line 225-227** — `AttributeError(f"{attr} is not available: fit MRMR with usability_aware_lists=True and a continuous target to populate it (the '{which}' usability list).")` Good, actionable — but raising `AttributeError` for what is really a "precondition not met, feature not enabled" situation is a slightly unusual choice (a user catching `ValueError` for config problems elsewhere in this same module — e.g. line 222 two lines above it — wouldn't catch this one). Minor inconsistency in exception type choice within the very same method.

### `_mrmr_param_constants.py`, `_mrmr_setstate_defaults.py`, `__init__.py`

No `raise`/`warnings.warn`/`logger.*` call sites in these three files (confirmed via grep — none matched). Nothing to audit there.

---

### Summary of the systemic issues (not per-line)

1. **`DeprecationWarning` misuse** (lines 2837, 2844): used for messages that are really "you passed a conflicting/deprecated value, here's what happened" — not pure API-deprecation notices — and `DeprecationWarning` is filtered by default in many contexts (plain `python script.py`, non-`__main__` code), unlike `UserWarning`. Both should be `UserWarning` (or the deprecation content split from the conflict-resolution content).

2. **The single biggest consistency bug**: functionally identical "requested feature X will be ignored/disabled for this fit" situations are reported through different, inconsistent channels across the file — `warnings.warn(UserWarning)` for `groups` ignored (3014) vs. `logger.warning` for `group_aware_mi` disabled (3382, 3384) vs. `logger.warning` for `mi_correction='chao_shen'` no-op (3371) vs. `logger.info` for `fe_auto` enabling generators (3292) and `redundancy_aggregator='auto'` decision (3349). A user running a plain script with default logging (the common case, called out explicitly in this module's own docstring at line 178-184 as the reason `_print_fit_summary` uses `print` instead of `logger`) will see the `groups` warning but miss every other equally-important "your setting was silently overridden/ignored" notice. These should all be unified to `warnings.warn(UserWarning)` (or all routed through the same guaranteed-visible channel the module already built for exactly this reason — `_print_fit_summary`).

3. **Raw exception leakage into user-facing warnings**: several `warnings.warn` calls (3078, 3108, 3253, 3298, 3352) interpolate `{type(_exc).__name__}: {_exc}` directly from internal helper failures (polars bridge, stability outer loop, SIS screen, fe_auto recommender, synergy detector) into the user-visible message. This is defensible as "better than nothing" but none of them add developer-vs-user framing (e.g., "this is likely a data-shape/environment issue, not something to configure differently") — a user sees a bare Python exception repr with no guidance on whether to act.

4. **Verbose-gated warnings** (`_mrmr_class_config.py:248-255`): gating a genuine behavioral notice behind `self.verbose` is a one-off pattern not used anywhere else in the audited files, and defeats the purpose of `logger.warning` (already semi-invisible) by adding a second opt-in layer on top.
