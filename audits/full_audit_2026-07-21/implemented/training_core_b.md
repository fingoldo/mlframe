# training/core phase machinery B (train-one-target body, predict pipeline, finalize/calibration) -- mlframe audit

## Scope

All 30 files were read in full (no file was too large to review end-to-end; the largest, `_phase_train_one_target_body.py`, is 955 LOC and was read completely in one pass).

- `src/mlframe/training/core/_phase_train_one_target_body.py` (955 LOC)
- `src/mlframe/training/core/_phase_helpers.py` (861 LOC)
- `src/mlframe/training/core/_phase_finalize.py` (761 LOC)
- `src/mlframe/training/core/_main_train_suite.py` (753 LOC)
- `src/mlframe/training/core/predict.py` (731 LOC)
- `src/mlframe/training/core/_phase_recurrent.py` (673 LOC)
- `src/mlframe/training/core/_phase_train_one_target_model_setup.py` (627 LOC)
- `src/mlframe/training/core/_predict_main_suite.py` (559 LOC)
- `src/mlframe/training/core/_predict_pre_pipeline.py` (559 LOC)
- `src/mlframe/training/core/_phase_finalize_calibration.py` (469 LOC)
- `src/mlframe/training/core/_phase_config_setup.py` (450 LOC)
- `src/mlframe/training/core/_main_train_suite_target_distribution.py` (425 LOC)
- `src/mlframe/training/core/_phase_train_one_target_dataset_cache.py` (348 LOC)
- `src/mlframe/training/core/_phase_train_one_target_helpers.py` (332 LOC)
- `src/mlframe/training/core/_phase_composite_post.py` (305 LOC)
- `src/mlframe/training/core/_setup_helpers_outliers.py` (285 LOC)
- `src/mlframe/training/core/_setup_helpers_pipeline_cache.py` (265 LOC)
- `src/mlframe/training/core/_phase_train_one_target_schema.py` (245 LOC)
- `src/mlframe/training/core/_setup_helpers_metadata.py` (234 LOC)
- `src/mlframe/training/core/_diagnostics_registry.py` (209 LOC)
- `src/mlframe/training/core/_volatility_lag_router.py` (198 LOC)
- `src/mlframe/training/core/_ensemble_chooser.py` (160 LOC)
- `src/mlframe/training/core/_phase_composite_discovery_helpers.py` (152 LOC)
- `src/mlframe/training/core/_phase_diagnostics.py` (149 LOC)
- `src/mlframe/training/core/_ood_lag_router.py` (126 LOC)
- `src/mlframe/training/core/_phase_train_one_target_cache_helpers.py` (111 LOC)
- `src/mlframe/training/core/_main_train_suite_encoding.py` (104 LOC)
- `src/mlframe/training/core/_ar_skip.py` (91 LOC)
- `src/mlframe/training/core/utils.py` (54 LOC)
- `src/mlframe/training/core/main.py` (36 LOC)

**Total: 30 files reviewed, 11227 LOC reviewed** (sum of the per-file `wc -l` counts above; matches the cluster's stated total exactly).

For context on the P0 finding below I also read (not audited -- no findings reported about them) the field declarations in `_training_context.py:253-254` (out of scope, sibling file) and a defensive workaround in `_benchmarks/_profile_fuzz_1m_run_suite.py:575-596` (out of scope) that independently corroborates the bug.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P0 | correctness / silent-failure | `_main_train_suite.py:686-698` | `slug_to_original_target_type` is built as a throwaway local variable and never written onto `ctx`, so the persisted metadata's target-type slug map is always empty. |
| F2 | P1 | correctness / silent-failure | `_predict_main_suite.py:290,307-321` | Confirmed downstream consequence of F1: every disk-loaded `predict_mlframe_models_suite` call falls back to the raw slug string for `target_type`, which breaks the multiclass-simplex renormalisation and quantile-alpha resolution in `_combine_probs`/`_resolve_quantile_alphas`, and spams a WARN per model. |
| F3 | P1 | silent-failure | `predict.py:600-624` (`_run_batched`) | On a `np.concatenate` `ValueError` while merging `ensemble_predictions`/`ensemble_probabilities` across batches, the code silently keeps only batch 0's array (a row-count truncated result) with no log message. |
| F4 | P2 | robustness | `_phase_train_one_target_dataset_cache.py:141` | `_kind is True` uses identity comparison against the literal `True` to detect a polars-tier cache sub-key; a numpy bool or any other truthy-but-not-identical value would silently fail to match and the entry would not be invalidated. |
| F5 | P2 | silent-failure / edge case | `_main_train_suite_target_distribution.py:174-176` | When the picked target array is shorter than `train_idx` implies, the mini-HPT analyzer silently falls back to analyzing the **entire unfiltered** target array (including val/test rows) instead of raising or warning. |
| F6 | P2 | ML best practice (documented) | `_phase_train_one_target_helpers.py:261-280` | `sample_weight` passed into `_maybe_run_feature_handling_apply` is accepted but silently discarded by the underlying `feature_handling_apply` for target-encoder handlers; a WARN is emitted (so not fully silent), but the gap is real and can degrade weighted-training AUC without failing loudly. |
| F7 | P2 | docs/comment accuracy | `_phase_config_setup.py:436-439` | Comment claims the `feature_handling_config` ctx-slot problem "needs a separate follow-up" to be read from `ctx.artifacts`; that follow-up was already implemented (`_phase_train_one_target_helpers.py:254-258`). The comment is stale and could mislead a future maintainer into re-doing already-done work. |
| F8 | P2 | docs/logic accuracy | `_main_train_suite_target_distribution.py:74-77` | The comment for the near-duplicate auto-drop says "if either is already in drop_set, skip so we don't drop both halves of the pair", but the code never skips -- `drop_set.add(...)` executes unconditionally every iteration and can end up adding both halves of a pair across iterations, contradicting the stated intent. |
| F9 | P2 | architecture | `_phase_train_one_target_body.py` (955 LOC), `_phase_helpers.py` (861 LOC) | Both files are within the repo's own "split before ~800-900 LOC" convention's danger zone (955 and 861 LOC respectively), already close to the 1000-LOC hard backstop test. Not a bug, but a maintenance-debt flag per the project's own stated convention. |

### F1 -- `slug_to_original_target_type` never reaches `ctx` (P0)

`_main_train_suite.py:686-687` declares two throwaway locals:
```python
slug_to_original_target_type: dict[str, Any] = {}
slug_to_original_target_name: dict[str, Any] = {}
```
Only `slug_to_original_target_type` is ever populated, at line 698 inside the per-target-type loop (`slug_to_original_target_type[slugify(str(target_type).lower())] = target_type`), and it is **never assigned back onto `ctx`**. `_setup_helpers_metadata.py:132-133` (`_finalize_and_save_metadata`, called both mid-suite and at `finalize_suite`) writes the persisted `metadata["slug_to_original_target_type"]` from `ctx.slug_to_original_target_type` -- a *different*, dataclass-default-initialised, permanently-empty dict (`_training_context.py:253`, `field(default_factory=dict)`). Because it is empty, the `if ctx.slug_to_original_target_type:` guard at line 132 is always falsy, so `metadata["slug_to_original_target_type"]` is **never written** for any suite trained through the public API. (By contrast, `slug_to_original_target_name` legitimately works: it is threaded through as a *reference* into `_setup_per_target_mlframe_models` -> `_phase_train_one_target_model_setup.py:236`, which mutates `ctx.slug_to_original_target_name` in place -- the asymmetry between the two maps is itself evidence this was an oversight, not an intentional design.)

I found independent corroboration that this bug is real and already known informally: `src/mlframe/training/_benchmarks/_profile_fuzz_1m_run_suite.py:587-592` (out of scope, not audited otherwise) contains a defensive workaround explicitly labelled `"Also stamp slug_to_original_target_type defensively."` that manually rebuilds the map from `trained_models.keys()` before saving -- because without it the map is empty, exactly as this audit found.

**Suggested fix direction**: in `_main_train_suite.py`, either write directly into `ctx.slug_to_original_target_type[...]` inside the per-target-type loop (mirroring the way `slug_to_original_target_name` is threaded and mutated in place), or delete the dead local and add the missing `ctx.slug_to_original_target_type[...] = target_type` line. Add a regression test that trains a small multi-target-type suite, asserts `metadata["slug_to_original_target_type"]` is non-empty and round-trips through `load_mlframe_suite`.

### F2 -- confirmed downstream breakage from F1 (P1)

In `_predict_main_suite.py:290` the loaded metadata's (always-empty, per F1) slug map is read: `_slug_to_tt = metadata.get("slug_to_original_target_type", {}) or {}`. For every `.dump` file, `_tt = _slug_to_tt.get(_tt_slug)` (line 307) is `None`, so the code falls into the WARN branch (lines 309-315) **on every single model, every single call** to `predict_mlframe_models_suite` when predicting from a saved/loaded suite (the standard production deployment pattern -- train once, predict many times from disk), and uses the raw slugified string (e.g. `"multiclass-classification"`, hyphenated by `slugify`) instead of the `TargetTypes` enum.

That raw string then flows into `_combine_probs(..., target_type=_tt_k)` (line 468) and `target_type=_suite_tt` (line 506) in `predict.py`. There, the multiclass-simplex-renormalisation gate
```python
_is_multiclass_tt = (
    target_type == TargetTypes.MULTICLASS_CLASSIFICATION
    or str(getattr(target_type, "value", target_type)) == TargetTypes.MULTICLASS_CLASSIFICATION.value
)
```
compares the slug string (hyphen-separated, e.g. `"multiclass-classification"`) against `TargetTypes.MULTICLASS_CLASSIFICATION.value` (underscore-separated: `"multiclass_classification"`) -- neither branch matches, so `_is_multiclass_tt` is silently `False` and the per-row renormalisation to a valid probability simplex is skipped for every multiclass suite predicted from disk with a non-trivial ensemble flavour (harm/geo/quad/qube). The same slug/enum mismatch also breaks `_resolve_quantile_alphas`'s `target_type not in ("quantile_regression", "regression_quantile")` check for quantile-regression targets loaded from disk.

**Suggested fix direction**: fixing F1 at the source resolves this transitively; additionally consider hardening `_combine_probs`'s multiclass check to also compare against the *slug* form (`slugify(str(TargetTypes.MULTICLASS_CLASSIFICATION.value))`) as defence-in-depth against future slug-map gaps.

### F3 -- `_run_batched` silently truncates ensemble output on concat failure (P1)

```python
for _key in ("ensemble_predictions", "ensemble_probabilities"):
    _parts: list = [b.get(_key) for b in batch_outs if b.get(_key) is not None]
    if _parts:
        try:
            merged[_key] = np.concatenate(_parts, axis=0)
        except ValueError:
            merged[_key] = _parts[0]
```
If any batch's ensemble output has a shape that can't be concatenated with the others (e.g. a differing number of quantile-alpha columns because one batch degenerately triggered a different code path, or a differing class count from a rare-class batch), the caller silently receives only the **first batch's** `ensemble_predictions`/`ensemble_probabilities` -- a row-count that is a small fraction of the true input length -- with no warning logged. Every other similar fallback in this codebase (see `_combine_probs`, `_apply_extensions_pipeline`, etc.) logs a WARN when it degrades; this one does not, so an operator using `predict_batch_rows` on a large frame could silently ship predictions covering only the first batch's rows for the ensemble keys, while per-model `predictions`/`probabilities` (built via `_concat_probs_dicts`, which has no such fallback path) are correctly sized -- creating a length mismatch between `results["predictions"][model]` and `results["ensemble_predictions"]` that a careless caller might not notice.

**Suggested fix direction**: log a WARN (row counts of both) whenever the `except ValueError` fallback fires, and/or raise if the caller has no way to detect the truncation from the returned dict shape alone.

### F4 -- fragile `is True` identity check (P2)

```python
if _kind == "pl" or _kind is True:
    _polars_sub_keys.append(_sub_key)
```
`prepared_frames` sub-keys are `(tier_tuple, supports_polars, strategy_class, cb_text_pass)`; `supports_polars` is read here as `_sub_key[1]`. `is True` only matches the literal singleton `True`; if `supports_polars` were ever a `numpy.bool_` or any other truthy-but-not-`True`-identical value, this polars-tier entry would silently be skipped during cache invalidation, leaving a stale pointer into a just-released polars frame in `_ensure_feature_side_cache`. In current code `supports_polars` is always set as a plain Python `bool` class attribute on `Strategy` subclasses, so this is not observed to fire today -- flagging as a latent fragility, not a live bug.

**Suggested fix direction**: use `bool(_kind) is True` or simply `_kind in ("pl", True)` (membership, not identity) to be robust to any future non-`bool`-typed truthy value.

### F5 -- mini-HPT analyzer silently widens to the full (unfiltered) target array (P2)

```python
_y_arr = np.asarray(_picked_target)
_y_train = _y_arr[train_idx] if _y_arr.size >= np.max(train_idx) + 1 else _y_arr
```
When the picked target's array is shorter than `max(train_idx) + 1` (a shape mismatch that should not normally happen, but is not otherwise guarded), the code falls back to using the *entire* `_y_arr` -- which includes val/test rows -- as if it were `_y_train`, rather than raising or logging. This is advisory-only (feeds hyperparameter *recommendations*, not the actual model fit), so the blast radius is low, but it is a silent train/val/test-boundary violation exactly in the class of bug the checklist calls out, and an empty/degenerate `train_idx` (`np.max` on an empty array raises `ValueError`) would also crash into the outer broad `except Exception` at line 417 and be swallowed with only a WARN.

**Suggested fix direction**: on the size mismatch, log a WARN and skip the analyzer for this target (return early) rather than silently using unfiltered data; guard `train_idx.size == 0` explicitly before calling `np.max`.

### F6 -- weighted-fit signal silently dropped for FHC target-encoder handlers (P2, already WARN-logged)

`_maybe_run_feature_handling_apply` accepts `sample_weight` "for forward compatibility" but the underlying `feature_handling_apply` does not yet consume it, so LeakageSafeEncoder / target-encoder OOF means are computed **unweighted** even on a recency- or fairness-weighted suite. This is not silent (a rate-limited WARN fires once per target), so it does not violate the "never silently drop weighting" rule outright, but it is a real, currently-shipping gap between the documented sample-weight-aware training contract and what the FHC path actually does.

**Suggested fix direction**: thread `sample_weight` through to `feature_handling_apply`'s target-encoder handlers (LeakageSafeEncoder already documented as capable of consuming it per the in-code note) and drop the WARN once wired.

### F7 -- stale comment claiming unfinished work that is actually done (P2)

`_phase_config_setup.py:436-439` says the FH consumer "needs a separate follow-up to fall back to `ctx.artifacts["feature_handling_config"]`" -- but `_phase_train_one_target_helpers.py:254-258` already implements exactly that fallback. A future maintainer trusting this comment could waste time re-implementing already-shipped behaviour, or conclude (incorrectly) that `feature_handling_config` is currently broken.

**Suggested fix direction**: update the comment to state the fallback is implemented (with a pointer to `_phase_train_one_target_helpers.py`), or delete the now-inaccurate sentence.

### F8 -- "skip" claimed but not implemented in near-duplicate auto-drop (P2)

```python
# Drop the alphabetically-larger; if either is already in
# drop_set, skip so we don't drop both halves of the pair.
drop_set.add(_b if _a not in drop_set and _b not in drop_set else (_a if _b in drop_set else _b))
```
`drop_set.add(...)` executes unconditionally on every iteration of the pairs loop -- there is no "skip" branch anywhere in the expression. Tracing a 3-node correlated chain (pairs `(A,B)` then `(B,C)`) shows the actual behaviour is "always add the not-yet-dropped member of the pair" -- which is a defensible greedy strategy for collapsing a redundant cluster to one survivor, but it is NOT what the comment describes, and the comment's claimed behaviour (leave `C` alone once `B` is already dropped) would be a materially different, less aggressive drop policy. Because the actual behaviour is at least plausible, this is filed as a documentation-accuracy issue rather than a functional bug, but it should be resolved one way or the other so the next reader isn't misled.

**Suggested fix direction**: either fix the comment to describe the actual greedy-chain-collapse behaviour, or fix the code to genuinely skip (only add a member when *neither* half of the pair is already in `drop_set`) if that was really the intended, more conservative policy.

### F9 -- two files approaching the LOC-split convention (P2)

`_phase_train_one_target_body.py` (955 LOC) and `_phase_helpers.py` (861 LOC) are both well past the project's own "carve before ~800-900 LOC" guidance (`CLAUDE.md`: *"New code goes in focused submodules from the start... Carve before a file nears ~800-900 LOC"*), though both are already under the 1000-LOC hard backstop. Both files are already heavily carved (many sibling `_phase_train_one_target_*` / `_misc_helpers` modules exist), so this is a genuine maintenance-debt signal rather than neglect, but is worth flagging before either file crosses 1000 LOC and trips the backstop test.

## Proposals

| ID | Category | File:Line | Summary |
|----|----------|-----------|---------|
| PR1 | test-gap | `_main_train_suite.py`, `_setup_helpers_metadata.py` | No test exercises the real `train_mlframe_models_suite` -> `finalize_suite` path and asserts `metadata["slug_to_original_target_type"]` is populated; existing tests (`test_security_io_validation.py:174-181`) only unit-test `_finalize_and_save_metadata` with a hand-constructed, already-populated `ctx`, which is exactly why F1 slipped through. |
| PR2 | test-gap | `predict.py` / `_predict_main_suite.py` | No test round-trips train (multiclass, >2 classes, ensembling on) -> save -> `load_mlframe_suite` -> `predict_mlframe_models_suite` and asserts the returned per-target/ensemble probability rows sum to 1 (or at least logs no `slug_to_original_target_type missing entry` WARN). Such a test would have caught F1/F2 directly and is a natural `test_biz_val_*`-style regression test once F1 is fixed. |
| PR3 | test-gap | `predict.py:_run_batched` | No test drives `predict_batch_rows` with more than one batch through a code path where `ensemble_predictions`/`ensemble_probabilities` shapes could legitimately differ across batches (F3); a synthetic test forcing the `except ValueError` branch would pin the desired (loud-log, or raise) behaviour. |
| PR4 | perf/observability | `_phase_train_one_target_body.py:756-786` | `t0_model = timer()` is captured even when `verbose` is False and the surrounding branch only logs `_elapsed_str(t0_model)` under `if verbose:`; the `timer()` call itself is cheap, but `_non_neural_train_times.append(timer() - t0_model)` always runs regardless of verbosity (this is intentional -- it feeds the neural-timeout P95 heuristic -- so no change needed; noted only because it looked at first glance like dead instrumentation before tracing the `_compute_neural_max_time` consumer). No action needed; documented here so a future auditor doesn't re-investigate the same false lead. |
| PR5 | ML best practice | `_phase_train_one_target_helpers.py:261-280` | Once `feature_handling_apply` grows a `sample_weight` parameter (tracked in-code), wire it through and add a `test_biz_val_feature_handling_weighted_target_encoder` regression test comparing weighted vs. unweighted OOF-mean quality on a synthetic recency-weighted fixture, per this repo's own "every ML trick gets a biz_value test" convention. |

## Coverage notes

- All 30 in-scope files were read in full; none were skipped or partially covered.
- I did not execute any test, benchmark, or training run (per the read-only mandate); F1/F2 are traced statically through the exact attribute names and call chains, cross-checked against the dataclass field declaration in `_training_context.py` (read for context only, not audited) and independently corroborated by the defensive workaround already present in `_benchmarks/_profile_fuzz_1m_run_suite.py` (also read for context only, not audited). I am highly confident in F1/F2 despite not running the code, because the data flow is unambiguous (a local variable that is provably never read, versus a same-named `ctx` attribute that is provably never written) and is corroborated by an independent, pre-existing workaround elsewhere in the codebase.
- I did not audit `_training_context.py`, `_phase_train_one_target.py`, `_phase_train_one_target_ensembling.py`, `_phase_train_one_target_polars_fastpath.py`, `_phase_train_one_target_pre_screen.py`, `_phase_train_one_target_post.py`, `_phase_train_one_target_mlp_helpers.py`, `_misc_helpers.py`, `_setup_helpers.py`, `_phase_helpers_fit_split.py`, `_phase_runners.py`, `_main_train_suite_defaults.py`, `_main_train_suite_phases.py`, `_predict_main.py`, `_phase_composite_wrapping.py`, `_phase_composite_post_xt_ensemble.py`, `_phase_composite_post_moe.py`, `_phase_composite_post_summary.py`, `_phase_drift_snapshot.py`, `_phase_dummy_baselines.py`, `_phase_temporal_audit.py` or any subdirectory of `training/core/` -- these are outside my assigned 30-file scope (owned by sibling clusters) and were only opened, when at all, to confirm an attribute's declaration/consumption for context, never to hunt for or report findings in them.
- The MRMR/SHAP-proxied-FS packages and their test mirrors were correctly excluded per the task instructions and were not read.
