# training/composite/discovery (composite model discovery & search engine) -- mlframe audit

## Scope

All 47 non-benchmark `.py` files under `src/mlframe/training/composite/discovery/**` were opened and read in full (the `_benchmarks/` subfolder, ~32 files, is excluded from the file/LOC count per the cluster definition -- it is a `python -m ...` manual-bench harness collection, not production code, and was only spot-checked for call-site correctness where a finding referenced it).

Files reviewed (relative to `src/mlframe/training/composite/discovery/`):

- `__init__.py` (632 LOC)
- `_auto_base.py` (867)
- `_auto_chain.py` (457)
- `_base_engineering.py` (220)
- `_calibration_gate.py` (201)
- `_causal_lag.py` (97)
- `_collinear_numba.py` (408)
- `_corr_numba.py` (205)
- `_eval.py` (699)
- `_eval_stats.py` (483)
- `_eval_waic.py` (319)
- `_filter.py` (263)
- `_filter_and_gate.py` (99)
- `_fit.py` (845)
- `_fit_helpers.py` (59)
- `_fit_multibase.py` (180)
- `_fit_ram.py` (95)
- `_fit_temporal.py` (85)
- `_grouped_causal_bases.py` (367)
- `_honest_holdout.py` (346)
- `_honest_oof_select.py` (207)
- `_honest_rmse_gate.py` (241)
- `_incremental.py` (307)
- `_interaction_bases.py` (229)
- `_knn_budget.py` (108)
- `_ktc_dispatch.py` (343)
- `_leakage.py` (277)
- `_mrmr_base_rank.py` (87)
- `_opt_in_steps.py` (327)
- `_per_group.py` (116)
- `_per_group_discovery.py` (88)
- `_region_adaptive.py` (261)
- `_rejection_ledger.py` (70)
- `_screening_mi_pair.py` (32)
- `_screening_tiny.py` (716)
- `_screening_tiny_perbin.py` (456)
- `_stability.py` (397)
- `_stability_check.py` (180)
- `_stacked.py` (434)
- `_structural_hints.py` (253)
- `_tiny_rerank.py` (951)
- `_tiny_rerank_waic.py` (111)
- `_yscale_holdout_gate.py` (458)
- `auto_detect.py` (485)
- `bayesian.py` (332)
- `forward_stepwise.py` (301)
- `screening.py` (938)

Every file above was read completely (not skimmed) -- no file in scope was too large to review in depth; the two largest (`_tiny_rerank.py` 951 LOC, `screening.py` 938 LOC) were each read start to end in a single pass.

**Total files reviewed: 47. Total LOC reviewed: 15715** (matches `wc -l` on the same file set exactly; excludes `_benchmarks/`).

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P1 | reproducibility | `_fit_multibase.py:107-119` | Multi-base forward-stepwise promotion never forwards `self.config.random_state`, `groups=`, or `time_aware=` to `forward_stepwise_multi_base`, so its CV split ignores the caller's seed and is group-blind on grouped datasets. |
| F2 | P1 | reproducibility | `_tiny_rerank_waic.py:65` | `getattr(self, "random_seed", 0)` reads a nonexistent attribute (the real field is `self.config.random_state`), so the WAIC tie-break's K-fold split always uses seed 0 regardless of configured `random_state`. |
| F3 | P2 | architecture/test-coverage | `_calibration_gate.py:1-202` | A fully implemented, documented "opt-in ranking signal" module has zero callers anywhere in `src/` or `tests/` -- dead code. |
| F4 | P2 | docs | `_interaction_bases.py:38-42` | Module docstring says wiring is gated on `config.interaction_base_discovery=True` (opt-in); the real field is `interaction_base_discovery_enabled`, defaulting `True` (default-on), not opt-in. |
| F5 | P2 | architecture | `screening.py` (938 LOC), `_tiny_rerank.py` (951 LOC) | Both exceed this repo's own "carve before ~800-900 LOC" convention (CLAUDE.md), though still under the hard 1000-LOC test gate. |
| F6 | P2 | perf | `_base_engineering.py:80-94` (`_causal_rolling`) | Per-row Python loop calling `np.median`/`.mean()` per window -- O(n * window), unvectorized, no numba dispatch, unlike every other numeric kernel in this package. |

### F1 -- Multi-base forward-stepwise promotion ignores config seed and group structure (P1, reproducibility + leakage-risk in selection)

`apply_multi_base_forward_stepwise` (`_fit_multibase.py:31-180`) is the **only production call site** of `forward_stepwise.forward_stepwise_multi_base` (confirmed via repo-wide grep; the function's only other callers are two standalone `_benchmarks/` scripts). The call at lines 107-119 passes `seed_bases`, `max_k`, `min_marginal_rmse_gain`, and the four `cv_selector_*` knobs, but **never** passes `random_state`, `groups`, or `time_aware`:

```python
_kept_bases, _fwd_diag = forward_stepwise_multi_base(
    _y_train_local, _pool_arrays,
    seed_bases=[_spec.base_column], max_k=_multi_max_k,
    min_marginal_rmse_gain=_multi_min_gain,
    cv_selector_mode=_cv_sel_mode, cv_selector_alpha=_cv_sel_alpha,
    cv_selector_confidence=_cv_sel_conf, cv_selector_quantile_level=_cv_sel_qlevel,
    cv_persist_fold_scores=_cv_persist,
)
```

`forward_stepwise_multi_base`'s signature (`forward_stepwise.py:38-68`) defaults these to `random_state: int = 42` and `time_aware: bool = True`, `groups: np.ndarray | None = None`. Two consequences, both real:

1. **Reproducibility**: every multi-base promotion CV always splits with the hardcoded seed 42, never `self.config.random_state`. A caller who sets `random_state=1` vs `random_state=2` for a multi-seed stability sweep (the exact use case `_stability_check.py` / `_stability.py` exist for) gets byte-identical CV folds for this stage on every replicate, silently defeating the decorrelation the rest of the pipeline works hard to guarantee (see `_stability_check.py`'s own docstring about a prior seed-collision bug in the same module family).
2. **Group-blindness**: because `time_aware` defaults `True` and is never overridden by the caller, the `elif time_aware:` branch in `_cv_rmse_with_folds` (`forward_stepwise.py:147-148`) always wins over the `elif _groups_eff is not None:` GroupKFold branch -- `groups` is never even passed in, so the group branch is unreachable from this call site regardless. On a grouped dataset (wells/entities), the greedy "should I add this base?" decision is therefore scored with `TimeSeriesSplit`, which has no notion of the group column: rows from the same group can appear on both sides of a fold's train/val split, so a base that only "helps" via within-group memorization can look like a genuine marginal-gain win and get folded into `extra_base_columns` of the resulting `linear_residual_multi` spec.

This is the exact leakage pattern the rest of the discovery pipeline goes to considerable lengths to prevent -- `_tiny_rerank.py`, `_yscale_holdout_gate.py`, `_honest_holdout.py`, `_honest_rmse_gate.py`, and `_stability.py` all explicitly thread `self._group_ids_for_rerank` / `groups=` into their CV, and `_yscale_holdout_gate.py`'s own module docstring describes in detail how a base-additive inverse trained without group-awareness "blows up" on unseen groups in production. The multi-base promotion step is the one CV-based selection stage in this file set that skips that discipline. The downstream `apply_structural_fragility_gate` / `apply_yscale_holdout_gate` / `apply_honest_rmse_gate` gates DO run after promotion and are group-aware, so a badly-fragile multi-base spec is likely to be caught before it ships -- but a *sub-optimal* base pick (one whose apparent 2%+ marginal gain came from in-fold group leakage rather than real orthogonal signal) can still survive if the resulting spec's aggregate RMSE still beats raw-y, since those downstream gates test the final spec's quality, not whether the *selection process* that produced `extra_base_columns` was sound.

`multi_base_enabled` defaults `True` (`_composite_target_discovery_config_base.py:231`), so this runs by default whenever any `linear_residual` spec survives single-base screening -- not a rare configuration. Corroborating the gap: every test in `tests/training/composite/discovery/` that exercises a grouped scenario (`test_composite_discovery_per_group.py`, `test_per_group_discovery.py`, `test_biz_val_grouped_causal_bases.py`) explicitly sets `multi_base_enabled=False`, so the grouped + multi-base combination that would exercise this bug has apparently never been run in CI.

**Suggested fix direction**: thread `random_state=self.config.random_state`, `groups=getattr(self, "_group_ids_for_rerank", None)`, and `time_aware=bool(getattr(self, "_screen_time_ordered_", False))` through the call in `_fit_multibase.py`, mirroring exactly what `_tiny_rerank.py`/`_yscale_holdout_gate.py` already do at their own call sites; add a regression test combining `multi_base_enabled=True` with a grouped fixture and two different `random_state` values.

### F2 -- WAIC tie-break K-fold always uses seed 0, ignoring `config.random_state` (P1, reproducibility)

`_apply_waic_tiebreak` (`_tiny_rerank_waic.py:57-111`), gated by the opt-in `config.transform_waic_validation_enabled` (default `False`), computes the RNG seed for its per-spec `compute_transform_waic(..., random_state=rs, ...)` calls as:

```python
rs = int(getattr(self, "random_seed", 0) or 0)
```

`self` here is the `CompositeTargetDiscovery` instance, which has **no** `random_seed` attribute anywhere in the class (confirmed via repo-wide grep of the whole `discovery/` package) -- the actual configured seed lives at `self.config.random_state`, which every other call site in the sibling `_tiny_rerank.py` module correctly reads (`self.config.random_state` appears 7 times in that file). Because the attribute genuinely does not exist, `getattr` always falls through to the default, so `rs` is always `0`. Every `KFold(n_splits=n_folds, shuffle=True, random_state=random_state)` inside `_oof_residuals_kfold` (`_eval_waic.py:219`) therefore uses the identical seed-0 split on every fit, regardless of what the caller configured `random_state` to.

Impact is narrower than F1 because the whole feature is opt-in and off by default, and WAIC only *re-orders* ties within a `rel_tol=0.02` RMSE band rather than deciding admission -- but when enabled it silently breaks the multi-seed reproducibility contract the rest of the codebase is careful to uphold (e.g. `_stability_check.py`'s `derive_seeds` decorrelation work would be for nothing on this one signal), and any test that varies `random_state` expecting the WAIC-based tie-break order to change accordingly would falsely see no effect.

**Suggested fix direction**: `rs = int(getattr(self.config, "random_state", 0) or 0)`.

### F3 -- `_calibration_gate.py` is fully implemented but never called anywhere (P2, dead code)

The module (201 LOC, `calibration_penalty` / `calibration_adjusted_score` / `CalibrationScore`) is documented as "a PURE, OPTIONAL ranking signal the rerank caller MAY consult" with a module-level `CALIBRATION_GATE_DEFAULT_ENABLED = False` toggle implying a wiring point exists elsewhere. A repo-wide grep for `calibration_adjusted_score` / `calibration_penalty` / `_calibration_gate` finds **zero** references outside the module's own file, in either `src/` or `tests/` (verified with two independent greps). This is a complete, well-tested-in-isolation-looking feature with no integration and no regression test exercising it as part of `fit`/`_tiny_model_rerank`. Not a correctness bug (nothing calls it, so it cannot misbehave), but it is either an abandoned feature that should be removed, or an intended-but-forgotten wiring point that should be finished and covered by a `test_biz_val_*` test per this repo's own convention.

### F4 -- `_interaction_bases.py` docstring names a nonexistent config attribute and mischaracterises the default (P2, docs)

Module docstring (lines 38-42):

> "This module is research-only by default. `discover_interaction_bases` is wired into the discovery base-resolution path ONLY when `config.interaction_base_discovery=True` (opt-in)."

The actual Pydantic field is `interaction_base_discovery_enabled` (`_composite_target_discovery_config_base.py:146`), defaulting `True` -- confirmed both at the config-class definition and at the `_opt_in_steps.py:291` read site, and corroborated by `_opt_in_steps.py`'s own accurate module docstring ("`interaction_base_discovery_enabled` ... default `True` ... each has test-confirmed business value"). The `_interaction_bases.py` docstring both names the wrong attribute (missing `_enabled` suffix) and inverts the actual default (calls it opt-in/off when it is on-by-default). A reader trusting this file's own docstring over the sibling's would misconfigure or misunderstand the feature's default behaviour.

### F5 -- Two files exceed the repo's own file-size convention (P2, architecture)

`screening.py` (938 LOC) and `_tiny_rerank.py` (951 LOC) both sit above the "carve before ~800-900 LOC" soft guidance in this repo's `CLAUDE.md` ("New code goes in focused submodules from the start ... Carve *before* a file nears ~800-900 LOC"), though both remain under the hard `test_no_file_over_1k_loc.py` gate at 1000. `screening.py` in particular is a grab-bag of column extraction, correlation, MI kernels, and re-exports from three already-carved siblings (`_screening_mi_pair`, `_screening_tiny`, `_screening_tiny_perbin`) -- a good chunk of it (the `_mi_per_feature_y_fixed*` / `_mi_to_target*` family, ~350 LOC) could be carved to a `_screening_mi.py` sibling the same way the pair-kernel and tiny-CV pieces already were, consistent with how every other large file in this package has been split.

### F6 -- `_causal_rolling` (temporal-base engineering) is an unvectorized per-row loop (P2, perf)

`_base_engineering.py:80-94`:

```python
def _causal_rolling(y_sorted, window, *, median):
    ...
    for i in range(window, n):
        past = y_sorted[i - window : i]
        out[i] = np.median(past) if median else past.mean()
    return out
```

This is O(n * window) with a Python-level loop and, for the mean case, a fresh `.mean()` reduction per row instead of an O(n) running-sum (the sibling `_grouped_causal_bases.py::_grouped_trailing_impl` in the SAME package already implements exactly this running-window optimisation, `@njit`-compiled, for the grouped-causal variant). The non-grouped `engineer_temporal_bases` this feeds is not auto-wired into `fit()` (only the grouped-causal path via `maybe_add_grouped_causal_bases` is default-wired; the global/non-grouped family is a public standalone helper re-exported from `composite/__init__.py`), so it is not on the discovery hot path today -- but per this repo's "always try njit when using numpy" convention it is a straightforward win (an O(n) running-sum for the mean case, and a numba running-median/deque for the median case) that would also let this helper's `rolling_mean` be reused for the causal-lag engineering fed into `_auto_base` if a future caller wires it in on large frames.

## Proposals

| ID | Category | File | Summary |
|----|----------|------|---------|
| PR1 | test-gap | `_fit_multibase.py` / tests | Add a `multi_base_enabled=True` + grouped-dataset + varying-`random_state` regression test to lock in the F1 fix and catch future regressions of this kind. |
| PR2 | test-gap | `_tiny_rerank_waic.py` / tests | Add a regression test asserting the WAIC tie-break's fold assignment changes when `config.random_state` changes (would have caught F2). |
| PR3 | architecture | `_calibration_gate.py` | Either wire `calibration_adjusted_score` into `_tiny_model_rerank`'s ranking (with a `test_biz_val_*` proving it changes a ranking outcome on a synthetic where a lucky-but-miscalibrated spec should be docked) or remove the module; dead, unreferenced, untested code should not persist indefinitely. |
| PR4 | refactor | `screening.py` | Carve the `_mi_per_feature_y_fixed*` / `_mi_to_target*` family (~350 LOC) into a `_screening_mi.py` sibling, matching the existing `_screening_mi_pair.py` / `_screening_tiny.py` split pattern, to bring the file back under the ~800-900 LOC soft guidance. |
| PR5 | perf | `_base_engineering.py` | Vectorize/numba `_causal_rolling`'s mean branch to an O(n) running sum (mirroring `_grouped_causal_bases._grouped_trailing_impl`'s existing kernel) and the median branch to a numba sliding-window median; low priority since the function is not on the default `fit()` hot path today. |
| PR6 | docs | `_interaction_bases.py` | Fix the module docstring's attribute name and default-value claim to match `interaction_base_discovery_enabled: bool = True` (see F4). |

## Coverage notes

- The two explicitly excluded packages (`feature_selection/filters/**`, `feature_selection/shap_proxied_fs/**`) and their test mirrors were not read, per the audit brief; this package imports nothing from either (confirmed via the import lists at the top of every reviewed file -- all imports resolve within `training/composite/**`, `training/_cv_aggregation.py`, `training/_ram_helpers.py`, or third-party libs).
- `_benchmarks/*.py` (32 files, manual `cProfile`/A-B harnesses invoked via `python -m ...`, all under `if __name__ == "__main__":` or dedicated `bench_*` entry points) were spot-checked only where a finding's call-site correctness needed cross-referencing (e.g. confirming `forward_stepwise_multi_base`'s only non-bench caller for F1); they are not production code and were not independently audited line-by-line, consistent with the cluster's stated ~47-file / ~15.7k-LOC scope which already excludes them.
- No test was executed (read-only audit per the task brief); test-coverage statements above (e.g. "no test exercises this combination") are grep-based absence-of-reference findings, not confirmed via a live pytest run.
- Sibling `_composite_target_discovery_config_base.py` (outside this cluster, under `training/`) was read only for the specific field defaults cited in F1/F4/F5's supporting evidence, not audited as a file in its own right.
