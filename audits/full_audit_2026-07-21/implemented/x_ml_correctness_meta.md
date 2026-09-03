# Cross-cutting: ML-practice meta-review (leakage, reproducibility, calibration, sample weights) -- mlframe audit

## Scope

This is a thematic sweep, not a file-by-file read of the whole tree. Files below were opened and read this
session (excluding `feature_selection/filters/**`, `feature_selection/shap_proxied_fs/**`, and their test
mirrors, per the audit's exclusion list).

Fully read:
- `src/mlframe/feature_selection/zero_importance_pruning.py` (122 LOC)
- `src/mlframe/feature_selection/stochastic_bandit_selection.py` (166 LOC)
- `src/mlframe/feature_selection/stochastic_bandit_selection_ensemble.py` (112 LOC)
- `src/mlframe/feature_engineering/two_step_target_encode.py` (118 LOC)
- `src/mlframe/feature_engineering/holiday_locale_target_encoding.py` (117 LOC)
- `src/mlframe/feature_engineering/cat_cooccurrence_svd.py` (315 LOC)
- `src/mlframe/feature_engineering/tfidf_svd_entity_embedding.py` (145 LOC)
- `src/mlframe/training/feature_handling/ordered_target_encoder.py` (184 LOC)
- `src/mlframe/calibration/post.py` (929 LOC)
- `src/mlframe/calibration/probabilities.py` (278 LOC)
- `src/mlframe/calibration/quality.py` (671 LOC)
- `src/mlframe/calibration/isotonic_risk.py` (144 LOC)
- `src/mlframe/metrics/_core_auc_brier.py` (988 LOC)
- `src/mlframe/metrics/regression/_regression_metrics.py` (826 LOC)
- `src/mlframe/metrics/classification/_weighted_kappa.py` (85 LOC)
- `src/mlframe/metrics/_fairness_metrics.py` (486 LOC)
- `tests/training/test_combine_probs_median_nanmedian.py` (69 LOC)

Partially read (specific line ranges, chosen because they are the sample-weight / metrics-dispatch call
sites relevant to this cluster's theme; the rest of each file is outside this sweep's reach and NOT claimed
as reviewed):
- `src/mlframe/training/reporting/_reporting_regression/__init__.py` (744 LOC total; read lines 1-250 -- the
  `report_regression_model_perf` signature/docstring and the fused-metrics-block dispatch)
- `src/mlframe/training/reporting/_reporting.py` (644 LOC total; read lines 275-365 -- the `report_model_perf`
  signature/docstring)
- `src/mlframe/models/ensembling/score.py` (476 LOC total; read lines 1-140 -- `score_ensemble` signature)
- `src/mlframe/models/ensembling/base.py` (981 LOC total; read lines 590-793 -- `combine_probs`)
- `src/mlframe/models/ensembling/predict.py` (457 LOC total; read lines 180-270 -- the outlier-gate ->
  `combine_probs` call site)
- `src/mlframe/models/ensembling/quality_gate.py` (247 LOC total; read lines 1-200 -- the sample_weight-aware
  member-quality gate)

Grepped for structural patterns (not opened via Read, findings not claimed from these): `random_state`/
`random_seed`/`seed` defaults and `x or DEFAULT` patterns across all of `src/mlframe/` outside the excluded
packages; `sample_weight` occurrences across `metrics/`, `calibration/`, `models/ensembling/`, `evaluation/`;
`src/mlframe/calibration/policy.py` (924 LOC, grepped only for `pick_best_calibrator`/multi-class/axis
patterns -- not read end-to-end, so no line-level findings are claimed there beyond what the grep directly
showed).

**Total: 17 files fully read (5755 LOC) + 6 files partially read (974 LOC of the ranges actually opened) =
23 files touched, 6729 LOC actually read this session.** (`policy.py` grepped only, not counted in either
total.)

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P0 | sample-weights / correctness | `src/mlframe/models/ensembling/base.py:699-704` | `combine_probs`'s `median` flavour passes a per-ROW `sample_weight` (length N) as `weights=` to `np.quantile(stacked, 0.5, axis=0, ...)`, where `axis=0` is the per-MEMBER axis (length M) -- a shape/semantic mismatch that either raises an uncaught `ValueError` or (if M happens to equal N) silently computes a meaningless result; the exact same bug class was already found and fixed at a sibling call site in the same module. |
| F2 | P1 | sample-weights / reporting | `src/mlframe/training/reporting/_reporting_regression/__init__.py:51-91,155-158,219-240` | `report_regression_model_perf` (the main honest-evaluation entry point for regression) has no `sample_weight` parameter at all, so every reported MAE/RMSE/R2/MaxError/MBE/MAPE/... is silently unweighted even when the underlying model was fit with `sample_weight`; an inline comment actively claims the fast metric helpers' full sklearn-signature support (incl. `sample_weight`) is exercised here, which is not true for this call site. |
| F3 | P1 | reproducibility | `src/mlframe/feature_selection/stochastic_bandit_selection.py:51,107,134-135` and `stochastic_bandit_selection_ensemble.py:50,88` | `_stochastic_bandit_selection_core`'s default `cv` (`KFold(random_state=0)`) is independent of the function's own `random_state` parameter; `stochastic_bandit_selection_ensemble` runs `len(seeds)` "independent" bandit searches that, when `cv` is left at its default, all share the exact same CV fold assignment -- only the bandit's own subset-sampling RNG varies across seeds, undermining the module's own stated rationale for the ensemble mode. |
| F4 | P1 | sample-weights / classification reporting | `src/mlframe/training/reporting/_reporting.py:275-320,347-349` | `report_model_perf` (the shared classifier+regressor reporting entry point used by `compare_postcalibrators`/`train_postcalibrators`) has no genuine per-sample `sample_weight` parameter; its `use_weights` flag is a different concept (bin-count weighting inside the calibration-MAE aggregation), so a model trained with row-level `sample_weight` gets its calibration/classification report computed fully unweighted with no way to opt back in. |
| F5 | P2 | leakage / documentation | `src/mlframe/training/feature_handling/ordered_target_encoder.py:79,150`, `src/mlframe/feature_engineering/holiday_locale_target_encoding.py:80` | The causal/leak-free ordered target encoders compute their smoothing `global_prior` (used to shrink every row's expanding-mean encoding, most heavily for early/low-count rows) as `np.mean(y)` over the WHOLE column, including rows that occur causally AFTER the row being encoded -- a small but real amount of future-target information leaks into the prior term for every row. |
| F6 | P2 | leakage / documentation | `src/mlframe/feature_engineering/tfidf_svd_entity_embedding.py:82-141` | `tfidf_svd_entity_embedding`'s single-call convenience wrapper fits the TF-IDF vocabulary and TruncatedSVD basis on the entire `df` passed to it; nothing in the docstring warns a caller that passing a full train+val/test frame here leaks distributional (not target) information -- vocabulary/IDF weights and the SVD basis reflecting val/test entities -- into the "training" embedding, unlike the explicit `return_fitted=True` + `transform_new_entities` path which is fit/transform-safe. |
| F7 | P2 | sample-weights / calibration | `src/mlframe/calibration/post.py:112-183,659-678`, module-wide | No function in `src/mlframe/calibration/` accepts a `sample_weight` parameter anywhere (`BinaryPostCalibrator.fit`, `get_postcalibrators`, `compare_postcalibrators`, `train_postcalibrators`) -- every third-party calibrator in the zoo is always fit unweighted even when the upstream model was trained with `sample_weight`, and this limitation is not documented anywhere in the module. |
| F8 | P2 | architecture | `src/mlframe/calibration/post.py` (929 LOC), `src/mlframe/calibration/policy.py` (924 LOC) | Both files are at/over the repo's own ~900-1000 LOC split convention; `post.py` in particular mixes the calibrator zoo, the binary-only guard, the inner-CV vs self-eval selection logic, and the calib/test-overlap safety checks in one file and is a natural split candidate (e.g. `post_zoo.py` / `post_train.py`). |
| F9 | P2 | ML best-practice / documentation | `src/mlframe/feature_engineering/two_step_target_encode.py:29,56-65` | `two_step_recency_weighted_target_encode` defaults to `causal=False`, which the docstring itself calls "a target-leakage source if the output is used as a per-EVENT training feature" -- the leaky mode is well-documented but is still the silent default a caller gets without reading the full docstring. |
| F10 | P2 | test-coverage | `tests/training/test_combine_probs_median_nanmedian.py` (whole file) | The only unit tests for `combine_probs(flavour="median")` never exercise the `sample_weight is not None` branch (see F1) -- the buggy branch has zero test coverage, which is how it went unnoticed while the sibling fix (in `_compute_outlier_gate`) has an explanatory comment referencing it. |

### F1 -- `combine_probs` median-flavour sample_weight axis mismatch (P0)

`combine_probs(stacked, "median", sample_weight=sw, ...)` is the single canonical ensemble-math helper shared
by both the training (`score_ensemble`) and prediction (`ensemble_probabilistic_predictions` /
`predict.py`) paths. `stacked` has shape `(M, N, ...)` (M = number of ensemble members, N = number of rows).
For the `median` flavour only, `base.py:700-702` does:

```python
if sample_weight is not None:
    try:
        combined = np.quantile(stacked, 0.5, axis=0, weights=sample_weight, method="inverted_cdf")
    except TypeError:
        combined = np.median(stacked, axis=0)
```

`axis=0` is the MEMBER axis (length M, typically 2-10), but `sample_weight` is documented elsewhere in the
very same package (`predict.py:196-205`) as "a per-ROW vector (length N), not a per-member vector". `numpy.quantile`'s
`weights` argument must align with the reduced axis's length, so `sample_weight` (length N) reaching a
reduction over an axis of length M will raise `ValueError` in realistic ensembles (M != N almost always) --
not caught by the `except TypeError:` clause, so this is a live crash, not a silent misresult, whenever a
caller selects `ensemble_method="median"` together with `sample_weight`. The `predict.py:196-205` comment
shows the exact same bug was previously found and *fixed* at a sibling call site
(`_compute_outlier_gate`'s median-based outlier anchor) with the explicit reasoning "applying it directly to
`np.quantile(axis=0)` is meaningless -- the member axis is uniformly weighted by construction" -- but the fix
was never propagated to `combine_probs`, which is the function that actually produces the final ensembled
output returned to callers (not just a diagnostic). `tests/training/test_combine_probs_median_nanmedian.py`
confirms this: its three tests exercise only the unweighted path. Suggested fix: drop the `sample_weight=`
kwarg from the `median` branch entirely (mirroring the `_compute_outlier_gate` fix and the `rrf`/`median`
"weights ignored silently" contract already documented in the function's own docstring at line 635), or
raise explicitly if a caller passes `sample_weight` with `flavour="median"` so the limitation is loud rather
than crash-on-first-use.

### F2 -- regression reporting has no `sample_weight` parameter (P1)

`report_regression_model_perf` (`_reporting_regression/__init__.py:51-91`) has 30 keyword parameters and none
of them is `sample_weight`. The 1-D fast path (lines 219-240, the common case) calls
`fast_regression_metrics_block_extended(targets_arr, preds_arr)` with no weight argument, even though the
underlying `mlframe.metrics.regression._regression_metrics` kernels (`fast_mean_absolute_error`,
`fast_mean_squared_error`, `fast_r2_score`, ...) fully support `sample_weight` (verified by reading that
module). The comment directly above the dispatch (lines 155-158) reads: "Numba fast helpers now cover full
sklearn signature (1-D / 2-D, sample_weight, multioutput) so all regression metric call sites here go
through the fast path" -- which is true of the underlying kernels but not of this call site, since no weight
is ever threaded in. Any suite that fits a regression model with `sample_weight` (e.g. business-value or
class-imbalance weighting) gets an "honest" test-set report that silently ignores those weights. Suggested
fix: add a `sample_weight: Optional[np.ndarray] = None` parameter, thread it into both the fused-block and
per-output `fast_*` calls, and update the misleading comment to state clearly whether/where it applies.

### F3 -- stochastic bandit ensemble's "independent" seeds share one CV split (P1)

`_stochastic_bandit_selection_core` (`stochastic_bandit_selection.py:32-92`) defaults `cv` to
`KFold(n_splits=3, shuffle=True, random_state=0)` (line 51) whenever the caller passes `cv=None`, completely
independent of the function's own `random_state` parameter (which only seeds `rng = np.random.default_rng(random_state)`,
line 52, used for the subset sampler). `stochastic_bandit_selection_ensemble` (`stochastic_bandit_selection_ensemble.py:42-108`)
runs `len(seeds)` calls to this core loop, each with a different `random_state=seed` (line 93) but the SAME
`cv` object propagated unchanged from the ensemble's own `cv=None` default (line 50) through to every seed's
core call (line 88). Concretely: unless the caller manually constructs and passes a `cv` splitter, every
"independent" seed in the ensemble sees identical train/test fold membership -- only the random-subset draw
differs. The module's own docstring (`stochastic_bandit_selection_ensemble.py:1-13`) frames the value
proposition as "Running several independent seeds ... recovers a more complete feature set" and treats the
per-feature stability fraction as measuring genuine independent-run agreement; with a shared CV split, part
of that intended independence (the data-partition randomness) is not actually varying across seeds. Suggested
fix: derive the default `cv`'s `random_state` from the caller's `random_state`/the per-seed loop variable
(e.g. `KFold(..., random_state=seed)` inside the ensemble loop when `cv is None`), or document explicitly
that callers wanting truly independent per-seed CV must pass a `cv` factory rather than a single splitter
instance.

### F4 -- classifier reporting has no genuine `sample_weight` parameter (P1)

`report_model_perf` (`_reporting.py:275-320`) is the shared entry point used by both `compare_postcalibrators`
and `train_postcalibrators` (`calibration/post.py`) to compute the "honest" evaluation metrics that select
the winning calibrator. Its `use_weights: bool = True` parameter (line 285, documented at line 347-349 as
"Whether to use weighted calibration metrics") is a bin-count weighting applied inside the calibration-MAE
aggregation (confirmed via `_reporting_probabilistic.py:102,162,468,532`), not a per-row `sample_weight`
array. There is no parameter anywhere in this function's 40+ arguments that accepts a genuine per-sample
weight vector. Combined with F7 (calibration/ having no sample_weight support at all), this means the entire
calibration-comparison and honest-metric pipeline is structurally unweighted, end to end, with no way for a
caller to propagate row-level importance weights from training through to reported metrics.

### F5 -- ordered/causal target encoders leak the global prior (P2, documented tradeoff)

`ordered_target_encode` (`ordered_target_encoder.py:79`) and `ordered_target_encode_batch` (line 150) both
compute `global_prior = float(np.mean(y)) if prior is None else float(prior)` over the FULL `y` array before
doing the causal expanding-mean walk. `holiday_name_target_encode_cross_locale`
(`holiday_locale_target_encoding.py:80`) does the same. Since the smoothing term is
`(running_sum + smoothing * global_prior) / (running_count + smoothing)`, the prior's weight is largest
exactly for early-in-order / low-count rows -- meaning the rows most reliant on the prior are the ones that
get the most future-derived information smuggled in via that scalar. This mirrors CatBoost's own published
"ordered target statistics" design (which also uses a single global-average prior), so it is very plausibly
an intentional simplification rather than an oversight, and the leak is bounded (a single scalar, not a
per-row statistic) -- but it is nonetheless a real, unacknowledged departure from a strictly zero-leakage
causal encoder, and none of the three docstrings mention it. Alternative reading: if this is intentional
(matching the reference algorithm), a one-line docstring note would prevent a future reader from assuming
the encoder is leak-free in the strictest sense. Suggested direction: either add the docstring caveat, or
offer an opt-in causal-prior variant (expanding mean of `y` up to each row) for callers who need the
stricter guarantee.

### F6 -- `tfidf_svd_entity_embedding` fits its unsupervised basis on the whole input frame (P2)

`tfidf_svd_entity_embedding` (`tfidf_svd_entity_embedding.py:82-141`) is target-free (never touches `y`), so
it cannot leak target information -- but its single-call convenience form fits both the `TfidfVectorizer` and
`TruncatedSVD` on whatever `df` is passed in (line 124-132), with no guidance that a caller who hands it a
combined train+val/test frame gets a vocabulary/IDF/SVD basis informed by entities that should be held out.
The module already ships the fix for this (`return_fitted=True` + `FittedTfidfSvdEntityEmbedding.transform_new_entities`,
lines 36-79) but the plain, more-likely-to-be-reached-for call signature doesn't warn against the naive
train+test-together usage. Suggested fix: a one-line docstring note recommending `return_fitted=True` +
per-split calls whenever CV/train-test discipline matters, mirroring the note already present in
`FittedTfidfSvdEntityEmbedding`'s own docstring.

### F7 -- calibration/ has zero sample_weight support anywhere (P2)

Grepping every `.py` file under `src/mlframe/calibration/` for `sample_weight` returns no matches. Both
`BinaryPostCalibrator.fit` (`post.py:154-183`) and every third-party calibrator instantiated by
`get_postcalibrators` (`post.py:278-348`) are always fit on `(calib_probs, calib_target)` with no weight
argument, and `train_postcalibrators`'s extensive docstring (`post.py:679-709`) never mentions the limitation.
For a suite that trains its base models with `sample_weight` (a common pattern for class-imbalance or
business-value weighting), the post-hoc calibration layer silently reverts to unweighted fitting with no
opt-out flag and no warning. Suggested direction: either thread `sample_weight` through to the calibrators
that support it (sklearn's `CalibratedClassifierCV` does) and document per-calibrator support/non-support in
the zoo, or add an explicit docstring note that calibration fitting is always unweighted today.

### F8 -- `post.py` / `policy.py` at the module-size split threshold (P2)

`post.py` (929 LOC) and `policy.py` (924 LOC) are both at/over the repo's documented ~900-1000 LOC
sibling-split convention. `post.py` in particular bundles four largely-independent concerns: the
`BinaryPostCalibrator` adapter + calibrator zoo construction, the `compare_postcalibrators` inner-CV/self-eval
selection engine, the calib/test-overlap safety-guard logic in `train_postcalibrators`, and the on-disk
persistence + sidecar-metadata writing -- a natural candidate for a `post_zoo.py` (adapter + zoo) /
`post_compare.py` (selection engine) / `post_train.py` (safety guards + persistence) split following the
project's own flat-sibling convention.

### F9 -- `two_step_recency_weighted_target_encode` defaults to the leaky mode (P2, alternative reading noted)

`two_step_recency_weighted_target_encode` (`two_step_target_encode.py:20-30`) exposes `causal: bool = False`
as its default. The docstring is exemplary about the tradeoff (lines 56-65: "a target-leakage source if the
output is used as a per-EVENT training feature"), so this reads as a deliberate default chosen for the
"one-shot entity score" use case rather than an oversight -- flagging it here only because a caller who skims
the signature (not the full docstring) and reaches for this as a row-level training feature would silently
get the leaky variant. No code change suggested beyond, optionally, a short inline comment at the call site
convention (e.g. requiring `causal=` to be passed explicitly with no default) if the maintainers want to force
the choice to be visible at every call site.

### F10 -- `combine_probs(median, sample_weight=...)` has zero test coverage (P2)

`tests/training/test_combine_probs_median_nanmedian.py` is the only test file exercising the `median` flavour
of `combine_probs`, and all three of its tests call `combine_probs(stacked, "median")` with no `sample_weight`
argument. This is precisely why F1 shipped unnoticed: the buggy branch is reachable from the public API
(`score_ensemble`/`ensemble_probabilistic_predictions`/`predict.py`, all of which forward a caller-supplied
`sample_weight`) but has no regression test. Suggested fix: add a test that calls
`combine_probs(stacked, "median", sample_weight=w)` with `M != N` and asserts it either raises a clear,
documented error or produces the intended per-row-weighted result (once F1 is fixed one way or the other).

## Proposals

| ID | Category | File | Idea |
|----|----------|------|------|
| PR1 | test-coverage | `src/mlframe/models/ensembling/base.py` | Add a dedicated `combine_probs` parity test matrix across every flavour x {sample_weight None/given} x {2-D/3-D `stacked`} combination, not just `median` -- `harm`/`arithm`/`quad`/`qube`/`geo` correctly ignore `sample_weight` in favour of `precomputed_weights` (member-axis weights) today, but that distinction is easy to blur in a future edit without a test pinning it. |
| PR2 | ML best-practice | `src/mlframe/calibration/` | Consider a documented, opt-in multi-class calibration path (e.g. a one-vs-rest wrapper around the existing binary zoo) -- today `compare_postcalibrators` explicitly raises on non-binary targets (`post.py:403-411`, correctly guarded), pushing the entire multi-class case onto the caller with only a pointer in the error message ("route multi-class calibration through a one-vs-rest wrapper upstream") and no such wrapper shipped in-repo that this sweep could find. |
| PR3 | reproducibility | `src/mlframe/feature_selection/stochastic_bandit_selection.py`, `zero_importance_pruning.py`, `greedy_backward_elimination.py` | These three selectors all default an internal `KFold(..., random_state=0)` when `cv=None`, independent of any caller-supplied seed. None of them are wrong in isolation (the caller can always pass an explicit `cv`), but a repo-wide convention -- e.g. a shared helper `_default_cv(random_state)` that seeds the default splitter FROM the function's own `random_state`/similar parameter when one exists -- would close this whole class at once rather than leaving each selector to reinvent (or, per F3, half-reinvent) it. |
| PR4 | leakage / test-coverage | `src/mlframe/training/feature_handling/ordered_target_encoder.py` | A `biz_val` test that measures the actual magnitude of the F5 prior-leakage effect (e.g. compare OOF AUC of a downstream model using the causal-encoded feature vs. a version with a strictly-causal expanding prior) would turn the "plausibly negligible" claim in F5 into a measured one. |
| PR5 | sample-weights | `src/mlframe/training/reporting/_reporting_regression/__init__.py`, `_reporting.py` | Once F2/F4 are fixed, add a `test_biz_val_*` regression test that fits a model with a strongly skewed `sample_weight` (e.g. 100x on a small subpopulation), confirms the reported metric changes materially between weighted and unweighted reporting, and pins that the weighted number matches an independent sklearn-with-`sample_weight` reference. |

## Coverage notes

- `src/mlframe/calibration/policy.py` (924 LOC) was only grepped, not read end-to-end; it is the module that
  implements `pick_best_calibrator` and `_stratified_inner_folds` (referenced by `post.py`) and would be the
  natural next stop for a deeper calibration-correctness pass -- specifically to verify its inner-CV fold
  construction doesn't have its own version of the same-OOF selection-optimism bug class it explicitly claims
  to have already fixed once (referenced from `post.py:372-373,428-432`).
- `src/mlframe/models/ensembling/base.py`, `score.py`, `predict.py`, `quality_gate.py` were read only in the
  sample-weight-relevant ranges; the rest of `models/ensembling/` (`score_flavours.py`, `score_gate.py`,
  `score_validate.py`, `selection.py`, and the full `process_method.py`) was not opened this session and could
  hide further instances of the F1 bug class (a `sample_weight` reaching a reduction over the wrong axis) that
  this sweep did not have time to chase down.
- `src/mlframe/metrics/` is 12,725 LOC across ~35 files; this sweep read 2 files in full
  (`_core_auc_brier.py`, `regression/_regression_metrics.py`) plus 2 smaller ones
  (`classification/_weighted_kappa.py`, `_fairness_metrics.py`) as the "8-10 non-trivial metrics" sample. The
  reviewed metrics (ROC AUC weighted/unweighted, PR AUC, Brier, MAE/MSE/RMSE/R2 weighted/unweighted, max_error,
  quadratic/linear weighted kappa, Tukey-fence fairness dispersion) were all found correct against their
  standard definitions with solid edge-case handling (empty input, single-class, NaN propagation) -- but
  `classification/_classification_report.py`, `classification/_gains_lift.py`, `classification/_ordinal_cutpoints.py`,
  `classification/_threshold_optimization.py`, `_multilabel_metrics.py`, `ranking.py`, `rank_correlation.py`,
  `_drift.py`, `_ice_metric.py`, and `quantile.py` were not opened and remain unreviewed by this cluster.
- `src/mlframe/feature_engineering/` (large tree, dozens of feature-builder modules) was sampled only at the
  four target-encoding-style modules the task explicitly named plus `cat_cooccurrence_svd.py`; other
  target/statistic-computing feature builders in that directory (there are 100+ modules) were not audited for
  the fit-before-split leakage pattern.
- `src/mlframe/training/composite/` (the largest single subtree in scope, per `find` output) was not opened at
  all this session beyond grep hits for hardcoded seeds -- a follow-up pass specifically on
  `training/composite/ensemble/`, `training/composite/discovery/`, and `training/composite/transforms/` for
  the leakage/reproducibility/sample-weight themes would be the highest-value next step given the size of that
  subtree relative to the time available here.
- `src/mlframe/votenrank/` was not opened this session despite being named explicitly in the cluster's brief
  ("does it recur in training/, calibration/, votenrank/, models/ensembling/") -- grep found no hardcoded-seed
  or `x or DEFAULT`-on-random_state hits there, but no file in that directory was read for the sample-weight or
  reproducibility themes beyond that grep.
