# Wave 1 (2026-11-05), attack class A6: numerics and degenerate-input semantics

Read-only audit by one Opus 5.5 low-effort agent. Only code was read; nothing was run and no file was edited. The report is the agent's returned text, condensed; each finding is re-checked on HEAD before it is fixed. Coverage was partial; the "Not covered" list says what was not read.

## Findings

| ID | Severity | Location | Defect |
|---|---|---|---|
| M6-01 | Medium | `src/mlframe/votenrank/rank_percentile_stacking.py:95-106` | A test value just outside the OOF range gets 0.0 or 1.0 instead of the OOF extreme percentile, so the test scale has a gap the meta-learner never saw. |
| M6-02 | Low | `src/mlframe/metrics/calibration/_calibration_metrics.py:78-83` | With bins present but every `hits` at 0, the function reports perfect calibration (MAE 0) instead of no measurement. |
| M6-03 | Low | `src/mlframe/core/stats.py:67-78` | `get_tukey_fences_multiplier_for_quantile(0.5)` returns `inf`; the input check accepts 0.5. |
| M6-04 | Low | `src/mlframe/core/proportion_stats.py:89-98` | `proportions_significantly_different` does not check `0 <= m <= n` (its sibling `wilson_interval` does), so an impossible input gets an answer. |

### M6-01

OOF percentiles run from `0.5/n` up to `(n-0.5)/n`, because `rankdata` is one-based. Test values are mapped with `(left+right)/2/n`. A test value equal to the OOF max gets `(n-0.5)/n`; any value even slightly above the max gets `right = left = n`, which gives exactly 1.0. With `oof=[1, 2]` and `test=[2.0, 2.0001, 0.9999]` the output is `[0.75, 1.0, 0.0]`, while the OOF scale only spans `[0.25, 0.75]`. The docstring promises "clamps to the nearest extreme"; the correct output is `[0.75, 0.75, 0.25]`, which needs clipping to `[0.5/n, (n-0.5)/n]`. The jump is `0.5/n`, large for small `n_oof`, the regime this stacking trick targets. Drifted test predictions are routinely outside the OOF range, so this is plausible input.

### M6-02

Take `hits=[0, 0]` with `use_weights=True`. `weights` is all zeros and `w_sum = 0`, so normalisation is skipped and `mae = sum(diffs * 0) = 0.0`, `std = 0.0`. Coverage is 0, but the MAE reads as a perfectly calibrated model. The empty-input branch two lines below returns `1.0, 1.0` (worst), and sibling kernels return NaN; this case should do the same.

### M6-03

At `quantile=0.5`, `norm.ppf(0.5) = 0`, so `(sd_sigma - ppf) / (2*ppf)` divides a NumPy float64 by zero. It returns `inf` with only a RuntimeWarning, and `lru_cache` keeps that value. The function's own fix comment exists to stop plus/minus inf from spreading, yet 0.5 still lets it through. Cleanest fix: raise a `ValueError` for 0.5.

### M6-04

With `m1=12, n1=10, m2=5, n2=10`: `p1 = 1.2`, `pooled = 0.85`, `se` is about 0.16, `z` is about 4.4, so the function returns `True` ("significantly different") on impossible counts instead of raising. If the pooled rate goes above 1, `math.sqrt` raises a domain error instead.

## Checked and found correct (traced by hand)

- `fast_classification_report`, `fast_precision`, and both MAPE kernels: max, mean, sMAPE, MdAPE, wMAPE, including the empty/NaN/zero cases.
- Batched and scalar Spearman: NaN handling, constant rows, average ranks for ties, `n < 2`.
- Binary log loss, sequential and parallel: one class, NaN labels, out-of-range probabilities, mapping two labels to 0/1.
- Brier, sequential and parallel: validation and the larger-label-is-positive mapping.
- Probability separation score; EWMA with `adjust=True/False` against pandas `ignore_na=False`, NaN included.
- Plug-in and debiased ECE, Murphy decomposition, and the `policy._ece_score` kernels.
- The BCa `z0` and acceleration formula (matches Efron); the bin smoother on constant or all-NaN input; the Wilson interval.

## Not covered

- `metrics/`: `ranking.py`, `_ranking_extras.py`, `quantile.py`, `_drift.py`, `_fairness_metrics.py`, and the AUC kernels (`_core_auc_brier.py`, `_auc_per_group.py`).
- `metrics/classification/`: `_weighted_kappa`, `_gains_lift`, ordinal cutpoints, threshold optimisation; all of `metrics/regression/` beyond the MAPE family.
- `calibration/` beyond `policy.py`, the ECE section, and the first 260 lines of `_calibration_metrics.py`.
- `evaluation/`, apart from the BCa reducer; `signal/`; `data_valuation/`; `competition/`.
- `votenrank/`, apart from `rank_percentile_stacking`.
- `preprocessing/` and the non-transformer `feature_engineering/`.
