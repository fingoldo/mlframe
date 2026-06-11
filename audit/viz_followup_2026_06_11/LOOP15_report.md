# LOOP15 -- Cross-split overfitting panel (UX/FUNCTIONALITY)

## Verdict: RESOLVED

A cross-split overfit view is NOT redundant with the per-split model card: the model card describes ONE split in
isolation and structurally cannot show a train-test gap. This panel puts the splits side by side and computes the
single most-asked first-check (train->test degradation), which no existing chart provided.

## Design

`compose_split_comparison_figure(per_split, task, ...)` -- one model, splits as the comparison axis. Two panels:

1. **Grouped-bar headline metrics per split.** One group per metric, one bar per split (train/val/test/oof, canonical
   left-to-right order, stable per-split color). Classification headline = ROC_AUC / PR_AUC / KS / ECE / Brier;
   regression = R2 / RMSE / MAE / bias. Every metric mapped to a comparable [0,1] "quality" (lower-is-better inverted;
   RMSE/MAE/|bias| normalized by the train target std) so bars share one axis and the cross-split comparison is direct.
2. **Delta table + traffic-light verdict.** Raw train->val, val->test, train->test change per metric, plus an
   `OverfitVerdict` traffic light with a stated reason.

**Overfit rule (stated, simple):** classification fires on the train->test ROC_AUC drop (>=0.10 RED, 0.03-0.10 AMBER,
else GREEN); regression on the test/train RMSE ratio (>=1.50x RED, 1.15-1.50x AMBER, else GREEN). Thresholds are module
constants (`AUC_GAP_RED/AMBER`, `RMSE_RATIO_RED/AMBER`). The verdict is reachable headless via `overfit_verdict(...)`.

**Efficiency:** metrics computed once per split through the SAME kernels the model card uses
(`_classification_metrics` over one `_ScoreSort`; `_regression_metrics`), no re-implementation. Raw arrays are
subsampled per split (cap 200k, aligned index) before the metric pass; aggregate-first, no per-row loops. Precomputed
`{"metrics": {...}}` entries short-circuit the raw path entirely.

**Edge-safe:** missing splits simply do not appear; single-class / no-finite-pair splits are annotated in the table and
excluded from bars + verdict; <2 usable splits degrades to an honest 1-panel note.

## biz_value (overfit gap + flag flip)

| synthetic | task | verdict | gap / ratio |
|---|---|---|---|
| memorize-train (train AUC ~0.99, test AUC ~0.70) | classification | **RED OVERFIT** | train->test AUC drop **+0.256** |
| generalizing (train~test) | classification | **GREEN GENERALIZES** | gap **+0.006** |
| memorize-train (train RMSE ~0, test RMSE large) | regression | **RED OVERFIT** | RMSE ratio **59.8x** |
| generalizing | regression | **GREEN GENERALIZES** | RMSE ratio **1.005x** |

Flag flips RED->GREEN purely on whether the model overfits; classification gap delta between the two synthetics
0.256 - 0.006 = **0.250** (test asserts >= 0.18). This is the core value the panel exists to surface.

## Tests (20, all pass, 8.3s)

Unit: figure structure (clf + reg), canonical split ordering, delta-table value equals the independent metric
difference, verdict red/amber/green at the AUC-gap thresholds, regression RMSE-ratio red/green, missing-split (train+test
only), single-class split annotated+excluded, <2 usable splits + no-splits degradation, precomputed-metrics short-circuit,
`overfit_verdict` raises on a single split. biz_value: 5 tests (clf RED big-gap, clf GREEN small-gap, reg RED, reg GREEN,
flag-flip delta). cProfile: 4-split x 80k-row build bounded < 5s.

## cProfile

The bounded-profile test builds the production-shape figure (4 splits x 80k rows = 320k metric rows after the per-split
200k cap is not hit) inside a `cProfile.Profile`; total wall asserted < 5.0s. Hotspots are the per-split `_ScoreSort`
sort + `_classification_metrics` (DeLong/ECE), already-tuned shared kernels reused from the model card -- no new hot
path introduced, so no additional optimization was warranted (subsample cap bounds the only superlinear step).

## Artifacts

- Module: `src/mlframe/reporting/charts/split_comparison.py`
- Test: `tests/reporting/test_charts_split_comparison.py`
- Gallery: `scripts/render_gallery.py` (new `split_comparison` entry, overfit synthetic) ->
  `docs/gallery/split_comparison/split_comparison.png` (rendered; bars show train towering over val/test, table shows
  train->test ROC_AUC -0.242, RED OVERFIT verdict)
- Re-export: `src/mlframe/reporting/charts/__init__.py` (parity with model_card / model_comparison)
