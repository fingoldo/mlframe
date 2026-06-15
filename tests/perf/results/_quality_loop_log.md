# mlframe ACCURACY / QUALITY loop log

This loop hunts measurable ML-QUALITY wins (calibration / probability post-processing / metric-correctness),
NOT speed (the perf loop is saturated -- see `_loop_iter_log.md`). Each iteration picks ONE self-contained
accuracy lever in a low-contention area, measures it on the HONEST metric (holdout / OOF / known ground-truth)
across MULTIPLE seeds AND scenarios, and either RESOLVES (flips the default, keeps the old as opt-in per
REJECTED!=DELETED, adds a `biz_value` test) or REJECTS (keeps the committed bench + verdict, no default change).
A flip requires a MAJORITY win across seeds/scenarios -- never a single-seed/single-scenario result.

| iter | lever | scenario / seeds | honest metric: old -> new | verdict | commit |
|------|-------|------------------|---------------------------|---------|--------|
| qual-1 | Debiased binned ECE (subtract per-bin Bernoulli noise floor) as headline-ECE default in `fast_calibration_report` (`ece_debiased=True`; plug-in kept as `ece_debiased=False`) | 3 perfectly-calibrated (uniform / beta-rare / bimodal, true ECE=0) + 2 miscalibrated (sigmoid-shift / overconf, fine-grid truth) x 7-8 seeds x nbins {10,15,20}; honest metric = \|estimate - ground-truth ECE\| | mean bias-vs-truth on calibrated scenarios @nbins=15: uniform 0.0294 -> 0.0127, beta-rare 0.0128 -> 0.0055, bimodal 0.0196 -> 0.0082 (>2x bias reduction); per-cell \|bias\| wins debiased/plugin = 29/6 @nbins=10, 28/7 @15, 32/8 @20 (debiased wins ~80% of cells, majority at EVERY bin count); miscalibrated model still flagged (true-gap term dominates) | RESOLVED (flip default) | 47713b8e |
