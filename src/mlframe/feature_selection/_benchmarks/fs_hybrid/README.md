# FS hybrid benchmark

Compares the four feature selectors alone and in hybrid combinations, scored by **honest-holdout AUC**
(selector never sees the test split) across three downstream model families with different inductive
biases: **LightGBM** (trees), **Logistic** (linear), **kNN** (distance). Also reports ground-truth
recovery, parsimony, and selector wall-time.

## Files
- `synth.py` — controlled generator. 8 causal latents (linear + a multiplicative interaction `inf_4*inf_5`
  + a quadratic `inf_6**2`), redundant correlated clusters around 3 of them, and pure noise. This shape
  is deliberate: trees capture the interaction/quadratic natively, a linear model needs the constructed
  features, kNN is hurt by noise — so the per-model divergence is observable.
- `fs_selectors.py` — uniform `.fit/.transform` adapters over MRMR / RFECV / BorutaShap / ShapProxiedFS,
  plus `Cascade` (chain) and `Ensemble` (union/intersect) combinators. All outputs are DataFrames with
  ASCII-safe column names (RFECV is DataFrame-first; LightGBM rejects names with `(),`).
- `run_experiment.py` — runs the strategy roster x seeds, writes `_results/results.jsonl` and a
  `_results/progress.txt` checkpoint per cell.
- `analyze.py` — aggregates `_results/results.jsonl` into per-model decision tables + a one-size-fits-all check.

## Run
```
python -m mlframe.feature_selection._benchmarks.fs_hybrid.run_experiment
python -m mlframe.feature_selection._benchmarks.fs_hybrid.analyze
```
Results land in `_results/` (git-ignored). The slow tail is ShapProxiedFS (gated to fewer cells).

## Swap in real data
Replace `synth.make_dataset(seed)` with a loader returning `(X_df, y_series, truth_dict)`, where `truth`
has keys `base` / `relevant` / `noise` (set them to `[]` if unknown — recovery metrics become N/A but the
honest-holdout AUC comparison across models still drives the verdict).
