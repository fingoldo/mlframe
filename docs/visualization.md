# Visualization & Diagnostics

Reference for the charts `train_mlframe_models_suite` emits and how to tune
them. For a quick overview see the README "Visualization & Diagnostics"
section; this page is the fuller catalogue.

All chart appearance is controlled by `ReportingConfig`
(`mlframe.training.configs.ReportingConfig`, passed as the suite's
`reporting_config`). Charts are emitted **only when the suite has somewhere
to write them** -- set `output_config.data_dir` (charts land under
`<data_dir>/charts/...`). A default run with no `data_dir` computes metrics
but saves no figures; the suite logs a one-line hint when that happens so the
absence is never silent.

Everything described here is **default-ON** unless noted. mlframe's policy is
that a correct diagnostic ships enabled; you remove tokens / flip flags to opt
*out*, not in.

---

## 1. How a chart is produced

Each chart is built in two stages:

1. A **spec builder** (`mlframe.reporting.charts.*`) turns arrays into a
   pure-data `FigureSpec` -- a backend-agnostic description of panels (no live
   matplotlib/plotly object). Specs are picklable and renderer-independent.
2. A **renderer** (`mlframe.reporting.renderers.render_and_save`) draws the
   spec once per requested backend and writes every requested format.

Which backends/formats are produced is the **plot-outputs DSL**
(`ReportingConfig.plot_outputs`); which panels appear is the **per-task panel
template** (`ReportingConfig.<task>_panels`). Both are validated at config
construction, so a typo fails before training starts, not mid-figure.

---

## 2. plot-outputs DSL

```
ReportingConfig.plot_outputs = "plotly[html] + matplotlib[png]"   # default
```

Grammar (`mlframe.reporting.output.parse_plot_output_dsl`): one or more
`<backend>[<fmt>,<fmt>,...]` clauses joined by `+`. Each clause renders the
figure on that backend and writes each listed format.

| Backend | Allowed formats |
|---|---|
| `matplotlib` | `png`, `pdf`, `svg`, `jpg`, `jpeg` |
| `plotly` | `html`, `png`, `svg`, `pdf`, `json` |

Rules: backend must be `matplotlib` or `plotly`; no duplicate backends in one
DSL; no duplicate formats within a backend; a backend may not write a format
it doesn't support (matplotlib can't write `html`, etc.).

**Why the default is `plotly[html] + matplotlib[png]`:** interactive HTML is
ideal for sharing / notebooks, while the static PNG comes from matplotlib
rather than plotly. Plotly PNG export goes through kaleido, which spends
~12-15s per figure on a Chromium `page.reload()`; on a multi-model x VAL+TEST
suite that is minutes of pure export wall-time. The matplotlib PNG path is
10-20x faster and needs no Chromium. If you specifically want plotly-rendered
PNGs, set `plot_outputs="plotly[html,png]"` (kaleido must be installed; absent
it, the plotly PNG silently falls back to HTML).

Render order is left-to-right, so listing `matplotlib` first makes it the
primary artifact.

---

## 3. Default panels per task type

The suite auto-detects the task from the prediction shapes (see
`mlframe.reporting.auto_dispatch.render_multi_target_panels`) and renders that
task's panel template. Pass an explicit `target_type` for unambiguous
dispatch; otherwise shape heuristics apply (1-D targets + 1/2-column proba ->
binary, K>=3 proba -> multiclass, 2-D targets -> multilabel, `group_ids` ->
LTR, `quantile_alphas` + 2-D preds -> quantile, regression has its own scatter
/ residual charts).

Discover every token at runtime:

```python
from mlframe.reporting import describe_available_panels
describe_available_panels()   # prints per-task tokens + one-line descriptions
# also returns the structured mapping: {task_type: [(token, description), ...]}
```

### Binary classification

`binary_panels = "ROC PR SCORE_DIST KS THRESHOLD GAIN PIT"`

Binary classification previously had **no curve charts at all** (the
dispatcher skipped it, leaving only a reliability diagram). The full curve set
is now default-ON:

| Token | What it shows |
|---|---|
| `ROC` | TPR vs FPR with the chance diagonal; AUC in the title. |
| `PR` | Precision vs recall with the prevalence no-skill baseline; AP in the title. |
| `SCORE_DIST` | Overlaid class-conditional score histograms (y=0 vs y=1) with a threshold marker. |
| `KS` | Class-conditional score ECDFs with the max-gap (KS statistic) marked. |
| `THRESHOLD` | Precision / recall / F1 vs threshold + queue-rate on a secondary axis -- the operating-point picker; supports a cost variant via `cost_ratio`. |
| `GAIN` | Cumulative-gain curve (positives captured vs population) with the lift area shaded; a decile gain/lift/KS table lands in `metrics`. |
| `PIT` | Probability-integral-transform histogram; uniform = well-calibrated. |

### Multiclass classification

`multiclass_panels = "CONFUSION CONFUSED_PAIRS PR_F1 ROC CALIB_GRID PROB_DIST TOP_K_ACC"`

| Token | What it shows |
|---|---|
| `CONFUSION` | Row-normalised confusion matrix heatmap (`P(pred \| true)`); cell annotations suppressed above K=15 to stay readable. |
| `CONFUSED_PAIRS` | Top-N most-confused (true -> pred) class pairs as horizontal bars. |
| `PR_F1` | Per-class precision / recall / F1 grouped bar. |
| `ROC` | Per-class one-vs-rest ROC curves overlaid (with chance diagonal). |
| `PR_CURVES` | Per-class one-vs-rest precision-recall curves overlaid (with per-class prevalence baselines). |
| `CALIB_GRID` | Per-class reliability curves overlaid. |
| `PROB_DIST` | Per-true-class violin of `P(y = true_class \| x)`. |
| `TOP_K_ACC` | Top-k accuracy curve for k = 1..K. |

`ROC` / `PR_CURVES` stratified-subsample to ~200k rows before curve fitting at
large n; grid cells scale with K and use `tab20` before recycling colors.

### Multilabel classification

`multilabel_panels = "PR_F1 CALIB_GRID COOCCURRENCE CARDINALITY JACCARD_DIST"`

| Token | What it shows |
|---|---|
| `PR_F1` | Per-label precision / recall / F1 grouped bar. |
| `ROC` | Per-label ROC curves overlaid. |
| `CALIB_GRID` | Per-label reliability curves overlaid. |
| `COOCCURRENCE` | True x predicted label co-occurrence heatmap. |
| `CARDINALITY` | Distribution of #labels per row (pred vs true grouped bar). |
| `JACCARD_DIST` | Per-row Jaccard-score histogram. |
| `HAMMING_DIST` | Per-row Hamming-distance histogram. |

### Learning-to-rank (LTR)

`ltr_panels = "NDCG_K NDCG_DIST NDCG_BY_QSIZE LIFT MRR_DIST SCORE_BY_REL"`

| Token | What it shows |
|---|---|
| `NDCG_K` | NDCG@k curve for k = 1..max_per_query. |
| `NDCG_DIST` | Per-query NDCG@10 distribution (violin). |
| `NDCG_BY_QSIZE` | Mean NDCG@10 binned by query size (exposes small-group inflation). |
| `LIFT` | Cumulative relevance vs rank position (lift / gain curve over queries). |
| `MRR_DIST` | Per-query reciprocal-rank histogram. |
| `SCORE_BY_REL` | Predicted-score distribution per relevance grade. |
| `TOP1_BY_QSIZE` | Top-1 accuracy as a function of query size. |

All per-query passes use njit batched kernels from `metrics.ranking` (the old
python per-query loops were quadratic at 2M rows).

### Quantile regression

```
quantile_panels = "RELIABILITY COVERAGE PINBALL_BY_ALPHA INTERVAL_BAND
                   WIDTH_DIST PIT_HIST QUANTILE_RELIABILITY PINBALL_DECOMP
                   QUANTILE_CROSSING"
```

| Token | What it shows |
|---|---|
| `RELIABILITY` | Empirical vs nominal coverage per alpha (diagonal = calibrated). |
| `COVERAGE` | Empirical vs nominal interval coverage per symmetric pair, with a 95% Wilson CI band and mean width on a secondary axis. |
| `PINBALL_BY_ALPHA` | Mean pinball loss per alpha (which tail the model is worst at). |
| `INTERVAL_BAND` | Per-row median line + filled lo..hi band, `y_true` as markers (caps at 1000 rows for readability). |
| `WIDTH_DIST` | Histogram of interval widths (sharpness diagnostic). |
| `PIT_HIST` | PIT histogram (uniform = calibrated); needs K >= 3 alphas, else an annotation placeholder. |
| `QUANTILE_RELIABILITY` | Per-tau isotonic-recalibrated observed coverage vs nominal tau (CORP-style). |
| `PINBALL_DECOMP` | CORP additive pinball decomposition (miscalibration - discrimination + uncertainty) per tau. |
| `QUANTILE_CROSSING` | Per adjacent tau-pair, the fraction of rows with `q_lo > q_hi` -- a monotonicity violation; any non-zero bar is a correctness bug in independent quantile heads. |

### Regression

`regression_panels = "SCATTER RESID_HIST RESID_VS_PRED ERR_BY_DECILE"`

| Token | What it shows |
|---|---|
| `SCATTER` | Predictions vs true with y=x, a robust trend line, and the worst-K residuals highlighted red. Above `hexbin_threshold` (50k) points the cloud renders as a log-density 2-D histogram; below it, a raw scatter with an extremes-preserving subsample. |
| `RESID_HIST` | Residual histogram + fitted-Normal overlay (noise hypothesis + suggested loss in the title). |
| `RESID_VS_PRED` | Residuals vs predicted with a running-median + IQR band (a funnel = heteroscedasticity; a sloped band = prediction-dependent bias). |
| `ERR_BY_DECILE` | Per-target-decile mean `|residual|` + mean signed residual (exposes the GBM extreme-compression pathology). |

---

## 4. Cross-cutting diagnostics (not panel-template tokens)

These render independently of the per-task templates, default-ON when charts
are being saved.

- **Target / prediction distribution overlay** (`target_dist_overlay`,
  `error_analysis.py`): per-split overlaid density histograms of `y` AND of
  predictions (incl. OOF-vs-test), with train p01/p99 reference lines; rendered
  per target near suite start. Aggregate-first, safe at 1M+ rows.
- **Weak-segment error heatmap** (`weak_segment_heatmap`): a shallow decision
  tree on per-row error picks the most error-discriminating features, then a
  1-2-feature grid is colored by mean error (darker = worse). Own FreaAI-style
  reimplementation; the tree fits on a 50k subsample (cell stats use all rows).
- **Error-bias-per-feature** (`error_bias_per_feature`): rows tagged
  OVER / UNDER / MAJORITY by signed-error quantile (5% tails); per feature the
  three groups' value distributions overlay plus a group-mean table. Own
  Evidently-style reimplementation.
- **Worst-K errors** (`worst_k_table`): top-K rows by `|resid|` (regression) or
  loss (classification) with id/timestamp/y/yhat/resid + top-FI feature values;
  the same K points are highlighted red on the scatter.
- **Segments bar** (`segments_bar`): per-subgroup metric bars with a
  global-reference hline (the previously table-only error segmentation, now
  plotted).
- **PSI drift heatmap** (`psi_heatmap`): Population Stability Index per feature
  per equal-count time bucket vs a baseline slice; `RdYlGn_r` scale with the
  industry-standard 0.10 (moderate) / 0.25 (significant) triage contours.
  10-bin equal-frequency binning; top-40 features by peak PSI kept.
- **Adversarial validation** (`adversarial_validation`): the Kaggle "will my CV
  transfer?" panel -- a LightGBM classifier separating train-vs-test (and
  train-vs-val) rows; out-of-fold ROC + AUC + a top-20 drifting-feature
  importance bar. AUC ~0.5 = same distribution; AUC well above ~0.6 = covariate
  shift, CV may not transfer. Each side subsamples to 200k rows.
- **Residual / metric over time** (`residual_vs_time`, `metric_over_time`):
  regression residual mean +- std per time bin (bias + variance drift), and any
  metric per time bucket with split/regime shading. Gated on timestamps being
  available.
- **Training curves** (`compose_training_curve_figure`,
  `ReportingConfig.training_curves`): per-model train vs validation metric over
  boosting iterations (from lgb `evals_result_` / xgb `evals_result()` /
  catboost `get_evals_result()`), with the early-stopping iteration marked and
  post-ES iterations shaded. No-op for models without boosting history.

---

## 5. Large-n rendering

The diagnostics stay cheap on multi-million-row frames; you do not need to
pre-subsample:

- **Regression scatter**: above `hexbin_threshold` (50k) the pred-vs-actual
  cloud is a **log-density 2-D histogram** (a hexbin/hist2d analogue) instead of
  a raw scatter -- a 2M-point scatter is both slow and an unreadable blob.
  Below the threshold, a **raw scatter with an extremes-preserving subsample**
  (`regression_scatter_sample_size`, default 5000) that always retains the
  worst-error and range-endpoint points (so the MaxError row quoted in the title
  is actually drawn).
- **Plotly scatter**: switches to `Scattergl` (WebGL) above 10k points and
  downsamples above 50k, passing ndarrays through natively (no `.tolist()`).
- **Histograms**: pre-binned with numpy at >=50k points so the renderer ships
  bar heights, not raw length-n arrays (the old path embedded 2M raw values =>
  ~37MB HTML; pre-binning drops it to ~14KB).
- **Curves** (ROC / PR / lift / metric-over-time): vertex-decimated to ~2000
  points; multiclass/LTR per-query work uses stratified subsampling + njit
  batched kernels.
- **Violins / KDE panels**: subsampled to 5000 (Scott's-rule plateau; visible
  shape diff below pixel precision).
- **Aggregate-first diagnostics** (PSI, residual-over-time, target overlay):
  one O(n) `np.histogram` / `bincount` pass per feature -- no per-row python.

---

## 6. Key ReportingConfig knobs

| Knob | Default | Effect |
|---|---|---|
| `plot_outputs` | `"plotly[html] + matplotlib[png]"` | backend x format DSL (section 2). |
| `binary_panels` / `multiclass_panels` / `multilabel_panels` / `ltr_panels` / `quantile_panels` / `regression_panels` | see section 3 | per-task panel template; remove tokens to skip panels. |
| `regression_scatter_sample_size` | `5000` | row cap for the regression scatter (extremes preserved). |
| `calibration_binning` | `"auto"` | reliability/ECE bins: `auto` picks quantile (equal-population) bins under a rare-event base rate where uniform bins collapse, uniform otherwise; or force `"uniform"` / `"quantile"`. |
| `reliability_show_ci` | `True` | Wilson 95% binomial CI band on per-bin empirical frequencies (tells noise from real miscalibration). |
| `training_curves` | `True` | per-model train/val metric-vs-iteration curves with the ES point marked. |
| `keep_figure_handles` | `False` | retain pure-data `FigureSpec` objects in `metrics["figure_specs"]` for programmatic re-render (section 7). |
| `prob_histogram_yscale` | `"auto"` | reliability prob-histogram y-scale (`auto` / `log` / `linear`). |
| `show_prob_histogram` / `show_inline_population_labels` | `True` | reliability sub-histogram + inline per-bin population labels. |
| `title_metrics_template` | `"ICE BR_DECOMP ECE CMAEW LL ROC_AUC PR_AUC KS MCC BSS"` | ordered token grammar composing the calibration-chart title. |
| `regression_title_metrics_tokens` | `("MAE","RMSE","MaxError","R2","RMSLE","Spearman","MBE")` | regression-chart title tokens. |
| `matplotlib_style` / `matplotlib_rcparams` / `plotly_template` | `None` | process-wide style overrides applied at suite entry. |
| `plot_dpi` | `None` | per-figure DPI for matplotlib PNG / inline render. |
| `plot_inline_display` | `None` | force/skip jupyter inline display (auto-detected when `None`). |

---

## 7. Getting figures programmatically

Two retrieval paths, both via the suite's returned `metrics` dict:

- **On-disk paths** -- `metrics["charts"]` is the rendering accounting:
  `{"saved": [...tags...], "failed": [...tags...], "paths": [...base paths...]}`.
  Saved/failed are stamped per chart even when a render is swallowed, so you can
  assert on what was produced. The suite-finalize step logs the saved count +
  destination (or a "0 charts saved; set output_config.data_dir" hint).
- **Live specs** -- set `keep_figure_handles=True` and the pure-data
  `FigureSpec` objects land in `metrics["figure_specs"]` keyed by chart tag
  (e.g. `"training_curve"`). These carry **no** live matplotlib/plotly handle,
  so they stay pickle-safe; re-render with:

  ```python
  from mlframe.reporting.output import parse_plot_output_dsl
  from mlframe.reporting.renderers import render_and_save

  spec = metrics["figure_specs"]["training_curve"]
  render_and_save(spec, parse_plot_output_dsl("matplotlib[svg]"), "out/curve")
  ```

---

## 8. Overriding panel templates

Set the relevant `*_panels` string on `ReportingConfig` -- any subset of that
task's allowed tokens, space-separated, no duplicates. Validation against the
chart modules' `ALLOWED_*_PANEL_TOKENS` frozensets happens at config
construction, so an unknown / duplicate token raises immediately with the
allowed set listed. An empty string for a task skips that task's panels
entirely.

```python
from mlframe.training.configs import ReportingConfig

reporting = ReportingConfig(
    binary_panels="ROC PR THRESHOLD",         # drop SCORE_DIST/KS/GAIN/PIT
    regression_panels="SCATTER RESID_VS_PRED", # scatter + heteroscedasticity only
    plot_outputs="matplotlib[png,pdf]",        # static only, two formats
)
```

The composers (`compose_binary_figure`, `compose_regression_figure`, ...) are
also directly callable with a `panels_template=` kwarg if you build figures
outside the suite.

---

## 9. Optional extras

- **CORP pinball decomposition** (`PINBALL_DECOMP`): the additive
  miscalibration / discrimination / uncertainty split uses the
  `model-diagnostics` package when importable; absent it, the panel falls back
  to plain mean pinball loss per tau as a bar. Install `model-diagnostics` to
  get the full decomposition.
- **datashader** (FUTURE): the regression density path currently uses a numpy
  log-density 2-D histogram, which is dependency-free and adequate at the row
  counts the suite sees. A `datashader` backend for billion-point clouds is a
  documented future optional extra (not added this round to avoid a Python 3.14
  wheel-availability risk); the hexbin/hist2d path covers present needs.
