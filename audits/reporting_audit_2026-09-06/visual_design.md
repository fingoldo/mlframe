# Visual quality & encoding-correctness audit — `src/mlframe/reporting/`

Date: 2026-09-06. Read-only research audit. Bug class: **visual quality and correctness of the encoding**
(what makes a chart look amateurish or mislead the reader), including matplotlib-vs-plotly renderer drift.

## Chart types this package actually generates

The package has two rendering paths.

**A. Spec-driven (both backends).** A builder in `charts/` returns a `FigureSpec` of panel specs
(`spec.py`), which `renderers/matplotlib.py` and `renderers/plotly.py` each render independently.
Nine panel kinds: `ScatterPanelSpec`, `HistogramPanelSpec`, `HeatmapPanelSpec`,
`ConfusionMarginsPanelSpec`, `BarPanelSpec`, `LinePanelSpec`, `ViolinPanelSpec`, `NetworkPanelSpec`,
`AnnotationPanelSpec`. Concrete charts built on them: reliability/calibration diagram + population
histogram, calibration-by-feature, calibration drift, 2-D calibration heatmap, fairness calibration
(per-subgroup reliability + ECE bar), binary ROC/PR/threshold-sweep/cost curves, decile table,
multiclass per-class ROC overlay + confusion matrix with margins, multilabel per-label P/R/F1 +
co-occurrence heatmap + cardinality histogram, regression pred-vs-true density heatmap with robust
trend, residual histogram/ACF/worm, quantile reliability + fan chart, decision curve, risk-coverage,
LTR NDCG@k, drift (PSI heatmap, adversarial train-vs-test ROC + importance bar, CUSUM), temporal
ACF/PACF bars, error-analysis worst-segment bars + slice finder, category discriminability (WoE),
model comparison (leaderboard bar + Spearman correlation heatmap), model card, PDP/ICE, 2-D PDP,
spectral embedding, feature friend-graph network, engineered separability, prediction stability,
class-structure heatmap, interaction strength, fuzzy membership.

**B. Raw-matplotlib (matplotlib only, no plotly twin).** `charts/confusion_matrix_plot.py`,
`charts/_binary_decile_table.py`, `charts/pdp_2d.py`, `charts/shap_panels.py`,
`charts/shap_interactions.py`, `charts/shap_per_instance.py` build `Figure`/`plt.subplots`
directly and `savefig` themselves.

Findings below are judged against those chart types.

---

## P1

### VIS-01 — P1 — `charts/model_comparison.py:325`
**A diverging colormap on an un-centred scale turns 0.90 correlation into "deep blue".**
The between-model Spearman correlation heatmap sets `colormap="RdBu_r"` with `cell_text=corr`. Neither
renderer ever pins a midpoint: `renderers/matplotlib.py:410` calls `ax.imshow(p.matrix, cmap=cm, ...)`
with no `vmin/vmax/norm`, and `renderers/_plotly_heatmap.py:196-203` builds `go.Heatmap` with no
`zmid`/`zmin`/`zmax`. `grep -rn "zmid|vcenter|TwoSlopeNorm"` over `renderers/` and `charts/` returns
nothing.
*What the reader sees:* a set of models that all agree (rho 0.90-0.99) rendered as a full blue-to-red
sweep, with the least-correlated pair painted the same saturated blue that RdBu_r reserves for
rho = -1. The reader reads anti-correlation off a matrix that contains none. Conversely a genuinely
negative rho in a mostly-negative matrix renders red.
*Fix:* add a `color_center: Optional[float]` (or `symmetric_scale: bool`) to `HeatmapPanelSpec`; when set,
matplotlib uses `matplotlib.colors.TwoSlopeNorm(vcenter=...)` (or symmetric `vmin=-m, vmax=+m`) and plotly
sets `zmid=`. Set it to `0.0` at this call site. Spearman rho has a fixed domain, so pinning
`vmin=-1, vmax=1` here is even better.

### VIS-02 — P1 — `renderers/plotly.py:620-623` vs `renderers/matplotlib.py:637-639`
**Horizontal bar charts have their two axis titles swapped on the plotly backend.**
Every horizontal builder uses one convention, stated explicitly at `charts/fairness_calibration.py:197`
("For a horizontal bar the VALUE axis is x and the CATEGORY axis is y, matching every other builder"):
`xlabel` = the measured quantity, `ylabel` = the category name. Verified at
`fairness_calibration.py:198-200`, `model_comparison.py:219-222`, `model_card.py:212-214`,
`slice_finder.py:487-489`, `category_discriminability.py:303-305`, `_multiclass_confusion.py:190-192`.
matplotlib honours it (`ax.set_xlabel(p.xlabel); ax.set_ylabel(p.ylabel)`). Plotly's horizontal branch
does `fig.update_xaxes(title_text=p.ylabel)` and `fig.update_yaxes(title_text=p.xlabel)`.
*What the reader sees:* the HTML fairness chart labels its value axis "subgroup" and its category axis
"ECE (lower = better calibrated)". Same for the leaderboard ("model" on the metric axis), the slice
finder, the WoE chart and the model card. Mislabelled axes on six chart types.
*Fix:* drop the swap — use `title_text=p.xlabel` on x and `title_text=p.ylabel` on y in both branches.

### VIS-03 — P1 — `charts/shap_panels.py:400`
**A mangled string escape is printed onto the chart.**
`fig.text(0.5, 0.5, "SHAP panels not produced:' + BS + 'n" + reason, ...)` — the intended newline escape
was corrupted into literal text. The two sibling notices use different, working mechanisms
(`shap_interactions.py:71` uses `chr(10)`, `shap_per_instance.py:196` uses `"\n"`).
*What the reader sees:* a figure in the report reading
`SHAP panels not produced:' + BS + 'n<reason>` — visibly broken source text on a delivered chart.
*Fix:* `"SHAP panels not produced:\n" + reason`. Also unify the three notice builders on one helper so
this cannot drift again.

---

## P2

### VIS-04 — P2 — `renderers/plotly.py:617-621` vs `renderers/matplotlib.py:596-612`
**Horizontal bar tick labels are thinned on matplotlib and not on plotly.**
matplotlib's horizontal branch truncates every label AND thins to ~20 evenly-spaced ticks past
`_BAR_TICK_THIN_THRESHOLD = 25`. Plotly's horizontal branch only truncates, and only
`if any(len(str(c)) > _BAR_XTICK_MAXLEN for c in cats)` — it never thins. The vertical branch on plotly
does thin (`renderers/plotly.py:630-637`), so this is the horizontal orientation alone.
*What the reader sees:* the 200-row horizontal feature-importance / slice-finder / WoE chart has a clean
20-label axis in the PNG and an unreadable overlapping black band in the HTML, from one spec.
*Fix:* apply the same `_BAR_XTICK_THIN_THRESHOLD` / `_BAR_XTICK_KEEP` subsampling to the horizontal
branch via `tickmode="array"` on the y-axis.

### VIS-05 — P2 — `charts/engineered_separability.py:213`
**A diverging continuous colormap encoding a nominal class label.**
`ScatterPanelSpec(point_color=yv, colormap="coolwarm", colorbar_label="class")` where `yv` is the class
label vector.
*What the reader sees:* a continuous blue-white-red colorbar with tick marks at 0.5 and 1.5 — values no
row can take. For a 3-class problem, class 1 lands on the pale white midpoint and is nearly invisible
against the panel; classes 0 and 2 read as "extremes of a quantity" rather than "two categories".
Colour is also the only channel, so red/blue is fine for CVD here but the sequential/nominal mismatch is not.
*Fix:* map classes through the discrete `colors.LINE_PALETTE` (`line_color(k)`) and emit a legend
instead of a colorbar, as every other categorical overlay in the package does.

### VIS-06 — P2 — `colors.py:155-159`, `charts/model_card.py:53-55`, `charts/category_discriminability.py:40-41`, `charts/fairness_calibration.py:61-64`, `charts/calibration_by_feature.py:60-63`, `charts/calibration_heatmap_2d.py:97-100`
**Red/green traffic lights with no redundant channel.**
`#2ca02c` (tab:green) vs `#d62728` (tab:red) is the exact pair `colors.py:98-102` documents as
indistinguishable under deuteranopia/protanopia. In `category_discriminability.py:297` the bar
*direction* also encodes the sign, so colour is redundant there and it is fine. In the model card
verdict, the fairness `_gap_traffic_light`, the calibration-by-feature `_het_traffic_light`, the 2-D
calibration verdict and the friend-graph node classes, **colour is the only channel**.
*What the reader sees:* a red-green colourblind reader (about 8% of men) cannot tell a PASS verdict from
a FAIL one on the model card. The friend-graph "unique feature" and "redundant sink" nodes read identical.
*Fix:* pair every traffic light with a text token already computed next to it (`_gap_traffic_light`
returns the string `"green"`/`"amber"`/`"red"` — print "PASS"/"WATCH"/"FAIL" beside the swatch), or use a
shape/hatch. `BarPanelSpec.hatches` and the plotly `_HATCH_TO_PATTERN` map at `renderers/plotly.py:558`
already exist for exactly this.

### VIS-07 — P2 — `renderers/_matplotlib_scatter.py:167-192` vs `renderers/_plotly_scatter.py:88-100`
**Inline point labels: matplotlib flips alignment at the panel edge and draws a contrast halo; plotly does neither.**
matplotlib computes `_ha`/`_va` from `_EDGE_LABEL_FLIP_FRACTION` and adds
`patheffects.withStroke(..., foreground=("black" if _colour == "white" else "white"))`. Plotly emits a
plain annotation with a fixed `yshift=8` and `font=dict(size=8, color=_lab_colors[i])`.
*What the reader sees:* on the reliability diagram, a bin in the busy bottom-left corner has its label
clipped by the axis on the HTML backend. Worse, `charts/calibration.py:424` deliberately picks
`auto_text_color(...)` = `"white"` for labels sitting on a dark bubble; with no halo and a fixed upward
shift, that white text lands on the white panel background and disappears entirely. This is precisely
the failure `colors.auto_text_color` exists to prevent, defeated by the missing halo.
*Fix:* port both behaviours to plotly — compute the flip against the axis range and set
`xanchor`/`yanchor` accordingly, and emit the halo via `bgcolor` + `bordercolor` on the annotation (or by
laying a contrasting text annotation underneath).

### VIS-08 — P2 — `renderers/_plotly_scatter.py:263-270` vs `renderers/_matplotlib_scatter.py:134-165`
**The perfect-fit scatter gets a different axis window on each backend.**
matplotlib sets `set_xlim/set_ylim(lo, hi)` only in the `equal_aspect and xlim is None and ylim is None`
sub-branch; the non-`equal_aspect` case (calibration, per the comment at line 143-146) deliberately falls
through to autoscale. Plotly unconditionally does `y_range = list(p.ylim) if p.ylim is not None else
[lo, hi]` and applies it whenever `perfect_fit_line` is set.
*What the reader sees:* the reliability scatter is windowed exactly to the data hull on plotly (points
sitting flush on the frame, no breathing room) and autoscaled with matplotlib's 5% margins in the PNG —
the same points, at visibly different scales, so a gap that looks large in one looks small in the other.
*Fix:* gate the plotly range assignment on the same condition matplotlib uses.

### VIS-09 — P2 — `renderers/_matplotlib_scatter.py:88` and `renderers/_plotly_scatter.py:171`
**When EVERY point is low-evidence, both backends draw them all as confident observations.**
The hollow/muted treatment is gated on `weak.any() and (~weak).any()`. If all bins are low-evidence the
`else` branch runs and every marker is drawn filled, with the normal solid error bars.
*What the reader sees:* a reliability diagram built entirely on 3-row bins renders pixel-identical to one
built on 300k-row bins. The chart's whole confidence signal silently vanishes at the exact moment it
matters most.
*Fix:* drop the `(~weak).any()` half of the guard and let the strong trace be empty; both backends
already skip empty traces cleanly. Also keep the "too few rows to read" legend entry, which is currently
lost in this case.

### VIS-10 — P2 — `charts/temporal.py:216-224` and `charts/temporal.py:242-250`
**The ACF/PACF significance band is drawn as one line but labelled as two.**
`hline=(band, "red", f"+-1.96/sqrt(n) = {band:.3f}")` draws a single horizontal line at `+band`, with a
legend entry claiming it is the `±` band. `BarPanelSpec.hline` is a single `(value, color, label)`
triple, so `-band` is never drawn.
*What the reader sees:* on the ACF chart a lag with autocorrelation -0.4 against a band of ±0.05 has no
line under it; the reader has to mentally mirror the red line to judge every negative lag, and the title
already counts those lags as significant (`{sig} of {n} lags beyond +-{band}`), so the count and the
picture disagree.
*Fix:* widen `BarPanelSpec.hline` to a tuple of reference lines (or add `hlines`), and pass both
`+band` and `-band`, labelling only the first.

### VIS-11 — P2 — `renderers/matplotlib.py:534-542` vs `renderers/_plotly_heatmap.py:117-133`
**The confusion-matrix marginal bars are three different colours across the two backends.**
matplotlib paints the top (predicted-volume) strip `"#4c72b0"` and the right (true-support) strip
`"#55a868"`. Plotly paints **both** strips `TREND_LINE` (darkorange) — a constant whose documented job
(`colors.py:91`) is the robust-fit overlay line, not a bar fill.
*What the reader sees:* the same confusion matrix has blue-and-green margins in the PNG and two identical
orange margins in the HTML. On the plotly version the two strips are no longer visually distinguishable
from each other, and orange collides with the trend line's meaning elsewhere in the same report.
*Fix:* add `CONFUSION_COL_MARGIN` / `CONFUSION_ROW_MARGIN` to `colors.py` (the module that exists to stop
exactly this) and use them from both renderers. The two matplotlib hex literals are the only remaining
hardcoded fills in that file.

### VIS-12 — P2 — `charts/fairness_calibration.py:34-37` and `charts/calibration_by_feature.py:40-43`
**Two verbatim copies of `colors.LINE_PALETTE`, cycled with no wrap disambiguation.**
`_GROUP_COLORS` and `_BIN_COLORS` are byte-identical to `colors.LINE_PALETTE:103-114` but private to
their modules, and are consumed as `_GROUP_COLORS[gi % len(_GROUP_COLORS)]`
(`fairness_calibration.py:138`). `colors.line_style(idx)` — which advances the dash pattern once per
palette wrap precisely so colours may repeat safely — is used by `multiclass.py:205`,
`multilabel.py:168,239` and `quantile.py:492`, but **not** by these two.
*What the reader sees:* an 11-subgroup fairness chart draws subgroup 0 and subgroup 10 as two solid lines
in the identical blue, with a legend that names them differently and nothing on the chart to tell them
apart. Repainting `LINE_PALETTE` also silently leaves these two charts on the old palette.
*Fix:* import `line_color`/`line_style` from `colors` and delete both local tuples; pass
`line_styles=tuple(line_style(i) for i in range(n_groups))` alongside the colours.

### VIS-13 — P2 — `charts/model_card.py:195-217`
**Unlike quantities sharing one unitless axis, unsorted, with colour carrying no information.**
`_headline_bar` maps AUC (chance = 0.5), accuracy (chance = base rate), and `1 - Brier` / `1 - ECE` onto a
single axis labelled "quality (higher is better)", draws a `hline` at 0.5 labelled "midpoint", and paints
every bar the same `_verdict_color_hex(verdict_color)`.
*What the reader sees:* four bars of comparable length that mean four unrelated things — a 0.5 AUC bar
(pure chance) sits level with a `1 - Brier = 0.5` bar (catastrophic) and with the "midpoint" reference,
implying the three are commensurate. Bars appear in the order the caller happened to build
`metric_fmt`, not by value. And every bar being one colour makes the reader hunt for a per-bar encoding
that does not exist.
*Fix:* label each bar with its raw value and unit in the category string (e.g. `"AUC 0.83"`), sort by
value, and either drop the 0.5 reference or replace it with a per-metric baseline; use a single neutral
fill and put the verdict colour on the card's verdict box only.

---

## P3

### VIS-14 — P3 — `renderers/plotly.py:508-521` vs `renderers/matplotlib.py:378-388`
**A histogram of all-NaN values says "no finite values" on matplotlib and renders a blank framed panel on plotly.**
matplotlib's raw-values fallback filters `np.isfinite` and, when nothing survives, draws the text
"no finite values". Plotly hands the raw array to `go.Histogram`, which silently drops non-finite values
and draws an empty panel with no note.
*Fix:* filter and emit the same note in the plotly branch.

**RESOLVED.** The plotly raw-values branch now filters `np.isfinite` and draws the same "no finite values"
note. Rendering it exposed a second, worse defect the audit had not named: a subplot cell holding no trace
at all is never laid out by plotly, so the annotation anchored to its axes was drawn on the NEIGHBOURING
panel -- "no finite values" written across a perfectly good histogram. Fixed by adding an empty scatter to
the cell, which forces the axes to exist and gives the reader the same framed-but-empty panel matplotlib
draws. `tests/reporting/test_histogram_all_nonfinite_notice.py` (5 tests, verified failing pre-fix with the
real signature: no notice, and the finite column binned 5 values instead of 3).

### VIS-15 — P3 — `renderers/plotly.py:550` vs `renderers/matplotlib.py:395-408`
**A log-scaled histogram gets readable tick values on matplotlib and one labelled decade on plotly.**
matplotlib installs `LogLocator(subs=(1.0, 2.0, 5.0), numticks=8)` + `LogFormatterSciNotation` for
`yscale == "log"` and a `MaxNLocator(nbins=6, ...)` for linear. Plotly only sets `type="log"` and takes
plotly's defaults for both. (No builder currently sets `yscale="log"`, so this is latent — but
`HistogramPanelSpec.yscale` is public API.) Related: nothing guards `yscale="log"` against a histogram
whose counts include 0; matplotlib drops those bars silently.
*Fix:* mirror the tick config via `dtick`/`tickformat`, and warn (or clamp) when a log axis is requested
over data containing non-positive values.

### VIS-16 — P3 — `renderers/matplotlib.py:83` (`_CAPTION_FONTSIZE = 7`) vs `renderers/plotly.py:69` (`_CAPTION_FONTSIZE = 10`, rendered at `font=dict(size=9)` on line 396)
**The "how to read" caption is three different sizes.**
matplotlib wraps and renders at 7pt in colour `"0.35"`. Plotly *wraps* against 10pt but *renders* at 9pt
in `#595959` — so even within plotly the wrap budget does not match the drawn glyphs, and the caption can
overrun its reserved band. Same class: `_CAPTION_WRAP_CHARS = 110` on matplotlib
(`renderers/matplotlib.py:81`) vs plotly reusing `_SUPTITLE_WRAP_CHARS = 90` for the caption
(`renderers/plotly.py:394`); and panel titles at `_TITLE_FONTSIZE = 10`
(`renderers/matplotlib.py:35`) vs `_PANEL_TITLE_FONTSIZE = 11` (`renderers/plotly.py:71`), whose own
comment claims "11 matches matplotlib's panel titles" — it does not.
*Fix:* move the four typography constants (suptitle/panel-title/caption size, caption wrap budget) into
`_shared_helpers.py` and import them into both renderers, the way `truncate_bar_label` and
`_HEATMAP_MAX_TICKS` already are.

**RESOLVED.** `PANEL_TITLE_FONTSIZE = 10`, `SUPTITLE_WRAP_CHARS = 90`, `CAPTION_FONTSIZE = 7` and
`CAPTION_WRAP_CHARS = 110` now live in `_shared_helpers.py`; both renderers bind their private names to
those objects. Plotly additionally drew its caption at a hardcoded 9pt (wrapped against 10) and folded it
against the narrower suptitle budget -- both now read the shared constants, so the measurement and the
drawn glyphs finally agree. `tests/reporting/test_typography_is_shared.py` (4 tests; three fail pre-fix
with the real signature -- panel title 10 vs 11, the identity check, and "wrapped against 10pt but drawn
at 9pt"). Rendered both backends side by side: single-line caption, same relative weight, no overlap.

### VIS-17 — P3 — `renderers/matplotlib.py:449-452` vs `renderers/_plotly_heatmap.py:252-269`
**PSI threshold contours are labelled on matplotlib and unlabelled on plotly.**
`HeatmapPanelSpec.threshold_contours` entries carry an optional 4th element, the label. matplotlib reads
it and calls `ax.clabel(cs, fmt={level: label})`. The plotly loop unpacks only `level`, `color`, `dash`
and sets `contours=dict(..., showlabels=False)`; the label is never read.
*What the reader sees:* the drift heatmap's two triage lines are annotated "moderate 0.1" / "significant
0.25" in the PNG and are two anonymous coloured squiggles in the HTML.
*Fix:* read `_entry[3]` and pass `contours.showlabels=True` with `contours.labelfont`, or add a legend
entry per contour level.

**RESOLVED, with a stated limit.** plotly can only write the LEVEL on a contour, never arbitrary text, so
the two mechanisms are used together: `showlabels=True` puts the number on the line inline, and the triage
wording ("moderate 0.1") rides in the trace name, which reaches the legend and the hover. Both beat the
anonymous squiggle. Verified by rendering the drift heatmap on both backends.

### VIS-18 — P3 — `renderers/_plotly_scatter.py:240`
**A shared overlay colour hardcoded on one backend only.**
`fillcolor="rgba(128,0,128,0.18)"` is the literal RGB of `colors.OVERLAY_LINE = "purple"`, while
matplotlib uses the constant (`renderers/_matplotlib_scatter.py:128`,
`color=OVERLAY_LINE, alpha=0.18`). The module already imports `_rgba` and uses it correctly at
`renderers/plotly.py:713` for the line-panel band.
*What the reader sees:* nothing today; on the next repaint of `OVERLAY_LINE` the curve-95%-band is
recoloured in PNGs and stays purple in HTML. This is the exact drift `colors.py:89-90` documents.
*Fix:* `fillcolor=_rgba(OVERLAY_LINE, 0.18)`.

**RESOLVED.** `_rgba(OVERLAY_LINE, 0.18)` produces the byte-identical `rgba(128,0,128,0.18)` today, so
nothing renders differently -- the point is that it now moves when the constant does.

### VIS-19 — P3 — `renderers/plotly.py:646-648, 730` vs `renderers/matplotlib.py:641-642, 205-207`
**Gridlines differ in axis coverage and weight between backends.**
matplotlib draws `ax.grid(True, alpha=0.3, axis="x" if horizontal else "y")` for bars — value axis only,
at 30% opacity. Plotly sets `showgrid=p.grid` on the value axis but never sets `showgrid=False` on the
category axis, and plotly's cartesian default is `showgrid=True`, so the category axis keeps its grid.
Scatter/line panels have the same 0.3-alpha vs full-strength-default mismatch.
*What the reader sees:* the plotly bar chart has vertical rules running between every category that the
PNG does not have, and all plotly gridlines read heavier. Two charts of the same data with different
visual density, printed in the same report.
*Fix:* set `showgrid=False` explicitly on the category axis and pin `gridcolor`/`gridwidth` in the plotly
layout to match matplotlib's `alpha=0.3`.

**RESOLVED.** `showgrid=False` on the category axis of all four plotly bar branches, and one
`update_xaxes/update_yaxes(gridcolor=_GRID_COLOR, gridwidth=1)` pass in `render()` -- matplotlib's default
`#b0b0b0` flattened against white at the `alpha=0.3` every panel draws it with.

### VIS-20 — P3 — `renderers/_matplotlib_scatter.py:116-117` vs `renderers/_plotly_scatter.py:212-214`
**The worst-K highlight ring is data-scaled on matplotlib and fixed-size on plotly.**
matplotlib: `s=base_s * 4.0` where `base_s` is the panel's own point size (or the median of the
per-point size array). Plotly: `marker=dict(symbol="circle-open", size=12, ...)`, a constant.
*What the reader sees:* on a panel with large bubbles the plotly highlight ring is smaller than the point
it is meant to highlight and disappears inside it.
*Fix:* convert `base_s * 4.0` through the same `sqrt(area) * 1.33` mapping the file already applies at
line 56-58.

**RESOLVED.** The ring is sized from the panel's own point area through the file's existing
area-to-diameter mapping, with an 8 px floor so a fine scatter still gets a visible ring. Confirmed by
rendering a 200 pt^2 bubble panel: the ring encloses the point instead of sitting inside it.

### VIS-21 — P3 — `charts/shap_panels.py:399,425,500`, `charts/shap_interactions.py:70,173,203`, `charts/shap_per_instance.py:195,221`, `charts/confusion_matrix_plot.py:157,177`, `charts/_binary_decile_table.py:123,151`
**The raw-matplotlib charts ignore the report's DPI and figure conventions.**
`FigureSpec.dpi` (`spec.py:459`) exists so `ReportingConfig.plot_dpi` can set one DPI for the run; only
`renderers/matplotlib.py:145-147` reads it. Path-B charts create their own figures with no `dpi`
(except `pdp_2d.py:126,135,149`, which threads one through), and `shap_panels.py:425` saves with
`bbox_inches="tight"` but no `pad_inches`, against the renderer's `pad_inches=0.15`
(`renderers/matplotlib.py:242`). They also use pyplot's global figure registry, which
`renderers/matplotlib.py:100-112` documents as deliberately avoided.
*What the reader sees:* in one report, SHAP and confusion-matrix images are rendered at a different pixel
density than every neighbouring chart (visibly different text weight when both are scaled to the same
width in HTML), and are cropped tighter at the edges.
*Fix:* thread the resolved DPI and pad into `_save_figure` for all Path-B charts, or migrate them onto
`FigureSpec`.

**RESOLVED, the first way.** All five Path-B charts the finding lists route through ONE `_save_figure`
(`shap_panels.py`, imported by `shap_interactions` and `shap_per_instance`), so the crop fix lands in a
single place: `pad_inches=0.15`, matching `renderers/matplotlib.py`. `confusion_matrix_plot` and
`_binary_decile_table` only CONSTRUCT figures -- they do not save -- so the DPI reaches them through the
same saver.

The DPI needed four hops, not the one this reads like: `_reporting.py` already had `plot_dpi`, but
`_render_post_fit_diagnostics`, `render_shap_diagnostic` and `shap_summary_and_dependence` all had to
accept and forward it before `_save_figure` could use it. Stopping at the crop would have left a parameter
nothing passes, which is not a fix.

`tests/reporting/test_path_b_save_conventions.py` (5 tests). One reads the pad back out of the RENDERER's
source, so the two cannot drift apart again silently, and one asserts every hop of the chain carries
`plot_dpi`. The pre-fix saver fails the DPI test.

### VIS-22 — P3 — `charts/error_analysis.py:246,257`, `charts/interaction_strength.py:80`, `charts/class_structure_heatmap.py:236`, `charts/spectral_embedding.py:164`, `charts/fuzzy_membership.py:85`, `charts/drift.py:248`, `charts/engineered_separability.py:246,257`
**Default figure sizes vary widely across charts that appear in the same report.**
Observed defaults: `(6.0, 3.0)`, `(6.0, 5.0)`, `(6.0, 5.5)`, `(7.0, 4.5)`, `(7.0, 5.0)`, `(7.0, 6.0)`,
`(8.0, 3.0)`, `(8.0, 6.0)`, `(10.0, h)`, against `FigureSpec.figsize`'s own default of `(12.0, 4.0)`
(`spec.py:455`). Aspect ratios range from 2.67:1 to 1:1.
*What the reader sees:* an HTML report where consecutive charts jump between wide-strip and near-square,
with font sizes appearing to change (the renderers use fixed point sizes, so text is relatively larger on
a small figure).
*Fix:* define a small set of named figure sizes in `spec.py` (`WIDE`, `STANDARD`, `SQUARE`,
`TALL_BAR(n)`) and have builders pick one, rather than each inventing a pair of floats.

### VIS-23 — P3 — `charts/temporal.py:222` (`colors=("steelblue",)`) and `charts/temporal.py:248` (`colors=("seagreen",)`)
**Two panels of the same conceptual chart in different colours for no encoding reason.**
The target ACF and target PACF bars are the same measurement family, drawn side by side, and differ only
in colour — which encodes nothing (the titles already say ACF and PACF).
*What the reader sees:* the eye reads a colour difference as a data difference, then finds none.
Neither literal comes from `colors.py`; `BAR_PRIMARY = "steelblue"` (`colors.py:88`) exists for this and
is not imported by any chart or renderer (both renderers instead hardcode the literal `"steelblue"`,
`renderers/matplotlib.py:41` and `renderers/plotly.py:600`).
*Fix:* use `colors.BAR_PRIMARY` for both panels, and replace the two renderer literals with the constant.

**RESOLVED.** Both panels take `BAR_PRIMARY`, and the two renderer literals now read the constant, so the
package has one definition of "single-series bar" again.

### VIS-24 — P3 — `charts/error_analysis.py:419-425`
**NaN subgroup metrics sort to the "worst" end and are rendered as empty slots.**
`order = np.argsort(metric)` then `order = order[::-1]` when `higher_is_worse`. numpy sorts NaN to the
end ascending, so reversing puts every NaN group first — into the `[:max_groups]` slice.
*What the reader sees:* the worst-first bar chart leads with blank slots (a NaN bar draws nothing but
keeps its tick label), and the title computed at line 430 reads
`worst segment <name> is nan x the global`.
*Fix:* drop non-finite metrics before sorting and note the count in the caption, per the package's own
"a silently dropped group reads as 'no problem here'" convention (see the violin handling at
`renderers/matplotlib.py:768-771`).

**RESOLVED, and the filter had to move further up than the audit said.** Dropping non-finite metrics just
before the sort fixes the bars, but `global_value` is a weighted mean over the SAME array, so the title
still read "global reference = nan; worst segment c is nanx the global". The filter now runs immediately
after the metric column is read, and the weights are filtered with it. The caption names the dropped
count.

### VIS-25 — P3 — `renderers/plotly.py:826-829` (docstring and code) vs `renderers/matplotlib.py:756-762`
**The violin inner box claims 5th/95th-percentile whiskers on plotly but does not deliver them.**
matplotlib overlays a real `ax.boxplot(..., whis=(5, 95))`. Plotly sets
`fig.update_traces(box=dict(visible=True), quartilemethod="linear", ...)`; `quartilemethod` selects how
quartiles are *computed* and has no effect on whisker extent, which `go.Violin`'s box draws to the data
range. The docstring at line 826-829 asserts parity ("5th/95th percentile whiskers, as matplotlib's
`whis=(5, 95)` draws").
*What the reader sees:* the same group's box shows a visibly longer whisker in the HTML than in the PNG,
so a reader comparing the two forms a different impression of the spread. The comment makes the
divergence invisible to the next maintainer.
*Fix:* compute the 5th/95th percentiles in the renderer and draw them as an explicit overlay
(`go.Box` with `lowerfence`/`upperfence`, or a shape), or correct the docstring and accept the
difference explicitly.

**RESOLVED, the first way.** `box_visible` is off and an explicit `go.Box` per group carries
`q1`/`median`/`q3` plus `lowerfence`/`upperfence` at the 5th/95th percentiles -- the quantities
matplotlib's `whis=(5, 95)` draws, not an approximation of them. The test asserts the upper fence is
strictly inside the data max, which is the exact defect the old code had.

### VIS-26 — P3 — `renderers/plotly.py:604-607` vs `renderers/matplotlib.py:626-635`
**A bar reference line becomes a legend entry on matplotlib and a floating corner annotation on plotly.**
matplotlib draws `axhline`/`axvline` with `label=hlabel` and calls `ax.legend(loc="best")`. Plotly uses
`annotation_text=hlabel, annotation_position="top right"`.
*What the reader sees:* on a horizontal bar chart the plotly annotation ("global = 0.31") is pinned to
the top-right corner, far from the vertical line it describes and often on top of the longest bar;
matplotlib puts it in a legend box that `loc="best"` moves out of the way.
*Fix:* place the plotly annotation at the line's own coordinate (`annotation_position="top"` for a vline)
or add a legend proxy trace, matching the `vspan` proxy pattern already used at
`renderers/plotly.py:756-762`.

**RESOLVED, but not with `annotation_position="top"` -- that produced a new collision.** Rendered, "top"
puts the label in the same strip as the subplot title and the two print over each other; plotly offers no
"inside top" for a vline annotation. The label is placed by hand at the line's x, anchored to the top of
the plot area and hanging INSIDE it. The horizontal (hline) case was already on its own value and is
unchanged.

---

## Dispositions (parent, 2026-09-06)

**VIS-01 FIXED.** `HeatmapPanelSpec` gained `color_vmin` / `color_vmax` -- the same field names and meaning
the scatter spec already had -- and both renderers now pass them through (`imshow(vmin=, vmax=)`,
`go.Heatmap(zmin=, zmax=)`). The correlation panel pins the scale to [-1, 1], the range rho can occupy, so
RdBu_r's white midpoint means "no rank agreement" instead of landing wherever the data happens to sit.
Confirmed in the rendered PNG: weak positive correlations (0.17-0.25) were deep blue and now read as pale
positive, with 1.00 on the diagonal dark red.

Pinning the image alone was not enough, and the render is what showed it: the per-cell TEXT colour samples
the colormap to decide black-or-white, and it was still sampling the data range, so the labels stayed white
on the newly pale cells and became unreadable. Both renderers now sample the same pinned scale. Pinned by
`tests/reporting/test_heatmap_scale_and_notice.py`, including a luminance assertion on the off-diagonal
labels; all five tests fail against the pre-fix code.

**VIS-03 FIXED.** The separator was a mangled escape that printed
``SHAP panels not produced:' + BS + 'n`` onto a figure handed to a user -- source-level debris, not a
formatting nit. Now a real newline. The test reads the text back off the FIGURE the production code built
rather than rebuilding the expression, which would have asserted against itself.

**VIS-02 FIXED.** ``xlabel`` names the VALUE and ``ylabel`` the CATEGORY whatever the orientation -- that is
what every horizontal-bar builder in ``charts/`` passes ("ECE (lower = better calibrated)" / "subgroup",
"quality (higher is better)" / "metric", "Weight of Evidence ..." / "feature=level") and what matplotlib
draws. The plotly branch had them the other way round, so the HTML report labelled the value axis "subgroup"
/ "model" / "metric" while the PNG of the SAME spec read correctly. Confirmed on a real figure before fixing:
the WoE panel's x axis carried "feature=level". Pinned by
``tests/reporting/test_horizontal_bar_axis_titles.py``, which asserts the two backends agree rather than
just checking plotly in isolation; all four tests fail with the swap restored.

**VIS-06 NOT CONFIRMED -- colour is not the only channel here, and the render is what settles it.** The
finding reads the traffic-light constants and concludes the verdict is encoded in colour alone. Rendering
the model card shows the verdict written in TEXT three times on the same figure: in the suptitle
("Model card -- lightgbm_dart_v2 (holdout) -- MISCALIBRATED"), in the header panel ("[AMBER] MISCALIBRATED"
plus the reason), and again in the caption ("VERDICT: ROC_AUC=0.802; ECE=0.275 (>= 0.15) drops it from
green"). ``fairness_calibration`` likewise puts "[green]" / "[amber]" / "[red]" in its panel title, and its
per-bar colours encode the GROUP, not the verdict. Adding hatches or symbols would be redundant noise on a
figure that already states the verdict in words. Recorded as not confirmed rather than fixed; if a future
panel encodes a verdict in fill alone, that panel is the finding.

## Found by rendering, not in the audit list (parent, 2026-09-06)

**CARD-01 [P1] the "mini gain" panel was crushed to a sliver.** ``col_width_ratios`` applies to the WHOLE
grid -- there is no per-row spanning -- and the model card collapsed its third column to 0.0001 to widen a
header whose own row leaves that cell empty. The row below puts a mini panel there, so "mini gain" rendered
narrower than its own legend on every model card this repo produces. Ratios are now (1.3, 1.3, 1.0): the
header keeps more width than a third, and the third mini panel keeps a panel's worth. Pinned by
``tests/reporting/test_model_card_layout.py`` as a GENERAL invariant -- a column occupied in any row must
keep at least 5% of the width -- so the next builder reaching for the same trick fails here instead of
shipping it.

**CARD-02 [P3] a stray hash printed in the card header.** ``f"{model_name}  # --  {split}"`` put a lone
"#" on the figure with nothing to its right; same class as the mangled escape in VIS-03, markup debris
shown to the user. Removed at both call sites.

**VIS-10 FIXED.** ``BarPanelSpec`` gained ``hline_symmetric``; when set, both renderers draw ``-hline[0]``
as well and annotate only the first, so the band is two lines and ONE legend entry rather than the same
threshold listed twice. The ACF and PACF panels set it, which is what their label ("+-1.96/sqrt(n)") and
their title (which counts ``|acf| > band``) were already claiming. Verified on an AR(1) with a NEGATIVE
coefficient, where the structure sits at about -0.6 and every informative lag previously had no reference
to be judged against. Pinned by ``tests/reporting/test_acf_symmetric_band.py``, which includes a guard that
the fixture really does have lags below the lower bound -- without it the test would pass on any series and
prove nothing; three of its six tests fail with the flag turned off.

**VIS-13 FIXED in part, and the parts NOT taken are deliberate.**

Taken: the shared 0.5 reference line is gone -- 0.5 is chance for ROC_AUC and for KS and means nothing for a
rescaled 1-Brier or 1-ECE, so one line across all of them invited a comparison that is not defined -- and
each bar now names its own raw value, because the bar LENGTH is a rescaled quality and the number behind it
was unrecoverable from the panel.

Caught while doing it: the first version labelled the bars from ``metric_fmt``'s own names, printing
"1-ECE 0.275" beside a bar of length 0.725. That states something false; 0.275 is the ECE. The label now
names the metric the raw value belongs to, and a test pins ``bar_length == 1 - labelled_value`` for the
inverted metrics.

NOT taken -- sort the bars by value. A model card is read against OTHER model cards, and a per-card
ordering puts the same metric on a different row on each one, which costs more than it buys. The fixed
order is now pinned by a test comparing two cards with different metrics.

NOT taken -- give the bars per-metric colours. One colour is the honest encoding when nothing per-bar is
being encoded; the finding's concern (a reader hunting for a meaning that is not there) is answered by the
title and caption saying what the length means, not by inventing a colour channel.

Pinned by ``tests/reporting/test_model_card_headline_bar.py``.

**VIS-11 FIXED.** ``CONFUSION_COL_MARGIN`` / ``CONFUSION_ROW_MARGIN`` are named in ``colors.py`` -- the
module that exists to stop exactly this drift -- and both renderers read them from there. matplotlib had
blue and green as literals; plotly painted BOTH strips with ``TREND_LINE``, whose documented job is the
robust-fit overlay line, so one confusion matrix came out blue-and-green as a PNG and two identical oranges
as HTML, with the orange also colliding with the trend line's meaning elsewhere in the same report.
Measured on rendered figures afterwards: both backends now emit ``#4c72b0`` and ``#55a868``. Pinned by
``tests/reporting/test_confusion_margin_colours.py``, which checks the two strips differ, that neither
borrows the trend-line colour, and that the backends agree; two of four fail with the plotly fills put
back on TREND_LINE.

**VIS-12 FIXED, with one correction to the finding and one departure from its fix.**

Both private palette copies are gone; the charts call ``colors.line_color`` now, so a repaint of the shared
palette reaches them. That half is exactly as reported -- the tuples were byte-identical to LINE_PALETTE.

Correction: the stated symptom ("subgroup 0 and subgroup 10 drawn as two solid lines in the identical
blue") is NOT reachable on defaults. ``max_groups`` defaults to 6, so the palette cannot wrap unless a
caller raises it past 10. It is reachable through the public argument, so it was still worth fixing, but
the finding overstates how easily it fires.

Departure: the proposed fix passes ``line_style(i)`` for every curve. In these two charts ``line_styles``
carries the DRAW MODE ("lines+markers"), not a dash pattern, so doing that would strip the markers from
every ordinary chart -- the 6-group default, where no colour repeats and nothing is gained. The style
switches only past the wrap, where colour has genuinely stopped identifying a group. Verified at 13
subgroups: the three repeated colours come back with distinct styles, and a separate test pins that a
4-group chart keeps its markers.

**VIS-09 FIXED, and the probe found a second instance the finding did not mention.** Dropping the
``(~weak).any()`` half of the guard restores the hollow markers and the "too few rows to read" legend entry
on a panel where EVERY bin is low-evidence -- previously it rendered pixel-identical to a 300k-row-per-bin
panel. Verified side by side in a rendered PNG.

The finding names one guard. matplotlib has TWO: one for the markers and one for the ERROR BARS. Fixing
only the first would have produced hollow markers beside ordinary solid grey error bars -- half the
confidence signal restored, half still missing -- and that was caught because the probe patched the wrong
line and the matplotlib assertion kept passing. The error-bar branch already contained its own
``strong.any()`` check, so the empty subset was always handled. Both are fixed and both are pinned.

**VIS-08 FIXED.** The plotly data-hull window is gated on ``equal_aspect``, matching matplotlib, which
applies ``lo..hi`` only in the square branch and lets the non-square calibration panel autoscale on
purpose. Checked across all three cases against matplotlib's own limits: calibration autoscales on both,
the square branch produces identical numbers, and an explicit limit wins.

That last case was a hole in the FIX rather than in the original code: gating the window left explicit
``xlim``/``ylim`` unapplied on the autoscaled branch, so a builder's own limit silently became a no-op.
Found by comparing the three cases against matplotlib rather than only the one the finding describes.

Both pinned by ``tests/reporting/test_scatter_backend_parity.py``.

**VIS-04 FIXED.** The plotly HORIZONTAL bar branch now thins its category labels with the same
threshold/keep policy matplotlib uses -- and that this renderer's own VERTICAL branch already used, which
is what made the omission easy to miss. Measured on a 200-row feature-importance chart: matplotlib drew 20
labels and plotly drew all 200 before; both draw 20 now. Verified at the boundaries too: 12 categories and
exactly the threshold keep every label, and the vertical branch is unchanged. Truncation and thinning are
independent guards, so a test pins that thinned labels are still truncated. Pinned by
``tests/reporting/test_horizontal_bar_tick_thinning.py``; three of seven fail with the thinning disabled.

**VIS-07 FIXED.** Both matplotlib protections are ported to plotly. The contrast halo is a tight opaque box
in the tone opposite the text (plotly has no text stroke), so a white label chosen by ``auto_text_color``
for a dark bubble stays legible when it runs off the fill onto the white panel -- previously it vanished
outright, defeating the very helper that picked the colour. The anchors now flip against the panel range
using the same fraction matplotlib uses, so a point in the busy bottom-left corner no longer has its label
clipped by the axis, and a top-edge label shifts DOWN instead of walking out of the panel. Verified on a
three-point fixture with points at both edges: the left-edge label anchors left, the top-edge one anchors
top with a negative shift, and each backing is the opposite tone of its own text. Pinned by
``tests/reporting/test_inline_label_contrast.py``; four of six fail with the halo removed.

**VIS-05 FIXED in the part that misleads; one part left open on purpose.** The class vector is remapped to
contiguous codes and drawn through ``tab10``, a QUALITATIVE colormap, with the scale pinned to
``-0.5 .. n_classes - 0.5`` so class k sits in the middle of band k regardless of how many classes a run
has. That removes what actually misled: classes no longer read as the extremes of one quantity, and the
middle class of a 3-class problem is no longer on a pale midpoint that vanishes against the panel.
Confirmed in a rendered PNG on a 3-class fixture.

Left open, and NOT worth pretending otherwise: the finding also asks for a discrete LEGEND instead of a
colourbar, so the ticks stop falling on values (0.5, 1.5) no row can take. ``ScatterPanelSpec`` has no way
to attach per-class legend entries to a point cloud -- it carries either a uniform colour or a numeric
colour array -- so that needs a new spec field and support in both renderers, which is a larger change than
this finding. The colourbar now at least names the class COUNT so the reader knows the scale is categorical.
Pinned by ``tests/reporting/test_separability_class_colours.py``.

### VIS-27 -- found by rendering, not in the audit list

**A heatmap colorbar's tick labels landed on the next panel's y-axis title.**
The colorbar is pinned just outside its own subplot's right edge; its tick labels stick out further still,
and the default 0.08 column gap (84 px on a 12-inch figure) does not hold the 12 px bar, its labels and the
neighbour's axis furniture. Seen directly in a two-heatmap render: "feature" written across "0.15".

**RESOLVED.** The column gap is now derived from what has to fit -- `_COLORBAR_GUTTER_PX` (95) plus
`_NEIGHBOUR_AXIS_PX` (60) over the figure width -- and applies only when a multi-column figure actually
draws a colorbar, so single-column layouts keep their full width.
`tests/reporting/test_backend_grid_and_contour_parity.py` (8 tests, every one verified failing pre-fix).
