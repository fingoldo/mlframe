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

### VIS-17 — P3 — `renderers/matplotlib.py:449-452` vs `renderers/_plotly_heatmap.py:252-269`
**PSI threshold contours are labelled on matplotlib and unlabelled on plotly.**
`HeatmapPanelSpec.threshold_contours` entries carry an optional 4th element, the label. matplotlib reads
it and calls `ax.clabel(cs, fmt={level: label})`. The plotly loop unpacks only `level`, `color`, `dash`
and sets `contours=dict(..., showlabels=False)`; the label is never read.
*What the reader sees:* the drift heatmap's two triage lines are annotated "moderate 0.1" / "significant
0.25" in the PNG and are two anonymous coloured squiggles in the HTML.
*Fix:* read `_entry[3]` and pass `contours.showlabels=True` with `contours.labelfont`, or add a legend
entry per contour level.

### VIS-18 — P3 — `renderers/_plotly_scatter.py:240`
**A shared overlay colour hardcoded on one backend only.**
`fillcolor="rgba(128,0,128,0.18)"` is the literal RGB of `colors.OVERLAY_LINE = "purple"`, while
matplotlib uses the constant (`renderers/_matplotlib_scatter.py:128`,
`color=OVERLAY_LINE, alpha=0.18`). The module already imports `_rgba` and uses it correctly at
`renderers/plotly.py:713` for the line-panel band.
*What the reader sees:* nothing today; on the next repaint of `OVERLAY_LINE` the curve-95%-band is
recoloured in PNGs and stays purple in HTML. This is the exact drift `colors.py:89-90` documents.
*Fix:* `fillcolor=_rgba(OVERLAY_LINE, 0.18)`.

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

### VIS-20 — P3 — `renderers/_matplotlib_scatter.py:116-117` vs `renderers/_plotly_scatter.py:212-214`
**The worst-K highlight ring is data-scaled on matplotlib and fixed-size on plotly.**
matplotlib: `s=base_s * 4.0` where `base_s` is the panel's own point size (or the median of the
per-point size array). Plotly: `marker=dict(symbol="circle-open", size=12, ...)`, a constant.
*What the reader sees:* on a panel with large bubbles the plotly highlight ring is smaller than the point
it is meant to highlight and disappears inside it.
*Fix:* convert `base_s * 4.0` through the same `sqrt(area) * 1.33` mapping the file already applies at
line 56-58.

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
