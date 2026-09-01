# reporting

Files reviewed: 23 read in full or targeted depth (~9,400 LOC); all 107 cluster files (26,841 LOC; 80 files /
25,133 LOC excluding `_benchmarks/`) pattern-scanned for the listed defect families.

## Summary

`src/mlframe/reporting/` is a three-layer stack: builders in `charts/` turn model outputs into frozen `spec.py`
dataclasses, `renderers/` renders one spec on matplotlib and plotly, and `save.py` / `report_html.py` /
`catalog.py` handle output. The stated design goal everywhere is "one spec, two visually equivalent backends",
and that is where the defects cluster.

Not found, having been looked for: no raw-power-sum moment formulas; no additive-epsilon denominators (the two
`1e-12` sites are FLOORS, `np.where(x > 1e-12, x, 1e-12)`, which is correct); no polars `min()==max()` constant
checks (all four are on numpy arrays); no process-wide `pl.Categorical`; no JSON-into-hash; no instance cache
missing from `__getstate__`. The broad-`except` sweep found 69 sites, all 69 logging, and the four that fall back
to a slower path name a cause -- this cluster does not carry the silent-swallow bug documented elsewhere in the
repo. The `add_annotation` O(n^2) trap is fixed at all three loop sites, each batching into one tuple assignment.

## Findings

### REPORTING-1 [P1] per-bar-colour-collapsed

**File:** `charts/category_discriminability.py` :297 (builder); `renderers/matplotlib.py` :558 and
`renderers/plotly.py` :593 (both renderers)

**Summary:** The WoE bar chart encodes the sign of each bar in a per-bar `colors` tuple, but on a single-series
`BarPanelSpec` both renderers read only `p.colors[0]` -- so every bar is painted the colour of the FIRST bar,
while the panel title and figure caption tell the reader the colour means the sign.

**Failure scenario:** `category_discriminability` is default-on (`catalog.py` :98). Rows are ranked by `|WoE|`
descending, so `colors[0]` is decided by whichever level has the largest absolute WoE. A dataset whose strongest
level tilts to `y=1` renders every bar green, including the ones with `WoE < 0`. The panel title reads
"green=>y=1, red=>y=0" and the caption reads "Positive (green) means the level tilts toward y=1, negative (red)
toward y=0". A reader following the stated key concludes every surfaced level tilts positive. Bar DIRECTION still
encodes the sign correctly, so the chart contradicts itself -- worse than being uniformly wrong, because the two
channels disagree with no indication which to trust.

**Suggested fix:** Teach both `_bar` single-series branches to accept a per-bar tuple -- matplotlib
`ax.bar(pos, values, color=list(p.colors))` and plotly `marker=dict(color=list(p.colors))` both accept a per-bar
sequence -- gated on `len(p.colors) == len(p.categories)`. Alternatively add a `bar_colors` field to
`BarPanelSpec` mirroring `HistogramPanelSpec.bar_colors`, which both renderers already honour per bar. The
builder's own comment at :296 ("Kept as a per-bar tuple so a signed-aware renderer can color each bar") shows the
first option was the intent. Either way, document `BarPanelSpec.colors` semantics and pin it with a test
asserting two differently-coloured bars come out of a two-row spec.

**Evidence:** Read the builder :295-307 and both renderer branches in full. Grepped every `colors=` call site in
`charts/`: this is the only single-series `BarPanelSpec` passing a length-N tuple; every other is a
`LinePanelSpec` (per-series, correct) or a length-1 tuple.

### REPORTING-2 [P2] index-misalignment -- FIXED 2026-09-01

**File:** `charts/calibration.py` :605 and :421-426

**Summary:** `inline_labels` was built by FILTERING OUT non-finite bins, but `_inline_label_colors` indexed the
unfiltered `point_color` / `point_size` arrays by LABEL position -- so one NaN bin shifted every subsequent
label's colour and size lookup by one bin.

**Failure scenario:** `hits = [500, 400, 0, 300, 200]` with `freqs_true[2] = nan` -- an empty bin, which the code
explicitly expects. The generator drops index 2, so `inline_labels` has 4 entries; `_inline_label_colors` then
reads `vals[3]` for the label belonging to bin 4. The contrast decision (`radius_pt > threshold`, then
`auto_text_color(vals[i], ...)`) is therefore made against the WRONG marker: a label sitting on a large dark
bubble gets the black chosen for a small pale neighbour, and is unreadable -- exactly the failure that helper
exists to prevent. Default path: `show_inline_population_labels=True` and `color_by="gap"` are both defaults.

**Disposition: RESOLVED.** A `label_keep` index array is now built once and used for the labels and for both
per-point arrays. Regression test `TestALabelIsColouredForItsOwnBin` added and proven to fail without the fix.

**Evidence:** Read :601-663 and :396-427 in full. Both renderers consume the two tuples positionally in parallel,
so the misalignment reached the canvas on both backends.

### REPORTING-3 [P2] verdict-on-biased-subsample

**File:** `charts/regression.py` :518 (and the `_shape` chain at :521-530)

**Summary:** The worm plot's "are residuals Gaussian?" verdict is computed on the DECIMATED plotting subset,
which deliberately over-represents the tails -- so the reported "% of points outside 95% CI" and the resulting
normality verdict are systematically biased toward "not normal" at large n.

**Failure scenario:** At `n = 1e6`, `_decimate_keep_tails(n, 2000, 100)` returns 100 head + 1800 middle + 100
tail indices. The 200 tail order statistics are 0.02% of the data but 10% of the 2000 points `_frac_out` averages
over, and tail order statistics are exactly where a QQ departure lives (the function's own docstring says so). A
residual distribution Gaussian in the body with a slightly heavy tail therefore reports several times the true
rate, trips `_frac_out >= 0.05`, and the title asserts "HEAVY TAILS -- a few errors far larger than Gaussian
(RMSE understates worst-case)" -- a modelling verdict a reader will act on. Below n = 2000 the decimation is a
no-op and the same figure reports the unbiased number, so the verdict CHANGES WITH ROW COUNT on identical
distributions. Secondarily, `_frac_out < 0.05` is the wrong threshold even unbiased: a pointwise 95% band is
expected to exclude ~5% of points under perfect normality, so the test sits exactly on its own null.

**Suggested fix:** Compute `_frac_out` on a uniform sample (or on the middle-strided subset alone, excluding the
verbatim-kept tails) and say which basis in the title; keep `keep` only for plotting. Then move the threshold off
the null. Pin with a test: a Gaussian `n = 1e6` residual vector must report "residuals ~ normal", and must report
the same verdict at n = 1500.

**Evidence:** Read `_decimate_keep_tails` :449-462 and `_worm_panel` :465-543 in full; `keep` flows straight into
`p_k` and `detrended`, and `_frac_out` reduces over that array with no reweighting. The title discloses the
sample size but not that the sample is non-uniform, and `_shape` carries no qualifier at all.

### REPORTING-4 [P2] renderer-divergence

**File:** `renderers/_plotly_heatmap.py` :283-303

**Summary:** The plotly heatmap's robust trend line maps its endpoints to the nearest category label via
`_to_cat`, which quantises AND CLAMPS them into `[0, nb-1]` -- so an extrapolated endpoint yields a segment with
a different SLOPE than matplotlib draws from the same spec. `_to_cat` additionally resolves the y endpoints
against `p.col_labels` instead of `p.row_labels`.

**Failure scenario:** `regression._scatter_panel` builds the hexbin spec with `trend_line="theil-sen"` (the
default) and `trend_xy=(yp, yt)`. `robust_fit_endpoints` returns y predicted at the x extremes, so with a slope
> 1 the predicted endpoints fall outside the data range. matplotlib's `_to_idx` is a pure affine map with no
clamp and `set_xlim`/`set_ylim` then CLIP the drawn segment, so the visible portion keeps the correct slope.
plotly's `_to_cat` moves the endpoint itself to the axis edge, so the two-point segment has a different slope and
the panel's whole purpose -- "so a systematic slope bias is visible even when the cloud hugs the diagonal" -- is
defeated on the HTML backend. The y/x label mix-up is latent today because the hexbin builder sets both from the
same centres, but any spec with `trend_line` and asymmetric labels puts the trend at a y category that does not
exist on the y axis, and plotly appends it as a new category rather than raising.

**Suggested fix:** Drop the category snapping -- plotly accepts fractional positions on a category axis, so pass
the continuous index and set explicit axis ranges to reproduce matplotlib's clip. Split `_to_cat` into `_to_x` /
`_to_y` so the y map reads `row_labels`, and factor the shared value-to-index affine map into `_shared_helpers.py`
so the two backends cannot drift.

**Evidence:** Read `_plotly_heatmap._heatmap` :268-303 and `matplotlib._heatmap` :439-468 side by side; the clamp
is literal at :286 and absent at `matplotlib.py` :455. `_trend.robust_fit_endpoints` :69-71 confirms the returned
y values are model predictions, unbounded by the data's y range.

### REPORTING-5 [P2] spec-field-dropped

**File:** `renderers/_plotly_network.py` :157-158 versus `renderers/plotly.py` :345-363

**Summary:** `_network` sets `showlegend=True` at figure level to make `NetworkPanelSpec.node_legend` visible, but
`render()` overwrites `layout.showlegend` AFTER every panel has rendered -- so the node-class legend is silently
dropped on the default interactive HTML backend while matplotlib draws it.

**Failure scenario:** Under the default `plot_outputs = "plotly[html] + matplotlib[png]"` the plotly path calls
`render()` with `static_legend=False`. The panel loop runs first, `_network` adds its legend-proxy traces and
calls `update_layout(showlegend=True)`; then :345-363 calls
`update_layout(showlegend=static_legend or _single_panel_has_labelled_series(spec))`. A network panel has no
`series_labels`, so that resolves False, and the three legend-proxy traces render as invisible `x=[None]` markers
with no key. The reader sees green / red / amber nodes with nothing saying which class is which -- while the
matplotlib PNG of the same spec carries the legend.

**Suggested fix:** Make the panel's need visible to `render()` rather than set from inside it: extend the
`showlegend` expression to also test `any(getattr(pn, "node_legend", None) ...)` over the panels, and delete the
`update_layout(showlegend=True)` in the panel body. Regression test: render a one-panel `NetworkPanelSpec` with
`node_legend` on plotly and assert `fig.layout.showlegend`.

**Evidence:** Read `plotly.render` :229-383 and `_plotly_network._network` :29-161 in full; the ordering is
unambiguous. `plotly.py` :585-589's own comment ("`barmode` is a FIGURE-level property, so setting it here from
inside one panel silently applies to every bar...") states the rule this site breaks.

### REPORTING-6 [P2] contract-drift

**File:** `charts/calibration.py` :562, :567, :573 versus the signature at :538-541

**Summary:** `build_calibration_spec`'s docstring documents three parameters as "(default on)" while all three
default to False, and inline comments 90 lines below in the same function state the opposite.

**Failure scenario:** A caller reads "(default on)" for `show_ece_annotation`, `reliability_smoothed` and
`reliability_band`, calls the builder expecting all three overlays, and gets none. The code comments at :613-617
("Both overlays are off by default") and :654-655 record the deliberate flip; the parameter docs were not updated
with it, and both halves live in the same docstring, so a reader cannot tell which is current.

**Suggested fix:** Change the three phrases to "(default off)" and fold the rationale from the inline comments
into the parameter docs so the WHY lives with the WHAT. `low_evidence_ci_width` and `color_by` are also
undocumented despite both changing what the chart means -- worth adding in the same pass.

**Evidence:** Read :522-733 in full; signature defaults and docstring claims are 25-50 lines apart in one
function body.

### REPORTING-7 [P3] renderer-divergence

**File:** `renderers/matplotlib.py` :596-612 and the comment at :581-584

**Summary:** The matplotlib `_bar` VERTICAL branch never calls `truncate_bar_label`, while the plotly twin
truncates on both orientations -- and the matplotlib horizontal branch's own comment asserts the vertical branch
already does it.

**Failure scenario:** A vertical bar panel with a 200-char generated column name: plotly truncates to 60 chars
plus ellipsis, matplotlib passes the label through untouched and the rotated label runs off the bottom of the
axis -- the exact case `truncate_bar_label` exists as "a safety valve against a pathological generated name
blowing out the axis". Two same-file comments contradict each other on this. Separately the vertical branch
hardcodes 25 and 20 instead of the module constants defined 520 lines above, so the same two numbers now exist in
four places across the two backends.

**Suggested fix:** Wrap both `set_xticklabels` argument lists in `truncate_bar_label`, replace the literals with
the constants, and correct the :581 comment. Ideally move both thresholds into `_shared_helpers.py` alongside
`_BAR_LABEL_MAXLEN` so one constant serves both backends.

**Evidence:** Read both `_bar` bodies in full; `truncate_bar_label` appears once in matplotlib (horizontal only)
and three times in plotly (both orientations).

### REPORTING-8 [P3] renderer-divergence / default_via_or

**File:** `renderers/matplotlib.py` :328 and :360

**Summary:** Two `x or default` expressions where the plotly twin uses the correct `x is not None` form, so a
legitimate falsy spec value produces different charts on the two backends.

**Failure scenario:** (a) `bin_width=0.0` -- matplotlib treats 0.0 as "unset" and derives a width from the centre
spacing; plotly honours it and draws zero-width bars. One spec, two pictures. (b) `overlay_label=""` --
matplotlib substitutes the auto label; plotly honours the deliberate blank. The
empty-string-means-"no label" distinction is one this cluster explicitly recognises elsewhere
(`calibration.py` :392 documents exactly it for `colorbar_label`).

**Suggested fix:** Change both matplotlib sites to the `is not None` form. Neither is reachable from a shipping
builder today, so this is latent -- but it is the same trap two other files in this cluster carry written
warnings about.

**Evidence:** Read all four expressions; the pairs are the same logic written two ways.

### REPORTING-9 [P3] renderer-divergence

**File:** `renderers/plotly.py` :797-818 versus `renderers/matplotlib.py` :716-759

**Summary:** The two `_violin` implementations disagree on two things from one spec: matplotlib drops empty
groups and names them in the title, plotly renders them as blank slots with no note; and matplotlib's box
whiskers are the 5th/95th percentiles while plotly's are the default 1.5x IQR fences.

**Failure scenario:** A spec with an empty group -- a multiclass panel where one class has no rows in the split.
matplotlib draws the non-empty violins and titles the panel "... (no data: class_c)", with the explicit rationale
that "a violin that silently vanishes reads as 'this group has no spread', which is a different statement from
'this group has no data'". plotly iterates every group with no filtering, emitting a labelled category with
nothing in it and no note anywhere. The two figures make different claims about the same data. Independently the
box whiskers mark different quantities on the two backends -- and matplotlib's own comment says the box was added
specifically so "one spec produced two different amounts of information" would stop happening.

**Suggested fix:** Move the empty-group filter and the "(no data: ...)" title suffix into a shared helper both
call. For the whiskers, set plotly's explicitly; plotly has no percentile-whisker mode, so either drop matplotlib
to 1.5x IQR or emit the 5/95 whiskers as an explicit box trace -- whichever is chosen, name the convention in the
panel or caption so the number a reader reads off is defined.

**Evidence:** Read both `_violin` bodies in full; matplotlib builds `kept`/`drawable`, plotly has no equivalent,
and `whis=(5, 95)` has no plotly counterpart.

### REPORTING-10 [P3] renderer-divergence

**File:** `renderers/_plotly_scatter.py` :141-155

**Summary:** The plotly low-evidence split narrows the error-bar arrays but never restyles them, unlike the
matplotlib twin -- and it sets `marker.color` to a fully transparent `rgba(0,0,0,0)`, which plotly's error bars
inherit when `error_y.color` is unset.

**Failure scenario:** A calibration reliability scatter where some bins trip `low_evidence_ci_width`. matplotlib
draws those intervals distinctly muted (`ecolor="0.75"`, `elinewidth=0.6`, `capsize=0`, `linestyle=":"`),
implementing the spec's documented intent that the muted interval is "the honest answer ('we know nothing
here')". plotly's `_sel_err` copies the error dict and narrows the arrays only; no colour, width or dash is set
on either trace, so the two are drawn identically and the distinction the spec field exists for is lost on the
HTML backend. The aggravating factor is inferred from plotly's documented inheritance rule rather than from this
repo: the transparent marker colour is what `error_y.color` falls back to, so the weak whiskers may render fully
transparent. Worth a one-line repro before fixing.

**Suggested fix:** Give `_sel_err` a `muted` flag and set `color="#c0c0c0"`, `thickness=0.6`, `width=0` on the
weak copy and an explicit `color="gray"`, `thickness=1.0` on the strong one, so neither inherits the marker. That
removes the transparency risk regardless of how the inheritance actually resolves.

**Evidence:** Read `_plotly_scatter.py` :101-155 and `_matplotlib_scatter.py` :53-70 side by side; the matplotlib
branch has four explicit style overrides for the weak errorbar call and the plotly branch has none. `spec.py`
:78-83 documents the intent.

### REPORTING-11 [P3] hovertext-silently-dropped

**File:** `renderers/_plotly_interactivity.py` :179-187

**Summary:** A builder's `hovertext` is attached only when the trace's point count exactly equals `len(sup)`,
which silently fails for every scatter panel that downsampled or split into strong/weak traces; and any other
trace on the same subplot with a coincidentally matching length gets that same hovertext.

**Failure scenario:** A `ScatterPanelSpec` with `hovertext` on a panel with `low_evidence_indices` set emits two
traces of `k` and `n-k` points, neither equal to `n`, so the per-point DENOMINATOR -- "without it a rate computed
from 3 rows renders identically to one from 300k" (`spec.py` :99-102) -- never reaches the tooltip. Same above
`_SCATTER_MAX_POINTS`. The fallback is the generic axis-name template, so the tooltip still looks plausible and
nothing signals the loss. Separately the support is keyed by subplot axis, not by trace, so a line panel's
hovertext of length L will also attach to an overlay, band or vline-proxy trace carrying L points.

**Suggested fix:** Have the renderers attach `hovertext` on the trace they build -- they already know the row
subset and can index it with the same mask via `select_per_point` -- and reduce `_apply_nonline_traces` to
templating only. If the length gate must stay, log at DEBUG naming the panel when a non-empty `hovertext` is
discarded.

**Evidence:** Read `_apply_nonline_traces` :149-196 and `_plotly_scatter._scatter` :40-168 in full; both the
downsample and the two-trace split change the per-trace point count away from `len(p.x)`, and
`_apply_nonline_traces` builds its supports with no knowledge of either.

### REPORTING-12 [P3] latent dense-axis-index assumption

**File:** `renderers/_plotly_interactivity.py` :89, :117, :159, :172; `renderers/_plotly_color.py` :29-30;
`renderers/_plotly_network.py` :112-113; `renderers/_plotly_heatmap.py` :39

**Summary:** Six sites derive a plotly axis id as `row * cols + col`, which assumes every grid cell got an axis --
but `plotly.py` :247-261 deliberately passes `None` in `specs` for empty cells, so plotly does not allocate axes
for them and the numbering of every later cell shifts.

**Failure scenario:** `panels = ((line_a, None), (line_b, heat))` -- a documented layout. plotly allocates three
axes for the three real panels, but the arithmetic computes index 4 for `line_b` and writes its hovertemplate
under a key no trace is bound to, so `line_b` keeps plotly's raw default hover and `heat`'s lookups are off by
one. `_axis_ref` feeds `scaleanchor`, so an `equal_aspect` scatter in a grid with a hole would square itself
against the wrong axis. Not reachable today: `pack_panels` only pads the LAST row, and all 56 hand-built
`panels=((...))` sites were checked for a mid-grid `None` -- none has one.

**Suggested fix:** Resolve the axis id from `fig._grid_ref[row-1][col-1][0].layout_keys` -- the lookup
`_plotly_heatmap._cell_domains` and `_colorbar_placement` already do correctly -- and put it in one shared helper
the six sites call.

**Evidence:** Read the `sub_specs` construction, whose comment states that `None` (not `{}`) is what suppresses
the axis; read all six `idx =` sites; read `pack_panels` and grepped every `panels=((` in the cluster.

### REPORTING-13 [P3] dead-code

**File:** `renderers/_plotly_scatter.py` :76, :159, :162-164

**Summary:** `text` is assigned None at :76 and never reassigned, so `mode="markers+text" if text else "markers"`,
`text=text`, the conditional `textposition` and `textfont` are unreachable branches.

**Failure scenario:** No wrong output today. The risk is that the surviving scaffolding reads as a live
per-point-text feature, so the next reader assumes `ScatterPanelSpec` text labels flow through here -- they do
not; `inline_labels` are handled as annotations at :87-99. The comment at :73-75 records why the per-point path
was removed.

**Suggested fix:** Delete `text = None` and the four dependent expressions; keep the comment.

**Evidence:** Read :73-168; `text` has exactly one assignment.

### REPORTING-14 [P3] colour-constant-bypassed

**File:** `renderers/_matplotlib_scatter.py` :139

**Summary:** The matplotlib perfect-fit diagonal is hardcoded `"g--"` while the plotly twin uses
`PERFECT_FIT_LINE` from `colors.py` -- the one overlay colour that escaped the centralisation `colors.py` :84-85
says was done to stop exactly this drift.

**Failure scenario:** No visible defect today (`PERFECT_FIT_LINE = "green"`). Changing it repaints the plotly y=x
line and leaves the matplotlib one green, from the same spec, with nothing flagging it -- the failure `colors.py`
:84-85 describes. `_matplotlib_scatter.py` already imports two other overlay colours from `colors`;
`PERFECT_FIT_LINE` is the one it did not pick up during the carve-out.

**Suggested fix:** Import it and replace `"g--"` with `color=PERFECT_FIT_LINE, linestyle="--"`.

**Evidence:** :139 versus `_plotly_scatter.py` :224; `colors.py` :89 and the `__all__` export at :231.

### REPORTING-15 [P3] duplicate-traces / hover-shadowed

**File:** `renderers/_plotly_network.py` :76-85 and :91-100

**Summary:** Edge midpoints get two overlapping marker traces -- a rich per-bucket one built inside the bin loop
and a global colorbar one added after -- and the later, poorer one wins the hover, so the descriptive text the
loop built is unreachable.

**Failure scenario:** Any spec with edges. The per-bucket trace carries "feat_a - feat_b<br>edge weight=0.1234";
the global trace carries "MI=0.1234" at the SAME coordinates and is added last, so it takes the hover. The reader
gets no node names and a hardcoded "MI=" label that ignores `p.colorbar_label`, which the per-bucket trace
resolves properly. Secondary cost: every edge midpoint is emitted twice, up to 8 extra traces plus a full
duplicate coordinate set per figure, and `_label` is recomputed once per bin for a value that does not vary.

**Suggested fix:** Delete the per-bucket midpoint trace and move its `hovertext` onto the single global trace,
built once over all edges. Hoist `_label` out of the loop.

**Evidence:** Read `_network` :43-100 in full; both traces are `go.Scattergl` markers at the same midpoints, the
second added unconditionally after the bin loop.

### REPORTING-16 [P3] contradictory-comments

**File:** `charts/slice_finder.py` :412-418

**Summary:** Two adjacent comment blocks state opposite display-ordering rules; the code implements the second.

**Failure scenario:** No wrong output -- the sort key is the degradation x sqrt(support) score, matching the
second block. The first block asserts "Display order is worst-ERROR-first: rank the surfaced slices by mean error
descending", describing a superseded implementation. The next reader debugging why the top bar is not the
highest-error slice will trust the wrong comment.

**Suggested fix:** Delete the stale block at :412-414.

**Evidence:** Read :402-419; the two blocks are consecutive and the sort key at :418 is `rec_score`.

### REPORTING-17 [P3] incomplete-dedupe

**File:** `report_html.py` :284-291

**Summary:** `_dedupe_plotly_js` only strips a repeated plotly.js bundle when it is the FIRST `<script>` in the
fragment; a fragment whose bundle sits in a later script passes through whole, contradicting the docstring's
"Keep the first, drop the rest".

**Failure scenario:** A fragment written with `include_plotlyjs=True` whose emitted order puts a config or
`require` shim before the bundle. The marker check runs against that first script only, is False, and the
function returns the fragment unmodified -- so a 20-chart report still ships 20 copies of a 3-4 MB bundle, the
exact cost the function exists to avoid. `seen` was already marked, so no later fragment is stripped either.

**Suggested fix:** Search for the script CONTAINING the marker rather than assuming it is first: find the marker,
walk back to the nearest preceding `<script`, forward to the following `</script>`. Guard the not-found case as
today.

**Evidence:** Read :277-293 in full; `find("<script")` takes the first occurrence unconditionally and the marker
check filters on that one script only.

### REPORTING-18 [P3] default_via_or

**File:** `colors.py` :61-62

**Summary:** `calibration_cmap()` tests the thread override with `if override:`, so an override deliberately set
to `""` is skipped and the env var or module default is used instead.

**Failure scenario:** `set_calibration_cmap("")` stores the empty string, and `if override:` is False, so the
call falls through. Narrow -- `""` is not a valid colormap name, so this is closer to input validation than to a
live wrong-chart path. It matters because the tri-state contract is `str` / `None` and the CLEAR path is
`set_calibration_cmap(None)`; `""` currently behaves as a silent second clear rather than as a bad value.
`save.py`'s two sibling overrides get this right with an explicit `_UNSET` sentinel.

**Suggested fix:** Either reject an empty string in `set_calibration_cmap`, or change the read to
`if override is not None:`. The sentinel pattern in `save.py` :42-49 is the in-repo precedent.

**Evidence:** Read `colors.py` :41-65 and `save.py` :42-68; the two tri-state overrides in this cluster are
implemented two different ways.

## Coverage

Read in full: `spec.py`, `colors.py`, `output.py`, `report_html.py`, `catalog.py`, `renderers/matplotlib.py`,
`renderers/plotly.py`, `renderers/_matplotlib_scatter.py`, `renderers/_plotly_scatter.py`,
`renderers/_plotly_heatmap.py`, `renderers/_plotly_network.py`, `renderers/_shared_helpers.py`,
`renderers/_trend.py`, `renderers/save.py`, `charts/calibration.py`, `charts/_calibration_chart_shared.py`.

Read in targeted depth: `renderers/_plotly_interactivity.py` (effectively full), `charts/_layout.py`,
`charts/regression.py` (:190-260, :440-545, :594-610), `charts/category_discriminability.py` (:255-344),
`charts/slice_finder.py` (:320-420, :480-500), `charts/error_analysis.py` (:55-115), `renderers/_plotly_color.py`.

Pattern-scanned across all 107 files: every `except` site with context (69 hits, all logged); every 1e-8 to 1e-15
literal (21 hits, all floors or clips); every `min() == max()` / `eq_missing` (4 hits, all numpy); every
`pl.Categorical` / `pd.Categorical` (3 hits, none process-wide-cache-exposed); `json.dumps` / `__getstate__` /
`_cache`; `@njit` / `prange` / `parallel=True` / `cuda.jit` / `cupy` / `kernel_tuning_cache` (3 real njit kernels,
all already `parallel=True`; no Python-loop-around-njit-kernel found); every `x or default` (52 hits, two
reportable); every `add_annotation` / `add_shape` loop (all three batched); every `colors=` / `value_err=` /
`hovertext=` / `trend_xy` builder call site.

Not read: the 27 files under `charts/_benchmarks/` and `_benchmarks/` (1,708 LOC, non-production); the bodies of
`diagnostics_dispatch.py` (974) and `_diagnostics_dispatch_extra.py` (816), `auto_dispatch.py`, and the ~20
remaining `charts/*.py` builders beyond the grep sweep and the ranges above. The two dispatch modules carry the
suite's gating logic and are the largest gap here.
