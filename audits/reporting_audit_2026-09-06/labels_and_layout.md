# Reporting audit 2026-09-06 — text & layout collisions

Scope: `src/mlframe/reporting/` (renderers + charts) plus `feature_selection/filters/friend_graph.py`
(the only chart builder outside `reporting/` that constructs a `NetworkPanelSpec`).
Read-only. Ranked by how likely the ugly input is given the defaults the real call sites pass.

General state: this module is already unusually well-hardened on this bug class — measured font-based
title/suptitle/caption wrapping, per-panel title wrap, bar-label truncation + thinning, heatmap tick
thinning, `_HEATMAP_CELL_TEXT_MAX`, outside-legend support, colorbar-per-subplot placement. The findings
below are the gaps left in that scheme, most of them where a guard exists on one panel type / one backend
and not on its twin, or where a guard is keyed on element COUNT while the figure size is keyed on the
same count (so the two cancel and the labels are dropped from a figure that had room for them).

---

## LBL-01 — P1 — network node labels: no cap, no truncation, no collision avoidance, fixed figsize

- `mlframe/reporting/renderers/matplotlib.py:836-837` — `for (x, y), label in zip(nx_pos, p.node_label): ax.annotate(label, (x, y), fontsize=7, ha="center", va="center")` — every node, unconditionally, centred ON the marker.
- `mlframe/reporting/renderers/_plotly_network.py:138-141` — `mode="markers+text"`, `text=list(p.node_label)`, `textposition="top center"` — same, every node.
- Producer: `mlframe/feature_selection/filters/friend_graph.py:637-651` — `node_label = tuple(n.name for n in nodes)`, `figsize=(11.0, 8.0)`, and the node set is capped at `max_nodes: int = 200` (`friend_graph.py:343`).

Bad input (routine here): a friend graph over 200 selected features whose names are generated FE columns
averaging 30-45 chars (`job_posted_at_day_of_year_cos`, `binned_numeric_agg__price__mean__by_region`).
200 labels x ~35 chars at 7 pt on an 11x8 in canvas is roughly 8x more text than the canvas holds; the
picture becomes a solid grey mat with the node markers invisible underneath. The edge-arrow path is
already capped (`_NETWORK_MAX_ARROWS = 500`, `_plotly_network.py:28`) — labels never got the same treatment.

Fix: cap the labelled nodes (e.g. label only the top-N by `node_size`/degree, N ~ 30, the rest reachable
via hover on plotly and via the legend/table on matplotlib), truncate each label through the existing
`truncate_bar_label` with a much shorter cap (~20) and, on matplotlib, offset the text (`xytext=(0, 6),
textcoords="offset points"`) so it does not sit on its own marker. A `label_max_nodes` field on
`NetworkPanelSpec` keeps the decision with the builder.

## LBL-02 — P1 — heatmap tick labels thinned to 8 regardless of how tall/wide the panel is

- `mlframe/reporting/renderers/_shared_helpers.py:19` — `_HEATMAP_MAX_TICKS = 8`, `:68-72` `_thin_tick_positions`.
- Applied unconditionally: `matplotlib.py:418-423`, `matplotlib.py:515-520` (confusion margins), `_plotly_heatmap.py:315-317, 324`.

Bad input (the DEFAULT for the drift chart): `charts/drift.py:230` `max_features: int = 40`, and
`charts/drift.py:284` sizes the figure as `max(3.0, 0.32 * n_feat + 1.5)` — at 40 features that is a
14.3-inch-tall figure, i.e. ~0.32 in (23 px) per row, ample for an 8 pt label. The renderer nevertheless
labels 8 of the 40 rows, so the reader can see a red band but cannot name the feature that drifted —
which is the entire purpose of the panel. Same for a 20-class confusion matrix (`_multiclass_confusion.py:79-80`)
and the between-model correlation heatmap (`model_comparison.py:322-323`).

Fix: make the ceiling space-aware, not count-aware: derive `max_ticks` from the axis length in inches
(matplotlib: `ax.get_position().height * fig height`; plotly: the subplot domain span x figure height)
divided by the label line height, and keep 8 only as the floor when that measurement is unavailable.
The measured-width machinery for this already exists in `_shared_helpers._char_advances`.

## LBL-03 — P1 — heatmap row/col tick labels are never truncated on either backend

- `matplotlib.py:420` / `:423` — `ax.set_xticklabels([p.col_labels[i] for i in _xt], ...)`, raw.
- `_plotly_heatmap.py:317` / `:324` — `tickvals=[p.col_labels[i] for i in _xt]`, raw.
- Bar labels DO get `truncate_bar_label` (`_shared_helpers.py:57-65`, cap 60) on both backends; heatmaps got no equivalent.

Bad input: `charts/class_structure_heatmap.py:211-212` sets `row_labels` from
`_group_codes_capped(group_arr, max_groups=30)` (`_group_codes.py:53`: `labels = [str(uniq[i]) for i in keep]`),
i.e. RAW values of an arbitrary categorical column — URLs, job titles, country+city strings — with the
figure pinned at `figsize=(7.0, 5.0)` (`class_structure_heatmap.py:236`). A single 90-char group value
consumes more than the whole 7-inch width. On matplotlib the `bbox_inches="tight"` save
(`matplotlib.py:246`) rescues it from clipping by inflating the PNG to a bizarre aspect ratio; on plotly
the fixed `margin=dict(l=60, ...)` (`plotly.py:369`) simply clips it.

Fix: run every heatmap tick label through `truncate_bar_label` (a shorter cap, ~28, is right for a y-axis
that has to leave room for the grid), and put the full value in the hover text, which
`HeatmapPanelSpec.cell_hovertext` already carries per cell.

## LBL-04 — P2 — plotly never thins HORIZONTAL bar category labels; matplotlib does

- `plotly.py:621-626` — the horizontal branch only truncates (`if any(len(str(c)) > _BAR_XTICK_MAXLEN ...)`), there is no thinning at all.
- `matplotlib.py:601-615` — the horizontal branch thins past `_BAR_TICK_THIN_THRESHOLD = 25` to `_BAR_TICK_KEEP = 20`.
- Vertical is symmetric on both (`plotly.py:632-639`, `matplotlib.py:627-635`); only horizontal diverges.

Bad input: `charts/category_discriminability.py:305` with `top_k` raised above 25, or
`charts/_multiclass_confusion.py:192` (confused-pairs bar) on a 30-class problem. One spec then renders a
readable subsampled y-axis in the PNG and an unreadable overlapping band in the interactive HTML — the
exact cross-backend drift `_shared_helpers` was created to prevent.

Fix: mirror the vertical branch's thinning into `plotly.py:621-626` using the same two constants.

## LBL-05 — P2 — bar-label truncation is head-preserving, but builders put the payload at the TAIL

- `_shared_helpers.py:57-65` — `truncate_bar_label` keeps the first `maxlen - 1` chars and appends `"..."`.
- `charts/slice_finder.py:468` — `cats = tuple(f"{bounds}  (n={support:_}, {ratio:.2g}x)" ...)`, and its own title (`:483`) tells the reader "label = support n + error ratio".
- `charts/category_discriminability.py:280` — `cats = tuple(f"{feat}={lbl}  (n={support:_}, p={p_rate:.2f})" ...)`, title `:302` again says "label = support n + P(y=1|level)".

Bad input: a 2-feature slice, which `slice_finder` produces by default — `"job_posted_at_day_of_year_cos
[0.12..0.45] & industry_code [3.00..9.00]  (n=12_345, 2.31x)"` is ~95 chars, so the 60-char cap severs
exactly the `(n=..., x)` suffix the title promises. The chart then documents a field it does not render.

Fix: give `truncate_bar_label` a middle-ellipsis mode (`head + "..." + tail`) and use it for these two
builders, or have the builders put the count/ratio FIRST. Either way the cap must not be allowed to
delete the annotated payload.

## LBL-06 — P2 — bar tick thinning is count-based while the figure height is also count-based, so they cancel

- `matplotlib.py:607-611` and `plotly.py:632-639` — drop to ~20 labels past 25 categories, independent of figure size.
- `charts/slice_finder.py:495` — `figsize=(10.0, max(5.0, 0.5 * len(table) + 2.0))`; `charts/category_discriminability.py:330` — `height = max(5.0, 0.5 * len(_cats) + 2.0)`.

Bad input: `top_k=40` on either chart. The builder deliberately buys 0.5 inch (50 px) of height per bar —
far more than an 8 pt label needs — and the renderer still hides 20 of the 40 labels, leaving 20 bars
whose identity is unrecoverable. Note `slice_finder`'s default `DEFAULT_TOP_K = 7` (`slice_finder.py:122`)
keeps the DEFAULT path safe; this bites the caller who raises it, which the parameter exists for.

Fix: same as LBL-02 — thin against the measured axis length per label, not against a raw category count.

## LBL-07 — P2 — spectral embedding labels every node on a fixed 7x6 figure

- `charts/spectral_embedding.py:133` — `node_label=tuple(str(i) for i in range(n_nodes))`, no cap.
- `charts/spectral_embedding.py:164` — `figsize=(7.0, 6.0)`, fixed.
- Rendered through the same uncapped label loops as LBL-01.

Bad input: any graph over a few hundred nodes (the sibling graph FE guards at `max_nodes: int = 2000`,
`feature_engineering/graph_spectral_features.py:77`). Node INDICES are short, so this is less severe than
LBL-01, but 2000 numerals on a 7x6 canvas is still a grey mat, and the labels carry almost no information
(the index is already in the hover text, `spectral_embedding.py:134`).

Fix: drop node labels entirely above ~50 nodes (hover already carries `node i / degree=k`), or label only
the highest-degree nodes.

## LBL-08 — P2 — multiclass overlay legends sit inside the axes and can carry 13 long entries

- `charts/multiclass.py:216-226` (ROC), `:307-316` (PR), `:385-395` (reliability) — build up to `_OVERLAY_MAX_CLASSES = 12` (`multiclass.py:61`) class series plus `chance` and `macro-avg`, and set neither `legend_outside` nor `legend_ncol`.
- `matplotlib.py:719` — the fallback is `ax.legend(handles, leg_labels, loc="best", fontsize=8, framealpha=0.7, ncol=1)`.
- Entry text: `f"{classes[k]} (AUC={roc_auc:.3f} [{lo:.3f}, {hi:.3f}])"` (`multiclass.py:196`) — a real class name plus 26 chars of CI.

Bad input: 12 classes whose names are strings ("electronics_accessories") rather than integers. 14 legend
rows at 8 pt is ~1.6 in tall and, with the CI suffix, ~2.6 in wide — on a `cell_width` panel that is most
of the plot, and `loc="best"` on an ROC panel lands it in the empty lower-right, exactly on top of the
region where the curves separate. `quantile.py:510` already sets `legend_outside=True` for precisely this
situation, so the field and the renderer support both backends (`matplotlib.py:715-717`, `plotly.py:384-394`).

Fix: set `legend_outside=True` (and `legend_ncol=2` on matplotlib) on these three panels once the drawn
series count exceeds ~5, and truncate each legend entry's class-name portion.

## LBL-09 — P2 — violin group labels are neither truncated nor thinned on either backend

- `matplotlib.py:779-780` — `ax.set_xticks(range(1, len(labels)+1)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)`.
- `plotly.py:854` — `fig.update_xaxes(title_text=p.xlabel, row=row, col=col, tickangle=-30)` — plotly gets rotation only; the category labels come straight off the trace names (`plotly.py:836` `name=label`).

Bad input: a per-class probability violin on a 20+ class problem with string class names, or a per-group
violin keyed on a raw categorical. At 30 degrees a 25-char label projects ~1.9 in horizontally, so 20 of
them overlap on any realistic cell width. Bars got both guards; violins got neither.

Fix: route `group_labels` through `truncate_bar_label` in both renderers and thin them with the same
`_BAR_TICK_*` policy (the violins themselves stay 1-per-group; only labels subsample).

## LBL-10 — P3 — the free-text panel measures its width before the layout engine has run

- `matplotlib.py:312-315` — `bbox = ax.get_window_extent()` then `panel_w_in = bbox.width / dpi`, fed to `wrap_annotation_text`.
- `render()` builds the figure with `layout="constrained"` (`matplotlib.py:155`), and a constrained layout does not compute final axes geometry until draw time.

So the wrap budget is measured against the PRE-layout axes rectangle. When constrained layout later
shrinks the panel (to make room for a colorbar on a sibling, or for the reserved suptitle/caption bands
set at `matplotlib.py:225`), the wrapped text is wider than the panel it ends up in — the overflow this
helper's docstring says it exists to prevent. Reproducer shape: a 2-panel figure with a heatmap+colorbar
beside an `AnnotationPanelSpec` (e.g. `model_card.py:412`).

Fix: derive the width from the gridspec fraction instead of the live extent —
`ax.get_subplotspec().get_position(fig).width * fig.get_size_inches()[0]` — which is layout-independent,
or wrap in a draw callback.

## LBL-11 — P3 — scatter inline labels: edge-flip only, no mutual collision avoidance, no cap

- `renderers/_matplotlib_scatter.py:182-192` — per-label `ha`/`va` flips within `_EDGE_LABEL_FLIP_FRACTION` of an axis edge (`matplotlib.py:91`), plus a contrast halo; nothing compares two labels to each other.
- `renderers/_plotly_scatter.py:81-100` — a fixed `yshift=8` for every label, no flip and no avoidance.
- Producer: `charts/calibration.py:691` `inline_labels=inline_labels`.

Bad input: a reliability diagram with 20 bins where the low-probability bins crowd into the bottom-left —
several bin labels land within a few points of each other and overprint. The halo makes the collision look
like a smudge rather than a clip, which is why it reads as low severity, but the numbers become unreadable.

Fix: greedy de-overlap (sort by y, push each label off the previous one's measured bbox, using the
existing `_measured_text_width_pt`), and cap the number of labelled points.

## LBL-12 — P3 — point-marker / vline annotations use fixed offsets with no de-collision

- `matplotlib.py:702` — `ax.annotate(mlabel, (mx, my), textcoords="offset points", xytext=(8, -10), fontsize=7)` for every entry of `point_markers`.
- `plotly.py:753-755` (vspan label) and `plotly.py:809` (datetime vline label) — both stamped at `y=1, yref="y domain", yanchor="bottom"`, i.e. all on the same horizontal line just above the panel.
- Specs allow arbitrarily many: `LinePanelSpec.point_markers` / `.vlines` / `.vspans` are unbounded tuples (`spec.py:319-326`).

Bad input: a threshold-sweep panel carrying two operating points a few pixels apart (e.g. F1-optimal and
Youden-optimal thresholds that nearly coincide), or a temporal panel with several change points close
together — the labels overprint exactly. On plotly all vspan labels share one y, so two adjacent regimes
always collide.

Fix: stagger the offset by index (alternate above/below, or step `yshift`), and skip a label whose
measured box overlaps the previously drawn one.

## LBL-13 — P3 — figures without a suptitle/caption get no layout engine at all

- `matplotlib.py:155` — `layout = "constrained" if (spec.constrained_layout or spec.suptitle or spec.caption) else None`.
- `FigureSpec.constrained_layout` defaults to `False` (`spec.py:465`) and many builders pass `suptitle=""` deliberately (e.g. `drift.py:295`, `decision_curve.py:252`, `risk_coverage.py:302`, `slice_finder.py:493`).

For those figures nothing reserves space for 45-degree-rotated x tick labels, a long y-axis title, or a
colorbar; the `savefig(bbox_inches="tight")` at `matplotlib.py:246` rescues the SAVED file by growing the
canvas (so the requested figsize/aspect is silently not what is produced), and the interactive
`show()` path (`matplotlib.py:248-279`, IPython `display(fig)`) has no tight bbox at all — in a notebook
those labels are simply clipped.

Fix: use `layout="constrained"` unconditionally (the ~800 ms note in `spec.py:461-465` is a per-figure cost
worth re-measuring against the correctness of every non-suptitled figure), or at minimum turn it on
whenever a panel sets `xtick_rotation`, `orientation="horizontal"`, or a `colorbar_label`.

## LBL-14 — P3 — `segments_bar` grows only its WIDTH with category count, on raw untruncated group strings

- `charts/error_analysis.py:442` — `figsize=(max(8.0, len(cats) * 0.5), 5.0)` with `max_groups: int = 30` (`:386`).
- `charts/error_analysis.py:409` — `groups = df[group_col].astype(str).to_numpy()`, raw values, `xtick_rotation=45.0` (`:434`).

At 30 groups the figure is 15 x 5 in — a 3:1 letterbox whose 5-inch height must absorb 45-degree-rotated
labels of arbitrary length; a 40-char group value projects ~2.0 in vertically, i.e. 40% of the figure
height goes to tick labels. Truncation does apply here (60 chars, `matplotlib.py:626`), which is why this
is P3 rather than P2, but 60 chars is far past what this aspect ratio can hold.

Fix: cap the width (~16 in), grow the height with the longest label's measured projection, or switch the
panel to `orientation="horizontal"` where long category names belong.

## LBL-15 — P3 — a long `colorbar_label` overflows the fixed right margin (plotly) and is never wrapped (both)

- `renderers/_plotly_heatmap.py:165-171` — the colorbar is pinned at `x = x_domain[1] + 0.01` with `thickness=12`; the label becomes the colorbar title.
- `plotly.py:369` — `margin=dict(l=60, r=40, ...)`, fixed, independent of the label.
- `matplotlib.py:489-490` / `_matplotlib_scatter.py:194-196` — `cbar.set_label(p.colorbar_label)` with no wrapping.

Bad input, already present in the codebase: `charts/error_analysis.py:358` passes
`colorbar_label="mean error (darker = worse); cell number = rows in cell"` (60 chars) and
`charts/interaction_strength.py:59` passes `"H (0 additive .. 1 pure interaction)"`. A vertical colorbar
title in plotly renders horizontally above the bar, so 60 chars centred on a bar sitting at the right edge
of the plot area runs past the 40 px right margin and is clipped.

Fix: wrap the colorbar label through the existing `wrap_text_to_width` (`<br>` for plotly) and widen `r`
in proportion to the wrapped width, the same way `top_margin` already grows with the suptitle line count
(`plotly.py:344`).

---

### Checked and found already handled (not findings)

- Suptitle / caption / panel-title wrapping is measured against the real font and honours explicit `\n` (`_shared_helpers.py:268-306`, `matplotlib.py:50-80`, `plotly.py:96-102`).
- Suptitle-vs-panel-title collision: both backends reserve an explicit band (`matplotlib.py:204-232`, `plotly.py:342-344`).
- Heatmap per-cell text is suppressed past `_HEATMAP_CELL_TEXT_MAX = 400` cells on both backends.
- Multi-heatmap colorbars are pinned per subplot rather than stacked (`_plotly_heatmap.py:143-172`).
- Date axes: `epoch_ns_ticks` picks a format by span and emits 6 ticks (`_shared_helpers.py:366-404`); both backends rotate.
- Numeric tick density on histograms is capped via `MaxNLocator`/`LogLocator` (`matplotlib.py:388-401`).
- Static plotly exports park the legend below the plot area (`plotly.py:377-383`).

---

## Dispositions (parent, 2026-09-06)

**LBL-02 FIXED.** The tick budget is no longer the constant 8. ``ticks_that_fit(extent_in, n)`` derives it
from the axis's real extent (one label per 0.18 in at the 8 pt these axes use), keeping 8 as a floor for an
unmeasurable or tiny axis. Verified on the case the finding names: the 40-feature drift heatmap sizes its
figure to 10 x 14.3 in and now names all 40 rows on BOTH backends, where it named 8. The small heatmaps in
the same sweep (6 x 6 multilabel co-occurrence) are unchanged.

Plotly needed one more thing to get there: panels are drawn before ``update_layout`` sets width/height, so a
panel asking how much room it has read ``None`` and silently took the floor. The renderer now stamps the
requested size on the figure before drawing, at the same px-per-inch the final layout uses -- and that
constant, previously declared in ``plotly.py`` alone, moved to ``_shared_helpers`` so the two cannot drift.

**LBL-03 FIXED.** Heatmap tick labels are truncated on both backends, which the bar branches have always
done and this one never did. Noted while writing the test: ``truncate_bar_label`` keeps ``maxlen - 1``
characters and appends an ellipsis, so a truncated label is two characters OVER its nominal cap. That is
pre-existing and deliberate -- ``test_renderer_audit_regressions.py`` documents the allowance -- so it was
left alone rather than quietly redefined here.

Pinned by ``tests/reporting/test_heatmap_tick_budget.py``; three of its five tests fail with the budget
reverted to the fixed cap.

**LBL-05 FIXED.** ``truncate_bar_label`` gained a middle-ellipsis mode, and ``BarPanelSpec.label_keep_tail``
lets a builder say how many trailing characters must survive. ``slice_finder`` (20) and
``category_discriminability`` (18) set it, because their own titles tell the reader the label carries the
support and the ratio -- and the head-preserving cut deleted exactly that on a two-feature slice, which is
what slice_finder produces by default. Verified on a rendered figure: labels now read
``job_posted_at_day_of_year_component_0...14]  (n=5_000, 1.7x)``. Deliberately opt-in rather than global:
two existing tests pin that an ordinary truncated label ends in an ellipsis, and that shape is fine for a
label with no payload. Pinned by ``tests/reporting/test_label_payload_survives_truncation.py``; both
end-to-end parametrisations fail with the flag turned off.

**LBL-01 FIXED.** Node labels are capped and truncated on both backends. The cap is spent on the LARGEST
nodes, because ``node_size`` carries the importance the graph is drawn to show; every other node keeps its
marker, size, colour and its full name in the hover. Confirmed by rendering a friend-graph-shaped panel at
120 nodes (the family default is 200): 120 names of ~35 chars were a single illegible mat across the middle
of the panel, and 25 truncated ones read cleanly. A graph smaller than the cap still names everything.

## Found by rendering, not in the audit list (parent, 2026-09-06)

**NET-01 [P1] a network panel CRASHED matplotlib on its own defaults.** ``NetworkPanelSpec.colormap``
defaults to ``HEATMAP_GENERIC``, a sentinel token rather than a colormap name, and the matplotlib network
branch subscripted ``matplotlib.colormaps[p.colormap]`` directly instead of going through
``resolve_heatmap_cmap`` the way every other lookup in that renderer does. Any network panel that did not
name a colormap raised ``KeyError('__mlframe_default_heatmap__')``. The plotly twin survived, but through
its unknown-name fallback (Viridis, with a WARN) -- right by accident, and it would have diverged silently
the moment the default changed. Both branches resolve the sentinel now. Found while rendering the panel for
LBL-01, not by reading the spec; pinned by ``tests/reporting/test_network_labels_and_colormap.py``, whose
render tests fail against the raw subscript.
