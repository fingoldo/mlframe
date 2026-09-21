# reporting audit 2026-09-06 -- master tracker

Four read-only reports on `src/mlframe/reporting/`: text and layout collisions ([labels_and_layout.md](labels_and_layout.md)),
rendering speed ([render_performance.md](render_performance.md)), visual quality and encoding correctness
([visual_design.md](visual_design.md)), and a walkthrough of rendered PNGs ([visual_walkthrough.md](visual_walkthrough.md)).
Each report also carries the parent's dispositions and the defects found by rendering (NET-01, CARD-01, CARD-02,
VIS-27, VIS-28), which are tracked here as rows of the report that records them.

This tracker was written after the fact, on 2026-09-19. Every status below was checked against the code and the tests
in the tree on that date rather than copied from the report: where a report's own disposition was overtaken by later
work (PERF-04, LBL-10) the row says so, and the WALK findings, which the walkthrough never dispositioned, were checked
one by one.

Statuses: **RESOLVED** (fixed in code; the note names the test that pins it, or says that none was found),
**PARTIAL** (part of the finding is fixed; the note says what remains), **TODO** (open; the note says what remains),
**REJECTED** (measured and declined; the evidence is in the report), **NOT A DEFECT** (the claimed behaviour does not
occur). Test paths are under `tests/reporting/`.

## Summary

| File | Findings | RESOLVED | PARTIAL | TODO | REJECTED | NOT A DEFECT |
|---|---|---|---|---|---|---|
| `labels_and_layout.md` | 16 | 16 | 0 | 0 | 0 | 0 |
| `render_performance.md` | 18 | 16 | 0 | 0 | 2 | 0 |
| `visual_design.md` | 30 | 28 | 1 | 0 | 0 | 1 |
| `visual_walkthrough.md` | 11 | 6 | 4 | 0 | 0 | 1 |
| **Total** | **75** | **66** | **5** | **0** | **2** | **2** |

### `labels_and_layout.md`

| Status | Sev | ID | Finding | Evidence / what remains |
|---|---|---|---|---|
| **RESOLVED** | P1 | `LBL-01` | network node labels: no cap, no truncation, no collision avoidance | Labels capped (spent on the largest nodes) and truncated on both backends. `test_network_labels_and_colormap.py`. |
| **RESOLVED** | P1 | `LBL-02` | heatmap tick labels thinned to 8 regardless of panel size | `ticks_that_fit` (`renderers/_shared_helpers.py`) budgets from the measured axis extent, rotated pitch via `rotated_tick_pitch_in`. `test_heatmap_tick_budget.py`. |
| **RESOLVED** | P1 | `LBL-03` | heatmap row/col tick labels never truncated | Truncated on both backends. `test_heatmap_tick_budget.py`. |
| **RESOLVED** | P2 | `LBL-04` | plotly never thins horizontal bar labels | Same defect as VIS-04; both orientations thin on both backends. `test_horizontal_bar_tick_thinning.py`. |
| **RESOLVED** | P2 | `LBL-05` | head-preserving bar-label truncation cuts the payload at the tail | `BarPanelSpec.label_keep_tail` (`spec.py`) plus a middle-ellipsis mode; set by `slice_finder` and `category_discriminability`. `test_label_payload_survives_truncation.py`. |
| **RESOLVED** | P2 | `LBL-06` | bar tick thinning count-based while figure height is count-based | Budget measured against the axis (`label_width_pitch_in`, `rotated_tick_pitch_in`); flat 25/20 constants deleted. `test_bar_tick_budget_follows_the_figure.py`. |
| **RESOLVED** | P2 | `LBL-07` | spectral embedding labels every node on a fixed figure | Shared `non_colliding_label_indices` picker; nodes sized by degree. `test_labels_do_not_overprint.py`. |
| **RESOLVED** | P2 | `LBL-08` | multiclass overlay legends inside the axes with 13 long entries | Not via the proposed `legend_outside` (rendered and rejected); class names shortened keeping the tail, `legend_ncol=2` past ten entries. `test_multiclass_overlay_legends.py`. |
| **RESOLVED** | P2 | `LBL-09` | violin group labels neither truncated nor thinned | Tail-keeping truncation on both backends, thinning by panel width. `test_violin_group_labels.py`. |
| **RESOLVED** | P3 | `LBL-10` | free-text panel measures its width before layout | The report recorded the overflow as not reproduced (the panel grows); the milder effect it measured, text under-using the panel by about a fifth, was fixed later by rewrapping on the first draw (commit `fce53eccd`). `test_annotation_panel_rewrap.py`. |
| **RESOLVED** | P3 | `LBL-11` | scatter inline labels: no mutual collision avoidance | Fixed with LBL-07 through the same picker on both backends. `test_labels_do_not_overprint.py`. |
| **RESOLVED** | P3 | `LBL-12` | point-marker / vline annotations use fixed offsets | Label rows chosen by measurement against the axis range, stack hangs inside the plot area. `test_marker_labels_stagger_by_measurement.py`, `test_stacked_band_labels.py`. |
| **RESOLVED** | P3 | `LBL-13` | figures without a suptitle/caption get no layout engine | Engine enabled when a panel carries rotated ticks, horizontal bars, a colorbar label or a heatmap/violin. `test_layout_engine_and_figure_growth.py`. |
| **RESOLVED** | P3 | `LBL-14` | `segments_bar` grows only its width | Width capped at 16 in, height grows with the projected label length. `test_layout_engine_and_figure_growth.py`. |
| **RESOLVED** | P3 | `LBL-15` | long `colorbar_label` overflows on plotly | Wrapped on plotly; matplotlib draws it vertically and is pinned unchanged. `test_colorbar_label_wrapping.py`. |
| **RESOLVED** | P1 | `NET-01` | network panel crashed matplotlib on its default colormap sentinel | Both branches resolve the sentinel through `resolve_heatmap_cmap`. `test_network_labels_and_colormap.py`. |

### `render_performance.md`

| Status | Sev | ID | Finding | Evidence / what remains |
|---|---|---|---|---|
| **RESOLVED** | P1 | `PERF-01` | AP bootstrap runs 500 serial numpy passes | `prange` kernel; end to end 1.2-2.2x rather than the reported 12.9x because the index draw dominates. Making the draw cheaper would move every seed's interval and was left for its own decision. `test_ap_bootstrap_kernel.py`. |
| **RESOLVED** | P1 | `PERF-02` | depth-3 `DecisionTreeRegressor` is the largest cost | Split-gain histogram ranker `charts/_split_gain_ranking.py`, 1.39x end to end. `test_split_gain_ranking.py`. |
| **RESOLVED** | P1 | `PERF-03` | `z=p.matrix.tolist()` forces plotly's list validator | ndarrays passed at all three sites in `renderers/_plotly_heatmap.py`, output byte-identical. No dedicated regression test found. |
| **RESOLVED** | P1 | `PERF-04` | PSI computed for every column, most discarded | The report says NOT taken pending proof that the kept feature set is unchanged; that proof was then written: commit `9a19dcfc5` (2026-09-07) screens on a stratified row sample (`PSI_SCREEN_OVERSAMPLE` in `charts/drift.py`) and recomputes survivors exactly. `test_psi_feature_screen.py` asserts the same drawn feature set as the exact ranking. |
| **RESOLVED** | P2 | `PERF-05` | Python append loop over every candidate cell | Per-combo numpy blocks, identical tables, 1.04-1.19x. No dedicated regression test named. |
| **RESOLVED** | P2 | `PERF-06` | Theil-Sen trend fit recomputed per backend | Memoised on array identity with weak-reference confirmation. `test_trend_fit_memoised.py`. |
| **RESOLVED** | P2 | `PERF-07` | same frame densified twice per run | `_diagnostics_prep.shared_error_prep` plus a memoised densify, both weak-referenced. `test_shared_error_prep.py`. |
| **RESOLVED** | P2 | `PERF-08` | per-row Python scan and slow string encode per object column | Already fixed earlier in the wave: `ordinal_codes` takes `pd.factorize(sort=True)` first. Exercised by `test_worst_k_membership_and_matrix_plane.py`. |
| **RESOLVED** | P2 | `PERF-09` | `_stratified_subsample` is O(n * K) | Not the proposed int64 argsort (measured 5x slower at K=10); sort key narrowed by `_narrowed` (`charts/multiclass.py`). `test_stratified_subsample_grouping.py`. |
| **REJECTED** | P3 | `PERF-10` | `argsort(kind="stable")` costs 2.4x a quicksort | Gated quicksort measured a 1.6x regression on quantised scores; stability is load-bearing for the AP label sequence. Bench kept at `src/mlframe/reporting/_benchmarks/bench_score_sort_tie_gate.py`. The tie observation it raised was fixed separately (`gain_curve_points`, `test_gain_curve_ties.py`). |
| **RESOLVED** | P3 | `PERF-11` | `np.unique(ys).size` to test for a constant score | `ys.min() == ys.max()` in `charts/binary.py`, 21x at n=2M. No dedicated regression test found. |
| **RESOLVED** | P3 | `PERF-12` | per-span / per-vline plotly calls | Batched shapes and annotations, 69x at 100 bands. `test_band_and_marker_batching.py`. |
| **RESOLVED** | P3 | `PERF-13` | uncapped per-edge / per-node `ax.annotate` | matplotlib honours `_NETWORK_MAX_ARROWS` imported from `renderers/_plotly_network.py`. No dedicated regression test found. |
| **RESOLVED** | P3 | `PERF-14` | `np.column_stack` contradicts "without a full frame copy" | Preallocated F-order plane, peak 144 MB -> 73 MB. `test_worst_k_membership_and_matrix_plane.py`. |
| **RESOLVED** | P3 | `PERF-15` | one full-length boolean mask per group | Factorised grouping in `charts/_grouping.py`, 2.4-3.8x. `test_label_grouping.py`. |
| **RESOLVED** | P3 | `PERF-16` | Python set of coordinate tuples plus a scan per worst-K row | One `lexsort` + `searchsorted`, 2.0x. `test_worst_k_membership_and_matrix_plane.py`. |
| **RESOLVED** | P3 | `PERF-17` | three `.iloc[i]` lookups per row | One `zip` over `to_numpy()` reads in `charts/slice_finder.py`. No dedicated regression test found; negligible at `top_k=7`. |
| **REJECTED** | P3 | `PERF-18` | 200 separate `np.quantile` calls | Batched quantile measured 0.92x (a regression); row sampling would move the printed slice bounds by up to 1.64. |

### `visual_design.md`

| Status | Sev | ID | Finding | Evidence / what remains |
|---|---|---|---|---|
| **RESOLVED** | P1 | `VIS-01` | correlation heatmap diverging scale not pinned at 0 | `color_vmin`/`color_vmax` on `HeatmapPanelSpec`, pinned to [-1, 1]; cell text samples the same scale. `test_heatmap_scale_and_notice.py`. |
| **RESOLVED** | P1 | `VIS-02` | plotly horizontal bar axis titles swapped | Value/category titles agree across backends. `test_horizontal_bar_axis_titles.py`. |
| **RESOLVED** | P1 | `VIS-03` | mangled escape printed on the SHAP notice | Real newline (`charts/shap_panels.py`). `test_heatmap_scale_and_notice.py` reads the text off the figure. |
| **RESOLVED** | P2 | `VIS-04` | plotly horizontal bars not thinned | Same policy as matplotlib. `test_horizontal_bar_tick_thinning.py`. |
| **PARTIAL** | P2 | `VIS-05` | separability classes drawn on a sequential colourbar | Classes remapped to contiguous codes on `tab10` with bounds pinned to `-0.5 .. n_classes - 0.5` (`charts/engineered_separability.py`); `test_separability_class_colours.py`. Remains: a discrete per-class legend in place of the colourbar, which needs a new `ScatterPanelSpec` field and support in both renderers. |
| **NOT A DEFECT** | P2 | `VIS-06` | traffic-light verdict encoded in colour alone | Rendered: the verdict is written in text three times on the model card; `fairness_calibration` colours encode the group, not the verdict. |
| **RESOLVED** | P2 | `VIS-07` | plotly inline labels lack the contrast halo and edge flip | Both protections ported. `test_inline_label_contrast.py`. |
| **RESOLVED** | P2 | `VIS-08` | plotly data-hull window applied on non-square panels | Gated on `equal_aspect`; explicit limits honoured. `test_scatter_backend_parity.py`. |
| **RESOLVED** | P2 | `VIS-09` | all-low-evidence panel hides the hollow markers | Guard dropped on markers and error bars. `test_scatter_backend_parity.py`. |
| **RESOLVED** | P2 | `VIS-10` | ACF/PACF band drawn on one side only | `BarPanelSpec.hline_symmetric`, set by `charts/temporal.py`. `test_acf_symmetric_band.py`. |
| **RESOLVED** | P2 | `VIS-11` | confusion margin colours differ per backend | `CONFUSION_COL_MARGIN` / `CONFUSION_ROW_MARGIN` in `colors.py`. `test_confusion_margin_colours.py`. |
| **RESOLVED** | P2 | `VIS-12` | private palette copies in two charts | `colors.line_color` used; styles switch only past the palette wrap. `test_group_palette_is_shared.py`. |
| **RESOLVED** | P2 | `VIS-13` | model-card headline bar: shared 0.5 line, raw values unrecoverable | 0.5 line removed, raw value labelled per bar. Sorting and per-metric colours declined deliberately, reasons in the report. `test_model_card_headline_bar.py`. |
| **RESOLVED** | P3 | `VIS-14` | all-NaN histogram blank on plotly | Same notice on both backends. `test_histogram_all_nonfinite_notice.py`. |
| **RESOLVED** | P3 | `VIS-15` | log histogram ticks differ; empty bins vanish silently | `dtick="D2"` on plotly, and both backends name the undrawable empty bins. `test_log_scale_histogram.py`. |
| **RESOLVED** | P3 | `VIS-16` | caption and panel-title typography differ | Shared constants in `renderers/_shared_helpers.py`. `test_typography_is_shared.py`. |
| **RESOLVED** | P3 | `VIS-17` | PSI threshold contours unlabelled on plotly | Level shown inline, wording in the trace name. `test_backend_grid_and_contour_parity.py`. |
| **RESOLVED** | P3 | `VIS-18` | overlay band colour hardcoded on plotly | `_rgba(OVERLAY_LINE, 0.18)` in `renderers/_plotly_scatter.py`. `test_backend_grid_and_contour_parity.py`. |
| **RESOLVED** | P3 | `VIS-19` | gridlines differ between backends | No category-axis grid; `_GRID_COLOR` pinned. `test_backend_grid_and_contour_parity.py`. |
| **RESOLVED** | P3 | `VIS-20` | worst-K ring fixed-size on plotly | Sized from the point area. `test_bar_violin_scatter_parity.py`. |
| **RESOLVED** | P3 | `VIS-21` | Path-B charts ignore the report DPI and pad | One `_save_figure` with `pad_inches=0.15`, `plot_dpi` forwarded. `test_path_b_save_conventions.py`. |
| **RESOLVED** | P3 | `VIS-22` | default figure sizes vary widely | Named `FIGSIZE_*` constants in `spec.py` used by 14 chart modules (commit `ce9a21359`); data-driven sizes deliberately left computed. No dedicated regression test found. |
| **RESOLVED** | P3 | `VIS-23` | ACF and PACF drawn in different colours | Both take `BAR_PRIMARY`. `test_bar_violin_scatter_parity.py::test_acf_and_pacf_are_the_same_colour`. |
| **RESOLVED** | P3 | `VIS-24` | NaN subgroup metrics lead the worst-first chart | Non-finite metrics filtered where the column is read, count named in the caption. `test_bar_violin_scatter_parity.py::test_unmeasurable_subgroups_do_not_lead_the_worst_first_chart`. |
| **RESOLVED** | P3 | `VIS-25` | plotly violin box claims 5th/95th whiskers | Explicit `go.Box` with `lowerfence`/`upperfence`. `test_bar_violin_scatter_parity.py::test_the_violin_box_whiskers_at_the_5th_and_95th_percentiles`. |
| **RESOLVED** | P3 | `VIS-26` | plotly bar reference label floats in a corner | Placed at the line's x inside the plot area. `test_bar_violin_scatter_parity.py::test_the_vertical_reference_label_sits_at_its_own_line_and_below_the_title`. |
| **RESOLVED** | - | `VIS-27` | colorbar tick labels land on the neighbour's axis title | Column gap derived from the colorbar gutter. `test_backend_grid_and_contour_parity.py`. |
| **RESOLVED** | - | `VIS-28` | PSI contour label drawn across the cell values | Inline label suppressed when cells carry text; legend placed in the calmest quadrant. `test_heatmap_contour_labels.py`. |
| **RESOLVED** | P1 | `CARD-01` | model-card mini gain panel crushed to a sliver | Column ratios (1.3, 1.3, 1.0), general 5% width invariant. `test_model_card_layout.py`. |
| **RESOLVED** | P3 | `CARD-02` | stray hash printed in the card header | Removed at both call sites. No dedicated regression test found. |

### `visual_walkthrough.md`

| Status | Sev | ID | Finding | Evidence / what remains |
|---|---|---|---|---|
| **PARTIAL** | P1 | `WALK-01` | LTR rotated tick label runs onto the panel below | The CI left the tick labels (now an error bar; `test_ltr_ndcg_bootstrap_ci.py::test_ndcg_by_qsize_title_and_bins_carry_ci`) and rotated labels get measured pitch and the layout engine (LBL-06, LBL-13). Remains: the NDCG_DIST instance (`all queries (n=300)` against its own axis title) has not been re-rendered, and no disposition was recorded. |
| **PARTIAL** | P1 | `WALK-02` | a single category drawn as a full-panel-width bar | matplotlib pads a sparse category axis to four slots (`_pad_sparse_category_axis`, `renderers/matplotlib.py`). Remains: plotly has no equivalent, and no test pins either backend. |
| **PARTIAL** | P1 | `WALK-03` | legends drawn inside the axes cover the data | matplotlib promotes an overflowing legend outside (`_legend_overflows` / `_place_legend`), but only on the rightmost panel with no secondary axis; plotly moves it only when a builder sets `legend_outside`; LBL-08 fixed the multiclass overlays. Remains: no test pins the automatic promotion, and the four named panels (multilabel CALIB_GRID, regression decile, binary SCORE_DIST, model_comparison overlay) have not been re-rendered. |
| **RESOLVED** | P1 | `WALK-04` | operating-point text duplicates the legend and overflows | The label rides the legend entry only, on both backends (`renderers/matplotlib.py` point-marker loop, `renderers/_plotly_line.py` marker traces). No test asserts the absence of the second copy. |
| **PARTIAL** | P2 | `WALK-05` | statistic-bearing panel titles wrap to two or three lines | The two titles the walkthrough named are cut to one line each while keeping every statistic: the reliability title drops the axis restatement, the interval-width title drops the span definition the xlabel already carries. Remains: a renderer-level length budget, so a future statistic-bearing title cannot wrap again. |
| **RESOLVED** | P2 | `WALK-06` | Spearman heatmap reads as anti-correlation | Same defect as VIS-01. `test_heatmap_scale_and_notice.py`. |
| **RESOLVED** | P2 | `WALK-07` | an empty panel is captioned for a metric never chosen | `_leaderboard_panel` now distinguishes the two cases: a named metric missing everywhere is still quoted so the caller can fix the call, while an empty headline (no model carries metrics at all) says exactly that. Tests: `tests/reporting/test_charts_model_comparison.py::test_metricless_models_say_so_instead_of_quoting_an_empty_metric` and `::test_a_named_metric_missing_everywhere_is_still_named`. |
| **RESOLVED** | P2 | `WALK-08` | unequal grid cells and a ragged last row | The matplotlib renderer spans a lone trailing panel across its row (`gs[r, :]`), so an odd panel count leaves no empty cell; full rows keep equal widths. Test: `tests/reporting/test_layout_engine_and_figure_growth.py::test_a_lone_trailing_panel_spans_its_row`. The plotly renderer still pads that row. |
| **RESOLVED** | P2 | `WALK-09` | heatmap axis labels full-length on both axes | Same defect as LBL-03. `test_heatmap_tick_budget.py`. |
| **NOT A DEFECT** | P3 | `WALK-10` | red/green is the only channel on the WoE bars | Confirmed in `charts/category_discriminability.py`: the bars are horizontal with a reference line at WoE = 0, so the sign is carried by which side of the line a bar extends to, and the green/red fill only reinforces it. Colour is not the only channel. |
| **RESOLVED** | P3 | `WALK-11` | suptitle and single panel title repeat each other | The panel title no longer repeats the figure's suptitle: it reads `Signed |WoE|: green => y=1, red => y=0; label = support n + P(y=1|level)`, and the empty-state panel is titled `Nothing cleared min_support`. |
