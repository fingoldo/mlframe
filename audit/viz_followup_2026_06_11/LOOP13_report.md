# LOOP13 — decision-threshold operating-point marker on ROC + PR

Axis: UI/FUNCTIONALITY. Status: RESOLVED.

## Design

The ROC/PR curves showed the whole threshold sweep but not WHERE the user operates. Added a distinct red star marker at the
chosen decision threshold on both curves, annotated with the threshold value + operating-point metrics:

- ROC marker at `(FPR, TPR)`, label `thr=0.50: TPR=0.81 FPR=0.12`.
- PR marker at `(recall, precision)`, label `thr=0.50: R=0.81 P=0.74`.

Mechanism:

- New `LinePanelSpec.point_markers` field: tuple of `(x, y, label, color, marker_symbol)`. Generic standalone-point overlay,
  rendered by BOTH the matplotlib renderer (`ax.plot` star + offset annotation) and the plotly renderer (`Scatter` markers+text,
  matplotlib->plotly symbol map). One marker per curve, drawn at `zorder=6` on top of the line, label rides the legend.
- `binary._operating_point(sort, threshold)` derives `(FPR, TPR, recall, precision)` from the SAME shared descending-score sort
  the ROC/PR/threshold panels already build. The threshold's rank is located via `searchsorted` on the existing sorted scores
  (`np.searchsorted(-scores_desc, -thr, side="right")` counts `score >= thr`) — O(log n), NO new full-n pass. TP=cum_tp[k-1],
  FP=cum_fp[k-1] read directly off the precomputed cumulative counts.
- `compose_binary_figure(..., operating_point=True)` (default ON, per "enable corrective mechanisms by default") + `threshold`
  (already a param, default 0.5) wire the marker into ROC and PR.

Edge-safe (iter-1 guard style): `_operating_point` returns None — marker omitted, curve kept — for single-class
(`n_pos==0 or n_neg==0`), empty input, or a threshold above every score (`k==0`, flags nobody).

## biz_value: marker == confusion-derived rates

`test_roc_marker_coords_equal_confusion_rates` / `test_pr_marker_coords_equal_confusion_rates` (parametrized thr in {0.3,0.5,0.7})
assert the marker coords equal a brute-force confusion-matrix computation of `(FPR,TPR)` / `(recall,precision)` at the exact
threshold, `abs=1e-9`. PASS at all three thresholds for both curves.

`test_operating_point_uses_searchsorted_not_full_pass` pins exactness on heavy-tied scores (the rank-location must be correct
under ties, where an ascending/descending mixup or off-by-one would diverge): PASS at thr in {0.9,0.5,0.2,0.1}.

Threshold-move check (`test_biz_val_operating_point_moves_monotonically_with_threshold`, n=8000 sep=2.0): recall at
thresholds [0.2,0.4,0.6,0.8] is monotone non-increasing AND strictly drops from thr=0.2 to thr=0.8 — a higher threshold flags
fewer positives -> lower recall, exactly as expected. PASS.

## cProfile (reuses the sweep)

`test_operating_point_cprofile_bounded_reuses_sweep`: ROC+PR figure build at n=1e6 completes in `total_tt < 2.0s`. The
operating point adds only one `searchsorted` (O(log n)) per curve on the already-built sort, so it contributes no measurable
overhead vs the irreducible single argsort the panels already pay. PASS.

## Tests

`tests/reporting/test_charts_binary.py` — 63 passed (added 13 operating-point tests: marker present + label format on ROC/PR,
disable toggle, confusion-rate equality x3 thr x2 curves, threshold-above-all omit, single-class None, tied-score searchsorted
exactness, monotone-move biz_value, matplotlib end-to-end render, cProfile bound).
`tests/reporting/test_renderers.py` — 18 passed (new `point_markers` spec field renders cleanly on both backends).

## Files

- src/mlframe/reporting/spec.py — `LinePanelSpec.point_markers`
- src/mlframe/reporting/renderers/matplotlib.py — `_line` marker draw + legend trigger
- src/mlframe/reporting/renderers/plotly.py — `_line` marker draw (symbol map)
- src/mlframe/reporting/charts/binary.py — `_operating_point`, `_operating_point_label`, ROC/PR marker wiring, composer toggle
- tests/reporting/test_charts_binary.py — 13 new tests
- scripts/render_gallery.py — unchanged (binary_full already renders ROC+PR; markers now appear)

## Gallery

docs/gallery/binary/binary_full.png — red star markers visible on ROC (top-left) and PR (top-right) with threshold+metric labels.

## Commit

(filled below)
