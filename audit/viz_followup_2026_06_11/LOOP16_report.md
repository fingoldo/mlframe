# LOOP16 — bootstrap confidence band around the smoothed isotonic reliability curve

Axis: ACCURACY. Status: **RESOLVED**.

## Goal
Per-bin Wilson CIs already say "is THIS bin's deviation real". They do not answer "is the calibration curve's
deviation from the perfect-fit diagonal statistically real OVERALL". This adds a bootstrap 95% confidence band around
the smoothed isotonic reliability curve plus a derived significance read so a user can judge that at a glance.

## Method
`bootstrap_reliability_band(y_score, y_true)` in `src/mlframe/reporting/charts/calibration.py`:
- Drop non-finite rows; guard single-class / all-equal-score / <50 finite rows -> return `None` (band omitted, curve kept).
- Subsample to `_BAND_MAX_ROWS=50_000` (band width is n-driven; the cap bounds the B-refit cost, a smaller n only widens the band conservatively).
- One RNG (`random_state`) reused for the subsample AND a single vectorised `rng.integers(0,n,size=(B,n))` resample-index draw.
- For each of `B=150` resamples, refit `IsotonicRegression` and evaluate on the shared `[smin,smax]` grid (100 pts). A degenerate resample (all-one-class / all-equal score) reuses the full-sample fit so it cannot flatten the percentile band.
- Per grid point: 2.5/97.5 percentiles -> `(grid, lower, upper)`.
- **Significant-fraction** = share of grid where the band fully excludes the diagonal: `(lower > grid) | (upper < grid)`.
- Surfaced as `ScatterPanelSpec.overlay_band` (new field) rendered as a shaded fill by both matplotlib (`fill_between`) and plotly (`tonexty`), under the purple smoothed curve; annotation `"miscal. significant on X% of range"` appended to the scatter title. Default-on (`reliability_band=True`); set `reliability_band=False` to disable.

## biz_value (n=40000)
- Perfectly-calibrated synthetic: **significant-fraction = 0.07** (band contains the diagonal across ~93% of range).
- Clearly over-confident synthetic (logit x2): **significant-fraction = 0.91** (band excludes the diagonal across most of the range).
- Clean separation: 0.91 - 0.07 = 0.84 (>> the 0.30 floor). Asserts cal < 0.10, mis > 0.40, gap > 0.30.
- **Narrows with n**: mean band width 0.129 (n=2000) -> 0.045 (n=40000); large-n band is < 0.7x the small-n width.
- Band contains the point-estimate curve (verified per grid point).
- Deterministic under a fixed seed.

## cProfile (warm, 50k cap, n_boot=150)
- ~1.39s wall. Hotspot is `sklearn.isotonic._build_y` (the B isotonic refits); PAVA core is ~0.06s, the rest is sklearn validation overhead.
- Sweep that fixed the knob: at the 50k cap, n_boot=250 -> 2.23s (over the ~1.5s target); n_boot=150 -> 1.39s with identical separation (mis 0.95 / cal 0.07). Picked B=150. Lowering the row cap also helps but trades band tightness, so the cap stays at 50k and B is the lever.

## Tests
`tests/reporting/test_calibration_bootstrap_band.py` (12 tests, all green): unit (band in spec, contains curve, sig-fraction computed, annotation in title, toggle-off keeps curve, three degenerate-omits, determinism), biz_value (calibrated-vs-miscalibrated sig-fraction split + narrows-with-n), cProfile-bounded. Existing `test_calibration_smoothed_overlay.py` + the wider reporting suite (`test_charts`, `test_reliability_overlay_spec`, `test_calibration_debiased_ece`, `test_edge_case_degradation`) stay green.

## Gallery
Re-rendered `docs/gallery/binary/calibration_reliability.png` (now shows the shaded band + significance annotation).

## Files
- `src/mlframe/reporting/charts/calibration.py` (helper + wiring; 600 LOC, under the 1000 limit — no carve needed)
- `src/mlframe/reporting/spec.py` (`ScatterPanelSpec.overlay_band` field)
- `src/mlframe/reporting/renderers/matplotlib.py`, `.../plotly.py` (band render)
- `scripts/render_gallery.py` (description), `tests/reporting/test_calibration_bootstrap_band.py`
- `docs/gallery/binary/calibration_reliability.png`

Commit hash: see git log (committed on branch `viz-loop10-confusion-margins`).
