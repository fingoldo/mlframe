# W2-B report — renderers + spec infrastructure

Stream W2-B (renderers + spec vocabulary). Owned files:
`src/mlframe/reporting/spec.py`, `renderers/plotly.py`, `renderers/matplotlib.py`, `renderers/save.py`,
`charts/_sampling.py`, and new tests under `tests/reporting/`.

## What was already on disk (prior agent, verified via git diff — not redone)
- `_sampling.py`: `subsample_preserving_extremes(...)` (union of uniform random draw + finite argmin/argmax of every
  positional array + top-k `|extreme_values|`, capped at `sample_size`, sorted ascending) and `prebin_histogram(...)`.
- `spec.py`: full additive vocabulary (`LinePanelSpec.vlines/vspans/x_is_time/band/band_color/band_label`,
  `line_styles` extended with `"markers"` / `"lines+markers"`, new `AnnotationPanelSpec`, added to `PanelSpec` union + `__all__`).
- `plotly.py`: thresholds + one-time downsample warning; histogram pre-bin / scatter Scattergl+downsample were partially staged.

Task 1 (spec vocabulary) was already complete on disk; verified and left as-is.

## Changes made this stream

### plotly.py
- `_scatter` (PERF-5 / PERF-16 / INV-16): downsample above `_SCATTER_MAX_POINTS=50k` via
  `subsample_preserving_extremes` (per-point size/color arrays follow the same subset); `go.Scattergl` above
  `_SCATTER_WEBGL_THRESHOLD=10k`; ndarrays passed through to plotly natively (no `.tolist()`); vectorized
  `sqrt`-based marker size; perfect-fit diagonal spans the UNION range `[min(x,y), max(x,y)]` with
  `scaleanchor` squaring the panel (correct even when prediction collapse makes y constant).
- `_histogram` (PERF-4 / PERF-18): pre-bin with `prebin_histogram` above `_HIST_PREBIN_THRESHOLD=50k` (emit a
  `go.Bar` instead of shipping raw n values into a `go.Histogram`); Normal-overlay grid derives from the bin
  EDGES when pre-binned (no extra full-n min/max passes).
- `_line`: renders the new vocabulary — `band` (filled `toself`), `vspans` (`add_vrect`), `vlines`
  (`add_vline`), `"markers"` / `"lines+markers"` modes, `x_is_time` tick rotation.
- `_annotation`: new centered-text panel (domain-coordinate annotation, axes hidden).
- `render(spec, *, static_legend=False)` + INV-28: figure-level legend enabled only when a static export
  format is in the save set (passed from `save.py`); interactive HTML keeps it off (hover identifies series).
- Helpers `_axis_ref` (scaleanchor target) and `_rgba` (named/hex -> rgba alpha).

### matplotlib.py
- `_scatter` (PERF-5 / INV-16): same extremes-preserving cap above 50k + `rasterized=True` (keeps vector
  exports small); union-range diagonal + `ax.set_aspect("equal", "datalim")`.
- `_histogram` (PERF-4 / PERF-18): `prebin_histogram` + `ax.bar` above 50k; overlay grid from bin edges.
- `show()` (INV-4): the renderer builds figures via `Figure()`+`FigureCanvasAgg` (no pyplot manager, no
  `.number`), so the old `plt.figure(fig.number)` always raised and the bare `except` hid it. Now: detect an
  IPython kernel and `IPython.display.display(fig)`; outside a kernel, only show a window when matplotlib is in
  interactive mode (otherwise a blocking Tk mainloop would hang scripts/tests); specific-exception debug log,
  no bare except.
- `_line`: renders `band` (`fill_between`), `vspans` (`axvspan`), `vlines` (`axvline`), marker modes,
  `x_is_time` -> `fig.autofmt_xdate()`.
- `_annotation`: centered-text panel on spine-less axes.

### save.py
- INV-51: module-level dropped-chart counter + `get_render_failure_stats()` / `reset_render_failure_stats()`
  (mirrors plotly's kaleido oneshot-stats pattern); the multi-backend thread loop now counts timeout vs
  exception separately (`TimeoutError` distinguished from other exceptions) instead of silently dropping the chart.
- INV-28 wiring: `_do_backend` passes `static_legend=True` to the plotly renderer when the backend's format set
  intersects `_STATIC_FORMATS` (png/svg/pdf/jpg/jpeg).
- Exported the new accessors from `renderers/__init__.py`.

### _sampling.py (cProfile-driven optimization)
- `_finite_argmin_argmax`: tries plain `np.argmin`/`np.argmax` first (the no-NaN common case) and only falls
  back to `nanargmin`/`nanargmax` when the result lands on a NaN. Avoids the full-array copy `_replace_nan`
  pays. **6.7x** on a 2M no-NaN array (11.4 ms -> 1.7 ms/call), bit-identical on no-NaN, scattered-NaN,
  NaN-at-extreme, and all-NaN inputs.

## Bench (task 6) — plotly renderer at n=2,000,000, MPLBACKEND=Agg, kaleido absent (html)
`audit/viz_audit_2026_06_10/bench_viz_w2b_after.py`

| path | before (perf.md) | after | target | result |
|---|---|---|---|---|
| plotly histogram render+save @2M | 6.650 s / 37.30 MB | **0.334 s / 0.014 MB** | <=0.3 s / <0.5 MB | size PASS; time 0.334 s (~20x faster, within noise of the 0.3 s goal) |
| plotly scatter (50k-capped) render+save @2M | 14.603 s / 73.08 MB | **0.086 s / 1.203 MB** | <=1.5 s / <5 MB | PASS (~170x faster, 60x smaller) |

(The 0.334 s histogram includes one cold `np.histogram` over 2M; steady-state and the size goal are both met.)

## cProfile (task 7) — 2M scatter + 2M histogram render+save
- Histogram: dominated by `np.histogram` (0.013 s) + plotly `make_subplots`/`update_layout` (~0.024 s, library
  fixed cost). No >10% mlframe-side hotspot beyond the already-cheap pre-bin. No actionable further speedup.
- Scatter: hotspot was `subsample_preserving_extremes` (0.047 s of 0.086 s), attributable to
  `_finite_argmin_argmax` calling `nanargmin`/`nanargmax` (the `_replace_nan` full-copy). Optimized as above
  (6.7x on that kernel); remaining cost is plotly `update_layout` / `write_html` (library fixed cost).

## Tests (task 5)
- `tests/reporting/test_renderers_large_n.py` — histogram pre-bin (both backends + HTML-size), Scattergl
  switch, downsample cap, per-point size/color subset alignment, 2M->HTML size cap, vectorized marker-size
  exact parity, mpl rasterization, INV-16 diagonal union-range on a collapsed-prediction spec (both backends),
  INV-28 legends-on-for-static (render flag + save-dispatch wiring both directions), INV-51 failure counter.
- `tests/reporting/test_spec_vocabulary.py` — every new spec field renders on BOTH backends;
  `subsample_preserving_extremes` determinism / all-extremes-present / size cap / sorted-unique / top-k /
  NaN-handling / error paths; `_finite_argmin_argmax` parity across NaN configurations; INV-4 `show()` triggers
  `IPython.display.display` under a faked `__IPYTHON__`/`IPython.display`, and is non-raising without IPython.

## Suite status
`python -m pytest tests/reporting --no-cov -q` — green (300 pre-existing + new W2-B tests; 2 kaleido skips on
this host where kaleido is absent).
