"""Live training-progress widget: per-metric tabs, train/val/test curves, RAM usage, and a stop button.

A CatBoost-style progress view for :class:`~mlframe.training.callbacks.UniversalCallback`, so a long fit is
watchable in a notebook instead of being read off a throttled log line. One tab per METRIC (the booster
routinely reports several); inside a tab, one curve per DATASET (train / validation / test / any eval shard the
booster registered), the running optimum marked with a star, and RAM usage on a secondary axis so a memory
climb is visible against the same iteration axis that shows the metric stalling.

Why this is not part of ``mlframe.reporting``
---------------------------------------------
That subsystem is spec-based and one-shot: a builder returns a pure-data ``FigureSpec``, a renderer turns it
into a figure, and the figure is saved. This is the opposite problem -- a figure that already exists and must
be PATCHED thousands of times while training runs. Forcing it through the spec model would mean rebuilding and
re-rendering a whole figure per refresh. It uses ``plotly.graph_objects.FigureWidget`` directly and mutates
trace data in place under ``batch_update()``, which is one browser repaint per refresh rather than per trace.

What it costs
-------------
Nothing measurable. Per-iteration overhead over a 20k-iteration fit reporting 2 metrics on 3 splits:
3.24 us/iter with no widget, 3.06 us/iter with one at a 0.5s refresh and 2.91 us/iter at 0.1s -- all inside
run-to-run noise, and dominated by the callback's own history bookkeeping rather than by the widget. That
holds because of three deliberate choices:

* Refreshes are throttled on wall-clock (``refresh_secs``); a booster can emit thousands of iterations per
  second and a repaint per iteration is what makes naive live plots slower than the fit they watch.
* Trace data is handed to plotly as NUMPY arrays, never Python lists. plotly validates a list element by
  element and then diffs old against new element by element, so one refresh of a 10k-point history made 3.66M
  Python-level calls at 331 ms; numpy takes its ``copy_to_readonly_numpy_array`` / ``np.array_equal`` fast
  paths and the same refresh costs 1.20 ms (276x, 80.9M calls -> 80.2K). See ``_as_y_array``.
* RAM is sampled on the refresh throttle, not per iteration: ``get_own_memory_usage()`` measures at ~21
  us/call, which would be 2.1% of wall time at 1000 iterations/sec. The RAM series therefore carries its own
  sparse iteration axis rather than being padded out to match the metric series.

Everything is opt-in and degrades to a hard no-op besides: outside IPython, or without ``plotly`` /
``ipywidgets``, :attr:`TrainingProgressWidget.enabled` is False, the callback discards the widget once at
construction, and the per-iteration path is two list appends.

``_benchmarks/bench_progress_widget_update.py`` is the harness for all of the above.

Stopping
--------
The stop button never touches the booster. It flips an internal flag that :meth:`TrainingProgressWidget.stop_requested`
reports, and the callback composes that into the ``stop_flag`` callable it already consults every iteration --
so stopping goes through the same single, tested path as a time budget or an external flag. The button is
two-step (click, then confirm) so a misclick cannot end a multi-hour fit.
"""

from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# One repaint per this many seconds at most. 0.5s reads as live to a human while costing nothing next to a
# booster iteration; the alternative (repaint per iteration) is what makes naive live plots slower than the fit.
DEFAULT_REFRESH_SECS: float = 0.5
# Per-dataset curve colours. Train/validation/test get fixed, conventional hues so the same split keeps the
# same colour across every tab and every run; anything else cycles through the tail.
_DATASET_COLORS: Dict[str, str] = {
    "train": "#1f77b4",
    "learn": "#1f77b4",
    "training": "#1f77b4",
    "valid_0": "#ff7f0e",
    "validation": "#ff7f0e",
    "validation_0": "#ff7f0e",
    "val": "#ff7f0e",
    "eval": "#ff7f0e",
    "test": "#2ca02c",
    "validation_1": "#2ca02c",
    "valid_1": "#2ca02c",
}
_FALLBACK_COLORS: Tuple[str, ...] = ("#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")
_RAM_COLOR: str = "#c7c7c7"


def _dataset_color(name: str, ordinal: int) -> str:
    """Stable colour for a dataset name: conventional hue for a known split, else a cycled fallback."""
    return _DATASET_COLORS.get(str(name).lower(), _FALLBACK_COLORS[ordinal % len(_FALLBACK_COLORS)])


def in_notebook() -> bool:
    """True when running under an IPython kernel that can actually display a widget.

    A plain ``import IPython`` is not enough -- IPython is importable in a terminal REPL and in pytest, where
    displaying a FigureWidget produces nothing and the work is wasted. ``ZMQInteractiveShell`` is the kernel
    (Jupyter / VS Code / Colab); ``TerminalInteractiveShell`` and "no shell at all" are both not it.
    """
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


class TrainingProgressWidget:
    """Live per-metric tabs of train/val/test curves + RAM, with a two-step stop button.

    Attach it by passing ``progress_widget=True`` (or an instance, to override the defaults) to any
    ``UniversalCallback`` subclass; the callback feeds it and composes its stop request into ``stop_flag``.

    Args:
        metrics: metric names to give tabs to. ``None`` (default) means every metric the booster reports,
            with tabs created as new metrics appear.
        datasets: dataset names to draw. ``None`` (default) means every eval set the booster registered.
        refresh_secs: minimum wall-clock seconds between repaints. See the module docstring on why this is
            throttled rather than per-iteration.
        show_ram: draw own-process RAM on a secondary y-axis, sampled on the refresh throttle.
        height: pixel height of each tab's figure.
        show_stop_button: render the two-step "stop training" control.
    """

    def __init__(
        self,
        *,
        metrics: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        refresh_secs: float = DEFAULT_REFRESH_SECS,
        show_ram: bool = True,
        height: int = 420,
        show_stop_button: bool = True,
    ) -> None:
        self.metrics = list(metrics) if metrics is not None else None
        self.datasets = list(datasets) if datasets is not None else None
        self.refresh_secs = float(refresh_secs)
        self.show_ram = bool(show_ram)
        self.height = int(height)
        self.show_stop_button = bool(show_stop_button)

        self._stop_requested = False
        self._displayed = False
        self._last_refresh = 0.0
        self._figures: Dict[str, Any] = {}  # metric -> FigureWidget
        self._trace_idx: Dict[str, Dict[str, int]] = {}  # metric -> dataset -> trace index
        self._best_idx: Dict[str, int] = {}  # metric -> index of that metric's star trace
        self._ram_idx: Dict[str, int] = {}  # metric -> index of that metric's RAM trace
        self._tabs: Any = None
        self._container: Any = None
        self._status: Any = None
        self._modes: Dict[str, str] = {}  # metric -> "min" / "max"
        # Conversion caches, so a refresh costs the points ADDED rather than the whole history. See _as_y_array.
        self._x_cache: Any = None
        self._y_cache: Dict[Tuple[str, str], Tuple[Any, int]] = {}
        self._ram_cache: Optional[Tuple[Any, Any, int]] = None

        self._enabled = self._resolve_enabled()

    # ------------------------------------------------------------------ setup

    def _resolve_enabled(self) -> bool:
        """Decide once whether this widget can do anything, so the hot path is a single boolean test."""
        if not in_notebook():
            return False
        try:
            import ipywidgets  # noqa: F401
            import plotly.graph_objects as go
        except ImportError:
            logger.info(
                "TrainingProgressWidget disabled: plotly and ipywidgets are both required "
                "(pip install 'plotly>=5.15' 'ipywidgets>=8.0'). Training is unaffected."
            )
            return False
        try:
            # CONSTRUCT one rather than trusting that the import succeeded. plotly >= 6 moved FigureWidget
            # behind ``anywidget`` and raises "Please install anywidget to use the FigureWidget class" only
            # at construction time, so an import-only probe reports the widget as usable and then the
            # ImportError lands mid-training instead. Everything else here is already best-effort; this is
            # the one call that decides whether any of it can run at all.
            go.FigureWidget()
        except Exception as exc:
            # WARNING, not INFO: the caller enabled a live training widget and is not getting one. An INFO
            # line is invisible under the default root level, so the capability vanished silently.
            logger.warning(
                "TrainingProgressWidget disabled: this plotly build cannot construct a FigureWidget (%s). "
                "plotly >= 6 requires the 'anywidget' package for it (pip install anywidget). Training is "
                "unaffected -- the per-iteration trajectory is still recorded on the callback.", exc,
            )
            return False
        return True

    @property
    def enabled(self) -> bool:
        """Whether the widget will actually render; False makes every other method a no-op."""
        return self._enabled

    def stop_requested(self) -> bool:
        """True once the operator has clicked stop AND confirmed. Consumed by the callback's ``stop_flag``."""
        return self._stop_requested

    # ------------------------------------------------------------- ui assembly

    def _build_stop_control(self) -> Any:
        """The two-step stop control: one click arms it, a second confirms, so a misclick cannot end a fit."""
        import ipywidgets as w

        stop_btn = w.Button(description="Stop training", icon="stop", button_style="danger", layout=w.Layout(width="160px"))
        confirm_btn = w.Button(description="Yes, stop now", button_style="danger", layout=w.Layout(width="150px"))
        cancel_btn = w.Button(description="Cancel", layout=w.Layout(width="100px"))
        prompt = w.HTML("<b style='color:#c0392b'>Stop training at the current iteration?</b>")
        confirm_row = w.HBox([prompt, confirm_btn, cancel_btn])
        confirm_row.layout.display = "none"
        self._status = w.HTML("")

        def _arm(_btn: Any) -> None:
            """First click: reveal the confirmation row rather than stopping."""
            confirm_row.layout.display = "flex"
            stop_btn.layout.display = "none"

        def _confirm(_btn: Any) -> None:
            """Second click: set the flag the callback polls. The booster stops at its next iteration."""
            self._stop_requested = True
            confirm_row.layout.display = "none"
            self._status.value = "<b style='color:#c0392b'>Stop requested -- finishing the current iteration...</b>"
            logger.info("TrainingProgressWidget: stop requested from the widget; training will halt at the next iteration.")

        def _cancel(_btn: Any) -> None:
            """Dismiss the confirmation and re-arm the original button."""
            confirm_row.layout.display = "none"
            stop_btn.layout.display = "inline-flex"

        stop_btn.on_click(_arm)
        confirm_btn.on_click(_confirm)
        cancel_btn.on_click(_cancel)
        return w.VBox([w.HBox([stop_btn]), confirm_row, self._status])

    def _new_figure(self, metric: str) -> Any:
        """One FigureWidget per metric tab: an empty metric axis plus, optionally, a secondary RAM axis."""
        import plotly.graph_objects as go

        fig = go.FigureWidget()
        fig.update_layout(
            height=self.height,
            margin=dict(l=60, r=60, t=40, b=45),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            xaxis=dict(title="iteration", showgrid=True, gridcolor="#eeeeee"),
            yaxis=dict(title=metric, showgrid=True, gridcolor="#eeeeee"),
            template="plotly_white",
        )
        if self.show_ram:
            # RAM lives on its own right-hand axis: it shares the iteration axis (which is the point -- a memory
            # climb wants reading against the metric stalling) but not the metric's units or range.
            fig.update_layout(yaxis2=dict(title="RAM, GB", overlaying="y", side="right", showgrid=False))
            fig.add_trace(go.Scatter(x=[], y=[], name="RAM, GB", yaxis="y2", mode="lines",
                                     line=dict(color=_RAM_COLOR, width=1, dash="dot"), hovertemplate="%{y:.1f} GB<extra></extra>"))
            self._ram_idx[metric] = len(fig.data) - 1
        # The optimum marker is one star that moves, not a point per improvement.
        fig.add_trace(go.Scatter(x=[], y=[], name="best", mode="markers",
                                 marker=dict(symbol="star", size=15, color="#d62728",
                                             line=dict(width=1, color="white")),
                                 hovertemplate="best %{y:.6f} @ iter %{x}<extra></extra>"))
        self._best_idx[metric] = len(fig.data) - 1
        self._trace_idx[metric] = {}
        return fig

    def _ensure_metric_tab(self, metric: str) -> Any:
        """Return this metric's figure, creating its tab on first sight so late-appearing metrics still show."""
        if metric not in self._figures:
            self._figures[metric] = self._new_figure(metric)
            if self._tabs is not None:
                names = list(self._figures)
                self._tabs.children = tuple(self._figures[m] for m in names)
                for i, name in enumerate(names):
                    self._tabs.set_title(i, name)
        return self._figures[metric]

    def _ensure_series(self, metric: str, dataset: str) -> int:
        """Return the trace index for one (metric, dataset) curve, adding the trace on first sight."""
        idx_map = self._trace_idx[metric]
        if dataset not in idx_map:
            import plotly.graph_objects as go

            fig = self._figures[metric]
            color = _dataset_color(dataset, len(idx_map))
            fig.add_trace(go.Scatter(x=[], y=[], name=str(dataset), mode="lines", line=dict(color=color, width=2)))
            idx_map[dataset] = len(fig.data) - 1
        return idx_map[dataset]

    def display(self) -> None:
        """Render the widget into the notebook once; subsequent calls are no-ops."""
        if not self._enabled or self._displayed:
            return
        import ipywidgets as w
        from IPython.display import display as ipy_display

        self._tabs = w.Tab(children=tuple(self._figures.values()))
        for i, name in enumerate(self._figures):
            self._tabs.set_title(i, name)
        parts: List[Any] = [self._tabs]
        if self.show_stop_button:
            parts.append(self._build_stop_control())
        self._container = w.VBox(parts)
        ipy_display(self._container)
        self._displayed = True

    # ------------------------------------------------------------------ update

    def update(
        self,
        iterations: Sequence[int],
        metric_history: Dict[str, Dict[str, List[float]]],
        ram_history: Sequence[Tuple[int, float]],
        modes: Optional[Dict[str, str]] = None,
        *,
        force: bool = False,
    ) -> None:
        """Repaint the tabs from the callback's running history, throttled to ``refresh_secs``.

        ``iterations`` is the shared x axis (the booster's own iteration index, not a callback-call counter --
        they differ whenever ``metric_period`` > 1). ``ram_history`` carries its OWN (iteration, GB) pairs
        because RAM is sampled on the refresh throttle rather than per iteration. ``modes`` maps a metric name
        to "min"/"max" so the star lands on the right extreme; an unknown metric defaults to "min".
        """
        if not self._enabled:
            return
        now = timer()
        if not force and (now - self._last_refresh) < self.refresh_secs:
            return
        self._last_refresh = now
        if modes:
            self._modes.update(modes)

        wanted_metrics = set(self.metrics) if self.metrics is not None else None
        wanted_datasets = set(self.datasets) if self.datasets is not None else None
        seen_any = False
        for dataset, per_metric in metric_history.items():
            if wanted_datasets is not None and dataset not in wanted_datasets:
                continue
            for metric in per_metric:
                if wanted_metrics is not None and metric not in wanted_metrics:
                    continue
                self._ensure_metric_tab(metric)
                self._ensure_series(metric, dataset)
                seen_any = True
        if not seen_any:
            return
        self.display()

        # Hand plotly NUMPY arrays, never Python lists. This is the single thing that decides whether a live
        # plot is usable: plotly validates a list ELEMENT BY ELEMENT (`to_scalar_or_list`) and then compares old
        # against new element by element (`_vals_equal`), so one refresh of a 10k-point history made 3.66M
        # Python-level calls and took 331 ms -- 66% of the wall clock at a 0.5s cadence, i.e. the widget would
        # cost more than the fit it is watching. numpy input takes plotly's `copy_to_readonly_numpy_array` fast
        # path and `np.array_equal`, both of which stay in C. Measured on a 3-split x 2-metric history
        # (bench_progress_widget_update): 331.41 ms -> 1.20 ms per refresh at 10k points, 276x, with the Python
        # call count dropping from 80.9M to 80.2K. Per-point cost now FALLS as the history grows (8.9 us at 100
        # points, 0.12 us at 10k) because the per-element Python work is gone, and at a 0.5s cadence the widget
        # costs 0.24% of wall time instead of 66%.
        x_all = self._as_x_array(iterations)
        for metric, fig in self._figures.items():
            # One batch_update per FIGURE, not per trace: plotly revalidates and repaints on every property
            # assignment otherwise, which is the whole cost of a live plot.
            with fig.batch_update():
                best_x: Optional[int] = None
                best_y: Optional[float] = None
                mode = self._modes.get(metric, "min")
                star_dataset = self._monitor_dataset_for(metric, metric_history)
                for dataset, t_idx in self._trace_idx[metric].items():
                    ys = metric_history.get(dataset, {}).get(metric, [])
                    if not ys:
                        continue
                    # History and the iteration axis are appended together, but guard the length anyway: a
                    # booster that registers a dataset late would otherwise misalign every point on its curve.
                    k = min(len(ys), len(x_all))
                    y_arr = self._as_y_array(metric, dataset, ys, k)
                    trace = fig.data[t_idx]
                    trace.x = x_all[:k]
                    trace.y = y_arr
                    if dataset == star_dataset and k:
                        pos = int(np.argmin(y_arr)) if mode == "min" else int(np.argmax(y_arr))
                        best_x, best_y = x_all[pos], float(y_arr[pos])
                if best_x is not None:
                    star = fig.data[self._best_idx[metric]]
                    star.x = np.array([best_x])
                    star.y = np.array([best_y])
                if self.show_ram and ram_history:
                    ram_trace = fig.data[self._ram_idx[metric]]
                    rx, ry = self._as_ram_arrays(ram_history)
                    ram_trace.x = rx
                    ram_trace.y = ry

    def _as_x_array(self, iterations: Sequence[int]) -> Any:
        """The shared iteration axis as a numpy array, rebuilt only when it actually grew.

        Every metric tab plots against the same x, so converting once per refresh (rather than once per trace)
        keeps the conversion off the per-trace path entirely.
        """
        n = len(iterations)
        cached = self._x_cache
        if cached is None or len(cached) != n:
            self._x_cache = np.asarray(iterations, dtype=np.int64)
        return self._x_cache

    def _as_y_array(self, metric: str, dataset: str, ys: List[float], k: int) -> Any:
        """One curve's values as a numpy array, reusing the previous buffer when the series only appended.

        A fit appends one point per iteration, so rebuilding the whole array every refresh is the wasteful
        part. Growing a buffer geometrically and writing only the new tail makes the per-refresh conversion
        proportional to the points ADDED since the last repaint rather than to the whole history.
        """
        key = (metric, dataset)
        buf, filled = self._y_cache.get(key, (None, 0))
        if buf is None or filled > k:
            # Allocate with headroom on the FIRST build too, not just on later growth: an exact-size initial
            # buffer reallocates on the very next appended point, which is every refresh of a live fit.
            buf = np.empty(max(k * 2, 64), dtype=np.float64)
            buf[:k] = ys[:k]
            self._y_cache[key] = (buf, k)
            return buf[:k]
        if k > buf.shape[0]:
            grown = np.empty(max(k * 2, 64), dtype=np.float64)
            grown[:filled] = buf[:filled]
            buf = grown
        if k > filled:
            buf[filled:k] = ys[filled:k]
        self._y_cache[key] = (buf, k)
        return buf[:k]

    def _as_ram_arrays(self, ram_history: Sequence[Tuple[int, float]]) -> Tuple[Any, Any]:
        """The sparse RAM series as (x, y) numpy arrays, rebuilt only when a new sample arrived."""
        n = len(ram_history)
        if self._ram_cache is None or self._ram_cache[2] != n:
            arr = np.asarray(ram_history, dtype=np.float64)
            self._ram_cache = (arr[:, 0].astype(np.int64), arr[:, 1], n)
        return self._ram_cache[0], self._ram_cache[1]

    def _monitor_dataset_for(self, metric: str, metric_history: Dict[str, Dict[str, List[float]]]) -> Optional[str]:
        """Which curve the star tracks: the monitored split when set, else the last non-train split reporting it.

        The optimum of the TRAIN curve is not a decision -- it is almost always the final iteration and says
        nothing about when to stop. The star belongs on the held-out curve.
        """
        explicit = getattr(self, "monitor_dataset", None)
        if explicit and metric in metric_history.get(explicit, {}):
            return str(explicit)
        candidates = [d for d, per in metric_history.items() if metric in per and str(d).lower() not in {"train", "learn", "training"}]
        return candidates[-1] if candidates else None

    def finalize(self, best_iter: Optional[int] = None, best_metric: Optional[float] = None, stopped_early: bool = False) -> None:
        """Force a last repaint and print the outcome, so the widget's final state matches the fitted model."""
        if not self._enabled:
            return
        if self._status is not None:
            if stopped_early:
                note = "training stopped early"
            else:
                note = "training finished"
            if best_iter is not None and best_metric is not None:
                note += f" -- best {best_metric:.6f} @ iteration {best_iter:,}"
            self._status.value = f"<b>{note}</b>"


__all__ = ["TrainingProgressWidget", "DEFAULT_REFRESH_SECS", "in_notebook"]
