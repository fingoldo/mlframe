"""Tests for the live training-progress widget and its wiring into UniversalCallback.

The widget is opt-in and notebook-only, so the two things that most need pinning are (a) it is a genuine
no-op everywhere else -- it must never be able to slow or break a headless fit -- and (b) when it IS on, the
star marks the right extreme of the right curve, since that is the one number an operator reads off it.
"""

from unittest.mock import patch

import numpy as np
import pytest

from mlframe.training.callbacks import LightGBMCallback, TrainingProgressWidget
from mlframe.training.callbacks.progress_widget import _dataset_color, in_notebook


def _figurewidget_available() -> bool:
    """Whether this plotly build can actually CONSTRUCT a FigureWidget.

    plotly >= 6 moved FigureWidget behind ``anywidget`` and raises only at construction time, so an
    environment can import plotly fine and still be unable to render this widget. The production code
    probes the same way (see ``TrainingProgressWidget._resolve_enabled``) and self-disables; the tests that
    exercise the drawing path have nothing to assert there, so they skip.
    """
    try:
        import plotly.graph_objects as go

        go.FigureWidget()
    except Exception:
        return False
    return True


_NEEDS_FIGUREWIDGET = pytest.mark.skipif(
    not _figurewidget_available(),
    reason="plotly build cannot construct a FigureWidget (plotly >= 6 needs the anywidget package)",
)


def _notebook():
    """Patch the kernel detection so the widget builds without a real frontend attached."""
    return patch("mlframe.training.callbacks.progress_widget.in_notebook", return_value=True)


def _no_display():
    """Swallow the IPython display call; there is no frontend to render into under pytest."""
    return patch("IPython.display.display", lambda *a, **k: None)


def _history(n_points: int = 60, period: int = 9):
    """Two metrics of opposite direction over three splits, reported every ``period`` booster iterations."""
    rng = np.random.default_rng(0)
    hist = {ds: {"ICE": [], "AUC": []} for ds in ("train", "validation", "test")}
    iters, ram = [], []
    for i in range(n_points):
        it = i * period
        iters.append(it)
        for ds, off in (("train", 0.0), ("validation", 0.02), ("test", 0.03)):
            hist[ds]["ICE"].append(-0.20 - 0.001 * i + off + rng.normal(0, 0.001))
            hist[ds]["AUC"].append(0.70 + 0.002 * i - off + rng.normal(0, 0.001))
        if i % 10 == 0:
            ram.append((it, 50.0 + i * 0.1))
    return iters, hist, ram


class TestDisabledEverywhereButANotebook:
    """A live widget must never be able to slow down or break a headless / CI fit."""

    def test_widget_is_disabled_outside_a_kernel(self):
        """pytest is not a notebook, so the widget must report itself unusable."""
        assert in_notebook() is False
        assert TrainingProgressWidget().enabled is False

    def test_a_disabled_widget_is_dropped_by_the_callback(self):
        """Resolving it to None once keeps the per-iteration path from ever testing for it again."""
        cb = LightGBMCallback(progress_widget=True)
        assert cb._widget is None

    def test_every_method_of_a_disabled_widget_is_a_no_op(self):
        """Calling into a disabled widget must be safe, not merely unused."""
        w = TrainingProgressWidget()
        iters, hist, ram = _history(5)
        w.update(iters, hist, ram, {"ICE": "min"})
        w.display()
        w.finalize(best_iter=3, best_metric=0.5)
        assert w._figures == {}

    def test_history_is_still_recorded_without_a_widget(self):
        """The iteration and metric series are useful on their own, so they are not gated on the widget."""
        cb = LightGBMCallback(progress_widget=None)
        cb.update_history({"valid_0": {"ICE": 0.5}}, iteration=41)
        cb.update_history({"valid_0": {"ICE": 0.4}}, iteration=50)
        assert cb.iter_history == [41, 50]
        assert cb.metric_history["valid_0"]["ICE"] == [0.5, 0.4]
        assert cb.ram_history == []


class TestIterationAxisIsTheBoostersOwn:
    """A callback-call counter is not the iteration axis whenever the booster reports every k-th round."""

    def test_reported_iteration_is_recorded_verbatim(self):
        """CatBoost with metric_period=9 must plot at 0, 9, 18 -- not 0, 1, 2."""
        cb = LightGBMCallback()
        for i in range(5):
            cb.update_history({"valid_0": {"ICE": float(i)}}, iteration=i * 9)
        assert cb.iter_history == [0, 9, 18, 27, 36]

    def test_missing_iteration_falls_back_to_a_dense_counter(self):
        """A booster that does not expose its round index still gets a monotonic axis."""
        cb = LightGBMCallback()
        for i in range(4):
            cb.update_history({"valid_0": {"ICE": float(i)}})
        assert cb.iter_history == [0, 1, 2, 3]


@_NEEDS_FIGUREWIDGET
class TestTabsAndCurves:
    """One tab per metric, one curve per dataset, discovered from whatever the booster reports."""

    def test_a_tab_is_created_per_metric_and_a_curve_per_dataset(self):
        """Both are discovered from what the booster reports; nothing has to be declared up front."""
        with _notebook(), _no_display():
            w = TrainingProgressWidget(refresh_secs=0.0)
            iters, hist, ram = _history()
            w.update(iters, hist, ram, {"ICE": "min", "AUC": "max"}, force=True)
        assert list(w._figures) == ["ICE", "AUC"]
        for metric in ("ICE", "AUC"):
            assert list(w._trace_idx[metric]) == ["train", "validation", "test"]

    def test_metrics_can_be_restricted(self):
        """`metrics=` narrows the tabs without the caller having to touch the booster's eval list."""
        with _notebook(), _no_display():
            w = TrainingProgressWidget(refresh_secs=0.0, metrics=["AUC"])
            iters, hist, ram = _history()
            w.update(iters, hist, ram, {"AUC": "max"}, force=True)
        assert list(w._figures) == ["AUC"]

    def test_datasets_can_be_restricted(self):
        """`datasets=` narrows the curves without touching the booster's eval sets."""
        with _notebook(), _no_display():
            w = TrainingProgressWidget(refresh_secs=0.0, datasets=["validation"])
            iters, hist, ram = _history()
            w.update(iters, hist, ram, {"ICE": "min"}, force=True)
        assert list(w._trace_idx["ICE"]) == ["validation"]

    def test_a_late_appearing_metric_still_gets_a_tab(self):
        """Boosters can register an eval metric after the first iteration."""
        with _notebook(), _no_display():
            w = TrainingProgressWidget(refresh_secs=0.0)
            w.update([0], {"validation": {"ICE": [0.5]}}, [], {"ICE": "min"}, force=True)
            assert list(w._figures) == ["ICE"]
            w.update([0, 1], {"validation": {"ICE": [0.5, 0.4], "AUC": [0.6, 0.7]}}, [], force=True)
        assert list(w._figures) == ["ICE", "AUC"]


@_NEEDS_FIGUREWIDGET
class TestOptimumStar:
    """The star is the one number an operator reads off this widget, so it must be on the right point."""

    def test_star_marks_the_minimum_for_a_lower_is_better_metric(self):
        """ICE is lower-is-better, so the star sits on the validation minimum."""
        with _notebook(), _no_display():
            w = TrainingProgressWidget(refresh_secs=0.0)
            w.monitor_dataset = "validation"
            iters, hist, ram = _history()
            w.update(iters, hist, ram, {"ICE": "min", "AUC": "max"}, force=True)
        star = w._figures["ICE"].data[w._best_idx["ICE"]]
        vals = hist["validation"]["ICE"]
        assert star.y[0] == pytest.approx(min(vals))
        assert star.x[0] == iters[int(np.argmin(vals))]

    def test_star_marks_the_maximum_for_a_higher_is_better_metric(self):
        """AUC is higher-is-better, so the star sits on the validation maximum."""
        with _notebook(), _no_display():
            w = TrainingProgressWidget(refresh_secs=0.0)
            w.monitor_dataset = "validation"
            iters, hist, ram = _history()
            w.update(iters, hist, ram, {"ICE": "min", "AUC": "max"}, force=True)
        star = w._figures["AUC"].data[w._best_idx["AUC"]]
        vals = hist["validation"]["AUC"]
        assert star.y[0] == pytest.approx(max(vals))
        assert star.x[0] == iters[int(np.argmax(vals))]

    def test_star_never_tracks_the_train_curve(self):
        """The optimum of the TRAIN curve is almost always the last iteration and is not a decision."""
        with _notebook(), _no_display():
            w = TrainingProgressWidget(refresh_secs=0.0)
            iters, hist, ram = _history()
            # Make train strictly better than every held-out split, so tracking it would be visible.
            hist["train"]["ICE"] = [v - 1.0 for v in hist["train"]["ICE"]]
            w.update(iters, hist, ram, {"ICE": "min"}, force=True)
        star = w._figures["ICE"].data[w._best_idx["ICE"]]
        assert star.y[0] > -1.0


@_NEEDS_FIGUREWIDGET
class TestRamSeries:
    """RAM shares the iteration axis but not the metric's units, and is sampled sparsely on purpose."""

    def test_ram_is_drawn_on_a_secondary_axis_with_its_own_sparse_x(self):
        """RAM shares the iteration axis but neither the units nor the sampling rate of the metric."""
        with _notebook(), _no_display():
            w = TrainingProgressWidget(refresh_secs=0.0)
            iters, hist, ram = _history()
            w.update(iters, hist, ram, {"ICE": "min"}, force=True)
        trace = w._figures["ICE"].data[w._ram_idx["ICE"]]
        assert trace.yaxis == "y2"
        assert list(trace.x) == [it for it, _ in ram]
        assert len(trace.x) < len(iters)  # sampled on the refresh throttle, not per iteration

    def test_ram_can_be_turned_off(self):
        """A caller who does not want the second axis gets no RAM trace at all."""
        with _notebook(), _no_display():
            w = TrainingProgressWidget(refresh_secs=0.0, show_ram=False)
            iters, hist, ram = _history()
            w.update(iters, hist, ram, {"ICE": "min"}, force=True)
        assert w._ram_idx == {}


class TestStopButton:
    """A misclick must not be able to end a multi-hour fit."""

    def _control(self, widget):
        """Unpack the stop control into (stop button, confirm row, confirm button, cancel button)."""
        ctl = widget._build_stop_control()
        row = ctl.children[1]
        return ctl.children[0].children[0], row, row.children[1], row.children[2]

    def test_one_click_only_arms_the_confirmation(self):
        """The first click reveals the confirmation and nothing else."""
        with _notebook():
            w = TrainingProgressWidget()
            stop, row, _confirm, _cancel = self._control(w)
            stop.click()
        assert w.stop_requested() is False
        assert row.layout.display != "none"

    def test_confirming_requests_the_stop(self):
        """Only the second, explicit click sets the flag the training loop polls."""
        with _notebook():
            w = TrainingProgressWidget()
            stop, _row, confirm, _cancel = self._control(w)
            stop.click()
            confirm.click()
        assert w.stop_requested() is True

    def test_cancelling_leaves_training_running(self):
        """Dismissing the confirmation must leave the fit untouched."""
        with _notebook():
            w = TrainingProgressWidget()
            stop, row, _confirm, cancel = self._control(w)
            stop.click()
            cancel.click()
        assert w.stop_requested() is False
        assert row.layout.display == "none"

    @_NEEDS_FIGUREWIDGET
    def test_stop_goes_through_the_callbacks_existing_stop_flag(self):
        """The widget must not get its own path into the training loop.

        Needs a genuinely ENABLED widget: the callback discards a self-disabled one at construction, which
        is the whole point of that design, so there is no stop_flag composition to assert without one. The
        button behaviour itself is covered by the tests above, which do not need a live figure.
        """
        with _notebook():
            w = TrainingProgressWidget()
            cb = LightGBMCallback(progress_widget=w)
            assert cb.stop_flag() is False
            stop, _row, confirm, _cancel = self._control(w)
            stop.click()
            confirm.click()
            assert cb.stop_flag() is True

    def test_a_user_supplied_stop_flag_is_preserved(self):
        """Composing must not discard the caller's own stop condition."""
        with _notebook():
            user_wants_stop = {"v": False}
            cb = LightGBMCallback(progress_widget=TrainingProgressWidget(), stop_flag=lambda: user_wants_stop["v"])
            assert cb.stop_flag() is False
            user_wants_stop["v"] = True
            assert cb.stop_flag() is True


@_NEEDS_FIGUREWIDGET
class TestRefreshThrottle:
    """A repaint per iteration would cost more than the fit it is watching."""

    def test_updates_inside_the_interval_are_skipped(self):
        """An unforced update inside the throttle window must not repaint."""
        with _notebook(), _no_display():
            w = TrainingProgressWidget(refresh_secs=3600.0)
            iters, hist, ram = _history(5)
            w.update(iters, hist, ram, {"ICE": "min"}, force=True)
            drawn = len(w._figures["ICE"].data[w._trace_idx["ICE"]["validation"]].x)
            hist["validation"]["ICE"].append(-99.0)
            iters.append(999)
            w.update(iters, hist, ram, {"ICE": "min"})  # not forced, inside the interval
            assert len(w._figures["ICE"].data[w._trace_idx["ICE"]["validation"]].x) == drawn

    def test_force_bypasses_the_throttle(self):
        """finalize() must be able to show the true last frame regardless of when the last repaint was."""
        with _notebook(), _no_display():
            w = TrainingProgressWidget(refresh_secs=3600.0)
            iters, hist, ram = _history(5)
            w.update(iters, hist, ram, {"ICE": "min"}, force=True)
            hist["validation"]["ICE"].append(-99.0)
            iters.append(999)
            w.update(iters, hist, ram, {"ICE": "min"}, force=True)
        assert len(w._figures["ICE"].data[w._trace_idx["ICE"]["validation"]].x) == len(iters)


class TestDatasetColors:
    """The same split keeps the same colour across every tab and every run."""

    @pytest.mark.parametrize(("name", "expected"), [("train", "#1f77b4"), ("validation", "#ff7f0e"), ("test", "#2ca02c")])
    def test_known_splits_get_conventional_hues(self, name, expected):
        """train/validation/test keep the same colour across every tab and run."""
        assert _dataset_color(name, 0) == expected

    def test_unknown_names_cycle_a_fallback_palette(self):
        """An arbitrary eval-shard name still gets a distinct, stable colour."""
        first = _dataset_color("shard_7", 0)
        assert first != _dataset_color("shard_8", 1)
        assert _dataset_color("shard_7", 0) == first  # stable for the same ordinal


@_NEEDS_FIGUREWIDGET
class TestRefreshCostDoesNotScaleWithHistory:
    """A live plot that re-validates its whole history per repaint costs more than the fit it is watching."""

    def test_plotly_receives_numpy_not_python_lists(self):
        """This is the whole optimisation: plotly validates a list element-by-element, numpy in C.

        Measured on a 3-split x 2-metric history: handing plotly Python lists cost 331.41 ms per refresh at
        10k points (80.9M Python calls, `to_scalar_or_list` fired once per element); numpy arrays cost 1.20 ms
        (80.2K calls). A single `list` slipping back in silently restores the 276x.
        """
        with _notebook(), _no_display():
            w = TrainingProgressWidget(refresh_secs=0.0)
            w.monitor_dataset = "validation"
            iters, hist, ram = _history(200)
            w.update(iters, hist, ram, {"ICE": "min", "AUC": "max"}, force=True)
        for metric, fig in w._figures.items():
            for trace in fig.data:
                for axis in ("x", "y"):
                    data = getattr(trace, axis)
                    if data is None or len(data) == 0:
                        continue
                    assert isinstance(data, np.ndarray), f"{metric}/{trace.name}.{axis} is {type(data).__name__}, not ndarray"

    def test_per_refresh_cost_grows_far_slower_than_the_history(self):
        """10x the history must not cost anywhere near 10x, or the widget is O(n^2) over a run."""
        from timeit import default_timer as timer

        def _cost(n: int) -> float:
            """Median-ish per-refresh wall time at a history of n points."""
            with _notebook(), _no_display():
                w = TrainingProgressWidget(refresh_secs=0.0)
                w.monitor_dataset = "validation"
                iters, hist, ram = _history(n)
                w.update(iters, hist, ram, {"ICE": "min", "AUC": "max"}, force=True)  # warm
                start = timer()
                for _ in range(20):
                    w.update(iters, hist, ram, {"ICE": "min", "AUC": "max"}, force=True)
                return (timer() - start) / 20

        small, large = _cost(200), _cost(2000)
        # Pre-fix this ratio was ~10 (strictly proportional to the history). The numpy path makes the refresh
        # dominated by fixed per-trace overhead instead, so a 10x history costs barely more. Generous bound:
        # the point is to catch a regression back to per-element validation, not to pin a wall-clock number.
        assert large < small * 4.0, f"refresh cost scaled {large / small:.1f}x for a 10x history"

    def test_appending_reuses_the_existing_buffer(self):
        """Growing the history must not reallocate the y buffer on every refresh."""
        with _notebook(), _no_display():
            w = TrainingProgressWidget(refresh_secs=0.0)
            iters, hist, ram = _history(100)
            w.update(iters, hist, ram, {"ICE": "min"}, force=True)
            buf_before = w._y_cache[("ICE", "validation")][0]
            for i in range(10):  # a few more reported points, well inside the geometric headroom
                iters.append((100 + i) * 9)
                for ds in ("train", "validation", "test"):
                    hist[ds]["ICE"].append(-0.3)
                    hist[ds]["AUC"].append(0.9)
            w.update(iters, hist, ram, {"ICE": "min"}, force=True)
            buf_after = w._y_cache[("ICE", "validation")][0]
        assert buf_after is buf_before  # same object: the tail was written in place, not rebuilt


@_NEEDS_FIGUREWIDGET
class TestCallbackIntegration:
    """The callback must feed the widget without changing anything about how it already trains."""

    def test_widget_receives_the_boosters_own_iteration_axis(self):
        """The callback forwards the real round index, not its own call count."""
        with _notebook(), _no_display():
            widget = TrainingProgressWidget(refresh_secs=0.0)
            cb = LightGBMCallback(progress_widget=widget, monitor_dataset="valid_0")
            cb.monitor_metric, cb.mode = "ICE", "min"
            for i in range(20):
                cb.update_history({"valid_0": {"ICE": 1.0 - i * 0.01}}, iteration=i * 9)
        assert list(widget._figures["ICE"].data[widget._trace_idx["ICE"]["valid_0"]].x) == [i * 9 for i in range(20)]

    def test_ram_is_sampled_and_forwarded(self):
        """The callback samples RAM on the widget throttle and records real values."""
        with _notebook(), _no_display():
            widget = TrainingProgressWidget(refresh_secs=0.0)
            cb = LightGBMCallback(progress_widget=widget)
            for i in range(5):
                cb.update_history({"valid_0": {"ICE": float(i)}}, iteration=i)
        assert len(cb.ram_history) > 0
        assert all(gb > 0.0 for _it, gb in cb.ram_history)

    def test_finalize_forces_a_last_repaint_past_the_throttle(self):
        """The widget's final frame must match the fitted model, not the last throttled refresh."""
        with _notebook(), _no_display():
            widget = TrainingProgressWidget(refresh_secs=3600.0)
            cb = LightGBMCallback(progress_widget=widget, monitor_dataset="valid_0")
            cb.monitor_metric, cb.mode = "ICE", "min"
            cb.update_history({"valid_0": {"ICE": 1.0}}, iteration=0)
            for i in range(1, 10):
                cb.update_history({"valid_0": {"ICE": 1.0 - i * 0.1}}, iteration=i)  # all throttled away
            cb.best_metric, cb.best_iter = 0.1, 9
            cb.finalize_widget(stopped_early=False)
        assert len(widget._figures["ICE"].data[widget._trace_idx["ICE"]["valid_0"]].x) == 10
        assert "best" in widget._status.value
