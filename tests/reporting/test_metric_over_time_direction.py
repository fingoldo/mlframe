"""Regression: metric_over_time drift annotation must use the canonical metric
direction, not a 2-item ("mse","brier") allowlist that mislabeled every other
loss (rmse/mae/mape/log_loss/ice/ece/pinball) as higher-is-better and printed an
inverted "(higher=better)" trend title on the default-ON temporal drift chart.

Pre-fix: diagnostics_dispatch computed `higher_is_better = metric not in ("mse",
"brier")`, so metric="rmse" -> True -> title "(higher=better)". This test pins
both the canonical lookup (known answers) and the rendered title direction.
"""

import numpy as np
import pytest

from mlframe.training.metrics_registry import metric_name_higher_is_better
from mlframe.reporting.charts.drift import metric_over_time


def test_loss_metrics_are_lower_is_better_in_canonical_table():
    # Each of these would be MISLABELED higher-is-better by the old
    # `metric not in ("mse","brier")` allowlist.
    """Loss metrics are lower is better in canonical table."""
    for loss in ("rmse", "mae", "mape", "log_loss", "ice", "ece", "pinball"):
        assert metric_name_higher_is_better(loss) is False, f"{loss} must be lower-is-better"
    # Quality metrics still higher-is-better.
    for q in ("roc_auc", "r2", "f1", "accuracy"):
        assert metric_name_higher_is_better(q) is True


def _direction_in_title(metric: str) -> str:
    """Render the chart and return every panel title, joined.

    The timestamps must give each daily bucket at least ``min_samples`` (100) rows. The previous fixture laid
    400 rows across 400 distinct days -- one sample each -- so no bucket ever cleared the threshold,
    ``metric_over_time`` returned its bare-metric-name annotation panel, and the two title assertions below
    sat behind an `if "over time" in title` that was never true. Both tests ran zero assertions, and an
    inverted direction label would have shipped green.
    """
    rng = np.random.default_rng(0)
    n_days, per_day = 5, 150
    n = n_days * per_day
    ts = np.repeat(np.arange(n_days), per_day).astype("datetime64[D]").astype("datetime64[ns]")
    yt = rng.normal(size=n)
    yp = yt + rng.normal(scale=0.3, size=n)
    hib = metric_name_higher_is_better(metric)
    hib = True if hib is None else hib
    spec = metric_over_time(yt, yp, ts, metric=metric, higher_is_better=hib)
    titles = []
    for row in spec.panels:
        for p in row:
            t = getattr(p, "title", "") or ""
            titles.append(t)
    return " ".join(titles)


def test_rmse_over_time_title_says_lower_is_better():
    """Rmse over time title says lower is better."""
    title = _direction_in_title("rmse")
    assert "over time" in title, f"the line panel did not render, so the direction was never checked: {title!r}"
    assert "lower=better" in title, title
    assert "higher=better" not in title, title


def test_roc_auc_over_time_title_says_higher_is_better():
    """Roc auc over time title says higher is better."""
    title = _direction_in_title("roc_auc")
    assert "over time" in title, f"the line panel did not render, so the direction was never checked: {title!r}"
    assert "higher=better" in title, title


def test_the_per_bucket_dispatcher_knows_every_metric_these_titles_claim():
    """A name the dispatcher rejects fails per bucket, is swallowed, and renders as "no buckets".

    That message is about sample counts, so an unsupported metric name is indistinguishable from a genuinely
    sparse time axis -- which is what hid `rmse` being uncomputable for as long as it was.
    """
    import numpy as np_

    from mlframe.training.evaluation import _compute_metric

    yt = np_.array([1.0, 2.0, 3.0, 4.0])
    yp = np_.array([1.5, 2.5, 2.5, 4.5])
    assert _compute_metric("rmse", yt, yp) == pytest.approx(0.5)
    assert _compute_metric("mae", yt, yp) == pytest.approx(0.5)
    assert _compute_metric("mse", yt, yp) == pytest.approx(0.25)


def _captured_hib_from_dispatch(metric, task, monkeypatch):
    """Call the real render_target_drift_diagnostics and capture the higher_is_better it passes to metric_over_time.

    This exercises the EXACT buggy line (the in-function direction computation), so it fails on the pre-fix
    `metric not in ("mse","brier")` allowlist (which yields True for rmse) and passes on the canonical-table fix.
    """
    import mlframe.reporting.charts.drift as drift_mod
    from mlframe.reporting.diagnostics_dispatch import render_target_drift_diagnostics

    captured = {}

    def _spy(yt, yp, ts, *, metric, higher_is_better):
        """Helper: Spy."""
        captured["higher_is_better"] = higher_is_better
        raise RuntimeError("stop after capture")  # short-circuit before rendering; the caller swallows + logs

    monkeypatch.setattr(drift_mod, "metric_over_time", _spy)
    n = 300
    ts = np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")
    rng = np.random.default_rng(0)
    yt = rng.normal(size=n)
    yp = yt + rng.normal(scale=0.3, size=n)
    render_target_drift_diagnostics(
        train_frame=None,
        test_frame=None,
        y_true=yt,
        y_pred=yp,
        timestamps=ts,
        task=task,
        plot_outputs="png",
        base_path="_unused",
        metric=metric,
        calibration_drift=False,
        target_acf=False,
        cusum_drift=False,
    )
    return captured.get("higher_is_better")


def test_dispatch_passes_lower_is_better_for_rmse(monkeypatch):
    """Dispatch passes lower is better for rmse."""
    assert _captured_hib_from_dispatch("rmse", "regression", monkeypatch) is False


def test_dispatch_passes_higher_is_better_for_roc_auc(monkeypatch):
    """Dispatch passes higher is better for roc auc."""
    assert _captured_hib_from_dispatch("roc_auc", "classification", monkeypatch) is True
