"""Regression: metric_over_time drift annotation must use the canonical metric
direction, not a 2-item ("mse","brier") allowlist that mislabeled every other
loss (rmse/mae/mape/log_loss/ice/ece/pinball) as higher-is-better and printed an
inverted "(higher=better)" trend title on the default-ON temporal drift chart.

Pre-fix: diagnostics_dispatch computed `higher_is_better = metric not in ("mse",
"brier")`, so metric="rmse" -> True -> title "(higher=better)". This test pins
both the canonical lookup (known answers) and the rendered title direction.
"""
import numpy as np

from mlframe.training.metrics_registry import metric_name_higher_is_better
from mlframe.reporting.charts.drift import metric_over_time


def test_loss_metrics_are_lower_is_better_in_canonical_table():
    # Each of these would be MISLABELED higher-is-better by the old
    # `metric not in ("mse","brier")` allowlist.
    for loss in ("rmse", "mae", "mape", "log_loss", "ice", "ece", "pinball"):
        assert metric_name_higher_is_better(loss) is False, (
            f"{loss} must be lower-is-better"
        )
    # Quality metrics still higher-is-better.
    for q in ("roc_auc", "r2", "f1", "accuracy"):
        assert metric_name_higher_is_better(q) is True


def _direction_in_title(metric: str) -> str:
    rng = np.random.default_rng(0)
    n = 400
    ts = np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")
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
    title = _direction_in_title("rmse")
    # Either a populated line panel title or the "no buckets" annotation;
    # when a line renders it must carry the correct direction.
    if "over time" in title:
        assert "lower=better" in title, title
        assert "higher=better" not in title, title


def test_roc_auc_over_time_title_says_higher_is_better():
    title = _direction_in_title("roc_auc")
    if "over time" in title:
        assert "higher=better" in title, title


def _captured_hib_from_dispatch(metric, task, monkeypatch):
    """Call the real render_target_drift_diagnostics and capture the higher_is_better it passes to metric_over_time.

    This exercises the EXACT buggy line (the in-function direction computation), so it fails on the pre-fix
    `metric not in ("mse","brier")` allowlist (which yields True for rmse) and passes on the canonical-table fix.
    """
    import mlframe.reporting.charts.drift as drift_mod
    from mlframe.reporting.diagnostics_dispatch import render_target_drift_diagnostics

    captured = {}

    def _spy(yt, yp, ts, *, metric, higher_is_better):
        captured["higher_is_better"] = higher_is_better
        raise RuntimeError("stop after capture")  # short-circuit before rendering; the caller swallows + logs

    monkeypatch.setattr(drift_mod, "metric_over_time", _spy)
    n = 300
    ts = np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")
    rng = np.random.default_rng(0)
    yt = rng.normal(size=n)
    yp = yt + rng.normal(scale=0.3, size=n)
    render_target_drift_diagnostics(
        train_frame=None, test_frame=None, y_true=yt, y_pred=yp, timestamps=ts,
        task=task, plot_outputs="png", base_path="_unused", metric=metric,
        calibration_drift=False, target_acf=False, cusum_drift=False,
    )
    return captured.get("higher_is_better")


def test_dispatch_passes_lower_is_better_for_rmse(monkeypatch):
    assert _captured_hib_from_dispatch("rmse", "regression", monkeypatch) is False


def test_dispatch_passes_higher_is_better_for_roc_auc(monkeypatch):
    assert _captured_hib_from_dispatch("roc_auc", "classification", monkeypatch) is True
