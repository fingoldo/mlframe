"""Integration tests for the per-model training-curve wiring in _reporting.py (INV-24).

Covers the booster-history extractor (lgb / xgb / catboost shapes + wrapper unwrap), the early-stopping marker, and
the render-to-disk path through report_model_perf (default-on, no-op without history / without saved charts).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mlframe.training.reporting._reporting import (
    _extract_training_history,
    _render_training_curves,
    _unwrap_booster,
)


class _FakeLGB:
    """Minimal lgb-shaped stub: evals_result_ is split-major {split: {metric: [...]}} + best_iteration_."""

    def __init__(self, evals, best_iter):
        self.evals_result_ = evals
        self.best_iteration_ = best_iter


class _FakeXGB:
    def __init__(self, evals, best_iter):
        self._evals = evals
        self.best_iteration = best_iter

    def evals_result(self):
        return self._evals


class _FakeCatBoost:
    def __init__(self, evals, best_iter):
        self._evals = evals
        self.best_iteration_ = best_iter

    def get_evals_result(self):
        return self._evals


class _Wrapper:
    def __init__(self, base):
        self.base_estimator = base


def _synthetic_evals(n=80, es=55):
    # Train keeps falling; val turns up after the ES point (overfitting signal).
    it = np.arange(n)
    train = 1.0 / (1.0 + it)
    val = 1.0 / (1.0 + it) + np.maximum(0.0, (it - es)) * 0.002
    # lgb positional eval-set names: valid_0 = train fold, valid_1 = holdout (mlframe convention).
    return {"valid_0": {"l2": train.tolist()}, "valid_1": {"l2": val.tolist()}}, es


def test_extract_lgb_history_canonicalises_positional_splits():
    evals, es = _synthetic_evals()
    hist, es_iter = _extract_training_history(_FakeLGB(evals, es))
    assert hist is not None
    assert "l2" in hist
    # valid_0 -> train, valid_1 -> val so both curves survive normalize_history.
    assert set(hist["l2"]) == {"train", "val"}
    assert es_iter == es


def test_extract_xgb_history():
    evals, es = _synthetic_evals()
    hist, es_iter = _extract_training_history(_FakeXGB(evals, es))
    assert hist is not None and "l2" in hist
    assert es_iter == es


def test_extract_catboost_history():
    evals, es = _synthetic_evals()
    hist, es_iter = _extract_training_history(_FakeCatBoost(evals, es))
    assert hist is not None and "l2" in hist
    assert es_iter == es


def test_extract_unwraps_wrapper():
    evals, es = _synthetic_evals()
    wrapped = _Wrapper(_FakeLGB(evals, es))
    assert _unwrap_booster(wrapped) is wrapped.base_estimator
    hist, es_iter = _extract_training_history(wrapped)
    assert hist is not None and es_iter == es


def test_extract_returns_none_for_non_booster():
    class _Plain:
        pass

    hist, es_iter = _extract_training_history(_Plain())
    assert hist is None and es_iter is None


def test_extract_returns_none_for_empty_history():
    hist, es_iter = _extract_training_history(_FakeLGB({}, 0))
    assert hist is None


def test_es_marker_sits_at_val_argmin_biz_value():
    """The composed panel's ES vline must land at the val curve's argmin (the iteration a no-ES fit would overshoot)."""
    from mlframe.reporting.charts.training_curve import compose_training_curve_figure

    evals, es = _synthetic_evals(n=100, es=60)
    hist, es_iter = _extract_training_history(_FakeLGB(evals, es))
    spec = compose_training_curve_figure(hist, es_iteration=es_iter)
    panel = [c for row in spec.panels for c in row if c is not None][0]
    val_series = [s for lab, s in zip(panel.series_labels, panel.y) if lab == "val"][0]
    val_argmin = int(np.argmin(val_series))
    assert abs(es_iter - val_argmin) <= 2, "ES marker must sit near the val argmin"
    assert panel.vlines is not None


def test_render_training_curves_writes_file(tmp_path):
    evals, es = _synthetic_evals()
    base = os.path.join(str(tmp_path), "model_x")

    class _Cfg:
        training_curves = True

    metrics: dict = {}
    _render_training_curves(
        _FakeLGB(evals, es),
        model_name="LGB",
        plot_file=base,
        plot_outputs="matplotlib[png]",
        plot_dpi=80,
        metrics=metrics,
        reporting_config=_Cfg(),
    )
    saved = metrics.get("charts", {}).get("saved", [])
    assert "training_curve" in saved
    paths = metrics["charts"].get("paths", [])
    assert any("training_curve" in p for p in paths)
    # The png landed on disk.
    assert any(f.endswith(".png") and "training_curve" in f for f in os.listdir(str(tmp_path)))


def test_keep_figure_handles_populates_figure_specs(tmp_path):
    """With keep_figure_handles=True the render path stores the pure-data FigureSpec in metrics["figure_specs"]
    (INV-56); default False leaves it absent. Pre-fix the config field did not exist so the getattr defaulted False
    and figure_specs was never populated even when requested."""
    from mlframe.training.configs import ReportingConfig

    assert "keep_figure_handles" in ReportingConfig.model_fields
    assert ReportingConfig().keep_figure_handles is False

    evals, es = _synthetic_evals()
    base = os.path.join(str(tmp_path), "model_keep")
    metrics: dict = {}
    _render_training_curves(
        _FakeLGB(evals, es),
        model_name="LGB",
        plot_file=base,
        plot_outputs="matplotlib[png]",
        plot_dpi=80,
        metrics=metrics,
        reporting_config=ReportingConfig(keep_figure_handles=True),
    )
    assert "training_curve" in metrics.get("figure_specs", {}), "figure_specs must be populated when keep_figure_handles=True"
    spec = metrics["figure_specs"]["training_curve"]
    assert hasattr(spec, "panels"), "stored object is the pure-data FigureSpec"
    # FigureSpec is pickle-safe (no live matplotlib/plotly handle).
    import pickle

    pickle.loads(pickle.dumps(spec))

    metrics_off: dict = {}
    _render_training_curves(
        _FakeLGB(evals, es),
        model_name="LGB",
        plot_file=os.path.join(str(tmp_path), "model_off"),
        plot_outputs="matplotlib[png]",
        plot_dpi=80,
        metrics=metrics_off,
        reporting_config=ReportingConfig(keep_figure_handles=False),
    )
    assert "figure_specs" not in metrics_off


def test_render_training_curves_noop_when_disabled(tmp_path):
    evals, es = _synthetic_evals()

    class _Cfg:
        training_curves = False

    metrics: dict = {}
    _render_training_curves(
        _FakeLGB(evals, es),
        model_name="LGB",
        plot_file=os.path.join(str(tmp_path), "m"),
        plot_outputs="matplotlib[png]",
        plot_dpi=None,
        metrics=metrics,
        reporting_config=_Cfg(),
    )
    assert metrics == {}
    assert not os.listdir(str(tmp_path))


def test_render_training_curves_noop_without_plot_outputs(tmp_path):
    evals, es = _synthetic_evals()

    class _Cfg:
        training_curves = True

    metrics: dict = {}
    _render_training_curves(
        _FakeLGB(evals, es),
        model_name="LGB",
        plot_file=os.path.join(str(tmp_path), "m"),
        plot_outputs=None,
        plot_dpi=None,
        metrics=metrics,
        reporting_config=_Cfg(),
    )
    assert metrics == {}


def test_real_lgb_fit_produces_curves(tmp_path):
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    n, d = 2000, 6
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    Xtr, Xva = X[:1500], X[1500:]
    ytr, yva = y[:1500], y[1500:]
    model = lgb.LGBMClassifier(n_estimators=80, num_leaves=15, verbosity=-1)
    model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], callbacks=[lgb.early_stopping(15, verbose=False)])
    hist, es_iter = _extract_training_history(model)
    assert hist is not None, "fitted lgb with eval_set must expose iteration history"
    # best_iteration_ is the ES point.
    assert es_iter is not None
    # Real lgb names the eval sets training / valid_1; both must survive into the canonical train + val curves
    # once normalize_history applies its alias collapse (the extractor maps the positional valid_N -> val).
    from mlframe.reporting.charts.training_curve import normalize_history

    norm = normalize_history(hist)
    a_metric = next(iter(norm.values()))
    assert "train" in a_metric and "val" in a_metric
