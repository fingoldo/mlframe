"""Tests for the one-glance MODEL CARD (charts/model_card.py).

Covers figure structure (header text + headline bar + up to 3 minis), headline-metric presence in the
header text, verdict color logic at the green/amber/red thresholds, the biz_value traffic-light flip
(strong synthetic -> green, near-random / high-noise -> red) for BOTH classification and regression,
degenerate-input degradation, and a bounded cProfile on a production-shape card.
"""

from __future__ import annotations

import cProfile
import io
import pstats

import numpy as np
import pytest

from mlframe.reporting.charts.model_card import (
    AUC_AMBER,
    AUC_GREEN,
    ECE_RED,
    R2_AMBER,
    R2_GREEN,
    compose_model_card_figure,
    model_card_verdict,
    _classification_verdict,
    _regression_verdict,
)
from mlframe.reporting.spec import (
    AnnotationPanelSpec,
    BarPanelSpec,
    FigureSpec,
    HistogramPanelSpec,
    LinePanelSpec,
    ScatterPanelSpec,
)


def _flat(fig: FigureSpec):
    """Helper: Flat."""
    return [p for row in fig.panels for p in row if p is not None]


def _separable_binary(n=4000, seed=0):
    """Strong AND well-calibrated binary: labels drawn FROM the score, so a high-AUC model is also low-ECE.

    Drawing ``y ~ Bernoulli(p)`` with a peaked latent score makes the score the true posterior -> high
    discrimination (AUC well above 0.8) with calibration honest by construction (ECE small)."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 2.5, n)
    p = 1.0 / (1.0 + np.exp(-raw))
    y = (rng.random(n) < p).astype(np.int8)
    return y, p


def _random_binary(n=4000, seed=1):
    """Score carries no signal -> AUC ~ 0.5."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return y, rng.random(n)


def _strong_regression(n=4000, seed=0):
    """Helper: Strong regression."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, n)
    yt = 2.0 * x + 5.0 + rng.normal(0.0, 0.3, n)
    yp = 2.0 * x + 5.0 + rng.normal(0.0, 0.3, n)
    return yt, yp


def _noisy_regression(n=4000, seed=2):
    """Prediction nearly uncorrelated with target -> R2 ~ 0."""
    rng = np.random.default_rng(seed)
    yt = rng.normal(0.0, 5.0, n)
    yp = rng.normal(0.0, 5.0, n)
    return yt, yp


# ----------------------------------------------------------------------------
# Unit: figure structure
# ----------------------------------------------------------------------------


def test_classification_card_structure():
    """Classification card structure."""
    y, s = _separable_binary()
    fig = compose_model_card_figure(task="classification", y_true=y, y_score=s, model_name="m", split="test")
    assert isinstance(fig, FigureSpec)
    # Row 0: header (annotation) + headline bar; row 1: 3 minis.
    assert isinstance(fig.panels[0][0], AnnotationPanelSpec)
    assert isinstance(fig.panels[0][1], BarPanelSpec)
    minis = [p for p in fig.panels[1] if p is not None]
    assert len(minis) == 3
    assert all(isinstance(p, (LinePanelSpec, AnnotationPanelSpec)) for p in minis)


def test_regression_card_structure():
    """Regression card structure."""
    yt, yp = _strong_regression()
    fig = compose_model_card_figure(task="regression", y_true=yt, y_pred=yp, model_name="m", split="oof")
    assert isinstance(fig.panels[0][0], AnnotationPanelSpec)
    assert isinstance(fig.panels[0][1], BarPanelSpec)
    minis = [p for p in fig.panels[1] if p is not None]
    assert len(minis) == 3
    assert isinstance(minis[0], ScatterPanelSpec)
    assert isinstance(minis[1], HistogramPanelSpec)
    assert isinstance(minis[2], ScatterPanelSpec)


def test_classification_header_lists_all_headline_metrics():
    """Classification header lists all headline metrics."""
    y, s = _separable_binary()
    fig = compose_model_card_figure(task="classification", y_true=y, y_score=s)
    header = fig.panels[0][0].text
    for key in ("ROC_AUC", "PR_AUC", "ECE", "Brier", "KS", "MCC"):
        assert key in header, f"{key} missing from header"


def test_regression_header_lists_all_headline_metrics():
    """Regression header lists all headline metrics."""
    yt, yp = _strong_regression()
    fig = compose_model_card_figure(task="regression", y_true=yt, y_pred=yp)
    header = fig.panels[0][0].text
    for key in ("RMSE", "MAE", "R2", "bias", "hetero"):
        assert key in header, f"{key} missing from header"


def test_bad_task_raises():
    """Bad task raises."""
    with pytest.raises(ValueError, match="unknown task"):
        compose_model_card_figure(task="ranking", y_true=np.zeros(3), y_score=np.zeros(3))


def test_classification_requires_score():
    """Classification requires score."""
    with pytest.raises(ValueError, match="requires y_score"):
        compose_model_card_figure(task="classification", y_true=np.array([0, 1]))


def test_regression_requires_pred():
    """Regression requires pred."""
    with pytest.raises(ValueError, match="requires y_pred"):
        compose_model_card_figure(task="regression", y_true=np.array([0.0, 1.0]))


# ----------------------------------------------------------------------------
# Unit: verdict color threshold logic
# ----------------------------------------------------------------------------


def test_classification_verdict_thresholds():
    """Classification verdict thresholds."""
    base = {"PR_AUC": 0.5, "Brier": 0.2, "KS": 0.5, "MCC": 0.5}
    green = _classification_verdict({**base, "ROC_AUC": AUC_GREEN + 0.05, "ECE": 0.02})
    amber = _classification_verdict({**base, "ROC_AUC": (AUC_GREEN + AUC_AMBER) / 2, "ECE": 0.02})
    red = _classification_verdict({**base, "ROC_AUC": AUC_AMBER - 0.05, "ECE": 0.02})
    assert green.color == "green"
    assert amber.color == "amber"
    assert red.color == "red"


def test_classification_calibration_downgrade():
    """A discriminating but badly miscalibrated model is knocked off green."""
    m = {"PR_AUC": 0.6, "Brier": 0.2, "KS": 0.6, "MCC": 0.6, "ROC_AUC": 0.9, "ECE": ECE_RED + 0.05}
    v = _classification_verdict(m)
    assert v.color == "amber"
    assert "ECE" in v.reason


def test_regression_verdict_thresholds():
    """Regression verdict thresholds."""
    green = _regression_verdict({"RMSE": 1.0, "MAE": 0.8, "R2": R2_GREEN + 0.1, "bias": 0.0, "hetero": 0.0}, y_std=10.0)
    amber = _regression_verdict({"RMSE": 1.0, "MAE": 0.8, "R2": (R2_GREEN + R2_AMBER) / 2, "bias": 0.0, "hetero": 0.0}, y_std=10.0)
    red = _regression_verdict({"RMSE": 1.0, "MAE": 0.8, "R2": R2_AMBER - 0.1, "bias": 0.0, "hetero": 0.0}, y_std=10.0)
    assert green.color == "green"
    assert amber.color == "amber"
    assert red.color == "red"


def test_regression_bias_downgrade():
    """High R2 but a large bias relative to target spread drops off green."""
    v = _regression_verdict({"RMSE": 1.0, "MAE": 0.8, "R2": 0.9, "bias": 5.0, "hetero": 0.0}, y_std=10.0)
    assert v.color == "amber"
    assert "bias" in v.reason


# ----------------------------------------------------------------------------
# biz_value: the traffic light must FLIP strong->green vs weak->red
# ----------------------------------------------------------------------------


def test_biz_classification_verdict_flips_green_to_red():
    """Biz classification verdict flips green to red."""
    y_g, s_g = _separable_binary()
    y_w, s_w = _random_binary()
    v_green = model_card_verdict(task="classification", y_true=y_g, y_score=s_g)
    v_red = model_card_verdict(task="classification", y_true=y_w, y_score=s_w)
    assert v_green.color == "green", f"strong synthetic should be green, got {v_green.color} ({v_green.reason})"
    assert v_red.color == "red", f"random synthetic should be red, got {v_red.color} ({v_red.reason})"
    # Headline AUC must reflect the gap (strong well above 0.8, weak near 0.5).
    assert v_green.headline["ROC_AUC"] >= 0.85
    assert v_red.headline["ROC_AUC"] <= 0.60


def test_biz_regression_verdict_flips_green_to_red():
    """Biz regression verdict flips green to red."""
    yt_g, yp_g = _strong_regression()
    yt_w, yp_w = _noisy_regression()
    v_green = model_card_verdict(task="regression", y_true=yt_g, y_pred=yp_g)
    v_red = model_card_verdict(task="regression", y_true=yt_w, y_pred=yp_w)
    assert v_green.color == "green", f"low-noise regression should be green, got {v_green.color} ({v_green.reason})"
    assert v_red.color == "red", f"high-noise regression should be red, got {v_red.color} ({v_red.reason})"
    assert v_green.headline["R2"] >= 0.9
    assert v_red.headline["R2"] <= 0.1


def test_verdict_label_in_suptitle():
    """Verdict label in suptitle."""
    y, s = _separable_binary()
    fig = compose_model_card_figure(task="classification", y_true=y, y_score=s, model_name="lgbm", split="test")
    assert "STRONG" in fig.suptitle
    assert "lgbm" in fig.suptitle


# ----------------------------------------------------------------------------
# Degenerate input degrades to an honest text card (no fake chart)
# ----------------------------------------------------------------------------


def test_single_class_degrades_to_text():
    """Single class degrades to text."""
    y = np.ones(500, dtype=np.int8)
    s = np.random.default_rng(0).random(500)
    fig = compose_model_card_figure(task="classification", y_true=y, y_score=s)
    panels = _flat(fig)
    assert len(panels) == 1
    assert isinstance(panels[0], AnnotationPanelSpec)
    assert "one class" in panels[0].text


def test_no_finite_pairs_degrades_to_text():
    """No finite pairs degrades to text."""
    yt = np.array([np.nan, np.nan, np.nan])
    yp = np.array([1.0, 2.0, 3.0])
    fig = compose_model_card_figure(task="regression", y_true=yt, y_pred=yp)
    panels = _flat(fig)
    assert len(panels) == 1
    assert isinstance(panels[0], AnnotationPanelSpec)


def test_verdict_standalone_raises_on_single_class():
    """Verdict standalone raises on single class."""
    y = np.zeros(100, dtype=np.int8)
    s = np.random.default_rng(0).random(100)
    with pytest.raises(ValueError, match="both classes"):
        model_card_verdict(task="classification", y_true=y, y_score=s)


# ----------------------------------------------------------------------------
# Render smoke: the card actually draws on the matplotlib backend
# ----------------------------------------------------------------------------


def test_card_renders_matplotlib(tmp_path):
    """Card renders matplotlib."""
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    from mlframe.reporting.renderers.base import get_renderer

    y, s = _separable_binary()
    fig = compose_model_card_figure(task="classification", y_true=y, y_score=s)
    rend = get_renderer("matplotlib")
    obj = rend.render(fig)
    out = tmp_path / "card.png"
    rend.save(obj, str(out), "png")
    assert out.exists() and out.stat().st_size > 0
    import matplotlib.pyplot as plt

    plt.close(obj)


# ----------------------------------------------------------------------------
# cProfile: bounded build at production shape (minis subsampled)
# ----------------------------------------------------------------------------


def test_cprofile_bounded_at_production_shape():
    """Spec build on a 1M-row card stays well bounded: minis decimate/subsample, metrics reuse one sort."""
    rng = np.random.default_rng(0)
    n = 1_000_000
    y = rng.integers(0, 2, n)
    s = 1.0 / (1.0 + np.exp(-(rng.standard_normal(n) + 1.5 * y)))
    pr = cProfile.Profile()
    pr.enable()
    fig = compose_model_card_figure(task="classification", y_true=y, y_score=s)
    pr.disable()
    st = pstats.Stats(pr, stream=io.StringIO())
    total = st.total_tt
    assert total < 5.0, f"model card build too slow at n=1M: {total:.2f}s"
    # Mini ROC / gain decimate to the coarse cap; verify the drawn curves are small.
    minis = [p for p in fig.panels[1] if p is not None]
    roc = minis[0]
    assert isinstance(roc, LinePanelSpec)
    assert len(roc.x) <= 400
