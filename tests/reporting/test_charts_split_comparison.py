"""Tests for the cross-split OVERFITTING panel (charts/split_comparison.py).

Covers figure structure (grouped-bar metrics-per-split + delta-table/verdict), per-split metric presence, delta-table
correctness, the overfit traffic-light flag logic (red/amber/green at the gap thresholds), missing-split + degenerate
(single-class) handling, precomputed-metrics short-circuit, the biz_value flip (memorize-train synthetic -> big gap +
RED, well-generalizing synthetic -> small gap + GREEN) for both classification and regression, and a bounded cProfile.
"""

from __future__ import annotations

import cProfile
import io
import pstats

import numpy as np
import pytest

from mlframe.reporting.charts.split_comparison import (
    AUC_GAP_AMBER, AUC_GAP_RED, RMSE_RATIO_AMBER, RMSE_RATIO_RED,
    OverfitVerdict, compose_split_comparison_figure, overfit_verdict,
)
from mlframe.reporting.spec import AnnotationPanelSpec, BarPanelSpec, FigureSpec


def _flat(fig: FigureSpec):
    return [p for row in fig.panels for p in row if p is not None]


def _clf_split(n, sep, seed):
    """Binary split: labels drawn from a latent score so a higher ``sep`` yields a higher (honest) ROC_AUC."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, sep, n)
    p = 1.0 / (1.0 + np.exp(-raw))
    y = (rng.random(n) < p).astype(np.int8)
    return {"y_true": y, "y_score": p}


def _reg_split(n, noise, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, n)
    yt = 2.0 * x + 5.0 + rng.normal(0.0, 0.3, n)
    yp = 2.0 * x + 5.0 + rng.normal(0.0, noise, n)
    return {"y_true": yt, "y_pred": yp}


# ---------------------------------------------------------------------------
# Overfit synthetics: train memorized (huge gap) vs generalizing (tiny gap).
# ---------------------------------------------------------------------------


def _overfit_clf(seed=0):
    """Train AUC ~0.99 (model memorized labels) but test AUC ~0.70 -> large train-test gap, RED flag."""
    rng = np.random.default_rng(seed)
    n = 4000
    # Train: score IS (almost) the label -> near-perfect discrimination.
    y_tr = rng.integers(0, 2, n).astype(np.int8)
    s_tr = np.clip(y_tr + rng.normal(0.0, 0.08, n), 0.0, 1.0)
    # Test: weak signal -> AUC ~0.70.
    y_te = rng.integers(0, 2, n).astype(np.int8)
    s_te = 1.0 / (1.0 + np.exp(-(0.9 * (y_te - 0.5) + rng.normal(0.0, 1.0, n))))
    return {"train": {"y_true": y_tr, "y_score": s_tr}, "test": {"y_true": y_te, "y_score": s_te}}


def _generalizing_clf(seed=1):
    """Train and test drawn the same way (labels from the score) -> AUCs match, small gap, GREEN flag."""
    return {"train": _clf_split(4000, 1.6, seed), "test": _clf_split(4000, 1.6, seed + 100)}


def _overfit_reg(seed=0):
    """Train RMSE tiny (memorized), test RMSE large -> high test/train RMSE ratio, RED flag."""
    rng = np.random.default_rng(seed)
    n = 4000
    x_tr = rng.uniform(0.0, 10.0, n)
    yt_tr = 2.0 * x_tr + 5.0 + rng.normal(0.0, 1.0, n)
    yp_tr = yt_tr + rng.normal(0.0, 0.05, n)        # train predictions nearly exact
    x_te = rng.uniform(0.0, 10.0, n)
    yt_te = 2.0 * x_te + 5.0 + rng.normal(0.0, 1.0, n)
    yp_te = yt_te + rng.normal(0.0, 3.0, n)         # test predictions badly off
    return {"train": {"y_true": yt_tr, "y_pred": yp_tr}, "test": {"y_true": yt_te, "y_pred": yp_te}}


def _generalizing_reg(seed=1):
    return {"train": _reg_split(4000, 0.4, seed), "test": _reg_split(4000, 0.4, seed + 100)}


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_figure_structure_classification():
    per_split = {"train": _clf_split(3000, 2.0, 0), "val": _clf_split(2000, 1.4, 1), "test": _clf_split(2000, 1.3, 2)}
    fig = compose_split_comparison_figure(per_split, task="classification", model_name="lgbm")
    assert isinstance(fig, FigureSpec)
    panels = _flat(fig)
    assert len(panels) == 2
    bar = next(p for p in panels if isinstance(p, BarPanelSpec))
    table = next(p for p in panels if isinstance(p, AnnotationPanelSpec))
    # Grouped bars: one series per split, one category per headline metric.
    assert isinstance(bar.values, tuple) and len(bar.values) == 3
    assert bar.series_labels == ("train", "val", "test")
    assert "ROC_AUC" in bar.categories and "ECE" in bar.categories
    assert "ROC_AUC" in table.text and "OVERFIT" in table.text or "GENERALIZES" in table.text


def test_figure_structure_regression():
    per_split = {"train": _reg_split(3000, 0.3, 0), "test": _reg_split(2000, 0.6, 2)}
    fig = compose_split_comparison_figure(per_split, task="regression", model_name="ridge")
    bar = next(p for p in _flat(fig) if isinstance(p, BarPanelSpec))
    assert "R2" in bar.categories and "RMSE" in bar.categories
    assert len(bar.values) == 2


def test_split_order_canonical():
    """Splits passed out of order still render train -> val -> test -> oof left to right."""
    per_split = {
        "test": _clf_split(2000, 1.3, 2), "oof": _clf_split(2000, 1.4, 3),
        "train": _clf_split(3000, 2.0, 0), "val": _clf_split(2000, 1.4, 1),
    }
    fig = compose_split_comparison_figure(per_split, task="classification")
    bar = next(p for p in _flat(fig) if isinstance(p, BarPanelSpec))
    assert bar.series_labels == ("train", "val", "test", "oof")


# ---------------------------------------------------------------------------
# Delta table correctness + verdict logic
# ---------------------------------------------------------------------------


def test_delta_table_matches_metric_difference():
    """The delta-table train->test cell equals the raw metric difference computed independently."""
    per_split = {
        "train": {"metrics": {"ROC_AUC": 0.95, "PR_AUC": 0.9, "KS": 0.8, "ECE": 0.02, "Brier": 0.08}},
        "test": {"metrics": {"ROC_AUC": 0.78, "PR_AUC": 0.7, "KS": 0.5, "ECE": 0.06, "Brier": 0.18}},
    }
    fig = compose_split_comparison_figure(per_split, task="classification")
    table = next(p for p in _flat(fig) if isinstance(p, AnnotationPanelSpec))
    # train->test ROC_AUC delta = 0.78 - 0.95 = -0.170.
    assert "-0.170" in table.text


def test_verdict_red_on_large_auc_gap():
    v = overfit_verdict(task="classification", per_split={
        "train": {"metrics": {"ROC_AUC": 0.99}}, "test": {"metrics": {"ROC_AUC": 0.70}},
    })
    assert v.color == "red" and v.label == "OVERFIT"
    assert v.gap == pytest.approx(0.29, abs=1e-9) and v.gap >= AUC_GAP_RED


def test_verdict_amber_on_moderate_auc_gap():
    v = overfit_verdict(task="classification", per_split={
        "train": {"metrics": {"ROC_AUC": 0.90}}, "test": {"metrics": {"ROC_AUC": 0.84}},
    })
    assert v.color == "amber" and AUC_GAP_AMBER <= v.gap < AUC_GAP_RED


def test_verdict_green_on_small_auc_gap():
    v = overfit_verdict(task="classification", per_split={
        "train": {"metrics": {"ROC_AUC": 0.85}}, "test": {"metrics": {"ROC_AUC": 0.84}},
    })
    assert v.color == "green" and v.gap < AUC_GAP_AMBER


def test_verdict_regression_rmse_ratio():
    v_red = overfit_verdict(task="regression", per_split={
        "train": {"metrics": {"RMSE": 1.0}}, "test": {"metrics": {"RMSE": 2.0}},
    })
    assert v_red.color == "red" and v_red.gap >= RMSE_RATIO_RED
    v_green = overfit_verdict(task="regression", per_split={
        "train": {"metrics": {"RMSE": 1.0}}, "test": {"metrics": {"RMSE": 1.05}},
    })
    assert v_green.color == "green" and v_green.gap < RMSE_RATIO_AMBER


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_missing_split_shows_what_is_present():
    """Only train + test present (no val) -> still renders, verdict uses train->test."""
    per_split = {"train": _clf_split(3000, 2.0, 0), "test": _clf_split(2000, 1.3, 2)}
    fig = compose_split_comparison_figure(per_split, task="classification")
    bar = next(p for p in _flat(fig) if isinstance(p, BarPanelSpec))
    assert bar.series_labels == ("train", "test")


def test_single_class_split_annotated_and_excluded():
    """A degenerate single-class split is annotated in the table and excluded from bars/verdict."""
    rng = np.random.default_rng(0)
    per_split = {
        "train": _clf_split(3000, 2.0, 0),
        "val": {"y_true": np.ones(2000, dtype=np.int8), "y_score": rng.random(2000)},  # one class
        "test": _clf_split(2000, 1.3, 2),
    }
    fig = compose_split_comparison_figure(per_split, task="classification")
    bar = next(p for p in _flat(fig) if isinstance(p, BarPanelSpec))
    assert bar.series_labels == ("train", "test")  # val excluded
    table = next(p for p in _flat(fig) if isinstance(p, AnnotationPanelSpec))
    assert "skipped" in table.text and "val" in table.text


def test_fewer_than_two_usable_splits_degrades():
    rng = np.random.default_rng(0)
    per_split = {"train": {"y_true": np.ones(100, dtype=np.int8), "y_score": rng.random(100)}}
    fig = compose_split_comparison_figure(per_split, task="classification")
    panels = _flat(fig)
    assert len(panels) == 1 and isinstance(panels[0], AnnotationPanelSpec)
    assert ">= 2 usable splits" in panels[0].text


def test_no_splits_degrades():
    fig = compose_split_comparison_figure({}, task="classification")
    panels = _flat(fig)
    assert len(panels) == 1 and isinstance(panels[0], AnnotationPanelSpec)


def test_precomputed_metrics_short_circuit():
    """Passing metrics directly skips the raw-array path entirely."""
    per_split = {
        "train": {"metrics": {"ROC_AUC": 0.9}}, "test": {"metrics": {"ROC_AUC": 0.85}},
    }
    fig = compose_split_comparison_figure(per_split, task="classification")
    assert isinstance(fig, FigureSpec)
    bar = next(p for p in _flat(fig) if isinstance(p, BarPanelSpec))
    assert bar.series_labels == ("train", "test")


def test_overfit_verdict_raises_on_single_split():
    with pytest.raises(ValueError):
        overfit_verdict(task="classification", per_split={"train": {"metrics": {"ROC_AUC": 0.9}}})


# ---------------------------------------------------------------------------
# biz_value: overfit -> RED + big gap; generalizing -> GREEN + small gap.
# ---------------------------------------------------------------------------


def test_biz_val_split_comparison_overfit_clf_red_big_gap():
    """Memorize-train classification synthetic: train-test ROC_AUC gap >= 0.20 and the flag fires RED."""
    per_split = _overfit_clf()
    v = overfit_verdict(task="classification", per_split=per_split)
    assert v.color == "red", f"expected RED, got {v.color} ({v.reason})"
    assert v.gap >= 0.20, f"train-test AUC gap should be large; got {v.gap:.3f}"


def test_biz_val_split_comparison_generalizing_clf_green_small_gap():
    """Well-generalizing classification synthetic: gap < AMBER threshold and the flag is GREEN."""
    per_split = _generalizing_clf()
    v = overfit_verdict(task="classification", per_split=per_split)
    assert v.color == "green", f"expected GREEN, got {v.color} ({v.reason})"
    assert abs(v.gap) < AUC_GAP_AMBER, f"train-test AUC gap should be small; got {v.gap:.3f}"


def test_biz_val_split_comparison_overfit_reg_red():
    """Memorize-train regression synthetic: test/train RMSE ratio >= RED threshold and the flag fires RED."""
    v = overfit_verdict(task="regression", per_split=_overfit_reg())
    assert v.color == "red", f"expected RED, got {v.color} ({v.reason})"
    assert v.gap >= RMSE_RATIO_RED, f"RMSE ratio should be large; got {v.gap:.2f}"


def test_biz_val_split_comparison_generalizing_reg_green():
    v = overfit_verdict(task="regression", per_split=_generalizing_reg())
    assert v.color == "green", f"expected GREEN, got {v.color} ({v.reason})"
    assert v.gap < RMSE_RATIO_AMBER, f"RMSE ratio should be near 1.0; got {v.gap:.2f}"


def test_biz_val_overfit_flag_flips_between_synthetics():
    """The same model on the same data type flips RED->GREEN purely on whether it overfits -- the core value."""
    red = overfit_verdict(task="classification", per_split=_overfit_clf())
    green = overfit_verdict(task="classification", per_split=_generalizing_clf())
    assert red.color == "red" and green.color == "green"
    assert red.gap - green.gap >= 0.18, "the overfit synthetic must have a materially larger gap"


# ---------------------------------------------------------------------------
# cProfile (bounded)
# ---------------------------------------------------------------------------


def test_cprofile_bounded():
    """Production-shape (4 splits x 80k rows) figure build stays well under a generous wall budget."""
    rng = np.random.default_rng(0)
    n = 80_000
    per_split = {}
    for name, sep in (("train", 2.0), ("val", 1.5), ("test", 1.4), ("oof", 1.5)):
        raw = rng.normal(0.0, sep, n)
        p = 1.0 / (1.0 + np.exp(-raw))
        y = (rng.random(n) < p).astype(np.int8)
        per_split[name] = {"y_true": y, "y_score": p}
    pr = cProfile.Profile()
    pr.enable()
    fig = compose_split_comparison_figure(per_split, task="classification", model_name="bench")
    pr.disable()
    assert isinstance(fig, FigureSpec)
    st = pstats.Stats(pr, stream=io.StringIO())
    total = st.total_tt
    assert total < 5.0, f"split-comparison build too slow: {total:.3f}s"
