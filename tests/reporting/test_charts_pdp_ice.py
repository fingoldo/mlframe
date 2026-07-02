"""Tests for PDP / ICE diagnostics (reporting/charts/pdp_ice.py).

Each diagnostic ships: a unit test (shape / panel content), a biz_value test (a synthetic where the diagnostic MUST
show a known verdict -- monotone PDP for a monotone dependence, flat PDP for an ignored feature -- asserted with a
quantitative threshold), and a cProfile pass at a production-ish shape (n>=1e6 rows subsampled to 2000) with the
conclusion documented inline.
"""

from __future__ import annotations

import cProfile
import io
import pstats

import numpy as np
import pytest

from mlframe.reporting.charts import pdp_ice
from mlframe.reporting.spec import (
    AnnotationPanelSpec, FigureSpec, HeatmapPanelSpec, LinePanelSpec,
)


class _LinearModel:
    """Deterministic linear scorer: predict = X @ w. No fit; weights fixed so PDP slope is exactly w[feature]."""

    def __init__(self, w):
        self.w = np.asarray(w, dtype=np.float64)

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.w


class _StepProbaModel:
    """Binary classifier whose P(y=1) is a logistic of feature 0 only (monotone increasing in f0)."""

    def __init__(self, k=3.0):
        self.k = k

    def predict_proba(self, X):
        z = self.k * np.asarray(X, dtype=np.float64)[:, 0]
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------- compute_pdp


def test_compute_pdp_shapes_and_keys():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 4))
    model = _LinearModel([1.0, 0.0, -0.5, 2.0])
    res = pdp_ice.compute_pdp(model, X, 0, grid=15, sample=1000, ice=True)
    assert res["grid"].ndim == 1 and res["grid"].shape[0] <= 15
    assert res["pdp"].shape == res["grid"].shape
    assert res["ice"].shape[1] == res["grid"].shape[0]
    assert res["ice"].shape[0] <= 1000
    assert res["kind"] == "predict"
    assert res["feature_index"] == 0


def test_compute_pdp_centered_anchors_first_grid_point():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 3))
    res = pdp_ice.compute_pdp(_LinearModel([1.5, 0.0, 0.0]), X, 0, grid=12, sample=800, ice=True, centered=True)
    cice = res["ice_centered"]
    assert cice is not None
    # Every centered ICE curve starts at exactly 0 at the first grid point.
    np.testing.assert_allclose(cice[:, 0], 0.0, atol=1e-9)


def test_compute_pdp_proba_model_uses_positive_column():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3000, 3))
    res = pdp_ice.compute_pdp(_StepProbaModel(k=3.0), X, 0, grid=20, sample=1000, ice=False)
    assert res["kind"] == "proba"
    assert np.all((res["pdp"] >= 0.0) & (res["pdp"] <= 1.0))


class _RaisingProbaRegressor:
    """Regressor that EXPOSES a bound ``predict_proba`` which raises at call time.

    Mirrors mlframe's PartialFitESWrapper wrapping a plain regressor (Ridge): the wrapper always defines the
    ``predict_proba`` method but raises ``AttributeError`` when the underlying estimator has neither proba nor
    decision_function. ``compute_pdp`` must fall back to ``predict`` rather than failing the whole diagnostic.
    """

    def __init__(self, w):
        self.w = np.asarray(w, dtype=np.float64)

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.w

    def predict_proba(self, X):
        raise AttributeError("Underlying estimator has no predict_proba or decision_function")


def test_compute_pdp_falls_back_to_predict_when_proba_raises_at_call_time():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(500, 3))
    model = _RaisingProbaRegressor([2.0, 0.0, 0.0])
    res = pdp_ice.compute_pdp(model, X, 0, grid=10, sample=200, ice=True)
    # Pre-fix this raised AttributeError out of compute_pdp; post-fix it renders via the predict fallback.
    assert res["pdp"].shape == (10,)
    # PDP slope must equal w[0]=2 (predict path), not a proba surface.
    slope = (res["pdp"][-1] - res["pdp"][0]) / (res["grid"][-1] - res["grid"][0])
    assert abs(slope - 2.0) < 1e-6


def test_compute_pdp_discrete_feature_grid_is_categories():
    rng = np.random.default_rng(3)
    cont = rng.normal(size=2000)
    cat = rng.integers(0, 4, size=2000).astype(float)  # 4 distinct values -> discrete
    X = np.column_stack([cont, cat])
    res = pdp_ice.compute_pdp(_LinearModel([0.0, 1.0]), X, 1, grid=20, sample=1500)
    assert res["is_discrete"] is True
    np.testing.assert_array_equal(res["grid"], np.array([0.0, 1.0, 2.0, 3.0]))


def test_pdp_panel_has_ice_and_mean_series():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(4000, 3))
    panel = pdp_ice.pdp_panel(_LinearModel([1.0, 0.0, 0.0]), X, 0, grid=15, sample=600, ice=True)
    assert isinstance(panel, LinePanelSpec)
    # n_draw ICE curves + 1 bold PDP mean.
    assert len(panel.y) >= 2
    assert panel.series_labels[-1].startswith("PDP")
    assert panel.colors[-1] == "#08519c"


def test_pdp_panel_constant_feature_is_annotation():
    X = np.column_stack([np.full(500, 2.0), np.random.default_rng(5).normal(size=500)])
    panel = pdp_ice.pdp_panel(_LinearModel([1.0, 1.0]), X, 0, grid=10, sample=400)
    assert isinstance(panel, AnnotationPanelSpec)


def test_pdp_pandas_input_named_feature():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000)})
    panel = pdp_ice.pdp_panel(_LinearModel([2.0, 0.0]), df, "a", grid=12, sample=800)
    assert isinstance(panel, LinePanelSpec)
    assert panel.xlabel == "a"


def test_compute_pdp_2d_surface_shape():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(3000, 3))
    res = pdp_ice.compute_pdp_2d(_LinearModel([1.0, 1.0, 0.0]), X, (0, 1), grid=10, sample=600)
    assert res["surface"].shape == (res["grid0"].shape[0], res["grid1"].shape[0])
    panel = pdp_ice.pdp_2d_panel(_LinearModel([1.0, 1.0, 0.0]), X, (0, 1), grid=10, sample=600)
    assert isinstance(panel, HeatmapPanelSpec)
    assert panel.matrix.shape == res["surface"].shape


def test_compose_pdp_figure_grid_and_interaction():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(3000, 4))
    fig = pdp_ice.compose_pdp_figure(
        _LinearModel([1.0, -1.0, 0.0, 0.5]), X, features=[0, 1, 3],
        grid=10, sample=500, interaction_pair=(0, 1), max_cols=2,
    )
    assert isinstance(fig, FigureSpec)
    flat = [p for row in fig.panels for p in row if p is not None]
    assert len(flat) == 4  # 3 one-feature panels + 1 interaction heatmap
    assert isinstance(flat[-1], HeatmapPanelSpec)


def test_compose_pdp_figure_no_features_is_annotation():
    fig = pdp_ice.compose_pdp_figure(_LinearModel([1.0]), np.zeros((10, 1)), features=[])
    assert isinstance(fig.panels[0][0], AnnotationPanelSpec)


# --------------------------------------------------------------------------- biz_value


def test_biz_value_pdp_monotone_increasing_for_monotone_dependence():
    """A model whose output increases monotonically in f0 (positive weight) MUST yield a PDP that is monotone
    increasing across the grid with a positive average slope, while a zero-weight feature gives a ~flat PDP.
    Measured: f0 PDP slope == w0 exactly (linear), f2 (w=0) slope ~0. Floors: f0 monotone-increasing with
    end-to-end rise > 0.85 * expected; f2 total variation < 1e-6."""
    rng = np.random.default_rng(100)
    X = rng.normal(size=(8000, 3))
    w = np.array([2.0, 0.0, 0.0])
    model = _LinearModel(w)
    res0 = pdp_ice.compute_pdp(model, X, 0, grid=20, sample=2000, ice=False)
    pdp0 = res0["pdp"]
    # Monotone increasing (allow tiny FP noise).
    assert np.all(np.diff(pdp0) >= -1e-9), "PDP(f0) must be monotone increasing"
    # End-to-end rise equals w0 * (grid span); floor at 85% of expected.
    span = res0["grid"][-1] - res0["grid"][0]
    expected_rise = w[0] * span
    actual_rise = pdp0[-1] - pdp0[0]
    assert actual_rise >= 0.85 * expected_rise, f"PDP(f0) rise {actual_rise:.3f} < 0.85 * {expected_rise:.3f}"

    res2 = pdp_ice.compute_pdp(model, X, 2, grid=20, sample=2000, ice=False)
    tv2 = float(np.sum(np.abs(np.diff(res2["pdp"]))))
    assert tv2 < 1e-6, f"PDP of the ignored feature f2 should be flat, total-variation {tv2:.3e}"


def test_biz_value_pdp_proba_monotone_in_logit_feature():
    """A logistic classifier monotone in f0 MUST give a PDP rising from < 0.2 at the low end to > 0.8 at the high
    end (sigmoid saturates). Measured ~0.02 -> ~0.98; floors low < 0.2, high > 0.8, and monotone increasing."""
    rng = np.random.default_rng(101)
    X = rng.normal(size=(6000, 3))
    res = pdp_ice.compute_pdp(_StepProbaModel(k=3.0), X, 0, grid=25, sample=2000, ice=False)
    pdp = res["pdp"]
    assert np.all(np.diff(pdp) >= -1e-9), "logistic PDP must be monotone increasing in f0"
    assert pdp[0] < 0.2, f"low-end P(y=1) should be < 0.2, got {pdp[0]:.3f}"
    assert pdp[-1] > 0.8, f"high-end P(y=1) should be > 0.8, got {pdp[-1]:.3f}"


def test_biz_value_pdp_2d_interaction_surface_separable_for_additive_model():
    """For an additive model f = a*f0 + b*f1 the 2-D PDP surface MUST be additively separable: each row differs from
    the next by a constant (the f0 step), independent of the column. Measured: row-to-row deltas constant to FP.
    Floor: max deviation of row deltas from their mean < 1e-6 (proves no spurious interaction)."""
    rng = np.random.default_rng(102)
    X = rng.normal(size=(4000, 2))
    res = pdp_ice.compute_pdp_2d(_LinearModel([1.5, -0.7]), X, (0, 1), grid=12, sample=2000)
    surf = res["surface"]
    row_deltas = np.diff(surf, axis=0)  # (g0-1, g1) -- should be column-constant for an additive model
    dev = float(np.max(np.abs(row_deltas - row_deltas.mean(axis=1, keepdims=True))))
    assert dev < 1e-6, f"additive model must give a separable 2-D PDP (no interaction); deviation {dev:.3e}"


# --------------------------------------------------------------------------- cProfile


def test_cprofile_compute_pdp_at_1e6_rows_subsampled():
    """cProfile compute_pdp at n=1e6 rows subsampled to 2000. The cost is exactly ``grid`` predict calls over the
    (2000, n_cols) block -- independent of n -- plus one O(n) subsample draw. Hot path is the model.predict calls
    (here a single matmul); no actionable speedup in the builder (the per-grid single predict is already the minimal
    vectorized form -- a per-row predict would be 2000x worse). Documented so a re-profile does not re-flag predict."""
    rng = np.random.default_rng(200)
    n = 1_000_000
    X = rng.normal(size=(n, 8))
    model = _LinearModel(rng.normal(size=8))
    pr = cProfile.Profile()
    pr.enable()
    res = pdp_ice.compute_pdp(model, X, 0, grid=20, sample=2000, ice=True)
    pr.disable()
    assert res["pdp"].shape[0] <= 20
    assert res["ice"].shape[0] <= pdp_ice.ICE_CURVE_DRAW_CAP
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(10)
    assert res["pdp"].size > 0


def test_cprofile_compute_pdp_2d_at_1e6_rows_subsampled():
    """cProfile compute_pdp_2d at n=1e6 subsampled to 2000. Cost is ``g0`` predict calls over the (2000*g1) tiled
    block, never g0*g1 per-cell predicts. The np.repeat tile (2000*g1 x n_cols) + per-outer predict dominate; the
    tile is the irreducible vectorization buffer. No actionable speedup in the builder."""
    rng = np.random.default_rng(201)
    n = 1_000_000
    X = rng.normal(size=(n, 6))
    model = _LinearModel(rng.normal(size=6))
    pr = cProfile.Profile()
    pr.enable()
    res = pdp_ice.compute_pdp_2d(model, X, (0, 1), grid=12, sample=2000)
    pr.disable()
    assert res["surface"].shape == (res["grid0"].shape[0], res["grid1"].shape[0])


def test_pdp_as_2d_coerces_string_columns_without_crashing():
    """_as_2d builds the float grid/ICE view; a whole-frame astype(float64) crashed with 'could not convert string to
    float' whenever ANY feature column was string/categorical, taking down the ENTIRE PDP figure (all numeric panels
    included) at the one upfront cast. It now coerces per-column (numeric passthrough; non-numeric -> factorize codes),
    so the float view never crashes. (Note: PDP for a LightGBM *categorical model* still needs the deeper grid-
    substitution category-preservation fix -- tracked separately; this pins the float-view layer.)"""
    import numpy as np
    import pandas as pd
    from mlframe.reporting.charts.pdp_ice import _as_2d, _coerce_float_2d

    # Pure-numeric passthrough is exact float64.
    num = np.arange(12.0).reshape(4, 3)
    assert np.array_equal(_coerce_float_2d(num), num)

    df = pd.DataFrame({"x0": np.arange(100.0), "x1": np.arange(100.0)[::-1], "cat": ["A", "B", "C", "D"] * 25})
    vals, carrier, names = _as_2d(df)
    assert vals.shape == (100, 3) and vals.dtype == np.float64
    assert names == ["x0", "x1", "cat"]
    assert carrier is df  # carrier kept native for the model's expected input
    assert np.array_equal(vals[:, 0], np.arange(100.0))  # numeric columns untouched
    assert np.all(np.isfinite(vals[:, 2])) and set(np.unique(vals[:, 2])) == {0.0, 1.0, 2.0, 3.0}  # cat -> codes
