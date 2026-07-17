"""Tests for the 2D partial-dependence-plot chart composer and its interaction-residual helper."""

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlframe.reporting.charts.pdp_2d import compose_pdp_2d_figure, interaction_residual
from mlframe.reporting.charts.pdp_ice import compute_pdp_2d


class _ProductModel:
    """Strong f0*f1 interaction: response is the product (non-additive)."""

    def predict(self, X):
        """Predict."""
        a = np.asarray(X)
        return a[:, 0] * a[:, 1]


class _AdditiveModel:
    """Separable f0 + f1 (no interaction)."""

    def predict(self, X):
        """Predict."""
        a = np.asarray(X)
        return 2.0 * a[:, 0] + 3.0 * a[:, 1]


def _grid_data(n=800, k=3, seed=0):
    """Helper: Grid data."""
    rng = np.random.default_rng(seed)
    import pandas as pd

    X = rng.uniform(-2.0, 2.0, size=(n, k))
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(k)])


def test_residual_metric_additive_is_separable():
    """Residual metric additive is separable."""
    gx = np.linspace(0, 1, 6)
    gy = np.linspace(0, 1, 5)
    surf = gx[:, None] + gy[None, :]  # purely additive
    m = interaction_residual(surf)
    assert m["residual_rms"] < 1e-9
    assert m["residual_ratio"] < 1e-6


def test_residual_metric_product_is_nonadditive():
    """Residual metric product is nonadditive."""
    gx = np.linspace(-1, 1, 6)
    gy = np.linspace(-1, 1, 5)
    surf = gx[:, None] * gy[None, :]  # pure interaction, zero main effects
    m = interaction_residual(surf)
    assert m["residual_rms"] > 0.1
    assert m["residual_ratio"] > 0.5


def test_residual_metric_constant_surface_ratio_zero():
    """Residual metric constant surface ratio zero."""
    surf = np.full((4, 4), 3.0)
    m = interaction_residual(surf)
    assert m["residual_ratio"] == 0.0


def test_surface_shape_and_bounded():
    """Surface shape and bounded."""
    X = _grid_data()
    res = compute_pdp_2d(_ProductModel(), X, ("f0", "f1"), grid=12, sample=400)
    surf = res["surface"]
    assert surf.shape == (res["grid0"].shape[0], res["grid1"].shape[0])
    assert np.all(np.isfinite(surf))
    # product over [-2,2] x [-2,2] grid means stay within the raw product range.
    assert surf.min() >= -4.0001 and surf.max() <= 4.0001


def test_compose_returns_figure_explicit_pair():
    """Compose returns figure explicit pair."""
    X = _grid_data()
    fig = compose_pdp_2d_figure(_ProductModel(), X, "f0", "f1", grid=12, sample_rows=400)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_compose_default_pair_from_importance():
    """Compose default pair from importance."""
    X = _grid_data()

    class _ImpModel:
        """Groups tests for: ImpModel."""
        feature_importances_ = np.array([0.6, 0.5, 0.01])

        def predict(self, Xa):
            """Predict."""
            a = np.asarray(Xa)
            return a[:, 0] * a[:, 1]

    fig = compose_pdp_2d_figure(_ImpModel(), X, grid=10, sample_rows=300)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_compose_constant_feature_annotates():
    """Compose constant feature annotates."""
    import pandas as pd

    rng = np.random.default_rng(1)
    X = pd.DataFrame({"f0": rng.uniform(-2, 2, 300), "f1": np.full(300, 1.0)})
    fig = compose_pdp_2d_figure(_AdditiveModel(), X, "f0", "f1", grid=10, sample_rows=200)
    txt = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "constant" in txt.lower()
    plt.close(fig)


def test_custom_axis_names_used():
    """Custom axis names used."""
    X = _grid_data()
    fig = compose_pdp_2d_figure(_ProductModel(), X, "f0", "f1", feat_x_name="alpha", feat_y_name="beta", grid=8, sample_rows=200)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "alpha" and ax.get_ylabel() == "beta"
    plt.close(fig)


def test_biz_value_interaction_residual_dominates_additive():
    """Interacting (product) surface has a much larger non-additive residual than an additive surface."""
    X = _grid_data(n=1200, seed=3)
    surf_int = compute_pdp_2d(_ProductModel(), X, ("f0", "f1"), grid=16, sample=600)["surface"]
    surf_add = compute_pdp_2d(_AdditiveModel(), X, ("f0", "f1"), grid=16, sample=600)["surface"]
    r_int = interaction_residual(surf_int)["residual_rms"]
    r_add = interaction_residual(surf_add)["residual_rms"]
    assert r_int > 3.0 * max(r_add, 1e-9), f"interacting {r_int:.4g} should be >3x additive {r_add:.4g}"
    # ratio-form check too: interacting surface is dominated by interaction, additive is near-separable.
    ratio_int = interaction_residual(surf_int)["residual_ratio"]
    ratio_add = interaction_residual(surf_add)["residual_ratio"]
    assert ratio_int > 0.4 and ratio_add < 0.05
