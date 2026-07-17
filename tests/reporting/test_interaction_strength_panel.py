"""Interaction-strength diagnostic panel + seriation consumer + Wilson-equivalence tests."""

from __future__ import annotations

import numpy as np

from mlframe.core.matrix_seriation import seriate
from mlframe.reporting.charts.interaction_strength import (
    compose_interaction_strength_figure,
    interaction_strength_panel,
)
from mlframe.reporting.spec import FigureSpec, HeatmapPanelSpec


class _ProductModel:
    """f(x) = x0 * x1 -> a strong (0, 1) interaction, additive elsewhere."""

    def predict(self, X):
        """Predict."""
        X = np.asarray(X, dtype=np.float64)
        return X[:, 0] * X[:, 1] + 0.5 * X[:, 2]


def _data(seed=0, n=1200, d=4):
    """Helper: Data."""
    return np.random.default_rng(seed).normal(size=(n, d))


def test_interaction_strength_panel_is_valid_symmetric_heatmap_in_unit_range():
    """Interaction strength panel is valid symmetric heatmap in unit range."""
    X = _data()
    panel = interaction_strength_panel(_ProductModel(), X, [0, 1, 2, 3], grid=15, sample=800)
    assert isinstance(panel, HeatmapPanelSpec)
    M = np.asarray(panel.matrix, dtype=np.float64)
    assert M.shape == (4, 4)
    assert np.all(M >= 0.0) and np.all(M <= 1.0), "H-statistic must lie in [0, 1]"
    assert np.allclose(M, M.T), "interaction matrix must be symmetric"
    assert np.allclose(np.diag(M), 0.0), "diagonal (self-interaction) must be 0"
    assert M[0, 1] > 0.3, f"x0*x1 should register a strong interaction, got {M[0, 1]:.3f}"
    assert len(panel.row_labels) == 4 and panel.row_labels == panel.col_labels


def test_compose_interaction_strength_figure_caps_features():
    """Compose interaction strength figure caps features."""
    X = _data(d=12)
    fig = compose_interaction_strength_figure(_ProductModel(), X, list(range(12)), max_features=8, grid=8, sample=400)
    assert isinstance(fig, FigureSpec)
    panel = fig.panels[0][0]
    assert np.asarray(panel.matrix).shape == (8, 8), "must cap to max_features to bound O(k^2) cost"


def test_seriation_makes_a_known_block_permutation_contiguous():
    # Two 3-node blocks (A: 0,2,4  B: 1,3,5) with high within-block similarity, interleaved so the raw order hides them.
    """Seriation makes a known block permutation contiguous."""
    blocks = [0, 1, 0, 1, 0, 1]
    n = len(blocks)
    M = np.array([[1.0 if blocks[i] == blocks[j] else 0.05 for j in range(n)] for i in range(n)], dtype=np.float64)
    reordered, perm = seriate(M)
    # After seriation the permuted block labels must be contiguous (all of one block, then the other).
    perm_blocks = [blocks[i] for i in perm]
    switches = sum(1 for a, b in zip(perm_blocks, perm_blocks[1:]) if a != b)
    assert switches == 1, f"a 2-block matrix must come back block-contiguous (1 switch), got {perm_blocks}"
    assert np.allclose(reordered, M[np.ix_(perm, perm)])


def test_wilson_local_matches_core_proportion_stats():
    # calibration.wilson_ci (vectorised, per-bin) and core.proportion_stats.wilson_interval (scalar) implement the
    # same Wilson score formula; they agree to ~1e-9 (the only gap is the z constant: exact vs Acklam approximation).
    """Wilson local matches core proportion stats."""
    from mlframe.core.proportion_stats import wilson_interval
    from mlframe.reporting.charts.calibration import wilson_ci

    for k, n in [(3, 10), (0, 5), (50, 50), (1, 1000), (137, 400)]:
        lo_v, hi_v = wilson_ci(np.array([k / n]), np.array([float(n)]))
        lo_s, hi_s = wilson_interval(k, n)
        assert abs(lo_v[0] - lo_s) < 1e-8, f"lower mismatch at ({k},{n}): {lo_v[0]} vs {lo_s}"
        assert abs(hi_v[0] - hi_s) < 1e-8, f"upper mismatch at ({k},{n}): {hi_v[0]} vs {hi_s}"
