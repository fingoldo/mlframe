"""_residual_variance_ratio (per-column OLS residual) must equal the closed form 1 - corr^2 the vectorised screen uses."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery._structural_hints import _residual_variance_ratio, structural_affinity_scores


@pytest.mark.parametrize("noise", [0.0, 0.3, 1.0, 5.0])
def test_ratio_equals_one_minus_squared_correlation(noise):
    """Single-feature OLS residual-variance ratio is exactly 1 - corr^2."""
    rng = np.random.default_rng(int(noise * 10) + 1)
    x = rng.normal(size=400)
    y = 2.0 * x + 1.0 + rng.normal(scale=max(noise, 1e-12), size=400)
    corr = np.corrcoef(x, y)[0, 1]
    assert _residual_variance_ratio(y, x) == pytest.approx(1.0 - corr * corr, abs=1e-12)


def test_degenerate_inputs_report_no_collapse():
    """Constant x or constant y leaves the ratio at 1.0 (no affine boost)."""
    rng = np.random.default_rng(0)
    assert _residual_variance_ratio(rng.normal(size=50), np.full(50, 3.0)) == 1.0
    assert _residual_variance_ratio(np.full(50, 2.0), rng.normal(size=50)) == 1.0


def test_vectorised_screen_scores_the_affine_column_and_not_noise():
    """The matrix pass built on the same closed form tags an affine column and leaves noise at zero."""
    rng = np.random.default_rng(2)
    y = rng.normal(size=500)
    x = np.column_stack([3.0 * y - 2.0 + rng.normal(scale=0.01, size=500), rng.normal(size=500)])
    scores, kinds = structural_affinity_scores(x, y, ["affine", "noise"])
    assert kinds.get("affine") == "linear_residual" and scores[0] > 0.9
    assert scores[1] == 0.0 and "noise" not in kinds
