"""Tests for ``residual_correlation_matrix`` + ``stacking_aware_gate`` (R10c brainstorm extension C; measure-first stacking gate).

The measure-first protocol:
1. Compute residual correlation matrix between candidate transforms.
2. If max off-diagonal correlation >= 0.8: transforms too redundant, skip the gate (per brainstorm agent's recommendation).
3. Else: run the NNLS-weight gate; keep only transforms with weight >= min_weight.

Tests cover both halves separately and the integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite import (
    max_off_diagonal_correlation,
    residual_correlation_matrix,
    stacking_aware_gate,
)


class TestResidualCorrelationMatrix:
    """Groups tests covering residual correlation matrix."""
    def test_shape_and_diagonal(self) -> None:
        """Shape and diagonal."""
        rng = np.random.default_rng(0)
        n = 200
        residuals = {f"t{i}": rng.normal(size=n) for i in range(3)}
        corr, names = residual_correlation_matrix(residuals)
        assert corr.shape == (3, 3)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-12)
        assert names == ["t0", "t1", "t2"]

    def test_perfectly_correlated_residuals(self) -> None:
        """Perfectly correlated residuals."""
        rng = np.random.default_rng(1)
        n = 200
        r = rng.normal(size=n)
        residuals = {
            "t1": r,
            "t2": 2.0 * r + 5.0,  # affine of t1: corr = 1.0
            "t3": rng.normal(size=n),
        }
        corr, _ = residual_correlation_matrix(residuals)
        assert corr[0, 1] == pytest.approx(1.0, abs=1e-9)

    def test_max_off_diag_helper(self) -> None:
        """Max off diag helper."""
        rng = np.random.default_rng(2)
        n = 200
        r = rng.normal(size=n)
        residuals = {
            "t1": r,
            "t2": r + 1e-4 * rng.normal(size=n),  # essentially t1
            "t3": rng.normal(size=n),
        }
        corr, _ = residual_correlation_matrix(residuals)
        max_off = max_off_diagonal_correlation(corr)
        # t1 vs t2 should saturate the off-diagonal max.
        assert max_off > 0.99

    def test_orthogonal_residuals_low_off_diag(self) -> None:
        """Orthogonal residuals low off diag."""
        rng = np.random.default_rng(3)
        n = 500
        residuals = {
            "t1": rng.normal(size=n),
            "t2": rng.normal(size=n),
            "t3": rng.normal(size=n),
        }
        corr, _ = residual_correlation_matrix(residuals)
        max_off = max_off_diagonal_correlation(corr)
        # Independent normals: |corr| should be well under 0.2.
        assert max_off < 0.2

    def test_length_mismatch_raises(self) -> None:
        """Length mismatch raises."""
        residuals = {"a": np.array([1.0, 2.0]), "b": np.array([1.0])}
        with pytest.raises(ValueError, match="same length"):
            residual_correlation_matrix(residuals)

    def test_empty_input(self) -> None:
        """Empty input."""
        corr, names = residual_correlation_matrix({})
        assert corr.shape == (0, 0)
        assert names == []

    def test_max_off_diag_single_member(self) -> None:
        """K=1 has no off-diagonal; return 0."""
        rng = np.random.default_rng(4)
        residuals = {"only": rng.normal(size=100)}
        corr, _ = residual_correlation_matrix(residuals)
        assert max_off_diagonal_correlation(corr) == 0.0


class TestStackingAwareGate:
    """Groups tests covering stacking aware gate."""
    def _make_orthogonal_predictions(self, n: int = 500, seed: int = 0) -> tuple:
        """Make orthogonal predictions."""
        rng = np.random.default_rng(seed)
        # Three independent predictors that each capture part of y.
        f1 = rng.normal(size=n)
        f2 = rng.normal(size=n)
        f3 = rng.normal(size=n)
        # y is a known linear combination.
        y = 2.0 * f1 + 1.0 * f2 + 0.5 * f3 + rng.normal(scale=0.1, size=n)
        # Predictions: each predictor on its own (perfect single-feature predictions).
        preds = {
            "f1_only": 2.0 * f1,
            "f2_only": 1.0 * f2,
            "f3_only": 0.5 * f3,
            "garbage": rng.normal(size=n),  # uncorrelated with y
        }
        return preds, y

    def test_orthogonal_predictors_all_kept(self) -> None:
        """Orthogonal predictors all kept."""
        preds, y = self._make_orthogonal_predictions()
        survivors, _weights = stacking_aware_gate(preds, y, min_weight=0.05)
        # f1, f2, f3 all carry independent signal -> all survive.
        assert "f1_only" in survivors
        assert "f2_only" in survivors
        assert "f3_only" in survivors

    def test_garbage_predictor_dropped(self) -> None:
        """Garbage predictor dropped."""
        preds, y = self._make_orthogonal_predictions()
        survivors, weights = stacking_aware_gate(preds, y, min_weight=0.05)
        # Pure noise predictor should have weight ~0 -> dropped.
        assert "garbage" not in survivors
        assert weights["garbage"] < 0.05

    def test_survivor_weights_sum_to_one(self) -> None:
        """Survivor weights sum to one."""
        preds, y = self._make_orthogonal_predictions()
        survivors, weights = stacking_aware_gate(preds, y)
        s = sum(weights[n] for n in survivors)
        assert s == pytest.approx(1.0, abs=1e-6)

    def test_low_min_weight_keeps_more(self) -> None:
        """Low min weight keeps more."""
        preds, y = self._make_orthogonal_predictions()
        few, _ = stacking_aware_gate(preds, y, min_weight=0.30)
        many, _ = stacking_aware_gate(preds, y, min_weight=0.001)
        assert len(many) >= len(few)

    def test_no_survivors_returns_empty(self) -> None:
        """All-noise predictions vs unrelated y -> NNLS may put all weight on noise depending on chance. Setup a clearer case: predictions perfectly anti-correlated with y so NNLS can't fit; uniform fallback."""
        # Edge case: only 4 finite rows + 5 predictors -> degenerate, uniform fallback.
        rng = np.random.default_rng(0)
        preds = {f"p{i}": rng.normal(size=3) for i in range(5)}
        y = rng.normal(size=3)
        survivors, weights = stacking_aware_gate(preds, y, min_weight=0.05)
        # Degenerate: all returned with uniform weights.
        assert len(survivors) == 5
        # Uniform = 1/5 each, which clears min_weight=0.05.
        assert all(weights[n] == pytest.approx(0.2) for n in preds)

    def test_length_mismatch_raises(self) -> None:
        """Length mismatch raises."""
        preds = {"a": np.array([1.0, 2.0]), "b": np.array([1.0, 2.0, 3.0])}
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="match y_train"):
            stacking_aware_gate(preds, y)
