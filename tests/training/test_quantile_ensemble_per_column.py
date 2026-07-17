"""Per-quantile ensemble blend test: members each emit (n_samples, n_quantiles); blend must be per-column."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite import predict_quantile_ensemble


class _MultiQuantileMember:
    """Test double: emits a fixed multi-quantile matrix for any X.

    Each member is parametrised by ``preds`` of shape ``(n_samples, n_quantiles)``
    and the ``quantiles`` it claims to support. Per-call returns one column of
    ``preds`` keyed by the matching alpha.
    """

    def __init__(self, preds: np.ndarray, quantiles: tuple[float, ...]) -> None:
        if preds.ndim != 2 or preds.shape[1] != len(quantiles):
            raise ValueError("preds shape must match (n_samples, len(quantiles))")
        self.preds = preds.astype(np.float64)
        self.quantiles = tuple(float(q) for q in quantiles)

    def predict_quantile(self, X, alpha):
        alpha = float(alpha)
        if alpha not in self.quantiles:
            raise KeyError(f"alpha {alpha} not in member quantiles {self.quantiles}")
        col = self.quantiles.index(alpha)
        return self.preds[:, col]


def _make_members_same_quantiles(n_samples: int = 100, n_quantiles: int = 5):
    """Two members emitting (n_samples, n_quantiles) on quantiles (0.1, 0.3, 0.5, 0.7, 0.9)."""
    rng = np.random.default_rng(42)
    quantiles = (0.1, 0.3, 0.5, 0.7, 0.9)
    preds_a = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_quantiles))
    preds_b = rng.normal(loc=2.0, scale=1.5, size=(n_samples, n_quantiles))
    # Sort each row so the multi-quantile output is non-crossing (matches the real ranker contract).
    preds_a = np.sort(preds_a, axis=1)
    preds_b = np.sort(preds_b, axis=1)
    return [_MultiQuantileMember(preds_a, quantiles), _MultiQuantileMember(preds_b, quantiles)], quantiles, preds_a, preds_b


def test_two_members_blend_preserves_quantile_dimension():
    """100x5 + 100x5 -> 100x5 ensemble; not collapsed to 100x1."""
    members, quantiles, preds_a, preds_b = _make_members_same_quantiles(n_samples=100, n_quantiles=5)

    blended = predict_quantile_ensemble(members, X=None, quantiles=quantiles)

    assert blended.shape == (100, 5), f"expected (100, 5), got {blended.shape!r}"
    expected = 0.5 * (preds_a + preds_b)
    np.testing.assert_allclose(blended, expected, rtol=1e-12, atol=1e-12)


def test_per_column_means_independent_per_quantile():
    """Per-quantile blending must average the q10 columns, the q90 columns separately -- not mix q10 of member A with q90 of member B."""
    quantiles = (0.1, 0.5, 0.9)
    preds_a = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    preds_b = np.array([[0.0, 4.0, 5.0], [100.0, 200.0, 300.0]])
    members = [_MultiQuantileMember(preds_a, quantiles), _MultiQuantileMember(preds_b, quantiles)]

    blended = predict_quantile_ensemble(members, X=None, quantiles=quantiles)

    assert blended.shape == (2, 3)
    np.testing.assert_allclose(blended[0, 0], 0.5)
    np.testing.assert_allclose(blended[0, 1], 3.0)
    np.testing.assert_allclose(blended[0, 2], 4.0)
    np.testing.assert_allclose(blended[1, 0], 55.0)
    np.testing.assert_allclose(blended[1, 1], 110.0)
    np.testing.assert_allclose(blended[1, 2], 165.0)


def test_weights_renormalised_and_applied():
    """Asymmetric weights produce a non-uniform blend; weights are renormalised to sum to 1."""
    quantiles = (0.25, 0.75)
    preds_a = np.array([[1.0, 2.0]])
    preds_b = np.array([[5.0, 6.0]])
    members = [_MultiQuantileMember(preds_a, quantiles), _MultiQuantileMember(preds_b, quantiles)]

    blended = predict_quantile_ensemble(members, X=None, quantiles=quantiles, weights=[3.0, 1.0])

    expected = 0.75 * preds_a + 0.25 * preds_b
    np.testing.assert_allclose(blended, expected, rtol=1e-12)


def test_different_quantile_sets_raises_value_error():
    """When members emit different quantile sets, blend is ambiguous -- raise rather than guess."""
    quantiles_full = (0.1, 0.5, 0.9)
    quantiles_short = (0.1, 0.5)
    preds_a = np.zeros((10, 3))
    preds_b = np.zeros((10, 2))
    members = [
        _MultiQuantileMember(preds_a, quantiles_full),
        _MultiQuantileMember(preds_b, quantiles_short),
    ]

    with pytest.raises(KeyError):
        # Second member raises on the third alpha (0.9) it doesn't know about.
        predict_quantile_ensemble(members, X=None, quantiles=quantiles_full)


def test_inconsistent_n_samples_raises():
    """Members emitting different n_samples (e.g. one trained on a sliced subset) -> ValueError."""
    quantiles = (0.5,)
    preds_a = np.zeros((10, 1))
    preds_b = np.zeros((20, 1))
    members = [_MultiQuantileMember(preds_a, quantiles), _MultiQuantileMember(preds_b, quantiles)]

    with pytest.raises(ValueError, match="produced shape"):
        predict_quantile_ensemble(members, X=None, quantiles=quantiles)


def test_empty_members_raises():
    with pytest.raises(ValueError, match="empty"):
        predict_quantile_ensemble([], X=None, quantiles=(0.5,))


def test_invalid_quantiles_raises():
    members, _, _, _ = _make_members_same_quantiles(n_samples=10, n_quantiles=5)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        predict_quantile_ensemble(members[:1], X=None, quantiles=(0.0, 0.5))
    with pytest.raises(ValueError, match="sorted ascending"):
        predict_quantile_ensemble(members[:1], X=None, quantiles=(0.7, 0.3))


def test_weights_zero_sum_raises():
    quantiles = (0.5,)
    preds = np.zeros((3, 1))
    members = [_MultiQuantileMember(preds, quantiles), _MultiQuantileMember(preds, quantiles)]
    with pytest.raises(ValueError, match="sum to zero"):
        predict_quantile_ensemble(members, X=None, quantiles=quantiles, weights=[0.0, 0.0])


def test_negative_weights_raise():
    quantiles = (0.5,)
    preds = np.zeros((3, 1))
    members = [_MultiQuantileMember(preds, quantiles), _MultiQuantileMember(preds, quantiles)]
    with pytest.raises(ValueError, match="non-negative"):
        predict_quantile_ensemble(members, X=None, quantiles=quantiles, weights=[1.0, -1.0])


def test_member_lacking_predict_quantile_raises():
    class _Bad:
        pass

    with pytest.raises(ValueError, match="lacks predict_quantile"):
        predict_quantile_ensemble([_Bad()], X=None, quantiles=(0.5,))
