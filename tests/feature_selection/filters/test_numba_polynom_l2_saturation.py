"""The numba polynomial optimizer honours ``l2_penalty_saturation`` like the reference ``_l2_penalty_value``."""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _numba_polynom_optimizer as opt


def _score(saturation: float) -> tuple:
    """Score one fixed candidate with an L2 penalty of 0.5 at the given saturation; returns ``(score, raw, sum of squared coefficients)``."""
    rng = np.random.default_rng(0)
    x_a = rng.normal(size=400)
    x_b = rng.normal(size=400)
    y = ((x_a * x_b) > 0).astype(np.int64)
    ca = np.array([0.0, 1.0, 0.5])
    cb = np.array([0.0, 1.0, 0.5])
    s = float((ca**2).sum() + (cb**2).sum())
    score, raw, _bf = opt._eval_one_candidate_njit(
        x_a, x_b, y, ca, cb, opt.BASIS_HERMITE, np.array([opt.BF_MUL], dtype=np.int64), 8, 0.5, False, True, saturation,
    )
    return score, raw, s


@pytest.mark.parametrize("saturation", [1.0, 4.0, 0.0])
def test_eval_candidate_penalty_follows_l2_saturation(saturation):
    """The penalty is l2 * s / (s + saturation), or the raw l2 * s for a non-positive saturation."""
    score, raw, s = _score(saturation)
    expected = 0.5 * s / (s + saturation) if saturation > 0 else 0.5 * s
    assert score == pytest.approx(raw - expected, abs=1e-12)


def test_run_numba_kernel_search_reads_l2_penalty_saturation():
    """A large saturation vanishes the penalty: the search score with saturation 1e9 exceeds the one with saturation 1 for the same seed."""
    rng = np.random.default_rng(3)
    x_a = rng.normal(size=600)
    x_b = rng.normal(size=600)
    y = ((x_a * x_b) > 0).astype(np.int64)
    X = np.column_stack([x_a, x_b])
    kw = dict(ca_size=3, cb_size=3, coef_range=(-2.0, 2.0), basis="hermite", bf_names=("mul",), n_trials=40, batch_size=10, elitism_k=2, l2_penalty=2.0, seed=5)
    low = opt.optimize_all_pairs_numba_kernel(X, y, np.array([[0, 1]], dtype=np.int64), l2_penalty_saturation=1.0, **kw)
    high = opt.optimize_all_pairs_numba_kernel(X, y, np.array([[0, 1]], dtype=np.int64), l2_penalty_saturation=1e9, **kw)
    assert high["best_scores"][0] > low["best_scores"][0] + 1e-3
