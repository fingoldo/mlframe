"""Unit + biz_value tests for LENKOR composite-similarity metric learning (PZAD recsys)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.core.composite_similarity import (
    combine_block_similarities,
    fit_composite_similarity,
)


def _gauss_sim(x, bw=1.0):
    """Gaussian similarity matrix from a 1-D feature."""
    d = x[:, None] - x[None, :]
    return np.exp(-(d**2) / (2 * bw**2))


# ---------------------------------------------------------------- unit
def test_combine_linear_and_sqrt():
    A = np.array([[1.0, 0.4], [0.4, 1.0]])
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    lin = combine_block_similarities([A, B], [2.0, 3.0], "linear")
    assert lin[0, 1] == 0.8  # 2*0.4 + 3*0.0
    sq = combine_block_similarities([A, B], [1.0, 1.0], "sqrt")
    assert abs(sq[0, 1] - np.sqrt(0.4)) < 1e-12


def test_guards():
    A = np.eye(4)
    with pytest.raises(ValueError):
        fit_composite_similarity([A], np.zeros(3))  # y length mismatch
    with pytest.raises(ValueError):
        fit_composite_similarity([A], np.arange(4), deformation="cube")
    with pytest.raises(ValueError):
        fit_composite_similarity([A], np.arange(4), k=4)  # k >= n


def test_returns_weights_and_score():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    y = (x > 0).astype(int)
    res = fit_composite_similarity([_gauss_sim(x)], y, k=5)
    assert res.weights.shape == (1,)
    assert 0.0 <= res.score <= 1.0
    S = res.combine([_gauss_sim(x)])
    assert S.shape == (40, 40)


def test_single_informative_block_scores_high():
    rng = np.random.default_rng(1)
    x = rng.normal(size=120)
    y = (x > 0).astype(int)  # perfectly separable by proximity in x
    res = fit_composite_similarity([_gauss_sim(x, bw=0.5)], y, k=7)
    assert res.score >= 0.9, f"kNN over a clean similarity should score >=0.9, got {res.score:.3f}"


# ---------------------------------------------------------------- biz_value
def test_biz_val_tuned_combination_beats_equal_weight_and_single_blocks():
    """LENKOR's value: two similarity blocks each carry partial signal, but block B is on a 100x larger scale and is
    the NOISIER one. Naive equal-weight (1,1) is dominated by the large-scale noisy block; coordinate-descent tuning
    rebalances toward the informative block, beating both equal-weight AND either single block on kNN accuracy."""
    rng = np.random.default_rng(2)
    n = 300
    # latent 2-D; label depends on BOTH coordinates (checkerboard-ish so neither block alone fully separates)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y = ((a > 0) ^ (b > 0)).astype(int)
    # block A: clean similarity on 'a'; block B: similarity on a NOISY 'b', and scaled up 100x
    A = _gauss_sim(a, bw=0.6)
    b_noisy = b + rng.normal(scale=1.2, size=n)
    B = 100.0 * _gauss_sim(b_noisy, bw=0.6)
    k = 9

    def knn_acc(S):
        res = fit_composite_similarity([S], y, k=k)  # single-block: weight is irrelevant to accuracy, just measures S
        return res.score

    acc_A = knn_acc(A)
    acc_B = knn_acc(B)
    # equal-weight naive combine
    acc_equal = fit_composite_similarity([A, B], y, k=k, grid=[1.0]).score  # grid pinned to 1 -> no tuning
    tuned = fit_composite_similarity([A, B], y, k=k)
    acc_tuned = tuned.score

    assert acc_tuned >= max(acc_A, acc_B) + 0.03, f"tuned {acc_tuned:.3f} should beat best single block {max(acc_A, acc_B):.3f}"
    assert acc_tuned >= acc_equal + 0.03, f"tuned {acc_tuned:.3f} should beat equal-weight {acc_equal:.3f} (coordinate descent earns its keep)"
