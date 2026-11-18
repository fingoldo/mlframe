"""Gate behaviour of ``detect_synergy``: a planted sign-product pair is found among many columns, additive/noise/decoy data never fires.

The detector scores every column pair by the no-three-way-interaction deviance, so recall must not depend on how many columns there are
(the earlier random column/pair sampling found the pair in about 8 % of calls at p=100) and the additive-link scenarios must stay silent.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._synergy_detector import detect_synergy

_SEEDS = (0, 1, 2, 3)


def _frame(p: int, n: int, seed: int) -> tuple[np.random.Generator, np.ndarray]:
    """Standard-normal float32 frame and its generator."""
    rng = np.random.default_rng(100 + seed)
    return rng, rng.standard_normal((n, p)).astype(np.float32)


@pytest.mark.parametrize("p", [100, 300])
@pytest.mark.parametrize("seed", _SEEDS)
def test_planted_sign_product_is_found_at_wide_p(p, seed):
    """Every pair is scored, so the planted pair is found at p=100 and p=300 and reported as the best pair."""
    _, X = _frame(p, 6000, seed)
    y = (np.sign(X[:, 0] * X[:, 1]) > 0).astype(np.int64)
    ok, info = detect_synergy(X, y, random_seed=seed)
    assert ok, info
    assert info["best_pair"] == (0, 1)
    assert info["n_pairs"] == p * (p - 1) // 2
    assert info["real_excess"] > 3 * info["threshold"]


def _additive_cases(seed: int):
    """Yield (name, X, y) for scenarios that carry no synergy."""
    rng, X = _frame(100, 6000, seed)
    n = X.shape[0]

    def sig(z):
        """Logistic link."""
        return 1.0 / (1.0 + np.exp(-z))

    yield "noise", X, rng.integers(0, 2, n)
    yield "logit2", X, (rng.random(n) < sig(1.5 * (X[:, 0] + X[:, 1]))).astype(np.int64)
    yield "logit3", X, (rng.random(n) < sig(1.5 * (X[:, 0] + X[:, 1] + X[:, 2]))).astype(np.int64)
    yield "thr2", X, ((X[:, 0] + X[:, 1] + 0.5 * rng.standard_normal(n)) > 0).astype(np.int64)
    yield "thr3", X, ((X[:, 0] + X[:, 1] + X[:, 2] + 0.5 * rng.standard_normal(n)) > 0).astype(np.int64)
    yield "regression", X, (X[:, 0] + X[:, 1] + 0.5 * rng.standard_normal(n)).astype(np.float32)
    Xd = X.copy()
    for j in range(1, 9):
        Xd[:, j] = Xd[:, 0] + (0.3 + 0.05 * j) * rng.standard_normal(n)
    yield "decoys", Xd, ((Xd[:, 0] + 0.5 * rng.standard_normal(n)) > 0).astype(np.int64)
    yield "main_effect", X, ((X[:, 0] ** 2 + 0.5 * X[:, 1] + 0.5 * rng.standard_normal(n)) > 0.8).astype(np.int64)


@pytest.mark.parametrize("seed", _SEEDS)
def test_no_false_positive_on_additive_noise_decoy_and_main_effect_data(seed):
    """Noise labels, logistic/threshold additive links, a quantised regression target, decoy copies and main effects all stay silent."""
    fired = [name for name, X, y in _additive_cases(seed) if detect_synergy(X, y, random_seed=seed)[0]]
    assert not fired, fired


def test_synergy_survives_label_noise_and_imbalance():
    """A sign-product pair flipped for 20 % of rows, and one active on a quarter of the rows, is still found."""
    rng, X = _frame(100, 8000, 7)
    sp = np.sign(X[:, 0] * X[:, 1]) > 0
    flip = rng.random(X.shape[0]) < 0.2
    assert detect_synergy(X, np.where(flip, ~sp, sp).astype(np.int64), random_seed=1)[0]
    rare = (sp & (rng.random(X.shape[0]) < 0.25)).astype(np.int64)
    assert detect_synergy(X, rare, random_seed=1)[0]


def test_info_keys_and_pair_cap_are_kept():
    """The info keys read by callers exist and ``max_pairs`` still caps the scored pair count."""
    _, X = _frame(20, 2000, 0)
    y = (np.sign(X[:, 0] * X[:, 1]) > 0).astype(np.int64)
    _, info = detect_synergy(X, y, max_pairs=50)
    for key in ("real_excess", "null_excess", "threshold", "n_pairs", "n_features"):
        assert key in info
    assert info["n_pairs"] == 50
