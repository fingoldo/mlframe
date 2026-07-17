"""Unit + biz_value tests for champion/challenger governance comparison."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.compare import compare_models, should_promote


class _ConstModel:
    """Predicts a fixed constant -- used to build a known-worse model."""

    def __init__(self, c: float):
        self.c = float(c)

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.c)


class _NoisyTruth:
    """Predicts y plus fixed additive noise scaled by `eps`."""

    def __init__(self, y: np.ndarray, eps: float, seed: int):
        rng = np.random.default_rng(seed)
        self._pred = y + rng.normal(0, eps, size=y.shape[0])

    def predict(self, X):
        return self._pred


def _reg_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X[:, 0] * 2.0 - X[:, 1] + rng.normal(0, 0.3, size=n)
    return X, y


# ---------------------------------------------------------------- unit


def test_identical_models_delta_zero_p_high():
    X, y = _reg_data()
    m = _NoisyTruth(y, 0.5, seed=1)
    res = compare_models(m, m, X, y, n_boot=500)
    assert abs(res["delta"]) < 1e-9
    assert res["p_value"] > 0.5
    assert res["challenger_wins"] is False
    assert res["ci_low"] <= 0.0 <= res["ci_high"]


def test_bootstrap_ci_brackets_true_delta():
    # Challenger strictly better -> true RMSE delta is positive & the CI
    # (loss-difference scale) should sit above zero.
    X, y = _reg_data()
    champ = _NoisyTruth(y, 1.0, seed=2)
    chall = _NoisyTruth(y, 0.2, seed=3)
    res = compare_models(champ, chall, X, y, n_boot=1000)
    assert res["delta"] > 0  # challenger better RMSE
    assert res["ci_low"] > 0  # loss-diff CI strictly positive
    assert res["ci_low"] <= res["challenger_score"] * 0 + np.inf  # sanity


def test_classification_accuracy_path():
    rng = np.random.default_rng(7)
    n = 2000
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] > 0).astype(float)
    # champion: random guess; challenger: near-perfect
    champ = _ConstModel(1.0)
    good = y.copy()
    flip = rng.choice(n, size=50, replace=False)
    good[flip] = 1.0 - good[flip]
    chall = _NoisyTruth(good, 0.0, seed=0)
    res = compare_models(champ, chall, X, y, metric="accuracy", n_boot=500)
    assert res["challenger_score"] > 0.9
    assert res["challenger_wins"] is True
    assert res["p_value"] < 0.05


def test_ttest_and_wilcoxon_paths():
    X, y = _reg_data()
    champ = _NoisyTruth(y, 1.0, seed=2)
    chall = _NoisyTruth(y, 0.2, seed=3)
    for test in ("ttest", "wilcoxon"):
        pytest.importorskip("scipy")
        res = compare_models(champ, chall, X, y, test=test, n_boot=500)
        assert res["challenger_wins"] is True
        assert res["p_value"] < 0.05


def test_should_promote_min_effect_gate():
    X, y = _reg_data()
    champ = _NoisyTruth(y, 0.55, seed=2)
    chall = _NoisyTruth(y, 0.45, seed=3)
    res = compare_models(champ, chall, X, y, n_boot=800)
    # Significant but small effect -> a large min_effect blocks promotion.
    blocked = should_promote(champ, chall, X, y, min_effect=10.0, n_boot=800)
    assert blocked["promote"] is False
    assert "effect too small" in blocked["reason"]
    allowed = should_promote(champ, chall, X, y, min_effect=0.0, n_boot=800)
    assert allowed["promote"] == res["challenger_wins"]


def test_custom_callable_metric():
    X, y = _reg_data()
    champ = _NoisyTruth(y, 1.0, seed=2)
    chall = _NoisyTruth(y, 0.2, seed=3)
    res = compare_models(champ, chall, X, y, metric=lambda yt, yp: np.abs(yt - yp), n_boot=500)
    assert res["challenger_wins"] is True


# ------------------------------------------------------------ biz_value


def test_biz_val_compare_models_flags_genuine_winner_not_equivalent():
    """A genuinely-better challenger must be flagged challenger_wins=True
    with p<0.05; two statistically-equivalent models must give p>0.05 and
    challenger_wins=False. Catches a broken significance gate that either
    rubber-stamps every challenger or never promotes a real winner."""
    X, y = _reg_data(n=2000, seed=11)

    # Equivalent: symmetric noise (+eps vs -eps) -> identical per-row
    # squared loss in expectation, so the true delta is ~0.
    rng_eq = np.random.default_rng(20)
    noise = rng_eq.normal(0, 0.6, size=y.shape[0])
    champ_eq = _NoisyTruth.__new__(_NoisyTruth)
    champ_eq._pred = y + noise
    chall_eq = _NoisyTruth.__new__(_NoisyTruth)
    chall_eq._pred = y - noise
    eq = compare_models(champ_eq, chall_eq, X, y, n_boot=1000)
    assert eq["challenger_wins"] is False
    assert eq["p_value"] > 0.05, f"equivalent models should be insignificant, p={eq['p_value']}"

    # Genuine winner: much lower noise -> clearly better RMSE.
    champ = _NoisyTruth(y, 1.0, seed=22)
    chall = _NoisyTruth(y, 0.2, seed=23)
    win = compare_models(champ, chall, X, y, n_boot=1000)
    assert win["challenger_wins"] is True
    assert win["p_value"] < 0.05, f"genuine winner should be significant, p={win['p_value']}"
    assert win["delta"] > 0
