"""biz_value + default tests for ``LeakageSafeEncoder`` smoothing default.

The smoothing constant pulls rare-category target-mean encodings toward the
global prior. The default was 10.0 (over-shrinks informative categories); an
isolated held-out sweep (``bench_target_encoder_smoothing``, 5 scenarios x 5
seeds) showed smoothing=3.0 wins the majority of cells on held-out log-loss
(17/25) and AUC (14/25), with less than half the posterior-MSE. Default
flipped 10.0 -> 3.0.

These tests exercise the REAL ``LeakageSafeEncoder`` (fit on a train split,
``transform`` a disjoint test split -- leak-free) and quantitatively assert
the win, so a silent revert of the default fails the suite.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from sklearn.metrics import log_loss

from mlframe.training.feature_handling.target_encoders import (
    LeakageSafeEncoder,
)


def test_default_smoothing_is_3():
    """The benched default must be 3.0; a silent bump back to 10.0 trips here."""
    assert LeakageSafeEncoder(method="target_mean").smoothing == 3.0


def _make_highcard_rare(seed):
    """High-cardinality binary target with many rare categories -- the regime
    where over-shrinking toward the prior hurts held-out calibration most."""
    rng = np.random.default_rng(seed)
    K = 300
    true_rate = rng.uniform(0.1, 0.9, size=K)
    counts = rng.integers(2, 30, size=K)
    cat_ids = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
    rng.shuffle(cat_ids)
    cats = np.array([f"c{int(i)}" for i in cat_ids], dtype=object)
    y = (rng.uniform(size=len(cat_ids)) < true_rate[cat_ids]).astype(np.float64)
    idx = rng.permutation(len(cats))
    k = len(cats) // 2
    return cats, y, idx[:k], idx[k:]


def _heldout_log_loss(cats, y, tr, te, smoothing, seed):
    """Fits LeakageSafeEncoder on the train split and returns held-out log-loss at a given smoothing."""
    enc = LeakageSafeEncoder(method="target_mean", smoothing=smoothing, cv=5, random_state=seed)
    enc.fit(cats[tr], y[tr])
    p = np.clip(enc.transform(cats[te]), 1e-6, 1 - 1e-6)
    return log_loss(y[te], p, labels=[0.0, 1.0])


def test_biz_val_leakage_safe_encoder_smoothing_beats_legacy_default():
    """smoothing=3.0 must beat the legacy 10.0 on held-out log-loss on the
    MAJORITY of seeds in the high-card-rare regime. Measured 5/5 wins; floor
    at 3/5 to absorb seed noise."""
    wins = 0
    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        cats, y, tr, te = _make_highcard_rare(seed)
        ll_new = _heldout_log_loss(cats, y, tr, te, 3.0, seed)
        ll_old = _heldout_log_loss(cats, y, tr, te, 10.0, seed)
        if ll_new < ll_old:
            wins += 1
    assert wins >= 3, f"smoothing=3.0 beat 10.0 only {wins}/{len(seeds)} seeds"


def test_biz_val_smoothing_3_lower_posterior_mse_than_10():
    """smoothing=3.0 encodes rare categories closer to their true posterior:
    mean held-out MSE-to-true-rate must be strictly lower than smoothing=10.0
    (averaged over seeds). Catches a regression that re-over-shrinks."""
    mse3, mse10 = [], []
    for seed in [0, 1, 2, 3, 4]:
        rng = np.random.default_rng(seed)
        K = 300
        true_rate = rng.uniform(0.1, 0.9, size=K)
        counts = rng.integers(2, 30, size=K)
        cat_ids = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
        rng.shuffle(cat_ids)
        cats = np.array([f"c{int(i)}" for i in cat_ids], dtype=object)
        post = true_rate[cat_ids]
        y = (rng.uniform(size=len(cat_ids)) < post).astype(np.float64)
        idx = rng.permutation(len(cats))
        k = len(cats) // 2
        tr, te = idx[:k], idx[k:]
        for sm, bucket in ((3.0, mse3), (10.0, mse10)):
            enc = LeakageSafeEncoder(method="target_mean", smoothing=sm, cv=5, random_state=seed)
            enc.fit(cats[tr], y[tr])
            pred = enc.transform(cats[te])
            bucket.append(float(np.mean((pred - post[te]) ** 2)))
    assert np.mean(mse3) < np.mean(mse10), f"mse3={np.mean(mse3):.5f} not < mse10={np.mean(mse10):.5f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--no-cov"]))
