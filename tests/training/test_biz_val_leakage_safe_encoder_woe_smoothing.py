"""biz_val + regression for the WoE Laplace-alpha default flip (10.0 -> 0.5).

The WoE method's Laplace smoothing is now a SEPARATE knob (``woe_smoothing``,
default 0.5) from the mean-encoder ``smoothing`` (default 10.0). A small alpha
preserves real per-category signal on rare / high-card / imbalanced data, where
the old shared alpha=10 over-shrank every WoE log-odds toward 0.

Bench: ``_benchmarks/bench_woe_laplace_alpha.py`` (alpha=0.5 wins 9/15 cells,
best mean test-AUC 0.7383 vs 0.7263 at alpha=10).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder


def _make_imbalanced_highcard(seed):
    rng = np.random.default_rng(seed * 100 + 7)
    n, K, base = 8000, 200, 0.01
    cat_logodds = rng.normal(np.log(base / (1 - base)), 1.3, size=K)
    cats = rng.integers(0, K, size=n)
    p = 1.0 / (1.0 + np.exp(-cat_logodds[cats]))
    y = (rng.random(n) < p).astype(np.float64)
    half = n // 2
    return cats[:half].astype(str), y[:half], cats[half:].astype(str), y[half:]


def _woe_test_auc(alpha, seed):
    ctr, ytr, cte, yte = _make_imbalanced_highcard(seed)
    enc = LeakageSafeEncoder(method="woe", woe_smoothing=alpha, random_state=0)
    enc.fit(ctr, ytr)
    return roc_auc_score(yte, enc.transform(cte))


def test_biz_val_woe_smoothing_default_is_half():
    """WoE cushion defaults to the Jeffreys 0.5, independent of the mean-encoder ``smoothing`` (now 3.0 per bench_target_encoder_smoothing)."""
    e = LeakageSafeEncoder(method="woe")
    assert e.woe_smoothing == 0.5
    assert e.smoothing == 3.0


def test_biz_val_woe_smoothing_small_alpha_beats_large_on_rare_highcard():
    """alpha=0.5 must beat alpha=10 in test-AUC on the majority of seeds.

    Regression sensor for the flip: pre-fix WoE used the shared smoothing=10,
    which on this rare/high-card synthetic loses ~0.04-0.06 AUC. Floor margin
    +0.02 (measured ~+0.04 mean) absorbs seed noise but trips a revert.
    """
    seeds = [0, 1, 2]
    deltas = [_woe_test_auc(0.5, s) - _woe_test_auc(10.0, s) for s in seeds]
    wins = sum(d > 0 for d in deltas)
    assert wins >= 2, f"alpha=0.5 should win majority of seeds; deltas={deltas}"
    assert np.mean(deltas) >= 0.02, f"mean AUC gain too small: {deltas}"


def test_biz_val_woe_smoothing_default_matches_explicit_half():
    """The new default (None -> 0.5) must reproduce explicit woe_smoothing=0.5."""
    ctr, ytr, cte, _yte = _make_imbalanced_highcard(0)
    e_def = LeakageSafeEncoder(method="woe", random_state=0).fit(ctr, ytr)
    e_half = LeakageSafeEncoder(method="woe", woe_smoothing=0.5, random_state=0).fit(ctr, ytr)
    np.testing.assert_allclose(e_def.transform(cte), e_half.transform(cte))


def test_woe_smoothing_negative_rejected():
    with pytest.raises(ValueError):
        LeakageSafeEncoder(method="woe", woe_smoothing=-1.0)


def test_woe_smoothing_independent_of_mean_smoothing():
    """Changing mean ``smoothing`` must NOT alter WoE output; only woe_smoothing does."""
    ctr, ytr, cte, _yte = _make_imbalanced_highcard(1)
    e_a = LeakageSafeEncoder(method="woe", smoothing=10.0, random_state=0).fit(ctr, ytr)
    e_b = LeakageSafeEncoder(method="woe", smoothing=999.0, random_state=0).fit(ctr, ytr)
    np.testing.assert_allclose(e_a.transform(cte), e_b.transform(cte))
