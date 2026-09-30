"""biz_value: CatBoost ``has_time=True`` on chronologically ordered rows beats the shuffled-permutation default under a categorical regime change.

Synthetic: 40-level categorical whose per-level positive propensity flips sign at the midpoint; train = first 80% (time-ordered), test = last 20%.
Measured over seeds 0-5 (150 iters, depth 4, 1 thread): mean test logloss 0.792 (has_time=False) -> 0.675 (True); mean test AUC 0.300 -> 0.605.
With a mild per-level random-walk drift the gain is small and seed-noisy (logloss 0.430 -> 0.417, AUC 0.886 -> 0.892), so this test pins the regime-change case only.
Thresholds sit ~15% below the measured gaps (logloss 0.117, AUC 0.305).
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("catboost")
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score


def _regime_change(seed, n=4000, k=40):
    """Categorical target relationship that flips halfway through the rows."""
    rng = np.random.default_rng(seed)
    c = rng.integers(0, k, n)
    base = rng.normal(size=k)
    logit = np.where(np.arange(n) / n < 0.5, base[c], -0.5 * base[c])
    y = (rng.random(n) < 1 / (1 + np.exp(-2 * logit))).astype(int)
    return pd.DataFrame({"c": c.astype(str), "x": rng.normal(size=n)}), y


def _mean_scores(has_time):
    """Mean log-loss and AUC over six seeds of a chronological split, with has_time on or off."""
    ll, auc = [], []
    for seed in range(6):
        X, y = _regime_change(seed)
        cut = int(len(y) * 0.8)
        m = CatBoostClassifier(iterations=150, learning_rate=0.1, depth=4, has_time=has_time, verbose=0, allow_writing_files=False, thread_count=1, random_seed=seed)
        m.fit(X[:cut], y[:cut], cat_features=["c"])
        p = m.predict_proba(X[cut:])[:, 1]
        ll.append(log_loss(y[cut:], p))
        auc.append(roc_auc_score(y[cut:], p))
    return float(np.mean(ll)), float(np.mean(auc))


def test_biz_val_catboost_has_time_regime_change_beats_shuffled_permutation():
    """Biz val catboost has time regime change beats shuffled permutation."""
    ll_off, auc_off = _mean_scores(False)
    ll_on, auc_on = _mean_scores(True)
    assert ll_off - ll_on >= 0.10, (ll_off, ll_on)
    assert auc_on - auc_off >= 0.26, (auc_off, auc_on)
