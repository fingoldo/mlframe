"""Null-scaled floor of the cat pair gate: noise pairs at few rows per cell are rejected, a real interaction is kept."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._cat_pair_fe import hybrid_cat_pair_fe


def _frame(seed: int, n: int, card: int):
    """Eight noise categoricals plus a parity interaction on c0, c1 driving a binary target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"c{i}": rng.integers(0, card, n).astype(str) for i in range(8)})
    parity = (X["c0"].astype(int) + X["c1"].astype(int)) % 2
    y = (rng.random(n) < np.where(parity == 0, 0.8, 0.2)).astype(int)
    return X, y


def test_noise_pairs_rejected_and_real_interaction_kept_over_seeds():
    """Measured over 5 seeds at 10 levels, n=1000: fixed 0.001 floor admitted noise pairs (max noise ii_mm 0.028); the null-scaled floor admits none and keeps cross_c0_c1 every seed."""
    noise_new = noise_old = kept = 0
    for seed in range(5):
        X, y = _frame(seed, 1000, 10)
        new = hybrid_cat_pair_fe(X, y, top_k=50)[1]
        old = hybrid_cat_pair_fe(X, y, top_k=50, null_floor_mult=0.0)[1]
        kept += "cross_c0_c1" in new
        noise_new += sum(c != "cross_c0_c1" for c in new)
        noise_old += sum(c != "cross_c0_c1" for c in old)
    assert kept == 5
    assert noise_new == 0
    assert noise_old >= 1
