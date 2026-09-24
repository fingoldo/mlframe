"""The permutation tests must compare observed and null MI on one scale."""

import numpy as np
import polars as pl

from mlframe.feature_selection.general import estimate_features_relevancy
from mlframe.feature_selection.mi import grok_compute_mutual_information


def _weak_signal_bins(n=300, seed=0):
    """A real but weak dependency on a high-cardinality pair, where the Miller-Madow floor is a large share of MI."""
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 15, size=n, dtype=np.int8)
    copy_mask = rng.random(n) < 0.5
    weak = np.where(copy_mask, target, rng.integers(0, 15, size=n)).astype(np.int8)
    noise = rng.integers(0, 15, size=n, dtype=np.int8)
    return pl.DataFrame(np.column_stack([target, weak, noise]).astype(np.int8), schema=["target", "weak", "noise"])


def test_a_weak_real_signal_is_not_out_scored_by_its_own_bias():
    """A debiased observation against a biased (raw) null rejected genuine signals whose bias floor was large.

    Measured over 20 seeds at the defaults: a 50%-copy dependency on a 15-bin pair was kept 3/20 times before and 20/20
    after, with pure noise kept 0/20 both times.
    """
    kept_weak, kept_noise = 0, 0
    for seed in range(5):
        cols_to_drop, *_ = estimate_features_relevancy(
            bins=_weak_signal_bins(seed=seed), target_columns=["target"], mi_algorithms_ranking=[grok_compute_mutual_information],
            benchmark_mi_algorithms=False, min_randomized_permutations=20, min_permuted_mi_evaluations=40, random_state=0,
            verbose=0,
        )
        kept_weak += "weak" not in cols_to_drop
        kept_noise += "noise" not in cols_to_drop
    assert kept_weak == 5, f"the weak dependency was dropped on {5 - kept_weak} of 5 seeds"
    assert kept_noise == 0, "pure noise must still be rejected"
