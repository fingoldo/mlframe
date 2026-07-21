"""Determinism regression tests for feature-selection wrapper subset search and fairness random baseline.

Both pin a same-seed reproducibility contract that was broken pre-fix:
- get_next_features_subset(ExhaustiveRandom) ignored the threaded rng and used the unseeded module-global random,
  and built its candidate pool from a set-difference whose order is PYTHONHASHSEED-dependent.
- create_robustness_standard_bins('**RANDOM**') shuffled via the unseeded global np.random -> different bins run-to-run.
"""

import numpy as np

from mlframe.feature_selection.wrappers._helpers import get_next_features_subset
from mlframe.feature_selection.wrappers._enums import OptimumSearch
from mlframe.metrics._fairness_metrics import create_robustness_standard_bins


def _pick(seed):
    """Test helper: orig = list(range(30)); evaluated = {3: 0.1, 9: 0.2, 17: 0.3}; rng = np.random.default_rng(seed)."""
    orig = list(range(30))
    evaluated = {3: 0.1, 9: 0.2, 17: 0.3}
    rng = np.random.default_rng(seed)
    # Non-empty FI so the returned subset length reflects the chosen N (the rng-driven decision).
    fi = {0: {str(i): float(30 - i) for i in range(30)}}
    return get_next_features_subset(
        nsteps=1,
        original_features=orig,
        feature_importances=fi,
        evaluated_scores_mean=evaluated,
        evaluated_scores_std={k: 0.0 for k in evaluated},
        use_all_fi_runs=False,
        use_last_fi_run_only=False,
        use_one_freshest_fi_run=False,
        use_fi_ranking=False,
        top_predictors_search_method=OptimumSearch.ExhaustiveRandom,
        rng=rng,
    )


def test_exhaustive_random_subset_reproducible_same_seed():
    # Two fresh generators with the same seed MUST pick the same candidate; pre-fix used global random -> flaky.
    """Exhaustive random subset reproducible same seed."""
    assert _pick(42) == _pick(42)
    assert _pick(7) == _pick(7)


def test_exhaustive_random_subset_seed_sensitive():
    # Across many seeds the pick must vary (proves the rng actually drives the choice, not a constant).
    """Exhaustive random subset seed sensitive."""
    picks = {tuple(_pick(s)) for s in range(40)}
    assert len(picks) > 1


def test_fairness_random_bins_reproducible_same_seed():
    """Fairness random bins reproducible same seed."""
    b1, _ = create_robustness_standard_bins("**RANDOM**", npoints=1000, cont_nbins=3)
    b2, _ = create_robustness_standard_bins("**RANDOM**", npoints=1000, cont_nbins=3)
    assert np.array_equal(b1, b2)


def test_fairness_random_bins_seed_sensitive():
    """Fairness random bins seed sensitive."""
    b0, _ = create_robustness_standard_bins("**RANDOM**", npoints=1000, cont_nbins=3, seed=0)
    b1, _ = create_robustness_standard_bins("**RANDOM**", npoints=1000, cont_nbins=3, seed=1)
    assert not np.array_equal(b0, b1)
