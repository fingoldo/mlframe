"""The usability candidate pool never builds more pair candidates than its cap on a wide feature set.

Every pair of ``base_names`` was materialised before the pair ranking cut it to ``max_pairs``. On a wide set that is a quadratic
list of tuples (about 2e8 at 20,000 columns) built only to be discarded. Above the cap the pool now keeps the base columns with the
largest marginal MI and pairs only those.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._usability_aware_selection as mod


def test_pair_candidates_stay_under_cap_on_wide_input(monkeypatch):
    """With 200 columns and a cap of 5000 the ranking step receives at most 5000 pairs."""
    monkeypatch.setattr(mod, "_MAX_BASE_PAIR_CANDIDATES", 5000, raising=False)
    seen = []

    class _Stop(Exception):
        """Raised by the spy to end the build once the pair list has been observed."""

    def _spy_rank(rank_pairs_by_joint_mi, base_names, base_f64, quantization_nbins, y_terms, _seleq, pairs, y_codes, _mi_key, max_pairs):
        """Record the number of pair candidates handed to the ranking step and stop the build."""
        seen.append(len(pairs))
        raise _Stop()

    monkeypatch.setattr(mod, "_build_usability_cand_step1_rank_pairs_joint", _spy_rank)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 200)), columns=[f"c{i}" for i in range(200)])
    y = rng.normal(size=200)

    try:
        mod.build_usability_candidate_pool(X, y, list(X.columns), max_pairs=10)
    except _Stop:
        pass

    assert seen, "the ranking step was never reached"
    assert seen[0] <= 5000, seen
