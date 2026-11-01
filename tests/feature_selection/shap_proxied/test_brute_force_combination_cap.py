"""The exhaustive subset search refuses a combination table it cannot hold, and still runs a small one (INP2-1).

``generate_combinations`` materialised every C(n, r) row at once, so ``max_card`` near ``n / 2`` on a wide frame allocated tens of
gigabytes before anything failed. The search now checks the table size first and raises a clear error.
"""

import numpy as np
import pytest

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n


def _inputs(n_rows, n_features, seed=0):
    """Random per-feature attributions, a base value and a regression target, shaped the way the search expects."""
    rng = np.random.default_rng(seed)
    phi = rng.normal(size=(n_rows, n_features))
    y = rng.normal(size=n_rows)
    base = np.zeros(n_rows)
    return phi, base, y


def test_exhaustive_search_over_an_unholdable_table_is_refused_before_allocation():
    """C(60, 30) rows cannot be materialised; the search raises an error that names the size instead of allocating it."""
    phi, base, y = _inputs(50, 60)
    with pytest.raises(ValueError, match="would materialise"):
        brute_force_top_n(phi, base, y, classification=False, metric="mae", min_card=30, max_card=30, top_n=3, parallel=False)


def test_small_exhaustive_search_still_returns_the_best_subsets():
    """A small table is far under the cap, so the search runs and returns candidates."""
    phi, base, y = _inputs(200, 8)
    out = brute_force_top_n(phi, base, y, classification=False, metric="mae", min_card=2, max_card=3, top_n=3, parallel=False)
    assert len(out) >= 3
    assert all(len(comb) in (2, 3) for _, comb in out)
    # The merge keeps the global top-N and force-includes the best subset of every cardinality, so both sizes must appear.
    assert {len(comb) for _, comb in out} == {2, 3}
