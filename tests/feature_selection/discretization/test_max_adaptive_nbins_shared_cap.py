"""Regression coverage for the shared ``max_adaptive_nbins`` ceiling (2026-07-21).

Found live during a 50k-row wellbore ground-truth sweep: ``knuth`` defaulted to a 500-bin cap while
``bayesian_blocks`` had NO cap at all (observed ~2470 bins on one real column), and
``freedman_diaconis``'s own ``sqrt(N)*4`` cap grows unbounded with N (~895 bins at N=50k). Any one of
these blows the downstream pairwise-MI joint cardinality (``nbins_a * nbins_b``) past both the CUDA
shared-memory budget and the row-chunked global-memory fallback's launch-count budget, forcing a
multi-thousand-second CPU njit fallback per column pair. ``MAX_ADAPTIVE_NBINS`` (256, matching MDLP's
own implicit ``2**max_depth`` ceiling) is now the single shared default across every strategy whose
formula has no natural upper bound, overridable per-fit via ``MRMR(max_adaptive_nbins=...)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._adaptive_nbins import (
    MAX_ADAPTIVE_NBINS,
    edges_bayesian_blocks,
    edges_freedman_diaconis,
    edges_knuth,
    freedman_diaconis_nbins,
)
from mlframe.feature_selection.filters.discretization._discretization_dataset import categorize_dataset


def test_max_adaptive_nbins_constant_matches_mdlp_implicit_ceiling():
    """MDLP's own default (max_depth=8 -> 2**8 leaves) is the ceiling everything else must match."""
    assert MAX_ADAPTIVE_NBINS == 256


@pytest.mark.parametrize("edge_fn,kwarg", [(edges_knuth, "m_max_cap"), (edges_bayesian_blocks, "m_max_cap"), (edges_freedman_diaconis, "max_bins")])
def test_uncapped_formulas_default_to_shared_ceiling(edge_fn, kwarg):
    """knuth / bayesian_blocks / freedman_diaconis must each default their cap kwarg to
    MAX_ADAPTIVE_NBINS (not a stale per-method literal like the old knuth=500 / bb=uncapped)."""
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(loc=i * 10, scale=0.02, size=50) for i in range(400)])  # many tight clusters
    default_edges = edge_fn(x)
    capped_edges = edge_fn(x, **{kwarg: 32})
    assert len(default_edges) - 1 <= MAX_ADAPTIVE_NBINS
    assert len(capped_edges) - 1 <= 32
    assert len(capped_edges) <= len(default_edges)


def test_freedman_diaconis_nbins_bounded_at_large_n_even_with_sqrt_cap_above_ceiling():
    """sqrt(N)*4 alone exceeds MAX_ADAPTIVE_NBINS once N > (MAX_ADAPTIVE_NBINS/4)**2 = 4096 -- pin
    that the absolute ceiling still applies at large N (this is the exact freedman_diaconis blowup
    that produced joint cardinality 801025 on 50k-row real data before this fix)."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(50_000)
    n_bins = freedman_diaconis_nbins(x)
    assert n_bins <= MAX_ADAPTIVE_NBINS, f"freedman_diaconis produced {n_bins} bins at n=50000, exceeding the shared cap {MAX_ADAPTIVE_NBINS}"


def test_max_adaptive_nbins_kwarg_propagates_through_categorize_dataset():
    """The single ``max_adaptive_nbins`` key in ``nbins_strategy_kwargs`` -- the same path
    ``MRMR(max_adaptive_nbins=...)`` feeds into -- must actually lower every uncapped-by-construction
    strategy's per-column bin count, not just the strategy it happens to be named after."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.standard_normal(5000) * 100, "b": rng.standard_normal(5000)})
    y = (X["a"] * 2 + rng.standard_normal(5000) * 0.3).to_numpy()
    for method in ("knuth", "freedman_diaconis"):
        _, cols, nbins_loose = categorize_dataset(
            df=X, method="quantile", n_bins=10, dtype=np.int32, nbins_strategy=method, nbins_strategy_kwargs={"max_adaptive_nbins": 256}, y_for_strategy=y,
        )
        _, _, nbins_tight = categorize_dataset(
            df=X, method="quantile", n_bins=10, dtype=np.int32, nbins_strategy=method, nbins_strategy_kwargs={"max_adaptive_nbins": 8}, y_for_strategy=y,
        )
        assert all(nb <= 8 for nb in nbins_tight), f"{method}: max_adaptive_nbins=8 not respected, got {dict(zip(cols, nbins_tight))}"
        assert any(nb > 8 for nb in nbins_loose), f"{method}: fixture didn't exercise the cap (all columns already <=8 bins at cap=256)"
