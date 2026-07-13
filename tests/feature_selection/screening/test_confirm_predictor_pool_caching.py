"""Regression + equivalence tests for Wave 13 findings 1 and 2 (``_confirm_predictor.py`` /
``_confirm_predictor_engineered.py``): both hoist a per-candidates-pool computation that was
previously rebuilt from scratch on every confirmed-predictor iteration within an interactions
order (~100-150 times/fit) into a cache keyed on the pool's object identity (``ctx.candidates``
is reassigned once per interactions order, never mutated in place -- see
``_screen_predictors.py``'s ``ctx.candidates = candidates`` refresh).

Finding 1: ``confirm_one_predictor`` rebuilt ``_cand_names`` (a ``get_candidate_name`` call per
candidate) + a full sorted ``_name_rank`` (an ``np.lexsort``-fed tie-break array) every call.
``get_candidate_name`` is a plain string join with no memoization of its own, so the waste was
real -- fixed by caching ``(candidates, cand_names, name_rank)`` on ``ScreenContext``.

Finding 2: ``_confirmable_engineered_child`` rescanned the full candidate pool with
``_extract_single_raw_parent`` per confirmed winner (gated behind
``prefer_engineered_rel_eps > 0``) -- fixed by precomputing a ``parent_name -> [idx, ...]`` index
once per pool identity.

Both fixes must be selection-equivalent: the greedy pick sequence (and hence ``support_``) is
unchanged versus the pre-fix per-call recompute.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _multi_signal_fixture(seed: int = 0, n: int = 500, n_features: int = 10):
    """Several independently-useful raw features so the greedy confirmation loop runs multiple
    ``confirm_one_predictor`` iterations within a single interactions_order (needed to exercise
    the cross-call cache)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = ((X[:, 0] + X[:, 1] - X[:, 2] + 0.3 * rng.standard_normal(n)) > 0).astype(np.int64)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
    return df, pd.Series(y, name="y")


def _mrmr_kw(**overrides):
    kw = dict(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        cat_fe_config=None,
        random_seed=0,
        use_simple_mode=False,
    )
    kw.update(overrides)
    return kw


# ---------------------------------------------------------------------------
# Finding 1: name_rank cache
# ---------------------------------------------------------------------------


class TestNameRankCache:
    def test_cache_field_present_and_used(self):
        from mlframe.feature_selection.filters._confirm_predictor import ScreenContext

        assert "_name_rank_cache" in ScreenContext.__dataclass_fields__

    def test_regression_get_candidate_name_called_at_most_once_per_pool(self, monkeypatch):
        """``get_candidate_name`` (used to build ``_cand_names``) must be called at most once per
        candidate per interactions_order -- not once per confirmed predictor. Pre-fix, N confirmed
        predictors meant the full ``[get_candidate_name(c, ...) for c in candidates]`` comprehension
        (len(candidates) calls) ran N times."""
        from mlframe.feature_selection.filters import _confirm_predictor as cp
        from mlframe.feature_selection.filters.mrmr import MRMR

        orig = cp.get_candidate_name
        calls = {"n": 0}

        def counted(*a, **kw):
            calls["n"] += 1
            return orig(*a, **kw)

        monkeypatch.setattr(cp, "get_candidate_name", counted)
        X, y = _multi_signal_fixture(seed=1, n=500, n_features=10)
        MRMR(**_mrmr_kw()).fit(X, y)

        # Exactly one comprehension pass over the (single, order-1) candidate pool: 10 raw
        # features -> 10 calls, REGARDLESS of how many predictors get confirmed inside that order.
        assert calls["n"] == 10, (
            f"get_candidate_name called {calls['n']} times for a 10-feature order-1 pool; "
            "expected exactly 10 (built once, cached across confirm_one_predictor calls)"
        )

    def test_equivalence_selection_unchanged_vs_forced_uncached(self):
        """Selection-equivalence: forcing the cache off (rebuild every call, the pre-fix behaviour)
        must produce the identical selected set as the cached path."""
        from mlframe.feature_selection.filters import _confirm_predictor as cp
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _multi_signal_fixture(seed=2, n=500, n_features=10)

        sel_cached = MRMR(**_mrmr_kw()).fit(X, y)
        support_cached = sorted(sel_cached.feature_names_in_[sel_cached.support_])

        orig_confirm_one_predictor = cp.confirm_one_predictor

        def uncached_confirm_one_predictor(ctx, *a, **kw):
            # Simulate the pre-fix behaviour: never let the cache survive across calls.
            ctx._name_rank_cache = None
            return orig_confirm_one_predictor(ctx, *a, **kw)

        import mlframe.feature_selection.filters._screen_predictors as sp

        sp.confirm_one_predictor = uncached_confirm_one_predictor
        try:
            sel_uncached = MRMR(**_mrmr_kw()).fit(X, y)
        finally:
            sp.confirm_one_predictor = orig_confirm_one_predictor
        support_uncached = sorted(sel_uncached.feature_names_in_[sel_uncached.support_])

        assert support_cached == support_uncached, f"cached path selected {support_cached}, forced-uncached path selected {support_uncached}"


# ---------------------------------------------------------------------------
# Finding 2: engineered-parent index cache (prefer_engineered_rel_eps path)
# ---------------------------------------------------------------------------


class TestEngineeredParentIndexCache:
    def test_cache_field_present(self):
        from mlframe.feature_selection.filters._confirm_predictor import ScreenContext

        assert "_engineered_parent_index_cache" in ScreenContext.__dataclass_fields__

    def test_regression_extract_single_raw_parent_scan_bounded_by_pool_identity(self, monkeypatch):
        """``_extract_single_raw_parent`` (the per-candidate regex/string match) must be called at
        most once per candidate per pool identity, not once per confirmed winner, when
        ``prefer_engineered_rel_eps`` is active."""
        from mlframe.feature_selection.filters import _confirm_predictor_engineered as cpe
        from mlframe.feature_selection.filters.mrmr import MRMR

        orig = cpe._extract_single_raw_parent
        calls = {"n": 0}

        def counted(*a, **kw):
            calls["n"] += 1
            return orig(*a, **kw)

        monkeypatch.setattr(cpe, "_extract_single_raw_parent", counted)
        X, y = _multi_signal_fixture(seed=3, n=500, n_features=6)
        # prefer_engineered_rel_eps is not an MRMR constructor kwarg; screen_predictors' own
        # default (0.15, nonzero) is what MRMR.fit uses, so the engineered-parent-index path is
        # already exercised without an explicit override.
        MRMR(**_mrmr_kw()).fit(X, y)

        # With the index cache, the scan is bounded by (number of DISTINCT candidate-pool
        # identities encountered) x (pool size) -- a small constant multiple of the pool size,
        # never growing with the number of confirmed winners within one order.
        assert calls["n"] <= 6 * 3, (
            f"_extract_single_raw_parent called {calls['n']} times on a 6-feature pool; "
            "expected a small bounded multiple of the pool size (cached per pool identity), "
            "not O(pool_size * n_confirmed_winners)"
        )

    def test_equivalence_selection_unchanged_with_prefer_engineered(self):
        """Selection-equivalence for the prefer-engineered path: forcing the index cache off
        (rebuild the parent index on every call, the pre-fix per-winner rescan) must produce the
        identical selected set as the cached path."""
        from mlframe.feature_selection.filters import _confirm_predictor_engineered as cpe
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _multi_signal_fixture(seed=4, n=500, n_features=6)
        kw = _mrmr_kw()

        sel_cached = MRMR(**kw).fit(X, y)
        support_cached = sorted(sel_cached.feature_names_in_[sel_cached.support_])

        orig_confirmable = cpe._confirmable_engineered_child

        def uncached_confirmable_engineered_child(ctx, *a, **kw2):
            ctx._engineered_parent_index_cache = None
            return orig_confirmable(ctx, *a, **kw2)

        import mlframe.feature_selection.filters._confirm_predictor as cp

        cp._confirmable_engineered_child = uncached_confirmable_engineered_child
        try:
            sel_uncached = MRMR(**kw).fit(X, y)
        finally:
            cp._confirmable_engineered_child = orig_confirmable
        support_uncached = sorted(sel_uncached.feature_names_in_[sel_uncached.support_])

        assert support_cached == support_uncached, f"cached path selected {support_cached}, forced-uncached path selected {support_uncached}"
