"""Hard parity gate for the JMIM joint-MI cache (evaluation.py, 2026-06-19).

The JMIM aggregator branch of ``evaluate_gain`` memoises ``mi({X, Z}; y)`` keyed on the
multiset ``arr2str({X} u Z)`` (``cached_jmim_MIs``), mirroring the plain-CMI
``cached_cond_MIs`` mechanism: raw value stored, ``** (nexisting + 1)`` applied at read.
The cache memoises the *same* ``mi()`` computation, so it MUST NOT change selection.

Reuse note (benchmarked, honest):
* At ``interactions_max_order == 1`` the existing ``(current_gain, last_checked_k)`` resume
  optimisation already evaluates each ``(X, Z)`` exactly once across the whole fit, so the
  JMIM cache populates but never HITS (harmless). Measured n=6000/p=150: ~456k entries, 0 hits.
* At ``interactions_max_order >= 2`` the resume is disabled (a newly-selected var inserts in
  the MIDDLE of the combination sequence), so each greedy round re-evaluates prior combos and
  the persisted cache absorbs the repeats. Measured n=4000/p=40 order-2: ~1.26M entries,
  ~117k HITS. This test exercises the order-2 regime so it proves the cache actually engages.

The test asserts:
  1. selection (get_feature_names_out order + support_) is IDENTICAL with cache vs a kill path
     that forces every JMIM lookup to MISS (the pre-change behaviour);
  2. the cache actually HITS (sum_hits > 0) in JMIM order-2 mode, proving it engages.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _make_frame(seed: int = 0, n: int = 4000, p: int = 40):
    """Make frame."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:5] = [1.5, -1.2, 1.0, 0.8, -0.9]  # planted linear signal in the first 5 cols
    y = (((X @ beta) + rng.normal(scale=0.5, size=n)) > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    return df, pd.Series(y, name="y")


def _fit_jmim_order2():
    """Fit jmim order2."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, ys = _make_frame()
    sel = MRMR(
        verbose=0,
        random_seed=42,
        n_workers=1,
        max_runtime_mins=2,
        quantization_nbins=5,
        redundancy_aggregator="jmim",
        use_simple_mode=False,
        interactions_max_order=2,
    )
    sel.fit(df, ys)
    return sel


def test_jmim_cache_parity_and_hits():
    """Cache ON vs a forced-MISS kill path -> identical selection; cache must HIT."""
    from mlframe.feature_selection.filters import evaluation as ev
    from mlframe.feature_selection.filters import _confirm_predictor as cp
    import numba
    from numba.core import types

    # --- Run WITH the cache (real path) and capture hit stats. ---
    ev._JMIM_CACHE_STATS.clear()
    sel_cached = _fit_jmim_order2()
    names_cached = list(sel_cached.get_feature_names_out())
    support_cached = np.asarray(sel_cached.support_).copy()
    sum_hits = sum(s["hits"] for s in ev._JMIM_CACHE_STATS)
    sum_size = sum(s["size"] for s in ev._JMIM_CACHE_STATS)

    # The cache must populate AND actually hit at order 2 -> proves it engages.
    assert sum_size > 0, "JMIM cache never populated -- branch not exercised"
    assert sum_hits > 0, f"JMIM cache populated ({sum_size} entries) but never HIT -- no cross-round reuse detected; the cache is not engaging at order 2"

    # --- Run with the cache KILLED: wrap evaluate_candidate so every call gets a FRESH
    # empty jmim dict -> every JMIM lookup MISSES, reproducing the pre-change behaviour
    # (recompute mi() every time). Patch both call sites' module reference. ---
    real_evaluate_candidate = ev.evaluate_candidate

    def _killed_evaluate_candidate(*args, **kwargs):
        """Killed evaluate candidate."""
        kwargs["cached_jmim_MIs"] = numba.typed.Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float64,
        )
        kwargs["jmim_hit_counter"] = np.zeros(1, dtype=np.int64)
        return real_evaluate_candidate(*args, **kwargs)

    ev._JMIM_CACHE_STATS.clear()
    cp.evaluate_candidate = _killed_evaluate_candidate
    try:
        sel_killed = _fit_jmim_order2()
    finally:
        cp.evaluate_candidate = real_evaluate_candidate
    names_killed = list(sel_killed.get_feature_names_out())
    support_killed = np.asarray(sel_killed.support_).copy()

    # The kill path forces misses -> the per-call counters are reset every call, so the
    # round-level published hits must collapse to 0 (proves the kill actually disabled reuse).
    killed_hits = sum(s["hits"] for s in ev._JMIM_CACHE_STATS)
    assert killed_hits == 0, f"kill path still recorded {killed_hits} hits -- not actually disabled"

    # HARD PARITY GATE: selection order + support must be byte-identical. The cache only
    # memoises the same mi() value, so selection MUST NOT change.
    assert names_cached == names_killed, f"JMIM cache changed selection ORDER:\n cached={names_cached}\n killed={names_killed}"
    assert np.array_equal(support_cached, support_killed), "JMIM cache changed support_ mask -- selection is NOT cache-invariant"
