"""screening-core: at max_veteranes_interactions_order >= 2 the partial-gain resume must NOT skip the
newest selected var's redundancy (audit P2, 2026-06-13).

The (current_gain, last_checked_k) cross-step resume is valid only when new selected vars APPEND to the
combination sequence (order==1). At order>=2 a new singleton inserts mid-sequence, so resuming from a
stale last_checked_k skipped measuring the new var's redundancy -> a feature fully redundant with the
most-recently-selected var could SURVIVE. The fix evaluates order>=2 from scratch. This pins that an
exact-duplicate column is dropped (not selected alongside its original) on the order>=2 path.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def test_order2_drops_exact_duplicate_redundant_feature():
    from mlframe.feature_selection.filters import MRMR
    n = 2500
    rng = np.random.default_rng(0)
    a = rng.random(n)
    b = rng.random(n)
    noise = rng.random(n)
    y = 2.0 * a + 1.5 * b + noise / 5.0
    dup = a.copy()                       # an EXACT duplicate of a -- must not be selected alongside a
    e = rng.random(n)                    # pure noise
    df = pd.DataFrame({"a": a, "b": b, "dup_a": dup, "e": e})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = MRMR(verbose=0, random_seed=0, max_veteranes_interactions_order=2).fit(
            X=df, y=pd.Series(y, name="y"))
    names = list(fs.get_feature_names_out())
    # the order>=2 path must complete and must not keep BOTH a and its exact duplicate (redundant).
    assert "a" in names or "dup_a" in names, f"lost the a-signal entirely: {names}"
    assert not ("a" in names and "dup_a" in names), (
        f"order>=2 kept BOTH a and its exact duplicate dup_a -- the redundancy of the duplicate against "
        f"the selected a was not measured (stale-resume bug): {names}"
    )


def test_order2_runs_and_selects_signal():
    """Smoke: the order>=2 research path completes and recovers the genuine signal features."""
    from mlframe.feature_selection.filters import MRMR
    n = 2500
    rng = np.random.default_rng(1)
    a, b, c, d, e = (rng.random(n) for _ in range(5))
    y = 2.0 * a + 1.5 * b + 0.3 * c
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = MRMR(verbose=0, random_seed=1, max_veteranes_interactions_order=2).fit(
            X=df, y=pd.Series(y, name="y"))
    names = list(fs.get_feature_names_out())
    assert names, "order>=2 fit selected nothing"
    assert "a" in names and "b" in names, f"order>=2 missed the strong linear signal a,b: {names}"
