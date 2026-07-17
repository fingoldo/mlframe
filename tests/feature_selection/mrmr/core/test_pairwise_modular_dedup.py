"""Wave 13 (4): _pairwise_modular_fe.py escalate_modulus/cheap_modular_scan dedup.

  * escalate_modulus reuses the c_arr ModularHit already carries from cheap_modular_scan instead of
    rebuilding it via _combine.
  * cheap_modular_scan(..., _cols_prefiltered=True) skips the redundant _is_integer_col re-scan when
    the caller (hybrid_pairwise_modular_fe_with_recipes) already ran the identical filter.

Both must be selection-equivalent to the pre-fix behavior.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest


def _build_modular_fixture(n=2000, seed=0):
    """y = (a + b) mod 5 -> pair-sum modular structure on (a, b); c is integer noise."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 50, n)
    b = rng.integers(0, 50, n)
    c = rng.integers(0, 50, n)
    y = ((a + b) % 5).astype(np.int64)
    return pd.DataFrame({"a": a, "b": b, "c": c}), y


def test_escalate_modulus_reuses_hit_c_arr():
    """escalate_modulus must return the SAME (best_m, best_mi) whether or not hit.c_arr is populated,
    and must NOT call _combine when c_arr is available."""
    from mlframe.feature_selection.filters._pairwise_modular_fe import (
        cheap_modular_scan,
        escalate_modulus,
    )
    import mlframe.feature_selection.filters._pairwise_modular_fe as mod

    df, y = _build_modular_fixture()
    hits = cheap_modular_scan(df, y, seed=0)
    responded = [h for h in hits if h.responded]
    assert responded, "sanity: this fixture must produce at least one responded hit"
    hit = responded[0]
    assert hit.c_arr is not None, "cheap_modular_scan must populate c_arr on every hit"

    real_combine = mod._combine
    calls = {"n": 0}

    def _counting_combine(*a, **kw):
        """Wrap mod._combine to count invocations, proving escalate_modulus reuses a pre-populated hit.c_arr instead of rebuilding it."""
        calls["n"] += 1
        return real_combine(*a, **kw)

    with mock.patch.object(mod, "_combine", _counting_combine):
        best_m_with, best_mi_with, residue_with = escalate_modulus(df, y, hit)
    assert calls["n"] == 0, "escalate_modulus must not rebuild c_arr via _combine when hit.c_arr is set"

    # Bare hit (no c_arr) must fall back to rebuilding and match exactly.
    from dataclasses import replace

    bare_hit = replace(hit, c_arr=None)
    with mock.patch.object(mod, "_combine", _counting_combine):
        best_m_without, best_mi_without, residue_without = escalate_modulus(df, y, bare_hit)
    assert calls["n"] == 1, "the bare (no c_arr) hit must fall back to rebuilding via _combine exactly once"

    assert best_m_with == best_m_without
    assert best_mi_with == pytest.approx(best_mi_without, abs=1e-12)
    np.testing.assert_array_equal(residue_with, residue_without)


def test_cols_prefiltered_matches_default_filtering():
    """cheap_modular_scan(cols=int_cols, _cols_prefiltered=True) must return IDENTICAL hits to the
    default (re-filtering) path when ``int_cols`` is already the correctly-filtered set."""
    from mlframe.feature_selection.filters._pairwise_modular_fe import (
        cheap_modular_scan,
        _is_integer_col,
    )

    df, y = _build_modular_fixture()
    int_cols = [c for c in df.columns if _is_integer_col(np.asarray(df[c]))]

    hits_default = cheap_modular_scan(df, y, int_cols, seed=0)
    hits_prefiltered = cheap_modular_scan(df, y, int_cols, seed=0, _cols_prefiltered=True)

    assert len(hits_default) == len(hits_prefiltered)
    for h0, h1 in zip(hits_default, hits_prefiltered):
        assert h0.op == h1.op
        assert h0.cols == h1.cols
        assert h0.modulus == h1.modulus
        assert h0.residue_mi == pytest.approx(h1.residue_mi, abs=1e-12)


def test_hybrid_recipes_end_to_end_unaffected():
    """hybrid_pairwise_modular_fe_with_recipes must still detect and emit the modular recipe."""
    from mlframe.feature_selection.filters._pairwise_modular_fe import hybrid_pairwise_modular_fe_with_recipes

    df, y = _build_modular_fixture()
    appended, _recipes = hybrid_pairwise_modular_fe_with_recipes(df, y, seed=0)
    assert appended, "sanity: (a+b) mod 5 structure must be detected and recipe-emitted"
