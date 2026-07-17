"""Regression: single-column ``merge_vars`` memoization across survivor pairs in
``_confirm_pairs_via_permutation``'s main loop.

Pre-fix, ``cls_x1``/``fq_x1``/``cls_x2``/``fq_x2`` were re-derived via a fresh
``merge_vars(vars_indices=[i])`` call for EVERY survivor pair, even when a
column recurs across multiple pairs (a common case: any "star" pattern where
several survivor pairs share one hub column). The sibling function
``_cat_confirm_bandit._confirm_pairs_bandit_ucb1`` already fixed this exact
pattern via a ``_single_merge_cache`` dict; this test pins the same fix now
ported into ``_confirm_pairs_via_permutation``.

Two invariants are pinned:
1. Call-count: with a "star" pair pattern (all pairs share column 0), the
   number of single-column ``merge_vars`` calls must equal the number of
   DISTINCT columns touched, not ``2 * n_pairs``.
2. Bit-identity: the per-pair confidence output must be EXACTLY the same as a
   reference re-implementation that recomputes the single-column merge fresh
   per pair (the pre-fix form) -- memoization must never change numerics.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig
from mlframe.feature_selection.filters import _cat_confirm_permutation as _ccp
from mlframe.feature_selection.filters._cat_confirm_permutation import _confirm_pairs_via_permutation
from mlframe.feature_selection.filters.info_theory._class_encoding import merge_vars


def _make_data(n=3000, n_cols=5, n_y=3, seed=11):
    """Make data."""
    rng = np.random.default_rng(seed)
    nbins = np.array([n_y] + [5] * n_cols, dtype=np.int64)
    cols = [rng.integers(0, nbins[c], size=n).astype(np.int32) for c in range(n_cols + 1)]
    # mild synergy so ii_obs isn't degenerate for at least one pair
    cols[1] = ((cols[0] + rng.integers(0, 2, size=n)) % nbins[1]).astype(np.int32)
    factors_data = np.column_stack(cols).astype(np.int32)
    return factors_data, nbins


def _reference_confidence(
    factors_data,
    pairs_a,
    pairs_b,
    selected_idx,
    ii_arr,
    nbins,
    classes_y,
    freqs_y,
    cfg,
    dtype,
):
    """Pre-fix replica: single-column merge_vars re-derived fresh per pair (no cache)."""
    from mlframe.feature_selection.filters._cat_confirm_permutation import (
        _count_nfailed_joint_indep_serial,
    )

    n_perms = cfg.full_npermutations
    factors_data.shape[0]
    confidence_dict = {}
    for j, k in enumerate(selected_idx):
        i = int(pairs_a[k])
        jj = int(pairs_b[k])
        ii_obs = float(ii_arr[k])
        cls_pair, fq_pair, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([i, jj], dtype=np.int64),
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=dtype,
        )
        cls_x1, fq_x1, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([i], dtype=np.int64),
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=dtype,
        )
        cls_x2, fq_x2, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([jj], dtype=np.int64),
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=dtype,
        )
        n_failed = _count_nfailed_joint_indep_serial(
            cls_pair,
            fq_pair,
            cls_x1,
            fq_x1,
            cls_x2,
            fq_x2,
            classes_y,
            freqs_y,
            ii_obs,
            n_perms,
            base_seed=int(j) * 1000003 + 7,
            dtype=dtype,
        )
        p = (n_failed + 1) / (n_perms + 1)
        confidence_dict[(i, jj)] = 1.0 - p
    return confidence_dict


def _star_pairs(n_cols=5):
    """All pairs share column 1 as the hub -- maximises single-column reuse."""
    pairs = [(1, c) for c in range(2, n_cols + 1)]
    pairs_a = np.array([p[0] for p in pairs], dtype=np.int64)
    pairs_b = np.array([p[1] for p in pairs], dtype=np.int64)
    return pairs, pairs_a, pairs_b


def test_single_merge_cache_reduces_merge_vars_call_count(monkeypatch):
    """Single merge cache reduces merge vars call count."""
    dtype = np.int32
    factors_data, nbins = _make_data()
    classes_y, freqs_y, _ = merge_vars(
        factors_data=factors_data,
        vars_indices=np.array([0], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=dtype,
    )
    pairs, pairs_a, pairs_b = _star_pairs(n_cols=5)
    selected_idx = np.arange(len(pairs))
    ii_arr = np.full(len(pairs), 0.0005, dtype=np.float64)

    calls: list = []
    real_merge_vars = _ccp.merge_vars

    def _counting_merge_vars(*args, **kwargs):
        """Counting merge vars."""
        vi = kwargs.get("vars_indices")
        calls.append(tuple(int(x) for x in vi))
        return real_merge_vars(*args, **kwargs)

    monkeypatch.setattr(_ccp, "merge_vars", _counting_merge_vars)

    cfg = CatFEConfig(full_npermutations=30, permutation_null="joint_independence", fwer_correction="none")
    _confirm_pairs_via_permutation(
        factors_data=factors_data,
        pairs_a=pairs_a,
        pairs_b=pairs_b,
        selected_idx=selected_idx,
        ii_arr=ii_arr,
        nbins=nbins,
        classes_y=classes_y,
        freqs_y=freqs_y,
        cfg=cfg,
        n_search_pairs=len(pairs),
        dtype=dtype,
        verbose=0,
    )

    single_col_calls = [c for c in calls if len(c) == 1]
    distinct_cols_touched = {c[0] for c in single_col_calls}
    # Star pattern: hub column 1 recurs in every pair; each leaf column (2..5) appears once.
    # Without the cache this would be 2 single-col merge_vars calls per pair (2 * n_pairs = 8);
    # with the cache it's exactly one call per DISTINCT column touched.
    assert len(single_col_calls) == len(distinct_cols_touched), (
        f"expected exactly one single-column merge_vars call per distinct column "
        f"({len(distinct_cols_touched)}), got {len(single_col_calls)} calls: {single_col_calls}"
    )
    assert len(single_col_calls) < 2 * len(pairs)


@pytest.mark.parametrize("n_perms", [30, 100])
def test_single_merge_cache_bit_identical_to_uncached_reference(n_perms):
    """Single merge cache bit identical to uncached reference."""
    dtype = np.int32
    factors_data, nbins = _make_data()
    classes_y, freqs_y, _ = merge_vars(
        factors_data=factors_data,
        vars_indices=np.array([0], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=dtype,
    )
    pairs, pairs_a, pairs_b = _star_pairs(n_cols=5)
    selected_idx = np.arange(len(pairs))
    ii_arr = np.full(len(pairs), 0.0005, dtype=np.float64)

    cfg = CatFEConfig(full_npermutations=n_perms, permutation_null="joint_independence", fwer_correction="none")

    _kept, conf = _confirm_pairs_via_permutation(
        factors_data=factors_data,
        pairs_a=pairs_a,
        pairs_b=pairs_b,
        selected_idx=selected_idx,
        ii_arr=ii_arr,
        nbins=nbins,
        classes_y=classes_y,
        freqs_y=freqs_y,
        cfg=cfg,
        n_search_pairs=len(pairs),
        dtype=dtype,
        verbose=0,
    )
    ref = _reference_confidence(
        factors_data,
        pairs_a,
        pairs_b,
        selected_idx,
        ii_arr,
        nbins,
        classes_y,
        freqs_y,
        cfg,
        dtype,
    )

    assert set(conf) == set(ref)
    for key in ref:
        assert conf[key] == ref[key], f"single-merge cache changed confidence for {key}: {conf[key]!r} != reference {ref[key]!r}"
