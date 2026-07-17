"""Regression: the single-column merge cache exposed by
``_cat_mm_correction._maybe_rerank_with_mm`` (``single_merge_cache_out``) can be
fed directly into ``_cat_confirm_permutation._confirm_pairs_via_permutation``
(``single_merge_cache``) to skip re-deriving the SAME single-column
``merge_vars`` result for the SAME survivor set.

Both functions independently scan the same top-K pairs on the same
factors_data/nbins/dtype and each maintain their OWN cache of single-column
``merge_vars`` results; this test pins that:
1. The values a shared dict collects from the MM pass are usable AS-IS by the
   permutation-confirm pass (bit-identical confidence vs the unshared path).
2. Feeding a pre-populated cache into the confirm pass measurably avoids
   re-deriving ``merge_vars`` for the columns the MM pass already touched.

No caller wires these two together yet (that needs
``_cat_interactions_step.py``, out of scope for this module) -- this test
exercises the two functions directly, standing in for that future orchestrator
wiring.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig
from mlframe.feature_selection.filters import _cat_confirm_permutation as _ccp
from mlframe.feature_selection.filters._cat_confirm_permutation import _confirm_pairs_via_permutation
from mlframe.feature_selection.filters._cat_mm_correction import _maybe_rerank_with_mm
from mlframe.feature_selection.filters.info_theory._class_encoding import merge_vars


def _make_data(n=3000, n_cols=5, n_y=3, seed=13):
    rng = np.random.default_rng(seed)
    nbins = np.array([n_y] + [5] * n_cols, dtype=np.int64)
    cols = [rng.integers(0, nbins[c], size=n).astype(np.int32) for c in range(n_cols + 1)]
    cols[1] = ((cols[0] + rng.integers(0, 2, size=n)) % nbins[1]).astype(np.int32)
    factors_data = np.column_stack(cols).astype(np.int32)
    return factors_data, nbins


def _setup():
    dtype = np.int32
    factors_data, nbins = _make_data()
    classes_y, freqs_y, _ = merge_vars(
        factors_data=factors_data,
        vars_indices=np.array([0], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=dtype,
    )
    pairs = [(1, 2), (1, 3), (2, 4), (3, 5)]  # columns 1/2/3 recur across both functions' pair sets
    pairs_a = np.array([p[0] for p in pairs], dtype=np.int64)
    pairs_b = np.array([p[1] for p in pairs], dtype=np.int64)
    selected_idx = np.arange(len(pairs))
    ii_arr = np.full(len(pairs), 0.0005, dtype=np.float64)
    target_indices = np.array([0], dtype=np.int64)
    return dtype, factors_data, nbins, classes_y, freqs_y, pairs_a, pairs_b, selected_idx, ii_arr, target_indices


def test_shared_cache_from_mm_rerank_is_bit_identical_in_confirm_pairs():
    dtype, factors_data, nbins, classes_y, freqs_y, pairs_a, pairs_b, selected_idx, ii_arr, target_indices = _setup()
    cfg = CatFEConfig(full_npermutations=30, permutation_null="joint_independence", fwer_correction="none", use_miller_madow=True)

    # Baseline: run confirm_pairs standalone, no shared cache.
    _kept_base, conf_base = _confirm_pairs_via_permutation(
        factors_data=factors_data,
        pairs_a=pairs_a,
        pairs_b=pairs_b,
        selected_idx=selected_idx,
        ii_arr=ii_arr,
        nbins=nbins,
        classes_y=classes_y,
        freqs_y=freqs_y,
        cfg=cfg,
        n_search_pairs=len(pairs_a),
        dtype=dtype,
        verbose=0,
    )

    # MM rerank populates a shared single-merge cache, then confirm_pairs consumes it.
    shared_cache: dict = {}
    _maybe_rerank_with_mm(
        factors_data=factors_data,
        pairs_a=pairs_a,
        pairs_b=pairs_b,
        selected_idx=selected_idx,
        ii_arr=ii_arr,
        nbins=nbins,
        target_indices=target_indices,
        classes_y=classes_y,
        freqs_y=freqs_y,
        cfg=cfg,
        dtype=dtype,
        verbose=0,
        single_merge_cache_out=shared_cache,
    )
    assert len(shared_cache) > 0, "MM rerank should have populated the shared cache for touched columns"

    _kept_shared, conf_shared = _confirm_pairs_via_permutation(
        factors_data=factors_data,
        pairs_a=pairs_a,
        pairs_b=pairs_b,
        selected_idx=selected_idx,
        ii_arr=ii_arr,
        nbins=nbins,
        classes_y=classes_y,
        freqs_y=freqs_y,
        cfg=cfg,
        n_search_pairs=len(pairs_a),
        dtype=dtype,
        verbose=0,
        single_merge_cache=shared_cache,
    )

    assert set(conf_base) == set(conf_shared)
    for key in conf_base:
        assert conf_base[key] == conf_shared[key], (
            f"shared single-merge cache changed confidence for {key}: {conf_shared[key]!r} != baseline {conf_base[key]!r}"
        )


def test_shared_cache_avoids_recomputing_merge_vars_for_precached_columns(monkeypatch):
    dtype, factors_data, nbins, classes_y, freqs_y, pairs_a, pairs_b, selected_idx, ii_arr, target_indices = _setup()
    cfg = CatFEConfig(full_npermutations=30, permutation_null="joint_independence", fwer_correction="none", use_miller_madow=True)

    shared_cache: dict = {}
    _maybe_rerank_with_mm(
        factors_data=factors_data,
        pairs_a=pairs_a,
        pairs_b=pairs_b,
        selected_idx=selected_idx,
        ii_arr=ii_arr,
        nbins=nbins,
        target_indices=target_indices,
        classes_y=classes_y,
        freqs_y=freqs_y,
        cfg=cfg,
        dtype=dtype,
        verbose=0,
        single_merge_cache_out=shared_cache,
    )
    precached_cols = set(shared_cache.keys())
    assert precached_cols, "expected at least one precached column from the MM rerank"

    calls: list = []
    real_merge_vars = _ccp.merge_vars

    def _counting_merge_vars(*args, **kwargs):
        vi = kwargs.get("vars_indices")
        calls.append(tuple(int(x) for x in vi))
        return real_merge_vars(*args, **kwargs)

    monkeypatch.setattr(_ccp, "merge_vars", _counting_merge_vars)

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
        n_search_pairs=len(pairs_a),
        dtype=dtype,
        verbose=0,
        single_merge_cache=shared_cache,
    )

    single_col_calls_on_precached = [c for c in calls if len(c) == 1 and c[0] in precached_cols]
    assert not single_col_calls_on_precached, (
        f"confirm_pairs re-derived merge_vars for column(s) already precached by the MM rerank: {single_col_calls_on_precached}"
    )
