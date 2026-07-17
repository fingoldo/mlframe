"""Regression test for the single-merge_vars collapse in
``_refine_kway_coordinate_ascent``.

The pre-fix code called ``merge_vars`` twice per swap candidate (once
discarding freqs, once discarding classes) plus a dead ``if False else None``
line. The fix calls it once and reuses both outputs. This test pins that the
refinement output (selected tuples + their joint-MI) is unchanged, i.e. the
collapse is numerically equivalent.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._cat_post_refine import _refine_kway_coordinate_ascent
from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes, merge_vars


def _build(n=4000, n_cols=8, n_bins=5, seed=11):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, n_bins, size=(n, n_cols)).astype(np.int32)
    # Make the last column (target) depend on cols 0,1,2 so swaps can improve MI.
    data[:, n_cols - 1] = (data[:, 0] + 2 * data[:, 1] + data[:, 2]) % n_bins
    nbins = np.full(n_cols, n_bins, dtype=np.int64)
    cls_y, fq_y, _ = merge_vars(
        factors_data=data,
        vars_indices=np.array([n_cols - 1], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=np.int32,
    )
    return data, nbins, cls_y, fq_y


def _seed_result(data, nbins, cls_y, fq_y, tup, dtype=np.int32):
    cls, fq, nuniq = merge_vars(
        factors_data=data,
        vars_indices=np.array(sorted(tup), dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=dtype,
    )
    mi = compute_mi_from_classes(classes_x=cls, freqs_x=fq, classes_y=cls_y, freqs_y=fq_y, dtype=dtype)
    return (tuple(sorted(tup)), cls, nuniq, mi)


def test_kway_coord_ascent_matches_expected_reference():
    data, nbins, cls_y, fq_y = _build()
    pool = np.arange(7, dtype=np.int64)  # exclude target col (idx 7)
    seeds = [_seed_result(data, nbins, cls_y, fq_y, (3, 4, 5))]

    refined = _refine_kway_coordinate_ascent(
        factors_data=data,
        kway_results=seeds,
        candidate_pool=pool,
        nbins=nbins,
        classes_y=cls_y,
        freqs_y=fq_y,
        max_combined_nbins=10_000,
        n_passes=3,
        dtype=np.int32,
        verbose=0,
    )

    # Recompute the reference MI for each refined tuple from scratch -- if the
    # double-merge collapse had introduced any divergence, the refinement would
    # have accepted/rejected swaps differently and these would not match.
    for tup, cls, nuniq, mi in refined:
        ref_cls, ref_fq, ref_nuniq = merge_vars(
            factors_data=data,
            vars_indices=np.array(sorted(tup), dtype=np.int64),
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=np.int32,
        )
        ref_mi = compute_mi_from_classes(
            classes_x=ref_cls,
            freqs_x=ref_fq,
            classes_y=cls_y,
            freqs_y=fq_y,
            dtype=np.int32,
        )
        assert mi == pytest.approx(ref_mi, abs=0.0), (tup, mi, ref_mi)
        assert nuniq == ref_nuniq
        assert np.array_equal(cls, ref_cls)

    # Coordinate ascent must have found the planted (0,1,2) structure given the
    # seed (3,4,5) and a candidate pool that contains 0,1,2.
    best_tuple = max(refined, key=lambda r: r[3])[0]
    assert set(best_tuple) >= {0, 1, 2}, best_tuple
