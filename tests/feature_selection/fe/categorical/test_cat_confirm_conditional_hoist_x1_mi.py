"""Regression: hoisting the loop-invariant I(X1; Y) out of the conditional-null
permutation loop in ``_confirm_pairs_via_permutation`` must NOT change the
confidence output (bit-identical by construction).

Under ``permutation_null="conditional"`` only X2 is shuffled inside the inner
permutation loop; X1 (cls_x1/fq_x1) and Y (classes_y/freqs_y) are never mutated,
so I(X1; Y) is loop-invariant. The optimization computes it once before the loop
instead of every permutation.

This test pins that the shipped (hoisted) ``_confirm_pairs_via_permutation``
produces exactly the same per-pair confidence as a reference re-implementation of
the SAME algorithm that recomputes I(X1; Y) every permutation (the pre-fix form).
Equality must be EXACT -- the optimization removes redundant work, never changes
numerics. A future "just recompute it / inline it differently" that breaks the
invariance assumption fails here.

Bench: src/mlframe/feature_selection/filters/_benchmarks/bench_cat_confirm_conditional_hoist_x1_mi.py
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig
from mlframe.feature_selection.filters._cat_confirm_permutation import (
    _confirm_pairs_via_permutation,
    _conditional_shuffle_within_strata,
    _CAT_CONFIRM_BASE_SEED,
)
from mlframe.feature_selection.filters.info_theory._class_mi_kernels import (
    compute_mi_from_classes,
)
from mlframe.feature_selection.filters.info_theory._class_encoding import merge_vars


def _make_data(n=4000, n_cols=4, n_y=3, seed=7):
    rng = np.random.default_rng(seed)
    # ordinal-encoded factors; column 0 = target carrier.
    nbins = np.array([n_y] + [6] * n_cols, dtype=np.int64)
    cols = [rng.integers(0, nbins[c], size=n).astype(np.int32) for c in range(n_cols + 1)]
    # inject a mild synergy so II_obs is positive for at least one pair
    cols[1] = ((cols[0] + rng.integers(0, 2, size=n)) % nbins[1]).astype(np.int32)
    factors_data = np.column_stack(cols).astype(np.int32)
    return factors_data, nbins


def _reference_conditional_confidence(
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
    """Pre-fix replica: conditional null with I(X1;Y) RECOMPUTED every perm."""
    n_perms = cfg.full_npermutations
    n_y_classes = int(classes_y.max()) + 1
    confidence_dict = {}
    for j, k in enumerate(selected_idx):
        i = int(pairs_a[k])
        jj = int(pairs_b[k])
        ii_obs = float(ii_arr[k])
        cls_x1, fq_x1, _ = merge_vars(
            factors_data=factors_data, vars_indices=np.array([i], dtype=np.int64), var_is_nominal=None, factors_nbins=nbins, dtype=dtype
        )
        cls_x2, _fq_x2, _ = merge_vars(
            factors_data=factors_data, vars_indices=np.array([jj], dtype=np.int64), var_is_nominal=None, factors_nbins=nbins, dtype=dtype
        )
        classes_x2_safe = cls_x2.astype(np.int64, copy=True)
        classes_x1_arr = cls_x1.astype(np.int64, copy=False)
        n = factors_data.shape[0]
        n_failed = 0
        for _perm in range(n_perms):
            _cond_seed = _CAT_CONFIRM_BASE_SEED + int(j) * 1000003 + _perm
            _conditional_shuffle_within_strata(classes_x2_safe, classes_y, n_y_classes, _cond_seed)
            local_data = np.empty((n, 2), dtype=dtype)
            local_data[:, 0] = classes_x1_arr.astype(dtype, copy=False)
            local_data[:, 1] = classes_x2_safe.astype(dtype, copy=False)
            local_nbins = np.array([int(cls_x1.max()) + 1, int(classes_x2_safe.max()) + 1], dtype=np.int64)
            cj, fj, _ = merge_vars(
                factors_data=local_data, vars_indices=np.array([0, 1], dtype=np.int64), var_is_nominal=None, factors_nbins=local_nbins, dtype=dtype
            )
            i_pair_p = compute_mi_from_classes(classes_x=cj, freqs_x=fj, classes_y=classes_y, freqs_y=freqs_y, dtype=dtype)
            i_x1_p = compute_mi_from_classes(classes_x=cls_x1, freqs_x=fq_x1, classes_y=classes_y, freqs_y=freqs_y, dtype=dtype)
            fq_x2_perm = np.bincount(classes_x2_safe.astype(np.int64), minlength=int(classes_x2_safe.max()) + 1).astype(np.float64) / n
            i_x2_p = compute_mi_from_classes(
                classes_x=classes_x2_safe.astype(dtype, copy=False), freqs_x=fq_x2_perm.astype(np.float64), classes_y=classes_y, freqs_y=freqs_y, dtype=dtype
            )
            if (i_pair_p - i_x1_p - i_x2_p) >= ii_obs:
                n_failed += 1
        p = (n_failed + 1) / (n_perms + 1)
        confidence_dict[(i, jj)] = 1.0 - p
    return confidence_dict


@pytest.mark.parametrize("n_perms", [50, 200])
def test_conditional_null_hoist_x1_mi_bit_identical(n_perms):
    dtype = np.int32
    factors_data, nbins = _make_data()
    factors_data.shape[0]
    classes_y, freqs_y, _ = merge_vars(
        factors_data=factors_data, vars_indices=np.array([0], dtype=np.int64), var_is_nominal=None, factors_nbins=nbins, dtype=dtype
    )

    # candidate pairs over the non-target columns
    pairs = [(1, 2), (1, 3), (2, 3)]
    pairs_a = np.array([p[0] for p in pairs], dtype=np.int64)
    pairs_b = np.array([p[1] for p in pairs], dtype=np.int64)
    selected_idx = np.arange(len(pairs))
    # observed II per pair (point estimate -- just needs to be a fixed reference)
    ii_arr = np.array([0.001, 0.0005, 0.0002], dtype=np.float64)

    cfg = CatFEConfig(
        full_npermutations=n_perms,
        permutation_null="conditional",
        fwer_correction="none",
    )

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

    ref = _reference_conditional_confidence(
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

    # fwer_correction="none" -> shipped confidence is the raw per-pair confidence,
    # which must match the per-perm-recompute reference EXACTLY.
    assert set(conf) == set(ref)
    for key in ref:
        assert conf[key] == ref[key], f"hoisted I(X1;Y) changed confidence for {key}: {conf[key]!r} != reference {ref[key]!r}"
