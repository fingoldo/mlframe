"""_maybe_rerank_with_mm (hoisted per-column entropies) must equal the unhoisted per-pair reference _compute_pair_ii_mm."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._cat_mm_correction import _compute_pair_ii_mm, _entropy_for_mode, _maybe_rerank_with_mm
from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig
from mlframe.feature_selection.filters.info_theory import merge_vars


def _fixture(seed: int, n: int = 500):
    """Four small-cardinality columns where column 3 is the target and (0, 1) carry an XOR-like synergy."""
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 4, n)
    x1 = rng.integers(0, 4, n)
    x2 = rng.integers(0, 5, n)
    y = ((x0 + x1) % 3 + (rng.random(n) < 0.1)).astype(np.int64) % 3
    data = np.column_stack([x0, x1, x2, y]).astype(np.int32)
    nbins = np.array([4, 4, 5, 3], dtype=np.int64)
    cls_y, fq_y, _ = merge_vars(factors_data=data, vars_indices=np.array([3], dtype=np.int64), var_is_nominal=None, factors_nbins=nbins, dtype=np.int32)
    return data, nbins, cls_y, fq_y


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hoisted_rerank_equals_per_pair_reference(seed):
    """Every pair's MM-corrected II from the batched re-rank equals the single-pair reference value."""
    data, nbins, cls_y, fq_y = _fixture(seed)
    pairs_a = np.array([0, 0, 1], dtype=np.int64)
    pairs_b = np.array([1, 2, 2], dtype=np.int64)
    selected = np.arange(3, dtype=np.int64)
    target = np.array([3], dtype=np.int64)
    cfg = CatFEConfig(enable=True, use_miller_madow=True)
    ii_mm, _ = _maybe_rerank_with_mm(
        factors_data=data, pairs_a=pairs_a, pairs_b=pairs_b, selected_idx=selected, ii_arr=np.zeros(3), nbins=nbins,
        target_indices=target, classes_y=cls_y, freqs_y=fq_y, cfg=cfg, dtype=np.int32, verbose=0,
    )
    h_y = _entropy_for_mode(fq_y, data.shape[0], False)
    for k in range(3):
        ref = _compute_pair_ii_mm(data, int(pairs_a[k]), int(pairs_b[k]), nbins, target, cls_y, fq_y, h_y, True, np.int32)
        assert ii_mm[k] == pytest.approx(ref, rel=1e-9, abs=1e-12)


def test_mm_bias_is_actually_applied():
    """Negative control: the reference with MM off differs from the MM value, so the parity above checks the bias term."""
    data, nbins, cls_y, fq_y = _fixture(0)
    target = np.array([3], dtype=np.int64)
    h_y = _entropy_for_mode(fq_y, data.shape[0], False)
    plug = _compute_pair_ii_mm(data, 0, 1, nbins, target, cls_y, fq_y, h_y, False, np.int32)
    mm = _compute_pair_ii_mm(data, 0, 1, nbins, target, cls_y, fq_y, h_y, True, np.int32)
    assert mm < plug
