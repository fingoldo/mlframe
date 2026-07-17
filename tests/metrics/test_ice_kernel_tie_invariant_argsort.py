"""Regression: _batch_per_class_ice_kernel now takes the descending score order as a precomputed arg (hoisted to
numpy's C argsort because numba's own argsort is ~3.6x slower). This is correct ONLY because the AUC/PR walk emits at
tie-run boundaries and is therefore invariant to the WITHIN-TIE order the sort chose. Pin that: feeding two DIFFERENT
valid descending permutations (numpy quicksort vs stable mergesort, which order ties differently) must yield a
bit-identical ICE. A future change that makes the walk tie-order-sensitive breaks this."""

import numpy as np
import pytest

from mlframe.metrics.core import _batch_per_class_ice_kernel

_KW = dict(
    nbins=10,
    use_weights=True,
    mae_weight=3.0,
    std_weight=2.0,
    brier_loss_weight=0.8,
    roc_auc_weight=1.5,
    pr_auc_weight=0.1,
    min_roc_auc=0.54,
    roc_auc_penalty=0.0,
)


@pytest.mark.parametrize("round_dec,n,k", [(2, 50_000, 1), (2, 20_000, 3), (1, 30_000, 2)])
def test_ice_kernel_invariant_to_within_tie_order(round_dec, n, k):
    rng = np.random.default_rng(round_dec * 100 + k)
    y_t = rng.integers(0, 2, (n, k)).astype(np.int8)
    # rounding forces heavy ties -- the regime the tie-run walk exists for
    p = np.clip(np.round(0.15 + 0.5 * y_t + rng.standard_normal((n, k)) * 0.3, round_dec), 1e-6, 1 - 1e-6)
    p = np.ascontiguousarray(p, dtype=np.float64)

    desc_qsort = np.ascontiguousarray(np.argsort(-p, axis=0).astype(np.int64))  # quicksort (default)
    desc_msort = np.ascontiguousarray(np.argsort(-p, axis=0, kind="mergesort").astype(np.int64))  # stable
    # sanity: the two orders genuinely differ on this tied data (else the test proves nothing)
    assert not np.array_equal(desc_qsort, desc_msort), "no ties present -- test would be vacuous"

    ice_q = _batch_per_class_ice_kernel(y_t, p, desc_qsort, **_KW)
    ice_m = _batch_per_class_ice_kernel(y_t, p, desc_msort, **_KW)
    assert np.array_equal(ice_q, ice_m), f"ICE differs by within-tie argsort order: quicksort={ice_q.tolist()} mergesort={ice_m.tolist()}"
