"""Stratified bootstrap keeps every class even when labels are fractional (audit 2026-11-20, finding 22-F8)."""
import numpy as np

from mlframe.evaluation._bootstrap_fused_binary_bundle import _iter_resample_idx_chunks


def test_fractional_labels_do_not_collapse_into_one_class():
    """Labels 0.2 and 0.7 both used to int() to 0; each resample must still draw from both classes."""
    rng = np.random.default_rng(0)
    stratify = np.array([0.2] * 30 + [0.7] * 30)
    n = stratify.size
    chunks = list(_iter_resample_idx_chunks(rng, n_bootstrap=5, n=n, stratify=stratify, chunk_size=5))
    _lo, _hi, idx = chunks[0]
    for row in idx:
        labels = set(np.round(stratify[row], 2).tolist())
        assert labels == {0.2, 0.7}
