"""The block-wise fractional ranks of ``gpu_multiple_roc_auc_scores`` equal the per-column result whatever the block size.

The columns of a block are laid end to end with a run boundary forced at each column start; heavily tied scores make a run that crosses a column boundary
visible, and the AUC of each column must still equal ``sklearn``'s.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from sklearn.metrics import roc_auc_score

from mlframe.metrics import _gpu_metrics as gm


def _tied_case(n=1500, m=9, seed=0):
    """Scores rounded to ten levels, so every column has long tied runs and the end of column 1 and the start of column 2 hold the same value."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) > 0.6).astype(np.int8)
    p = np.round(rng.random((n, m)) * 10) / 10
    p[:, 1] = np.round(rng.random(n) * 5) / 10  # values 0.0 .. 0.5: its sorted end is 0.5
    p[:, 2] = 0.5 + np.round(rng.random(n) * 5) / 10  # values 0.5 .. 1.0: its sorted start is 0.5, the same value, so a run would cross the column boundary
    return y, p.astype(np.float32)


@pytest.mark.parametrize("cols_per_block", [1, 2, 4, 100])
def test_block_size_does_not_change_the_aucs(cols_per_block, monkeypatch):
    """Every block size gives the sklearn AUC of each column, to round-off."""
    y, p = _tied_case()
    monkeypatch.setattr(gm, "_ROC_BLOCK_ELEMENTS", cols_per_block * p.shape[0])
    got = cp.asnumpy(gm.gpu_multiple_roc_auc_scores(cp.asarray(y), cp.asarray(p)))
    want = np.array([roc_auc_score(y, p[:, j]) for j in range(p.shape[1])])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_blocks_equal_one_block_exactly(monkeypatch):
    """Two columns per block and one block over all columns give identical floats."""
    y, p = _tied_case(seed=3)
    monkeypatch.setattr(gm, "_ROC_BLOCK_ELEMENTS", 1 << 40)
    whole = cp.asnumpy(gm.gpu_multiple_roc_auc_scores(cp.asarray(y), cp.asarray(p)))
    monkeypatch.setattr(gm, "_ROC_BLOCK_ELEMENTS", 2 * p.shape[0])
    blocked = cp.asnumpy(gm.gpu_multiple_roc_auc_scores(cp.asarray(y), cp.asarray(p)))
    assert np.array_equal(whole, blocked)
