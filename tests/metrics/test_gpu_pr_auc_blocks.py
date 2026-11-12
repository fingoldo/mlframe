"""The block-wise average precision of ``gpu_multiple_pr_auc_scores`` equals ``sklearn`` whatever the block size.

Scores are rounded to a few levels so every column has long tie runs; the last score of one column equals the first of the next, so a run that leaked across the
column boundary of the flat boundary list would change an AP.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from sklearn.metrics import average_precision_score

from mlframe.metrics import _gpu_metrics as gm


def _tied_case(n=1500, m=9, seed=0):
    """Ten-level scores plus two columns whose sorted end and start hold the same value."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) > 0.6).astype(np.int8)
    p = np.round(rng.random((n, m)) * 10) / 10
    p[:, 1] = np.round(rng.random(n) * 5) / 10
    p[:, 2] = 0.5 + np.round(rng.random(n) * 5) / 10
    return y, p.astype(np.float32)


@pytest.mark.parametrize("cols_per_block", [1, 2, 4, 100])
def test_block_size_does_not_change_the_average_precisions(cols_per_block, monkeypatch):
    """Every block size gives the sklearn AP of each column, to round-off."""
    y, p = _tied_case()
    monkeypatch.setattr(gm, "_ROC_BLOCK_ELEMENTS", cols_per_block * p.shape[0])
    got = cp.asnumpy(gm.gpu_multiple_pr_auc_scores(cp.asarray(y), cp.asarray(p)))
    want = np.array([average_precision_score(y, p[:, j]) for j in range(p.shape[1])])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_a_constant_column_and_continuous_columns():
    """A column of one repeated score has a single threshold (AP = prevalence); continuous columns match too."""
    rng = np.random.default_rng(5)
    n = 800
    y = (rng.random(n) > 0.7).astype(np.int8)
    p = rng.random((n, 3)).astype(np.float32)
    p[:, 1] = 0.25
    got = cp.asnumpy(gm.gpu_multiple_pr_auc_scores(cp.asarray(y), cp.asarray(p)))
    want = np.array([average_precision_score(y, p[:, j]) for j in range(3)])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    assert abs(got[1] - y.mean()) < 1e-12


def test_a_single_class_label_gives_nan():
    """Undefined AP stays NaN for every column."""
    p = cp.asarray(np.random.default_rng(1).random((50, 3)).astype(np.float32))
    assert np.isnan(cp.asnumpy(gm.gpu_multiple_pr_auc_scores(cp.zeros(50, dtype=cp.int8), p))).all()
