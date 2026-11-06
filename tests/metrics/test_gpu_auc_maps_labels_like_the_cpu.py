"""The GPU ROC and PR AUC accept {-1, 1} labels and agree with sklearn, as the CPU path does.

The GPU kernels counted ``1 - y`` as negatives, so {-1, 1} labels gave NaN on the device while the CPU returned the AUC (audit PARG4-1, wave 3, High).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover
    _HAS_DEVICE = False

pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")


def test_gpu_auc_with_minus_one_one_labels_matches_sklearn():
    """{-1, 1} labels give the sklearn ROC and PR AUC on the device, for every column."""
    from mlframe.metrics import _gpu_metrics as gm

    rng = np.random.default_rng(0)
    n = 4000
    y01 = rng.integers(0, 2, n)
    y = np.where(y01 == 1, 1, -1)
    scores = rng.normal(size=(n, 3)) + 0.8 * y01[:, None]
    roc = cp.asnumpy(gm.gpu_multiple_roc_auc_scores(y, scores))
    pr = cp.asnumpy(gm.gpu_multiple_pr_auc_scores(y, scores))
    for j in range(3):
        assert roc[j] == pytest.approx(roc_auc_score(y01, scores[:, j]), abs=1e-9)
        assert pr[j] == pytest.approx(average_precision_score(y01, scores[:, j]), abs=1e-9)
