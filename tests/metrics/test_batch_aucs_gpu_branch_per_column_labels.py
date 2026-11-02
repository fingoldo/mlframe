"""compute_batch_aucs hands the GPU kernels one 1-D label vector per score column when labels are an (N, M) matrix (tracker 2026-11-02 wave 12, PAR12-1)."""

import numpy as np
import pytest

pytest.importorskip("cupy")

from mlframe.metrics import _gpu_metrics as g


def test_gpu_branch_splits_a_label_matrix_by_column(monkeypatch):
    """The kernels pool the positive count over a label matrix; each call must see a 1-D label vector and one score column."""
    seen = []

    def fake_roc(y, s):
        """Record the shapes the kernel receives."""
        seen.append((np.ndim(y), np.shape(s)))
        return np.array([0.5])

    monkeypatch.setattr(g, "_resolve_backend", lambda *a, **k: True)
    monkeypatch.setattr(g, "gpu_multiple_roc_auc_scores", fake_roc)
    monkeypatch.setattr(g, "gpu_multiple_pr_auc_scores", lambda y, s: np.array([0.25]))
    rng = np.random.default_rng(0)
    yt = (rng.random((50, 3)) > 0.5).astype(np.int8)
    roc, pr = g.compute_batch_aucs(yt, rng.random((50, 3)))
    assert roc.shape == (3,) and pr.shape == (3,)
    assert seen == [(1, (50, 1))] * 3
