"""A failing GPU combiner pass must fall back to the host scan with a WARNING, not a silent debug line."""

import logging

import numpy as np

from mlframe.feature_selection.filters import _gpu_strict_fe
from mlframe.feature_selection.filters import _pairwise_modular_resident as pmr


def test_combiner_bases_resident_failure_warns_and_returns_none(monkeypatch, caplog):
    """A raised GPU failure returns None (the caller's host path) and is logged at WARNING so the slower path is visible."""
    monkeypatch.setattr(_gpu_strict_fe, "fe_gpu_device_born_modular_enabled", lambda: True)

    def _boom(*_a, **_k):
        """Stand in for the GPU rank-binning kernel and fail the way a device fault would."""
        raise RuntimeError("synthetic device fault")

    from mlframe.feature_selection.filters import _gpu_resident_rank_bin

    monkeypatch.setattr(_gpu_resident_rank_bin, "plugin_mi_classif_batch_rank_cuda_resident", _boom)
    combiners = [np.arange(40, dtype=np.int64)]
    y = np.arange(40) % 3
    with caplog.at_level(logging.WARNING, logger=pmr.__name__):
        out = pmr.combiner_bases_and_grids_resident(combiners, y, [3, 5], nbins=4, rank_binning=True)
    assert out is None
    assert any(r.levelno == logging.WARNING and "per-combiner scan" in r.getMessage() for r in caplog.records)
