"""The additive-fusion device upload is admitted by the VRAM cushion before any copy."""

import numpy as np
import pytest


def test_upload_refused_by_the_vram_cushion_raises_the_device_fault_the_caller_handles(monkeypatch):
    """The bulk upload is refused before any copy when the VRAM cushion refuses it: the device out-of-memory error the fusion caller falls back on is raised; with the real cushion the upload runs."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters import _fe_additive_fusion_gpu_resident as G
    from mlframe.feature_selection.filters import _fe_gpu_vram

    rng = np.random.default_rng(0)
    n, H = 2000, 3
    vals_cols = [rng.standard_normal(n) for _ in range(H)]
    y = rng.integers(0, 3, n)
    out = G._upload_and_subsample(cp, vals_cols, n, y, H)
    assert len(out) == 6
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda *args, **kwargs: False)
    with pytest.raises(cp.cuda.memory.OutOfMemoryError):
        G._upload_and_subsample(cp, vals_cols, n, y, H)
