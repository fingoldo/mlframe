"""The radix-select launchers say so when they fall back, and skip the large compaction scratch when it would not fit in VRAM (tracker 2026-11-02, LNC6-1, LNC6-2)."""

import logging

import numpy as np
import pytest

from mlframe.feature_selection.filters import _gpu_resident_select as grs


def test_scratch_fits_compares_against_a_fraction_of_the_free_memory(monkeypatch):
    """A scratch buffer within half the free VRAM fits; one over it does not; unknown free memory means try."""
    monkeypatch.setattr(grs, "_free_vram_bytes", lambda: 1000)
    assert grs._scratch_fits(500)
    assert not grs._scratch_fits(501)
    monkeypatch.setattr(grs, "_free_vram_bytes", lambda: None)
    assert grs._scratch_fits(10**18)


def _gpu_ok():
    """Whether cupy and a device are present."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_ok(), reason="no CUDA device")
def test_a_failing_fused_quantile_kernel_is_logged_and_the_two_kernel_path_answers(monkeypatch, caplog):
    """When the fused f64 kernel raises, a warning names the exception, and the result still equals cp.percentile."""
    import cupy as cp

    def boom():
        """Stand in for a fused kernel whose launch fails."""
        raise RuntimeError("simulated launch failure")

    monkeypatch.setattr(grs, "_get_radix_select_interp_f64_kernel", boom)
    rng = np.random.default_rng(0)
    data = cp.asarray(rng.normal(size=(2000, 3)))
    with caplog.at_level(logging.WARNING, logger=grs.logger.name):
        out = grs._radix_quantiles(data, [0.25, 0.5, 0.75])
    assert out is not None
    assert any("simulated launch failure" in r.getMessage() for r in caplog.records)
    expected = cp.percentile(data, [25.0, 50.0, 75.0], axis=0)
    np.testing.assert_allclose(cp.asnumpy(out), cp.asnumpy(expected), rtol=0, atol=1e-12)
