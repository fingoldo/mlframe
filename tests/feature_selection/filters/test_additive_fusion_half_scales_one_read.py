"""``_raw_half_scales`` reads the scales of all halves back with one copy and gives the per-column values (audit 2026-11-20-perf F-F04)."""

from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._fe_additive_fusion_gpu_resident import _raw_half_scales


def _reference(cols):
    """The former loop: one float per column, 1.0 for a non-finite or zero spread."""
    out = np.ones(len(cols))
    for j, c in enumerate(cols):
        arr = cp.asnumpy(c.dev) if hasattr(c, "dev") else np.asarray(c)
        sd = float(np.nanstd(arr.astype(np.float64)))
        out[j] = sd if np.isfinite(sd) and sd > 0.0 else 1.0
    return out


def test_half_scales_equal_the_per_column_loop_for_device_and_host_columns():
    """Device columns (with NaN, a constant and an all-NaN one) and host columns give the scales of the per-column loop."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=500) * 3.0
    b = rng.normal(size=500)
    b[::7] = np.nan
    cols = [SimpleNamespace(dev=cp.asarray(a)), SimpleNamespace(dev=cp.asarray(b)), SimpleNamespace(dev=cp.asarray(np.full(500, 2.0))), rng.normal(size=500) * 0.1, SimpleNamespace(dev=cp.asarray(np.full(500, np.nan)))]
    np.testing.assert_allclose(_raw_half_scales(cp, cols), _reference(cols), rtol=1e-12)
    assert _raw_half_scales(cp, cols)[2] == 1.0 and _raw_half_scales(cp, cols)[4] == 1.0
