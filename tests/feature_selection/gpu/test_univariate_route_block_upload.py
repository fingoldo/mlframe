"""``_upload_columns`` builds the same device matrix in one transfer as the column-by-column fill."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _univariate_route_device as mod


def _columns() -> list:
    """Five host columns of mixed float width (the generator hands columns in their own dtype)."""
    rng = np.random.default_rng(0)
    cols = [rng.standard_normal(2000).astype(np.float32) for _ in range(4)]
    cols.append(rng.standard_normal(2000) * 1e3)
    return cols


def test_block_upload_equals_the_column_fill(monkeypatch):
    """One transfer for the block (two device launches) and the per-column fill give the same C-contiguous matrix; the block path makes a single host-to-device transfer."""
    cols = _columns()
    transfers = []
    real = cp.asarray

    def spy(a, *args, **kwargs):
        """Count host arrays handed to the device."""
        if isinstance(a, np.ndarray):
            transfers.append(a.shape)
        return real(a, *args, **kwargs)

    monkeypatch.setattr(cp, "asarray", spy)
    block = mod._upload_columns(cp, cols, np.float32)
    assert transfers == [(5, 2000)]
    monkeypatch.setattr(mod, "_BLOCK_UPLOAD_FRACTION", 0.0)
    transfers.clear()
    filled = mod._upload_columns(cp, cols, np.float32)
    assert len(transfers) == 5
    assert block.shape == (2000, 5) and block.flags.c_contiguous and block.dtype == np.float32
    assert bool((block == filled).all())
    np.testing.assert_array_equal(cp.asnumpy(block)[:, 4], cols[4].astype(np.float32))
