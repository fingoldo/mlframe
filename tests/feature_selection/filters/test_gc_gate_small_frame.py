"""The pre-discretisation garbage collection runs only for a frame that is a visible share of the free RAM."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._helpers import collect_garbage_if_frame_is_large


def test_small_frame_skips_collection():
    """A 10k x 20 frame is far below 1/512 of the free RAM."""
    assert collect_garbage_if_frame_is_large(pd.DataFrame(np.zeros((10_000, 20)))) is False


def test_frame_large_against_free_ram_collects(monkeypatch):
    """With 1 MB of free RAM reported, the same frame is a large share."""
    import psutil

    class _Mem:
        """Fake psutil memory record."""

        available = 2**20

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _Mem())
    assert collect_garbage_if_frame_is_large(pd.DataFrame(np.zeros((10_000, 20)))) is True
