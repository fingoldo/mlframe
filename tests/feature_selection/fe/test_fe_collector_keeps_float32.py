"""The FE batch collector and dispatcher keep float32 candidate columns float32 on the way to the device scorer, and promote a mixed set to double precision."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import _fe_batch_dispatch as dispatch
from mlframe.feature_selection.filters._fe_gpu_batch import _collector as collector


def _families(dtypes, n=300):
    """One (n, 3) candidate matrix per dtype."""
    rng = np.random.default_rng(0)
    return [rng.standard_normal((n, 3)).astype(dt) for dt in dtypes]


def _spy(monkeypatch):
    """Replace the scorer with a recorder of the matrix it receives; returns the list of received matrices."""
    seen = []
    monkeypatch.setattr(dispatch, "fe_batch_mi", lambda big, y, nbins=10, **kw: (seen.append(big), np.arange(big.shape[1], dtype=np.float64))[1])
    return seen


def test_all_float32_families_reach_the_scorer_as_float32(monkeypatch):
    """No upcast: the stacked matrix is float32 and the per-family results are reassembled by column offset."""
    seen = _spy(monkeypatch)
    out = collector.collect_and_score(_families([np.float32, np.float32]), np.zeros(300, dtype=np.int64))
    assert seen[0].dtype == np.float32 and seen[0].shape == (300, 6)
    assert [o.tolist() for o in out] == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_a_wide_family_promotes_the_whole_stack_to_double_precision(monkeypatch):
    """One float64 family among float32 ones: everything is stacked as float64."""
    seen = _spy(monkeypatch)
    collector.collect_and_score(_families([np.float32, np.float64]), np.zeros(300, dtype=np.int64))
    assert seen[0].dtype == np.float64


def test_dispatch_scores_a_float32_matrix_in_float32_on_the_device(monkeypatch):
    """``fe_batch_mi`` on the GPU backend hands a float32 matrix to the device scorer as float32 and a float64 one as float64 (the opt-in knob unset)."""
    import mlframe.feature_selection.filters._fe_gpu_batch as gpu_batch

    monkeypatch.delenv("MLFRAME_FE_VRAM_F32", raising=False)
    got = []
    monkeypatch.setattr(gpu_batch, "multi_gpu_fe_batch_mi", lambda X, y, nbins, dtype=None: (got.append(dtype), np.zeros(X.shape[1]))[1])
    y = np.zeros(300, dtype=np.int64)
    dispatch.fe_batch_mi(_families([np.float32])[0], y, backend="gpu")
    dispatch.fe_batch_mi(_families([np.float64])[0], y, backend="gpu")
    assert got == [np.float32, np.float64]
