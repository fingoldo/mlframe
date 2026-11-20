"""Host fallbacks of the device pair kernels are logged, and the device pair search does not invent joint cell counts."""
import logging
from types import SimpleNamespace

import numpy as np

from mlframe.utils.log_throttle import reset_throttle_counts


def test_pair_joint_dispatch_logs_pairs_left_to_the_host(monkeypatch, caplog):
    """Pairs whose table exceeds the shared memory are scored on the host and the dispatcher says so."""
    from mlframe.feature_selection.filters import _cat_pair_joint_gpu as mod
    from mlframe.feature_selection.filters import _gpu_policy

    reset_throttle_counts()
    codes = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 0]], dtype=np.int64)
    radix = np.array([2, 2, 2], dtype=np.int64)
    first = np.array([0, 1], dtype=np.int64)
    second = np.array([1, 2], dtype=np.int64)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    monkeypatch.setattr(mod, "pair_device_chosen", lambda n, p: True)
    monkeypatch.setattr(_gpu_policy, "cuda_available_for_run", lambda: True)
    monkeypatch.setattr(mod, "pair_joint_mi_gpu", lambda *a: np.array([0.1, np.nan]))
    with caplog.at_level(logging.INFO, logger=mod.logger.name):
        out = mod.pair_joint_mi_dispatch(codes, radix, first, second, y, 2)
    assert np.isfinite(out).all()
    assert any("exceed" in r.getMessage() and r.levelno == logging.INFO for r in caplog.records)


def test_weighted_dispatch_logs_pairs_left_to_the_host(monkeypatch, caplog):
    """Weighted pairs the device cannot hold run on the host kernel and the dispatcher says so."""
    from mlframe.feature_selection.filters import _cat_weighted_pair_gpu as mod
    from mlframe.feature_selection.filters import _gpu_policy

    reset_throttle_counts()
    monkeypatch.setattr(mod, "weighted_device_chosen", lambda n, p: True)
    monkeypatch.setattr(_gpu_policy, "cuda_available_for_run", lambda: True)
    fits = np.array([True, False])
    monkeypatch.setattr(mod, "weighted_pair_search_gpu", lambda *a: (np.zeros(2), np.zeros(2), np.zeros(2, dtype=np.int64), fits))

    def host(*a):
        """Host kernel stand-in scoring the one remaining pair."""
        return np.ones(1), np.ones(1), np.ones(1, dtype=np.int64)

    data = np.zeros((4, 3), dtype=np.int32)
    with caplog.at_level(logging.INFO, logger=mod.logger.name):
        joint, _, _ = mod.weighted_pair_search_dispatch(
            host, data, np.array([0, 1]), np.array([1, 2]), np.zeros(3), np.array([2, 2, 2]), np.zeros(4, dtype=np.int64), np.ones(4), np.int32
        )
    assert joint[1] == 1.0
    assert any("exceed" in r.getMessage() for r in caplog.records)


def test_gpu_pair_search_leaves_joint_cell_count_unknown_and_diagnostics_count_it():
    """The device path reports -1 for n_uniq and the diagnostics replace it by the occupied cell count of the selected pair."""
    from mlframe.feature_selection.filters._cat_interactions_step import _gpu_pair_search_or_none, _run_cat_interactio_diagnostics_always_cheap_gated

    a = np.array([0, 1, 2, 0, 1, 2])
    data = np.column_stack([a, a]).astype(np.int32)
    nbins = np.array([3, 3])
    pa, pb = np.array([0]), np.array([1])
    used, ii, joint, n_uniq = _gpu_pair_search_or_none(
        SimpleNamespace(backend="auto"), lambda **kw: np.zeros(1), data, pa, pb, nbins, np.zeros(6, dtype=np.int64), np.ones(1), np.int32, np.zeros(2),
    )
    assert used and int(n_uniq[0]) == -1
    state = SimpleNamespace(diagnostics={})
    cfg = SimpleNamespace(emit_diagnostics=True)
    _run_cat_interactio_diagnostics_always_cheap_gated(cfg, np.array([0]), pa, pb, ii, joint, np.zeros(2), n_uniq, ["a", "b"], 6, {}, {}, state, ["new"], data, nbins)
    assert state.diagnostics["new"]["joint_nclasses"] == 3
