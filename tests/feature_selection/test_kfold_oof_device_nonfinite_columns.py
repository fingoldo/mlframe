"""A NaN / inf column no longer sends the whole k-fold OOF block to the host: finite columns run on the device, the odd ones on the host binner; values match the all-host fold."""
import numpy as np
import pytest


def _cuda_device_count() -> int:
    """Number of CUDA devices visible to cupy, 0 when the runtime cannot be queried."""
    import cupy as cp

    try:
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def test_split_columns_match_host_fold(monkeypatch):
    """_folds_on_device with NaN and inf columns equals _kfold_one_fold column by column (device round-off only)."""
    pytest.importorskip("cupy")
    if _cuda_device_count() == 0:
        pytest.skip("no CUDA device")
    from mlframe.feature_selection.filters import _kfold_oof_mi_gpu as K
    from mlframe.feature_selection.filters import _orthogonal_three_gate_mi_fe as M

    monkeypatch.setattr(K, "kfold_oof_mi_device_enabled", lambda n, c: True)
    rng = np.random.default_rng(5)
    n = 6000
    raw = rng.standard_normal((n, 6)).astype(np.float32)
    eng = rng.standard_normal((n, 5)).astype(np.float32)
    y = (raw[:, 0] + 0.5 * rng.standard_normal(n) > 0).astype(np.int64)
    raw[rng.choice(n, 40, replace=False), 2] = np.nan
    eng[rng.choice(n, 25, replace=False), 3] = np.inf
    folds = M._stratified_fold_indices(y, 3, 0)
    dev = M._folds_on_device(raw, eng, y, folds, n, 10)
    assert dev is not None
    assert len(dev) == len(folds)
    for test_idx, got in zip(folds, dev):
        want = M._kfold_one_fold(test_idx, n, y, raw, eng, raw.shape[1], eng.shape[1], 10)
        np.testing.assert_allclose(got[0], want[0], atol=1e-6)
        np.testing.assert_allclose(got[1], want[1], atol=1e-6)
