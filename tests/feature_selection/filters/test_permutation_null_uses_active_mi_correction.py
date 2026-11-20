"""mi_direct's permutation null must score shuffles with the same MI estimator as the observed value."""
import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory import merge_vars, set_mi_miller_madow
from mlframe.feature_selection.filters.permutation import _perm_pvalue, mi_direct, parallel_mi_prange_with_null
from mlframe.feature_selection.filters.info_theory import compute_relevance_score


@pytest.fixture(autouse=True)
def _mm_off_after():
    """Reset the thread-local Miller-Madow toggle after each test."""
    yield
    set_mi_miller_madow(False)


def _weak_signal(n=3000, k=12, seed=0):
    """Return (factors_data, factors_nbins): x and y share a weak signal below the plug-in bias."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, k, n)
    y = np.where(rng.random(n) < 0.06, x, rng.integers(0, k, n))
    return np.column_stack([x, y]).astype(np.int32), np.array([k, k], dtype=np.int64)


def test_permutation_null_uses_miller_madow_like_observed():
    """mi_direct's exceedance count equals the one from shuffles scored with Miller-Madow, and differs from the plug-in one."""
    data, nb = _weak_signal()
    cx, fx, _ = merge_vars(data, (0,), None, nb, np.int32)
    cy, fy, _ = merge_vars(data, (1,), None, nb, np.int32)
    set_mi_miller_madow(True)
    obs = compute_relevance_score(False, cx, fx, cy, fy, use_mm=True)
    nperm, seed = 64, 3
    nf_mm, _, _ = parallel_mi_prange_with_null(cx, fx, cy, fy, nperm, obs, np.uint64(seed), np.int32, False, True, False)
    nf_plain, _, _ = parallel_mi_prange_with_null(cx, fx, cy, fy, nperm, obs, np.uint64(seed), np.int32, False, False, False)
    assert nf_mm != nf_plain
    _mi_val, conf = mi_direct(data, (0,), (1,), nb, npermutations=nperm, min_nonzero_confidence=0.0, parallelism="outer", base_seed=seed)
    assert conf == pytest.approx(1.0 - _perm_pvalue(int(nf_mm), nperm, full_budget=nperm))


def test_gpu_path_is_skipped_while_a_mi_correction_is_active(monkeypatch, caplog):
    """The device permutation path has no Miller-Madow/Chao-Shen, so ``mi_direct`` must not route there while one is active, and it says so."""
    import logging

    from mlframe.feature_selection.filters import permutation as P

    def _boom(*a, **k):
        """Fail loudly if the device path is taken."""
        raise AssertionError("GPU path taken under an active MI correction")

    monkeypatch.setattr(P, "_mi_direct_try_gpu", _boom)
    data, nb = _weak_signal()
    set_mi_miller_madow(True)
    with caplog.at_level(logging.INFO, logger=P.logger.name):
        mi_direct(data, (0,), (1,), nb, npermutations=64, min_nonzero_confidence=0.0, parallelism="outer", base_seed=3, prefer_gpu=True)
    assert any("GPU permutation path skipped" in r.getMessage() for r in caplog.records)
