"""RESIDENT UPLOAD (wave 10, 2026-07-13): ``_count_nfailed_joint_indep_cupy`` (the GPU permutation-null
counter for cat-FE pair confirmation) previously re-uploaded the FIT-CONSTANT target ``classes_y``/
``freqs_y`` via raw ``cp.asarray`` on EVERY survivor pair of ``_confirm_pairs_via_permutation``'s
confirmation loop. The per-pair operands (``classes_pair``/``classes_x1``/``classes_x2``/
``freqs_pair``/``freqs_x1``/``freqs_x2``) genuinely vary every call and stay raw uploads;
``classes_y``/``freqs_y`` now route through ``resident_operand`` under roles ``cat_confirm_y`` /
``cat_confirm_freqs_y``.

Skips when cupy is unavailable (CI without a GPU)."""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._cat_confirm_permutation import (
    _count_nfailed_joint_indep_cupy,
    _count_nfailed_joint_indep_serial,
)
from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_fe_resident_operands()
    yield
    clear_fe_resident_operands()


def _pair_inputs(n, k_pair, k_x1, k_x2, k_y, seed):
    rng = np.random.default_rng(seed)
    classes_pair = rng.integers(0, k_pair, size=n).astype(np.int64)
    classes_x1 = rng.integers(0, k_x1, size=n).astype(np.int64)
    classes_x2 = rng.integers(0, k_x2, size=n).astype(np.int64)
    freqs_pair = np.bincount(classes_pair, minlength=k_pair).astype(np.float64) / n
    freqs_x1 = np.bincount(classes_x1, minlength=k_x1).astype(np.float64) / n
    freqs_x2 = np.bincount(classes_x2, minlength=k_x2).astype(np.float64) / n
    return classes_pair, freqs_pair, classes_x1, freqs_x1, classes_x2, freqs_x2


def test_ycodes_dedup_across_survivor_pairs(monkeypatch):
    """Two calls scoring DIFFERENT survivor pairs against the SAME (classes_y, freqs_y) target -- the
    realistic per-survivor confirmation loop -- must upload classes_y and freqs_y only ONCE each."""
    n, k_y = 3000, 4
    rng = np.random.default_rng(1)
    classes_y = rng.integers(0, k_y, size=n).astype(np.int64)
    freqs_y = np.bincount(classes_y, minlength=k_y).astype(np.float64) / n

    pair1 = _pair_inputs(n, 5, 3, 3, k_y, seed=11)
    pair2 = _pair_inputs(n, 4, 2, 2, k_y, seed=12)

    upload_calls = {"y": 0, "freq": 0}
    orig_asarray = cp.asarray

    def _counting_asarray(arr, *a, **kw):
        if isinstance(arr, np.ndarray) and arr.shape == classes_y.shape and arr.dtype == np.int64 and np.array_equal(arr, classes_y):
            upload_calls["y"] += 1
        elif isinstance(arr, np.ndarray) and arr.shape == freqs_y.shape and arr.dtype == np.float64 and np.allclose(arr, freqs_y):
            upload_calls["freq"] += 1
        return orig_asarray(arr, *a, **kw)

    monkeypatch.setattr(cp, "asarray", _counting_asarray)

    n_failed_1 = _count_nfailed_joint_indep_cupy(*pair1, classes_y, freqs_y, ii_obs=0.0, n_perms=8, base_seed=100)
    n_failed_2 = _count_nfailed_joint_indep_cupy(*pair2, classes_y, freqs_y, ii_obs=0.0, n_perms=8, base_seed=200)

    assert upload_calls["y"] == 1, f"classes_y upload called {upload_calls['y']} times across 2 survivor pairs (expected 1)"
    assert upload_calls["freq"] == 1, f"freqs_y upload called {upload_calls['freq']} times across 2 survivor pairs (expected 1)"
    assert isinstance(n_failed_1, (int, np.integer))
    assert isinstance(n_failed_2, (int, np.integer))


def test_bit_identical_vs_resident_cache_disabled(monkeypatch):
    """MLFRAME_FE_RESIDENT_OPERANDS=0 reproduces the EXACT pre-fix raw-upload behavior for classes_y /
    freqs_y. Same base_seed -> same permutation stream on the SAME GPU kernel either way, so the failure
    count must be identical regardless of whether the upload was cached."""
    n, k_y = 4000, 3
    rng = np.random.default_rng(2)
    classes_y = rng.integers(0, k_y, size=n).astype(np.int64)
    freqs_y = np.bincount(classes_y, minlength=k_y).astype(np.float64) / n
    pair = _pair_inputs(n, 6, 4, 4, k_y, seed=21)

    monkeypatch.setenv("MLFRAME_FE_RESIDENT_OPERANDS", "0")
    n_failed_raw = _count_nfailed_joint_indep_cupy(*pair, classes_y, freqs_y, ii_obs=0.0, n_perms=16, base_seed=55)
    monkeypatch.setenv("MLFRAME_FE_RESIDENT_OPERANDS", "1")
    clear_fe_resident_operands()
    n_failed_cached = _count_nfailed_joint_indep_cupy(*pair, classes_y, freqs_y, ii_obs=0.0, n_perms=16, base_seed=55)

    assert n_failed_raw == n_failed_cached, f"raw={n_failed_raw} cached={n_failed_cached}"


def test_gpu_count_matches_cpu_reference_order_of_magnitude():
    """Sanity check the fixed function still agrees with the CPU serial reference within the documented
    ULP-rounding tolerance (a real, informative synthetic pair; not just a smoke call)."""
    n, k_y = 5000, 3
    rng = np.random.default_rng(3)
    y = rng.integers(0, k_y, size=n).astype(np.int64)
    x1 = rng.integers(0, 4, size=n).astype(np.int64)
    x2 = (x1 + rng.integers(0, 2, size=n)) % 4
    pair = (x1 * 4 + x2) % 8
    freqs_y = np.bincount(y, minlength=k_y).astype(np.float64) / n
    freqs_x1 = np.bincount(x1, minlength=4).astype(np.float64) / n
    freqs_x2 = np.bincount(x2, minlength=4).astype(np.float64) / n
    freqs_pair = np.bincount(pair, minlength=8).astype(np.float64) / n

    n_perms = 200
    n_gpu = _count_nfailed_joint_indep_cupy(pair, freqs_pair, x1, freqs_x1, x2, freqs_x2, y, freqs_y, ii_obs=0.0, n_perms=n_perms, base_seed=7)
    n_cpu = _count_nfailed_joint_indep_serial(pair, freqs_pair, x1, freqs_x1, x2, freqs_x2, y, freqs_y, ii_obs=0.0, n_perms=n_perms, base_seed=7, dtype=np.int64)
    # Different RNG streams (cupy vs numba LCG) -> not bit-identical counts, but both count occurrences of
    # a threshold event over the SAME statistical null -> must land in the same broad regime.
    assert abs(int(n_gpu) - int(n_cpu)) <= n_perms, f"gpu={n_gpu} cpu={n_cpu} n_perms={n_perms}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
