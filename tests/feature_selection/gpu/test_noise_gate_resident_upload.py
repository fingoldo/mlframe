"""RESIDENT UPLOAD (wave 10, 2026-07-13): ``batch_mi_noise_gate_gpu``'s fit-constant operands must be
uploaded ONCE across repeated dispatches on the SAME target, not re-uploaded on every call. Covers three
independent fixes in ``batch_mi_noise_gate_gpu.py``:

  1. The ``use_su=True`` delegate inside ``batch_mi_with_noise_gate_cuda_resident`` now threads the
     resident (P, n) shuffled-y matrix (``_resident_y_all_device``) through to
     ``batch_mi_with_noise_gate_cuda`` instead of rebuilding + re-uploading it every dispatch.
  2. The non-SU resident path's ``offsets``/``nbins``/``freqs_y`` uploads now route through
     ``resident_operand`` (content-keyed) instead of a fresh ``_nb_cuda.to_device`` every call.
  3. The cupy backend's (P, n) shuffle-matrix upload now shares the SAME resident cache the numba.cuda
     path already had (``_resident_y_all_device_for_cupy``), including a genuine CROSS-BACKEND share when
     numba.cuda is also present (zero-copy ``cp.asarray`` view of the numba device array).

Mirrors ``test_batch_pair_mi_resident_upload.py``'s monkeypatch-and-count template.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlframe.feature_selection.filters.batch_mi_noise_gate_gpu as bg
from mlframe.feature_selection.filters.info_theory import batch_mi_with_noise_gate, merge_vars
from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands

_HAS_CUDA = bg._CUDA_AVAIL
_HAS_CUPY = bg._CUPY_AVAIL


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear caches."""
    bg._DY_DEVICE_CACHE.clear()
    bg._DY_DEVICE_CACHE_CUPY.clear()
    clear_fe_resident_operands()
    yield
    bg._DY_DEVICE_CACHE.clear()
    bg._DY_DEVICE_CACHE_CUPY.clear()
    clear_fe_resident_operands()


def _make_frame(n, K, nbins, n_classes_y, seed):
    """(n, K) int frame mixing informative / pure-noise / strongly-informative columns; target built via
    merge_vars exactly like the sibling bit-identity suite (test_batch_mi_noise_gate_gpu.py)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes_y, size=n).astype(np.int32)
    cols = np.empty((n, K), dtype=np.int32)
    for k in range(K):
        kind = k % 3
        if kind == 0:
            c = (y + rng.integers(0, 2, size=n)) % nbins
        elif kind == 1:
            c = rng.integers(0, nbins, size=n)
        else:
            c = y.copy().astype(np.int64)
        cols[:, k] = (c % nbins).astype(np.int32)
    classes_y, freqs_y, _ = merge_vars(
        factors_data=y.reshape(-1, 1),
        vars_indices=np.array([0], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=np.array([int(y.max()) + 1], dtype=np.int64),
        dtype=np.int32,
    )
    return cols, classes_y, freqs_y


def _cpu_ref(disc_2d, factors_nbins, classes_y, freqs_y, npermutations, use_su):
    """Cpu ref."""
    return batch_mi_with_noise_gate(
        disc_2d=disc_2d,
        factors_nbins=factors_nbins,
        classes_y=classes_y,
        classes_y_safe=classes_y.copy(),
        freqs_y=freqs_y,
        npermutations=npermutations,
        base_seed=np.uint64(0),
        min_nonzero_confidence=0.99,
        use_su=use_su,
        dtype=np.int32,
    )


def _assert_close_to_cpu_ref(got, ref):
    """GPU-vs-CPU comparison at the task's sanctioned ~1e-9 tolerance (not exact array_equal): a PRE-EXISTING,
    shape-dependent ULP-level divergence in ``batch_mi_with_noise_gate_cuda_resident``'s non-SU reduction was
    found while writing this suite (e.g. n=2000 K=64 nbins=6 n_classes_y=4 nperm=3 seed=21, use_su=False: 14/64
    columns differ by 1-2 ULP) and CONFIRMED pre-existing (bit-for-bit reproduced on a clean
    ``git worktree add <tmp> HEAD`` checkout, unrelated to the resident-upload caching fixes in this module) --
    out of scope for this residency-upload wave, so this helper isolates the caching proof from that
    unrelated numerical quirk instead of asserting exact equality."""
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=1e-9)


@pytest.mark.skipif(not _HAS_CUDA, reason="numba.cuda not available on this host")
def test_su_branch_reuses_resident_y_matrix(monkeypatch):
    """The use_su=True delegate in batch_mi_with_noise_gate_cuda_resident previously rebuilt (CPU
    Fisher-Yates) AND re-uploaded the shuffled-y codes from scratch on EVERY dispatch, bypassing the
    _resident_y_all_device cache the non-SU path already used. Two SU dispatches on the SAME target must
    now upload the (P, n) shuffle matrix only ONCE."""
    n, K, nbins, n_classes_y, nperm = 1500, 40, 8, 4, 4
    disc_2d, classes_y, freqs_y = _make_frame(n, K, nbins, n_classes_y, seed=11)
    factors_nbins = np.full(K, nbins, dtype=np.int64)
    classes_y_safe = classes_y.copy()
    P = nperm + 1

    orig_to_device = bg._nb_cuda.to_device
    upload_calls = {"n": 0}

    def _counting_to_device(arr, *a, **kw):
        """Counting to device."""
        if getattr(arr, "shape", None) == (P, n):
            upload_calls["n"] += 1
        return orig_to_device(arr, *a, **kw)

    monkeypatch.setattr(bg._nb_cuda, "to_device", _counting_to_device)

    kwargs = dict(
        disc_2d=disc_2d,
        factors_nbins=factors_nbins,
        classes_y=classes_y,
        classes_y_safe=classes_y_safe,
        freqs_y=freqs_y,
        npermutations=nperm,
        base_seed=np.uint64(0),
        min_nonzero_confidence=0.99,
        use_su=True,
        dtype=np.int32,
    )
    out1 = bg.batch_mi_with_noise_gate_cuda_resident(**kwargs)
    out2 = bg.batch_mi_with_noise_gate_cuda_resident(**kwargs)

    assert upload_calls["n"] == 1, f"y_all-shaped to_device called {upload_calls['n']} times across 2 SU dispatches (expected 1)"
    assert np.array_equal(out1, out2)

    ref = _cpu_ref(disc_2d, factors_nbins, classes_y, freqs_y, nperm, use_su=True)
    _assert_close_to_cpu_ref(out1, ref)


@pytest.mark.skipif(not (_HAS_CUDA and _HAS_CUPY), reason="both numba.cuda and cupy required for the resident histgate path")
def test_histgate_off_nb_freq_dedup_via_resident_operand(monkeypatch):
    """batch_mi_with_noise_gate_cuda_resident's non-SU path uploads offsets/nbins/freqs_y via
    resident_operand (content-keyed). Two dispatches sharing the SAME factors_nbins/freqs_y content but a
    DIFFERENT candidate-column chunk (disc_2d) -- the realistic same-K-different-round scenario -- must
    upload each of the 3 fit-constant operands only ONCE."""
    n, K, nbins, n_classes_y, nperm = 2000, 64, 6, 4, 3
    disc_2d_a, classes_y, freqs_y = _make_frame(n, K, nbins, n_classes_y, seed=21)
    disc_2d_b, _classes_y_b, _freqs_y_b = _make_frame(n, K, nbins, n_classes_y, seed=22)
    factors_nbins = np.full(K, nbins, dtype=np.int64)
    classes_y_safe = classes_y.copy()

    import cupy as cp

    orig_asarray = cp.asarray
    upload_calls = {"off": 0, "nb": 0, "freq": 0}

    def _counting_asarray(arr, *a, **kw):
        """Counting asarray."""
        shp = getattr(arr, "shape", None)
        dt = str(getattr(arr, "dtype", ""))
        if shp == (K,) and dt == "int64":
            upload_calls["off"] += 1
        elif shp == (K,) and dt == "int32":
            upload_calls["nb"] += 1
        elif shp == freqs_y.shape and dt == "float64":
            upload_calls["freq"] += 1
        return orig_asarray(arr, *a, **kw)

    monkeypatch.setattr(cp, "asarray", _counting_asarray)

    kwargs_common = dict(
        factors_nbins=factors_nbins,
        classes_y=classes_y,
        classes_y_safe=classes_y_safe,
        freqs_y=freqs_y,
        npermutations=nperm,
        base_seed=np.uint64(0),
        min_nonzero_confidence=0.99,
        use_su=False,
        dtype=np.int32,
    )
    out1 = bg.batch_mi_with_noise_gate_cuda_resident(disc_2d=disc_2d_a, **kwargs_common)
    out2 = bg.batch_mi_with_noise_gate_cuda_resident(disc_2d=disc_2d_b, **kwargs_common)

    assert upload_calls["off"] == 1, f"offsets upload count={upload_calls}"
    assert upload_calls["nb"] == 1, f"nbins upload count={upload_calls}"
    assert upload_calls["freq"] == 1, f"freqs_y upload count={upload_calls}"

    ref1 = _cpu_ref(disc_2d_a, factors_nbins, classes_y, freqs_y, nperm, use_su=False)
    ref2 = _cpu_ref(disc_2d_b, factors_nbins, classes_y, freqs_y, nperm, use_su=False)
    _assert_close_to_cpu_ref(out1, ref1)
    _assert_close_to_cpu_ref(out2, ref2)


@pytest.mark.skipif(not _HAS_CUPY, reason="cupy not available on this host")
def test_cupy_y_all_matrix_resident_reuse(monkeypatch):
    """batch_mi_with_noise_gate_cupy previously rebuilt + re-uploaded the (P, n) shuffled-y matrix on
    EVERY call (no cache analog for this backend). Two dispatches on the SAME target must upload it only
    ONCE."""
    n, K, nbins, n_classes_y, nperm = 1800, 48, 7, 3, 3
    disc_2d, classes_y, freqs_y = _make_frame(n, K, nbins, n_classes_y, seed=31)
    factors_nbins = np.full(K, nbins, dtype=np.int64)
    classes_y_safe = classes_y.copy()
    P = nperm + 1

    if _HAS_CUDA:
        # numba.cuda present -> the cupy path shares _resident_y_all_device's cache via a zero-copy view,
        # so the real upload call is _nb_cuda.to_device, not cp.asarray.
        orig_upload = bg._nb_cuda.to_device
        patch_target = (bg._nb_cuda, "to_device")
    else:
        import cupy as cp

        orig_upload = cp.asarray
        patch_target = (cp, "asarray")

    upload_calls = {"n": 0}

    def _counting(arr, *a, **kw):
        """Helper that counting."""
        if getattr(arr, "shape", None) == (P, n):
            upload_calls["n"] += 1
        return orig_upload(arr, *a, **kw)

    monkeypatch.setattr(*patch_target, _counting)

    kwargs = dict(
        disc_2d=disc_2d,
        factors_nbins=factors_nbins,
        classes_y=classes_y,
        classes_y_safe=classes_y_safe,
        freqs_y=freqs_y,
        npermutations=nperm,
        base_seed=np.uint64(0),
        min_nonzero_confidence=0.99,
        use_su=False,
        dtype=np.int32,
    )
    out1 = bg.batch_mi_with_noise_gate_cupy(**kwargs)
    out2 = bg.batch_mi_with_noise_gate_cupy(**kwargs)

    assert upload_calls["n"] == 1, f"y_all-shaped upload called {upload_calls['n']} times across 2 cupy dispatches (expected 1)"
    assert np.array_equal(out1, out2)

    ref = _cpu_ref(disc_2d, factors_nbins, classes_y, freqs_y, nperm, use_su=False)
    _assert_close_to_cpu_ref(out1, ref)


@pytest.mark.skipif(not (_HAS_CUDA and _HAS_CUPY), reason="both backends required for the cross-backend share proof")
def test_cuda_resident_and_cupy_share_one_resident_y_upload(monkeypatch):
    """resident_operand-style cross-backend dedup: the numba.cuda resident gate and the cupy gate must
    share ONE y_all device upload for the SAME target -- a cuda_resident dispatch followed by a cupy
    dispatch uploads the shuffle matrix only once total, not once per backend (_resident_y_all_device_for_cupy's
    zero-copy cp.asarray view of the numba device array)."""
    n, K, nbins, n_classes_y, nperm = 1600, 32, 5, 3, 3
    disc_2d, classes_y, freqs_y = _make_frame(n, K, nbins, n_classes_y, seed=41)
    factors_nbins = np.full(K, nbins, dtype=np.int64)
    classes_y_safe = classes_y.copy()
    P = nperm + 1

    orig_to_device = bg._nb_cuda.to_device
    upload_calls = {"n": 0}

    def _counting_to_device(arr, *a, **kw):
        """Counting to device."""
        if getattr(arr, "shape", None) == (P, n):
            upload_calls["n"] += 1
        return orig_to_device(arr, *a, **kw)

    monkeypatch.setattr(bg._nb_cuda, "to_device", _counting_to_device)

    kwargs = dict(
        disc_2d=disc_2d,
        factors_nbins=factors_nbins,
        classes_y=classes_y,
        classes_y_safe=classes_y_safe,
        freqs_y=freqs_y,
        npermutations=nperm,
        base_seed=np.uint64(0),
        min_nonzero_confidence=0.99,
        use_su=False,
        dtype=np.int32,
    )
    out_cuda = bg.batch_mi_with_noise_gate_cuda_resident(**kwargs)
    out_cupy = bg.batch_mi_with_noise_gate_cupy(**kwargs)

    assert upload_calls["n"] == 1, (
        f"y_all-shaped to_device called {upload_calls['n']} times across a cuda_resident+cupy dispatch pair "
        "on the same target (expected 1 -- cross-backend share)"
    )

    ref = _cpu_ref(disc_2d, factors_nbins, classes_y, freqs_y, nperm, use_su=False)
    _assert_close_to_cpu_ref(out_cuda, ref)
    _assert_close_to_cpu_ref(out_cupy, ref)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
