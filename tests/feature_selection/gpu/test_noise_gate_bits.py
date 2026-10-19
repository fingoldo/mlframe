"""The bit-set noise gate returns the plug-in MI of the count-table reduction, takes the same gate decisions as the histogram pair, declines what it cannot hold, and follows the backend choice."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_DEVICE = False
pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")


def _frame(n, k, bins, classes, seed=0, signal_columns=6):
    """Integer codes with a few columns that carry signal about the target, the target, its class frequencies, and a ``(P, n)`` matrix of the original plus shuffled targets."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, n).astype(np.int64)
    codes = rng.integers(0, bins, (n, k)).astype(np.int8)
    for j in range(min(signal_columns, k)):
        keep = rng.random(n) < 0.15 * (j + 1)
        codes[:, j] = np.where(keep, codes[:, j], (y * bins // classes)).astype(np.int8)
    freqs = np.bincount(y, minlength=classes).astype(np.float64) / n
    y_all = np.stack([y] + [rng.permutation(y) for _ in range(19)]).astype(np.int32)
    return codes, y, freqs, y_all


@pytest.mark.parametrize("n,bins,classes", [(999, 10, 10), (4096, 2, 2), (3001, 32, 5), (1500, 7, 25)])
def test_matrix_matches_the_count_table_reduction(n, bins, classes):
    """Every sampled (permutation, column) MI equals the count-table MI of the same joint histogram to a few ulps, including n that is not a multiple of 64."""
    from mlframe.feature_selection.filters._batch_mi_noise_gate_bits import bits_gate_mi
    from mlframe.feature_selection.filters._batch_mi_noise_gate_kernels import _mi_from_counts_cpu

    codes, _y, freqs, y_all = _frame(n, 40, bins, classes)
    nbins = np.full(codes.shape[1], bins, dtype=np.int64)
    got = bits_gate_mi(cp.asarray(codes), nbins, cp.asarray(y_all), classes)
    assert got is not None and got.shape == (y_all.shape[0], codes.shape[1])
    for p in (0, 1, 7, 19):
        for k in (0, 3, 17, 39):
            joint = np.zeros((bins, classes), dtype=np.int64)
            np.add.at(joint, (codes[:, k].astype(np.int64), y_all[p].astype(np.int64)), 1)
            assert got[p, k] == pytest.approx(_mi_from_counts_cpu(joint, bins, freqs, n, False), abs=1e-12)


def test_gate_decision_equals_the_histogram_pair():
    """With the backend forced each way, the resident gate keeps the same columns with the same MI values (to float round-off)."""
    from mlframe.feature_selection.filters import batch_mi_noise_gate_gpu as gate

    codes, y, freqs, _ = _frame(6000, 300, 10, 10, seed=2)
    nbins = np.full(codes.shape[1], 10, dtype=np.int64)
    args = (codes, nbins, y, y.astype(np.int32), freqs, 49, np.uint64(7), 0.99, False, np.int32, None, cp.asarray(codes))
    previous = gate.force_noise_gate_backend("hist")
    try:
        ref = np.asarray(gate.batch_mi_with_noise_gate_cuda_resident(*args))
        gate.force_noise_gate_backend("bits")
        got = np.asarray(gate.batch_mi_with_noise_gate_cuda_resident(*args))
    finally:
        gate.force_noise_gate_backend(previous)
    assert np.array_equal(ref > 0, got > 0) and int((ref > 0).sum()) > 0
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


def test_kernel_declines_what_it_cannot_hold():
    """More bins than the register accumulators, too many classes for a block's shared memory, and a float code type are declined with ``None``."""
    from mlframe.feature_selection.filters import _batch_mi_noise_gate_bits as bits

    codes, _y, _f, y_all = _frame(1000, 8, 10, 10)
    y_d = cp.asarray(y_all)
    nbins = np.full(8, 10, dtype=np.int64)
    assert bits.bits_gate_mi(cp.asarray(codes), np.full(8, bits.MAX_BINS + 1, dtype=np.int64), y_d, 10) is None
    assert bits.bits_gate_mi(cp.asarray(codes), nbins, y_d, 40) is None
    assert bits.bits_gate_mi(cp.asarray(codes.astype(np.float32)), nbins, y_d, 10) is None


def test_column_groups_do_not_change_the_result(monkeypatch):
    """A memory budget that forces one column per group gives the same matrix as a single group."""
    from mlframe.feature_selection.filters import _batch_mi_noise_gate_bits as bits

    codes, _y, _f, y_all = _frame(2000, 37, 10, 10)
    nbins = np.full(37, 10, dtype=np.int64)
    whole = bits.bits_gate_mi(cp.asarray(codes), nbins, cp.asarray(y_all), 10)
    words = (2000 + 63) // 64
    monkeypatch.setattr(bits, "_BUDGET_CAP_BYTES", 10 * words * 8 * 4)  # four columns per group
    grouped = bits.bits_gate_mi(cp.asarray(codes), nbins, cp.asarray(y_all), 10)
    assert whole is not None and grouped is not None
    assert whole.view(np.uint64).tolist() == grouped.view(np.uint64).tolist()


def test_backend_choice_decides_the_path_and_a_forced_decline_raises(monkeypatch):
    """The tuning-cache choice routes to the bit-set gate or the histogram pair; forcing the bit-set gate on a shape it declines raises so a sweep leaves that cell out."""
    from mlframe.feature_selection.filters import _batch_mi_noise_gate_bits as bits
    from mlframe.feature_selection.filters import batch_mi_noise_gate_gpu as gate

    codes, y, freqs, _ = _frame(3000, 60, 10, 10, seed=4)
    nbins = np.full(60, 10, dtype=np.int64)
    args = (codes, nbins, y, y.astype(np.int32), freqs, 19, np.uint64(3), 0.99, False, np.int32, None, cp.asarray(codes))
    calls = []
    real = bits.bits_gate_mi
    monkeypatch.setattr(bits, "bits_gate_mi", lambda *a: (calls.append(1), real(*a))[1])
    monkeypatch.setattr(bits, "noise_gate_backend_choice", lambda n, k, ny, nb: "hist")
    gate.batch_mi_with_noise_gate_cuda_resident(*args)
    assert calls == []
    monkeypatch.setattr(bits, "noise_gate_backend_choice", lambda n, k, ny, nb: "bits")
    gate.batch_mi_with_noise_gate_cuda_resident(*args)
    assert calls == [1]
    monkeypatch.setattr(bits, "bits_gate_mi", lambda *a: None)
    previous = gate.force_noise_gate_backend("bits")
    try:
        with pytest.raises(RuntimeError):
            gate.batch_mi_with_noise_gate_cuda_resident(*args)
    finally:
        gate.force_noise_gate_backend(previous)


def test_fallback_choice_follows_the_measured_crossover():
    """Without a cache entry the bit-set gate is chosen wherever its kernel applies."""
    from mlframe.feature_selection.filters._batch_mi_noise_gate_bits import _fallback_choice

    assert _fallback_choice(10000, 9720, 10, 10) == "bits"
    assert _fallback_choice(10000, 9720, 25, 32) == "bits"
    assert _fallback_choice(10000, 9720, 26, 10) == "hist"
    assert _fallback_choice(10000, 9720, 10, 33) == "hist"
