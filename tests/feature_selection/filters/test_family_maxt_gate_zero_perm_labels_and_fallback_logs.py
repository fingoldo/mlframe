"""Family maxT gate: empty permutation draws on the device path, label/code row mismatches and the logged host fallbacks."""

import logging

import numpy as np
import pytest

from mlframe.feature_selection.filters import _family_maxt_gate as G


def test_device_bits_null_with_zero_permutations_returns_empty_instead_of_indexerror():
    """``n_perm == 0`` (the second draw of a gate run with at most 32 permutations) must give an empty null, not ``starts[0]`` IndexError."""
    out = np.empty(0, dtype=np.float64)
    rng = np.random.default_rng(0)
    y = np.zeros(8, dtype=np.int64)
    got = G._family_null_max_device_bits(None, rng, y, 2, 0, 256, 1, 0, 0.0, out, None)
    assert got is out and got.shape == (0,)
    assert G._family_null_max_device_bits(None, rng, y, 2, 0, 256, 1, 0, 0.0, out, object()) is None


def test_family_maxt_pass_codes_rejects_labels_shorter_than_the_code_rows():
    """A label vector with fewer rows than the code matrix would be read out of bounds by the compiled kernels: it must raise."""
    cols = np.zeros((2, 100), dtype=np.int16)
    nx = np.array([1, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="rows"):
        G.family_maxt_pass_codes(cols, nx, np.arange(50) % 2)


def test_family_pass_batches_rejects_labels_longer_than_the_code_rows():
    """The host batch gate checks the label length against the stacked code rows before any kernel runs."""
    batch = np.zeros((40, 3), dtype=np.int64)
    with pytest.raises(ValueError, match="rows"):
        G.family_pass_batches([batch], np.arange(60) % 2)


def test_device_oom_block_falls_back_to_the_host_kernel_and_logs_a_warning(monkeypatch, caplog):
    """A device out-of-memory on a null block scores it on the host and says so at WARNING."""
    from mlframe.feature_selection.filters import _family_maxt_cuda as C
    from mlframe.utils.log_throttle import reset_throttle_counts

    class _Oom(RuntimeError):
        """Stand-in for the device pool's out-of-memory error."""

    def _boom(*_a, **_k):
        """Device scorer that always refuses the block."""
        raise _Oom("pool limit")

    reset_throttle_counts()
    monkeypatch.setattr(G, "_null_device_wanted", lambda *a, **k: True)
    monkeypatch.setattr(G, "_family_bit_prep", lambda *a, **k: (None, None))
    monkeypatch.setattr(G, "_device_null_oom_errors", lambda: (_Oom,))
    monkeypatch.setattr(C, "device_supported", lambda *a, **k: True)
    monkeypatch.setattr(C, "family_null_max_device", _boom)
    rng = np.random.default_rng(1)
    cols = rng.integers(0, 3, (4, 200)).astype(np.int16)
    nx = np.full(4, 3, dtype=np.int64)
    y = rng.integers(0, 2, 200).astype(np.int64)
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters.mrmr"):
        out = G.family_null_max(cols, nx, y, 2, 6, np.random.default_rng(2))
    assert out is not None and out.shape == (6,) and np.all(np.isfinite(out))
    assert any("host kernel scores it" in r.getMessage() for r in caplog.records)
