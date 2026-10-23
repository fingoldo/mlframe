"""MI path chooser: resident / streamed / cpu_blocks decided from free memory; routing and the reactive OOM net."""

import logging

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_mi_backends as bk
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _streamed_gpu_mi as sg

GiB = 1 << 30
DTYPES = [np.float32, np.float64]


# --- pure chooser, synthetic device / host numbers (no GPU) ----------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_no_device_small_matrix_is_resident(dtype):
    """No GPU and a whole-matrix CPU working set well inside free RAM: one resident batch."""
    assert sg.choose_mi_path(1_000, 50, dtype, [], 16 * GiB) == "resident"


@pytest.mark.parametrize("dtype", DTYPES)
def test_no_device_large_matrix_is_cpu_blocks(dtype):
    """No GPU and the whole-matrix float64 working set exceeds the host budget: column blocks on the CPU."""
    assert sg.choose_mi_path(10_000_000, 1_000, dtype, [], 2 * GiB) == "cpu_blocks"


@pytest.mark.parametrize("dtype", DTYPES)
def test_device_holding_all_columns_is_resident(dtype):
    """The planner's block width covers every column: the whole matrix goes to the GPU in one batch."""
    n, p = 100_000, 8
    assert sg.choose_mi_path(n, p, dtype, [(0, 8 * GiB)], 16 * GiB) == "resident"


@pytest.mark.parametrize("dtype", DTYPES)
def test_device_too_small_for_matrix_is_streamed(dtype):
    """The block width is positive but smaller than the matrix: streamed through VRAM blocks."""
    n = 1_000_000
    plan = sg.plan_streamed_mi(n, [(0, 4 * GiB)], 16 * GiB, dtype)
    p = plan["width"] + 1
    assert plan["width"] > 0
    assert sg.choose_mi_path(n, p, dtype, [(0, 4 * GiB)], 16 * GiB) == "streamed"


@pytest.mark.parametrize("dtype", DTYPES)
def test_device_that_cannot_hold_one_column_falls_back_to_cpu(dtype):
    """A device whose planned width is 0 is no device: the CPU rule decides (here a small matrix stays resident on the CPU)."""
    assert sg.plan_streamed_mi(1_000_000, [(0, 64 << 20)], 16 * GiB, dtype)["width"] == 0
    assert sg.choose_mi_path(100, 4, dtype, [(0, 64 << 20)], 16 * GiB) == "resident"


def test_multi_device_width_is_set_by_the_smallest_device():
    """Fake two-device list: the smallest device's VRAM decides the block width, so the same matrix flips from resident to streamed."""
    n = 100_000
    p = sg.plan_streamed_mi(n, [(0, 8 * GiB)], 16 * GiB, np.float32)["width"]  # widest block one big device holds
    small = sg.plan_streamed_mi(n, [(1, 1 * GiB)], 16 * GiB, np.float32)["width"]
    assert 0 < small < p
    assert sg.choose_mi_path(n, p, np.float32, [(0, 8 * GiB)], 16 * GiB) == "resident"
    assert sg.choose_mi_path(n, p, np.float32, [(0, 8 * GiB), (1, 1 * GiB)], 16 * GiB) == "streamed"


def test_decision_follows_free_vram_not_absolute_sizes():
    """Growing the free VRAM flips a streamed decision to resident for the same n and p: no row or column constant involved."""
    n, p = 100_000, 1_000
    assert sg.choose_mi_path(n, p, np.float32, [(0, 2 * GiB)], 16 * GiB) == "streamed"
    assert sg.choose_mi_path(n, p, np.float32, [(0, 8 * GiB)], 16 * GiB) == "resident"


def test_float64_needs_more_vram_than_float32():
    """The same device and shape: float32 can hold the matrix where float64 must stream."""
    n, p = 100_000, 800
    dev = [(0, 3 * GiB)]
    assert sg.choose_mi_path(n, p, np.float32, dev, 16 * GiB) == "resident"
    assert sg.choose_mi_path(n, p, np.float64, dev, 16 * GiB) == "streamed"


def test_small_host_ram_pushes_cpu_to_blocks_when_no_gpu():
    """Fail loudly if the streamed GPU path is entered."""
    """Free host RAM alone moves the CPU decision: the same matrix is resident with plenty of RAM and column-blocked without."""
    n, p = 1_000_000, 100
    assert sg.choose_mi_path(n, p, np.float32, [], 64 * GiB) == "resident"
    assert sg.choose_mi_path(n, p, np.float32, [], 1 * GiB) == "cpu_blocks"


def test_empty_matrix_is_resident():
    """Zero columns has nothing to stream."""
    assert sg.choose_mi_path(1000, 0, np.float32, [(0, 4 * GiB)], 16 * GiB) == "resident"


# --- routing inside _mi_classif_batch_numba (monkeypatched chooser, devices and kernels) -------------------------------


def _data(n=2_000, p=12, seed=3):
    """Small float32 matrix with a binary label that depends on columns 0-2."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.standard_normal(n) > 0).astype(np.int64)
    return X, y


def _cpu_reference(X, y):
    """CPU batch MI of every column, the reference the GPU and chunked paths must match."""
    from mlframe.feature_selection.filters.hermite_fe import plugin_mi_classif_batch_dispatch

    return np.asarray(plugin_mi_classif_batch_dispatch(np.ascontiguousarray(X), y, 10), dtype=np.float64)


def test_chooser_streamed_routes_through_mi_classif_streamed(monkeypatch):
    """A 'streamed' decision reaches mi_classif_streamed (the production caller) and its output is what the batch returns."""
    X, y = _data()
    calls = []

    def fake_streamed(blocks, y_arg, *, nbins, n_rows, dtype, max_devices=None, n_cols=None):
        """Stand-in for mi_classif_streamed that records the call and returns the CPU reference."""
        calls.append((len(list(blocks)), n_rows))
        return _cpu_reference(X, y)

    monkeypatch.setattr(bk, "_mi_path_for", lambda *a, **k: "streamed")
    monkeypatch.setattr(sg, "_devices", lambda max_devices=None: [(0, 4 * GiB)])
    monkeypatch.setattr(sg, "mi_classif_streamed", fake_streamed)
    out = bk._mi_classif_batch_numba(X, y, nbins=10)
    assert calls and calls[0][1] == X.shape[0]
    np.testing.assert_allclose(out, _cpu_reference(X, y), atol=1e-12)


def test_chooser_cpu_blocks_keeps_column_block_numba_path(monkeypatch):
    """A 'cpu_blocks' decision never touches the GPU and returns the CPU batch values."""
    X, y = _data()
    monkeypatch.setattr(bk, "_mi_path_for", lambda *a, **k: "cpu_blocks")
    monkeypatch.setattr(bk, "_mi_chunk_cols_for", lambda n: 5)

    def no_gpu(*a, **k):
        """Fail loudly if the streamed GPU path is entered."""
        raise AssertionError("cpu_blocks must not call the streamed GPU path")

    monkeypatch.setattr(bk, "_streamed_block_mi", no_gpu)
    out = bk._mi_classif_batch_numba(X, y, nbins=10)
    np.testing.assert_allclose(out, _cpu_reference(X, y), atol=1e-12)


def test_rank_binning_never_takes_the_streamed_path(monkeypatch):
    """rank_binning=True always resolves to the resident path, since the streamed GPU twin bins by percentile edge."""
    monkeypatch.setattr(sg, "_devices", lambda max_devices=None: [(0, 4 * GiB)])
    assert bk._mi_path_for(1_000_000, 400, np.float32, True) == "resident"


def test_chooser_is_called_at_the_top_of_the_batch(monkeypatch):
    """The decision is taken once per batch with the caller's dtype (float32 by default)."""
    X, y = _data(n=500, p=4)
    seen = {}

    def spy(n, p, dtype, rank_binning):
        """Record the chooser's arguments and return the resident path."""
        seen.update(n=n, p=p, dtype=np.dtype(dtype), rank=rank_binning)
        return "resident"

    monkeypatch.setattr(bk, "_mi_path_for", spy)
    bk._mi_classif_batch_numba(X, y, nbins=10)
    assert seen == {"n": 500, "p": 4, "dtype": np.dtype(np.float32), "rank": False}


# --- reactive OOM safety net ------------------------------------------------------------------------------------------


def test_reactive_oom_fallback_still_returns_cpu_values(monkeypatch, caplog):
    """Resident batch raises MemoryError: the warning fires and the column-block retry returns the same MI as the clean batch."""
    from mlframe.feature_selection.filters import hermite_fe

    X, y = _data()
    expected = _cpu_reference(X, y)
    real = hermite_fe.plugin_mi_classif_batch_dispatch
    state = {"calls": 0}

    def flaky(Xa, ya, nb, *a, **k):
        """Raise MemoryError on the first full-width call, then delegate to the real dispatcher."""
        state["calls"] += 1
        if state["calls"] == 1 and Xa.shape[1] == X.shape[1]:
            raise MemoryError("simulated OOM on the full slice")
        return real(Xa, ya, nb, *a, **k)

    monkeypatch.setattr(bk, "_mi_path_for", lambda *a, **k: "resident")
    monkeypatch.setattr(sg, "_devices", lambda max_devices=None: [])
    monkeypatch.setattr(bk, "_block_mi_gpu_fn", lambda: None)
    monkeypatch.setattr(bk, "_mi_chunk_cols_for", lambda n: 4)
    monkeypatch.setattr(hermite_fe, "plugin_mi_classif_batch_dispatch", flaky)
    with caplog.at_level(logging.WARNING, logger=bk.logger.name):
        out = bk._mi_classif_batch_numba(X, y, nbins=10)
    assert any("retrying in column blocks" in r.getMessage() for r in caplog.records)
    assert state["calls"] > 1
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_failed_streamed_route_falls_back_to_resident(monkeypatch):
    """A 'streamed' decision whose GPU path fails returns None internally and the resident batch still produces the values."""
    X, y = _data()
    monkeypatch.setattr(bk, "_mi_path_for", lambda *a, **k: "streamed")
    monkeypatch.setattr(sg, "_devices", lambda max_devices=None: [(0, 4 * GiB)])

    def broken(*a, **k):
        """Stand-in for a streamed path whose GPU kernel fails."""
        raise RuntimeError("no kernel")

    monkeypatch.setattr(sg, "mi_classif_streamed", broken)
    out = bk._mi_classif_batch_numba(X, y, nbins=10)
    np.testing.assert_allclose(out, _cpu_reference(X, y), atol=1e-12)


# --- GPU parity (skipped without CUDA) --------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_streamed_matches_resident_on_small_data(monkeypatch, dtype):
    """Small float32 matrix with a binary label that depends on columns 0-2."""
    """Forced streamed vs forced resident through the batch: same MI up to the existing 1e-6 tolerance on tie-free data."""
    pytest.importorskip("cupy")
    if sg._devices() == []:
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(5)
    n, p = 200_000, 24
    X = rng.standard_normal((n, p)).astype(dtype)
    y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.standard_normal(n) > 0).astype(np.int64)
    monkeypatch.setattr(sg, "plan_streamed_mi", _narrow_plan)
    monkeypatch.setattr(bk, "_mi_path_for", lambda *a, **k: "streamed")
    streamed = bk._mi_classif_batch_numba(X, y, nbins=10)
    monkeypatch.setattr(bk, "_mi_path_for", lambda *a, **k: "resident")
    resident = bk._mi_classif_batch_numba(X, y, nbins=10)
    assert streamed.shape == resident.shape == (p,)
    np.testing.assert_allclose(streamed, resident, atol=1e-6)


_REAL_PLAN = sg.plan_streamed_mi


def _narrow_plan(n, devs, host, dt):
    """The real planner with the block width capped at 8 columns, so the small test matrix really spans several blocks."""
    plan = _REAL_PLAN(n, devs, host, dt)
    plan["width"] = min(plan["width"], 8)
    plan["block_bytes"] = plan["width"] * n * np.dtype(dt).itemsize
    return plan
