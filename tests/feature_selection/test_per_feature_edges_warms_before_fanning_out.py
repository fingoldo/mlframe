"""The first column is binned on the calling thread, so numba compiles before the pool exists.

``per_feature_edges`` fans out over columns with a ThreadPoolExecutor, and the njit kernels it reaches
(``_mdlp_recurse_validated_bfs`` and friends, all ``cache=True``) compile on first call. Several threads
entering numba's compiler at once aborts the process on macOS: mlframe's first macOS CI run crashed 150
xdist workers across ten shards with ``Fatal Python error: Aborted``, and every faulthandler dump showed
two or more threads inside ``_mdlp_recurse_validated_bfs`` -> ``dispatcher.compile``.

Linux tolerates it, which is why the threaded path has run there since its threshold was lowered from 128
columns to 2 -- an optimisation that was right on its own terms and made concurrent compilation the normal
case rather than an exotic one.

Asserted on WHERE the first call runs rather than on wall time: the defect is that compilation happens
concurrently, and a timing test for it would only ever reproduce on the platform that aborts.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from mlframe.feature_selection.filters import _adaptive_nbins


def _numeric_columns(n_cols: int, n_rows: int = 400):
    """An (n_rows, n_cols) frame with enough structure that MDLP recurses instead of bailing at one split."""
    rng = np.random.default_rng(0)
    y = (rng.random(n_rows) < 0.5).astype(np.int64)
    return np.column_stack([rng.normal(size=n_rows) + y * 1.5 for _ in range(n_cols)]), y


def test_the_first_column_is_computed_before_any_pool_thread_starts(monkeypatch):
    """Warming on the calling thread is what keeps numba's compiler single-threaded here."""
    numba = pytest.importorskip("numba")
    if numba.config.DISABLE_JIT:
        pytest.skip("the threaded path is gated off when JIT is disabled, so there is nothing to warm")

    cols, y = _numeric_columns(6)
    threads: list[str] = []
    # ``edges_fayyad_irani`` is what the per-column closure calls, and unlike that closure it is a module
    # attribute, so it is the observable seam. It is also the frame the macOS dumps died in.
    real = _adaptive_nbins.edges_fayyad_irani

    def _record(*args, **kwargs):
        """Note which thread each per-column computation runs on, in call order."""
        threads.append(threading.current_thread().name)
        return real(*args, **kwargs)

    monkeypatch.setattr(_adaptive_nbins, "edges_fayyad_irani", _record)
    monkeypatch.setattr(_adaptive_nbins, "_PARALLEL_EDGES_MIN_COLS", 2, raising=False)
    _adaptive_nbins.per_feature_edges(cols, y, method="fayyad_irani", n_jobs=4)

    assert threads, "no column was computed; the fixture no longer reaches the per-column path"
    assert threads[0] == threading.main_thread().name, (
        f"the first column ran on {threads[0]!r}, so numba's first compile happens inside the pool -- " "which is what aborts the process on macOS"
    )


def test_the_threaded_and_serial_paths_still_agree(monkeypatch):
    """Warming must not change a single edge: it moves where the work happens, not what it computes."""
    numba = pytest.importorskip("numba")
    if numba.config.DISABLE_JIT:
        pytest.skip("the threaded path is gated off when JIT is disabled")

    cols, y = _numeric_columns(5)
    threaded = _adaptive_nbins.per_feature_edges(cols, y, method="fayyad_irani", n_jobs=4)
    serial = _adaptive_nbins.per_feature_edges(cols, y, method="fayyad_irani", n_jobs=1)
    assert len(threaded) == len(serial) == cols.shape[1]
    for i, (a, b) in enumerate(zip(threaded, serial)):
        assert np.array_equal(np.asarray(a), np.asarray(b)), f"column {i} differs between the threaded and serial paths"
