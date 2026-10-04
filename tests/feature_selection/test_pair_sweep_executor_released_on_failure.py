"""The pair-sweep executor must be released on the failure path, not only at the end of a clean sweep.

`_chunk_state["pipeline_ex"] = ThreadPoolExecutor(max_workers=1)` was created about two hundred lines
before its `shutdown(wait=True)`, with nothing covering the span. Any exception in the pair loop -- a cupy
OutOfMemoryError, a kernel launch failure, a KeyboardInterrupt -- skipped the shutdown.
`ThreadPoolExecutor` registers its worker through `threading._register_atexit`, so the thread survived to
interpreter exit, and a still-pending future held the shared double buffer with it: at a 2M-row chunk over
40 operands that is 640 MB per buffer, 1.28 GB for the pair, retained for the rest of the process.

The real chunk pipeline needs cupy and a device, so these tests drive the code paths on CPU instead:

* the setup helper is run for real with its GPU gates faked open, proving it is what installs the executor
  and the double buffer into ``chunk_state`` (the contract the sweep test below relies on), and that its own
  ``except`` shuts a half-built executor down;
* ``check_prospective_fe_pairs`` is run for real with the pair scorer forced to raise mid-loop while a
  pending prefetch future is outstanding; the executor must come out shut down, the future must have run to
  completion, and the buffers must be dropped from ``chunk_state``.
"""

from __future__ import annotations

import threading
import types
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _fe_gpu_strict, _gpu_resident_fe
from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_core, _pairs_core_helpers


def _open_pipeline_gates(monkeypatch) -> None:
    """Make the setup helper believe the strict GPU-resident path is on, without a GPU."""
    monkeypatch.setenv("MLFRAME_FE_PIPELINE_CHUNKS", "1")
    monkeypatch.setattr(_fe_gpu_strict, "fe_gpu_strict_enabled", lambda **kw: True)
    monkeypatch.setitem(__import__("sys").modules, "cupy", types.ModuleType("cupy"))
    monkeypatch.setattr(_gpu_resident_fe, "_resident_operand_table", lambda cp, tv: None, raising=False)


def _run_setup(chunk_state: dict) -> None:
    """Run the real pipeline-setup helper with a two-chunk plan, as the sweep does."""
    buf = np.zeros((8, 3), dtype=np.float32)
    _pairs_core_helpers._check_prospective__read_weakref_cache_no(True, buf, [[(0, 1)], [(0, 2)]], np.zeros((8, 3)), 3, np.zeros((8, 3)), chunk_state, 0)


def test_setup_installs_the_executor_and_double_buffer(monkeypatch):
    """The helper the sweep calls is what builds the worker thread and the two buffers it shares."""
    _open_pipeline_gates(monkeypatch)
    state: dict = {}
    _run_setup(state)
    try:
        assert isinstance(state.get("pipeline_ex"), ThreadPoolExecutor), "the pipeline executor is no longer built by the setup helper"
        assert len(state["pipeline_buffers"]) == 2
        assert state["pipeline_futures"] == {}
    finally:
        ex = state.get("pipeline_ex")
        if ex is not None:
            ex.shutdown(wait=True)


class _FailsAfterExecutorBuilt(dict):
    """A chunk_state whose write right after the executor construction fails, as a mid-setup error would."""

    built: list = []

    def __setitem__(self, key, value):
        """Record the executor, then fail on the write that follows its construction."""
        if key == "pipeline_ex":
            type(self).built.append(value)
        if key == "pipeline_futures":
            raise RuntimeError("simulated failure after the executor was built")
        super().__setitem__(key, value)


def test_the_construction_is_still_guarded_by_its_own_setup_except(monkeypatch):
    """A setup failure after the executor exists must shut it down and leave no half-built state behind."""
    _open_pipeline_gates(monkeypatch)
    _FailsAfterExecutorBuilt.built = []
    state = _FailsAfterExecutorBuilt()
    _run_setup(state)  # the helper's own except swallows the error and falls back to the synchronous path
    assert len(_FailsAfterExecutorBuilt.built) == 1, "the simulated failure did not hit after construction; the test lost its subject"
    ex = _FailsAfterExecutorBuilt.built[0]
    assert ex._shutdown, "the setup-failure path no longer shuts down a partially built executor"
    assert "pipeline_ex" not in state and "pipeline_buffers" not in state


def _sweep_inputs():
    """Small real inputs for check_prospective_fe_pairs (one prospective pair)."""
    from mlframe.feature_selection.filters.discretization import discretize_array
    from mlframe.feature_selection.filters.feature_engineering import create_binary_transformations, create_unary_transformations
    from mlframe.feature_selection.filters.info_theory import merge_vars

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"a": rng.uniform(0.5, 5.0, n), "b": rng.uniform(-2.0, 2.0, n), "c": rng.uniform(0.1, 1.0, n)}).astype(np.float32)
    data = np.column_stack([discretize_array(df[c].to_numpy(), n_bins=4, method="quantile", dtype=np.int32) for c in "abc"])
    target = ((df["a"].to_numpy() > df["a"].mean()) ^ (df["b"].to_numpy() > df["b"].mean())).astype(np.int32)
    data = np.column_stack([data, target])
    classes_y, freqs_y, _ = merge_vars(
        factors_data=data,
        vars_indices=np.array([3], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=np.array([4, 4, 4, 2], dtype=np.int64),
        dtype=np.int32,
    )
    return dict(
        prospective_pairs={((0, 1), 1.0): 1.5},
        X=df,
        unary_transformations=create_unary_transformations(preset="minimal"),
        binary_transformations=create_binary_transformations(preset="minimal"),
        classes_y=classes_y,
        classes_y_safe=classes_y.copy(),
        freqs_y=freqs_y,
        num_fs_steps=0,
        cols=["a", "b", "c"],
        original_cols={0: 0, 1: 1, 2: 2},
        fe_max_steps=1,
        fe_npermutations=1,
        fe_max_pair_features=2,
        fe_print_best_mis_only=True,
        fe_min_nonzero_confidence=0.0,
        fe_min_engineered_mi_prevalence=0.0,
        fe_good_to_best_feature_mi_threshold=0.5,
        fe_max_external_validation_factors=0,
        numeric_vars_to_consider=[0, 1, 2],
        quantization_nbins=4,
        quantization_method="quantile",
        quantization_dtype=np.int32,
        times_spent=defaultdict(float),
        verbose=0,
    )


def test_the_blocking_shutdown_runs_when_the_pair_loop_raises(monkeypatch):
    """An exception in the pair loop must not leak the worker thread or the buffers its pending future holds."""
    captured: dict = {}
    release = threading.Event()
    finished = threading.Event()

    def _prefetch():
        """A pending prefetch that only completes once the loop has failed."""
        release.wait(5.0)
        finished.set()

    def _fake_setup(_chunk_global_batch, _chunk_buffer, _fe_chunks, X, _chunk_buf_width, transformed_vars, chunk_state, verbose):
        """Install the executor, buffers and a pending future the way the real setup does."""
        # Same contract as the real helper (pinned by test_setup_installs_the_executor_and_double_buffer),
        # plus an unconsumed prefetch so shutdown(wait=True) has something to wait for.
        ex = ThreadPoolExecutor(max_workers=1)
        chunk_state["pipeline_buffers"] = [np.zeros(4), np.zeros(4)]
        chunk_state["pipeline_ex"] = ex
        chunk_state["pipeline_futures"] = {1: ex.submit(_prefetch)}
        captured["ex"] = ex
        captured["state"] = chunk_state

    def _raising_scorer(**kwargs):
        """Fail mid-loop, releasing the prefetch shortly after."""
        threading.Timer(0.2, release.set).start()
        raise MemoryError("simulated device OOM in the pair loop")

    monkeypatch.setattr(_pairs_core, "_check_prospective__read_weakref_cache_no", _fake_setup)
    monkeypatch.setattr(_pairs_core, "_score_one_pair", _raising_scorer)

    with pytest.raises(MemoryError, match="simulated device OOM"):
        _pairs_core.check_prospective_fe_pairs(**_sweep_inputs())

    assert "ex" in captured, "the sweep no longer calls the pipeline setup; the test lost its subject"
    assert captured["ex"]._shutdown, "an exception in the pair loop leaked the pipeline executor's worker thread"
    assert finished.is_set(), "the executor was not shut down with wait=True; the pending prefetch was abandoned"
    assert "pipeline_ex" not in captured["state"]
    assert "pipeline_buffers" not in captured["state"], "the double chunk buffer is not released with the executor"
    assert "pipeline_futures" not in captured["state"]
