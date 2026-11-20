"""Regression tests for the grouped-aggregate / grouped-quantile / hermite MI audit fixes."""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _grouped_agg_fe as GA
from mlframe.feature_selection.filters import _grouped_quantile_fe as GQ
from mlframe.feature_selection.filters import _grouped_quantile_gpu as G
from mlframe.utils.log_throttle import reset_throttle_counts


def test_relevance_filter_does_not_truncate_continuous_target():
    """A continuous target in [0, 1) is decile-binned, so the noise column is filtered out and the signal column kept."""
    rng = np.random.default_rng(0)
    n = 4000
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1.0 / (1.0 + np.exp(-3.0 * x1)) * 0.98
    X = pd.DataFrame({"x1": x1, "x2": x2})
    assert GA._filter_num_cols_by_relevance(X, y, ["x1", "x2"]) == ["x1"]


def test_target_qcut_failure_is_logged(monkeypatch, caplog):
    """A failing decile binning of the target warns instead of being swallowed at debug level."""
    reset_throttle_counts()

    def boom(*a, **k):
        """Always fails."""
        raise ValueError("qcut boom")

    monkeypatch.setattr(pd, "qcut", boom)
    with caplog.at_level(logging.WARNING, logger=GA.logger.name):
        GA._dense_target_codes(np.linspace(0.0, 1.0, 100))
    assert any("decile-binning" in r.getMessage() for r in caplog.records)


def test_device_mi_densifies_sparse_integer_target(monkeypatch):
    """The device MI batch receives 0-based dense class codes even when the integer target has negative or gapped labels."""
    from mlframe.feature_selection.filters import _device_blocks, _gpu_policy, _hermite_fe_mi, _scorer_device_dispatch

    seen = {}
    stub = types.SimpleNamespace(asarray=np.asarray)
    monkeypatch.setitem(sys.modules, "cupy", stub)
    monkeypatch.setattr(_gpu_policy, "cuda_available_for_run", lambda: True)
    monkeypatch.setattr(_scorer_device_dispatch, "scorer_device_choice", lambda *a, **k: True)
    monkeypatch.setattr(_device_blocks, "first_block", lambda *a, **k: 100)
    monkeypatch.setattr(_device_blocks, "run_column_blocks", lambda cp, k, block, fn: [fn(0, k)])

    def fake_mi(Xd, yd, n_bins, **kw):
        """Capture the label vector and class count handed to the kernel."""
        seen["y"] = np.asarray(yd)
        seen["n_classes"] = kw["n_classes"]
        return np.zeros(Xd.shape[1])

    monkeypatch.setattr(_hermite_fe_mi, "_plugin_mi_classif_batch_cuda_resident", fake_mi)
    rng = np.random.default_rng(0)
    eng = pd.DataFrame({"a": rng.standard_normal(300).astype(np.float32)})
    y_bin = rng.choice([-1, 5], size=300).astype(np.int64)
    GQ._grouped_quantile_mi_device(eng, ["a"], y_bin, 10)
    assert seen["y"].min() == 0 and seen["y"].max() == 1 and seen["n_classes"] == 2


def test_grouped_quantile_device_failure_is_logged(monkeypatch, caplog):
    """A device-form failure warns before the host form runs."""
    reset_throttle_counts()
    monkeypatch.setitem(sys.modules, "cupy", None)
    xs = np.random.default_rng(0).standard_normal((2, 50)).astype(np.float32)
    with caplog.at_level(logging.WARNING, logger=G.logger.name):
        assert G._stats_device(xs, np.zeros(50, dtype=np.int64), 1) is None
    assert any("device form is unavailable" in r.getMessage() for r in caplog.records)


def test_grouped_quantile_ktc_choose_failure_is_logged(monkeypatch, caplog):
    """A kernel-tuning choose() failure warns and falls back to the row-count mark."""
    reset_throttle_counts()
    monkeypatch.delenv("MLFRAME_GROUPED_QUANTILE_DEVICE", raising=False)

    class Spec:
        """Tuner whose choose() always fails."""

        def choose(self, **kw):
            """Always fails."""
            raise RuntimeError("no entry")

    monkeypatch.setattr(G, "_SPEC", Spec())
    with caplog.at_level(logging.WARNING, logger=G.logger.name):
        assert G.grouped_stats_on_device(10) is False
    assert any("choose() failed" in r.getMessage() for r in caplog.records)


def test_radix_column_block_edges_are_kept_after_oom(monkeypatch):
    """Edges recomputed in column blocks after a whole-matrix out-of-memory are used, not discarded for the percentile path."""
    from mlframe.feature_selection.filters import _gpu_resident_select, _hermite_fe_mi

    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(float32=np.float32, float64=np.float64))
    sentinel = object()

    def oom(*a, **k):
        """Whole-matrix radix select running out of memory."""
        raise RuntimeError("Out of memory")

    monkeypatch.setattr(_gpu_resident_select, "_radix_select_interior_edges", oom)
    monkeypatch.setattr(_gpu_resident_select, "_transpose_to_cm", lambda x: None)
    monkeypatch.setattr(_gpu_resident_select, "fe_gpu_radix_edges_enabled", lambda: True)
    monkeypatch.setattr(_gpu_resident_select, "radix_quantiles_grouped", lambda *a, **k: "percentile-path")
    monkeypatch.setattr(_hermite_fe_mi, "_radix_edges_in_column_blocks", lambda cp, X, n_bins: sentinel)
    X = np.random.default_rng(0).standard_normal((100, 3))
    out = _hermite_fe_mi._resident_mi_candidates(X, 10, keep_dtype=False, relax_binning=False, x_is_cm=False)
    assert out[3] is sentinel


def test_dense_target_codes_decile_bins_a_continuous_ndarray_target():
    """A continuous ndarray target is decile-binned into at most ten classes (qcut returns an ndarray, which has no to_numpy)."""
    codes = GA._dense_target_codes(np.random.default_rng(0).random(500))
    assert 2 <= codes.max() + 1 <= 10
