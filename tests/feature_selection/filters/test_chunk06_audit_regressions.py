"""Regressions for the chunk-06 line-by-line audit of the filters helpers (arena, hierarchy, CMIM, UAED, composite blockwise, gate logging)."""

import logging
import threading
import types

import numpy as np
import pandas as pd
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.fixture(autouse=True)
def _fresh_throttles():
    """Forget the per-key log counts so each test sees its first log call."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


class _GatedArray(np.ndarray):
    """ndarray whose slice assignment blocks until the test lets it through."""

    entered = threading.Event()
    gate = threading.Event()

    def __setitem__(self, key, value):
        """Signal entry, wait for the gate, then assign."""
        _GatedArray.entered.set()
        _GatedArray.gate.wait(5)
        super().__setitem__(key, value)


def test_arena_append_of_a_name_being_written_waits_for_the_write():
    """A second append of a name whose column is still being copied must not return before the data is readable."""
    from mlframe.feature_selection.filters._column_arena import ColumnArena

    _GatedArray.entered.clear()
    _GatedArray.gate.clear()
    arena = ColumnArena(4)
    arena._new_chunk = lambda: np.zeros((arena.chunk_columns, arena.n_rows), dtype=np.float32).view(_GatedArray)  # type: ignore[method-assign]
    vals = np.array([1, 2, 3, 4], dtype=np.float32)
    first = threading.Thread(target=arena.append, args=("a", vals))
    first.start()
    assert _GatedArray.entered.wait(5)
    done = threading.Event()
    second = threading.Thread(target=lambda: (arena.append("a", vals), done.set()))
    second.start()
    assert not done.wait(0.3), "the second append returned while the column was still unwritten"
    _GatedArray.gate.set()
    first.join(5)
    second.join(5)
    assert done.is_set()
    np.testing.assert_array_equal(np.asarray(arena.column("a")), vals)


def test_arena_failed_write_does_not_leave_a_phantom_column():
    """A column whose copy raises must not stay registered with uninitialised memory behind it."""
    from mlframe.feature_selection.filters._column_arena import ColumnArena

    arena = ColumnArena(3)
    with pytest.raises(ValueError):
        arena.append("bad", np.array(["x", "y", "z"]))
    assert "bad" not in arena and len(arena) == 0
    slot = arena.append("good", np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(np.asarray(arena.column("good")), [1.0, 2.0, 3.0])
    assert arena.names == ["good"] and slot >= 0


def test_blockwise_composite_scoring_applies_the_source_uplift_gate_like_the_full_path(monkeypatch):
    """With no caller-supplied sources the block path used a zero uplift baseline and kept aggregates of an informative source the full path rejects."""
    import mlframe.feature_selection.filters._composite_group_agg_fe as comp
    from mlframe.feature_selection.filters._grouped_agg_fe import score_grouped_agg_by_cmi_uplift

    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({"g1": rng.integers(0, 6, n), "g2": rng.integers(0, 5, n)})
    for i in range(4):
        X[f"x{i}"] = rng.normal(size=n)
    key_mean = X.groupby(["g1", "g2"])["x0"].transform("mean").to_numpy()
    y = (X["x0"].to_numpy() + key_mean + 0.5 * rng.normal(size=n) > 0).astype(np.int64)
    num_cols = [f"x{i}" for i in range(4)]
    sets = [("g1", "g2")]
    enc, recipes = comp.generate_composite_group_agg_features(X, sets, num_cols)
    sources = {name: recipes[name]["num_col"] for name in enc.columns}
    scores = score_grouped_agg_by_cmi_uplift(X, enc, y, num_cols, n_bins=10, eng_to_source=sources)
    keep = scores[(scores["cmi"] >= 1e-4) & (scores["uplift"] >= 0.0)]
    want = set(keep["engineered_col"].head(20))
    assert any("x0" in c for c in scores["engineered_col"] if c not in want), "fixture must contain a candidate the uplift gate rejects"
    monkeypatch.setenv("MLFRAME_COMPOSITE_BLOCK_MB", "0.02")
    frame, _scores, _rec = comp.composite_group_agg_top_blockwise(
        X, y, sets, num_cols, base_cols=num_cols, top_k=20, n_bins=10, min_cmi=1e-4, min_uplift=0.0, eng_to_source={}
    )
    assert set(frame.columns) == want


def test_cluster_hierarchy_reports_pairs_that_could_not_be_scored(monkeypatch, caplog):
    """When every pair SU fails the build says so once, instead of returning an empty hierarchy as if the anchors had stabilised."""
    import mlframe.feature_selection.filters._dcd_pair_su_batch as batch_mod
    import mlframe.feature_selection.filters._dynamic_cluster_discovery as dcd_mod
    from mlframe.feature_selection.filters._cluster_hierarchy import build_cluster_hierarchy

    def boom(*_a, **_k):
        """Fail like a broken SU kernel."""
        raise RuntimeError("su broke")

    monkeypatch.setattr(batch_mod, "pair_su_batch", boom)
    monkeypatch.setattr(dcd_mod, "pair_su", boom)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200), "c": rng.normal(size=200)})
    with caplog.at_level(logging.WARNING):
        out = build_cluster_hierarchy({"cluster_anchors_names": {"a": [], "b": [], "c": []}}, X)
    assert out == {}
    assert any("could not be scored" in r.getMessage() for r in caplog.records)


def test_cmim_device_failure_is_logged_not_debug(monkeypatch, caplog):
    """A device fault in the CMIM block scorer is a warning, not a silent debug line before the slower host form."""
    from mlframe.feature_selection.filters import _cmim_min_gpu as mod

    def boom(*_a, **_k):
        """Fail like a device fault."""
        raise RuntimeError("device fault")

    monkeypatch.setattr(mod, "_prepare", boom)
    x = np.zeros((2, 10), dtype=np.int16)
    with caplog.at_level(logging.WARNING):
        got = mod._scores_device(x, np.zeros(10, dtype=np.int64), [np.zeros(10, dtype=np.int64)], np.ones((1, 2), dtype=bool), 4)
    assert got is None
    assert any(r.levelno >= logging.WARNING and "cmim min" in r.getMessage() for r in caplog.records)


def test_uaed_elbow_weights_are_float64():
    """Near-tied elbow scores resolve as in double precision: float32 position weights flipped this curve's elbow from 1 to 2."""
    from mlframe.feature_selection.filters._cmi_perm_stop import uaed_elbow

    m1, m2 = 2.0, 3.0 * (1 - 1e-9)
    c = np.zeros(5)
    c[2] = -m1
    c[3] = -m2 + 2 * c[2] - c[1]
    c[4] = c[3]
    assert uaed_elbow(c) == 1


def test_cell_moments_masked_gate_logs_when_too_many_cells(caplog):
    """The scatter-add fallback for a table over the shared-memory limit is announced with the numbers that decided it."""
    from mlframe.feature_selection.filters import _cell_moments_kernel as mod

    with caplog.at_level(logging.INFO):
        assert mod.masked_cell_moments(object(), None, None, None, mod.MAX_CELLS + 1) is None
    assert any("shared-memory limit" in r.getMessage() for r in caplog.records)


def test_cell_moments_fold_gate_logs_when_table_too_large(caplog):
    """The per-fold fallback for an all-folds table over the shared-memory limit is announced."""
    from mlframe.feature_selection.filters import _cell_moments_kernel as mod

    with caplog.at_level(logging.INFO):
        assert mod.fold_cell_moments(object(), None, None, None, None, 10, 1000) is None
    assert any("per-fold form" in r.getMessage() for r in caplog.records)


def test_cell_moments_fold_import_error_is_logged(monkeypatch, caplog):
    """A missing toolkit pins the per-fold form for the process and says so, as the single-fold kernels do."""
    from mlframe.feature_selection.filters import _cell_moments_kernel as mod

    def no_toolkit(*_a, **_k):
        """Fail like a missing CUDA toolkit."""
        raise ImportError("no nvrtc")

    monkeypatch.setattr(mod, "_FOLD_KERNELS", None)
    monkeypatch.setattr(mod, "_FOLD_UNAVAILABLE", False)
    with caplog.at_level(logging.WARNING):
        assert mod._fold_kernels(types.SimpleNamespace(RawKernel=no_toolkit)) is None
    assert any("all-folds kernels are unavailable" in r.getMessage() for r in caplog.records)


def test_strata_null_cmi_shape_decline_is_logged(caplog):
    """A table too wide for the per-stratum kernel sends the null to the dense route and logs the shape that decided it."""
    from mlframe.feature_selection.filters._cmi_null_strata_gpu import strata_null_cmi

    n = 20
    z_rank = np.arange(n)
    with caplog.at_level(logging.INFO):
        got = strata_null_cmi(None, np.zeros((1, n), dtype=np.int64), np.zeros(n, dtype=np.int64), z_rank, z_rank, np.zeros(n, dtype=np.int64), 100, None, np.zeros(n, dtype=np.int64))
    assert got is None
    assert any("declined" in r.getMessage() for r in caplog.records)
