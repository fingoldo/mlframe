"""Audit chunk 34 regressions: orth MI backends, pair-cross recipes, scoring memo, streamed planner, xi/pair KTC lookups, pair residual screen."""

import logging
import sys
import types

import numpy as np
import pandas as pd
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.fixture(autouse=True)
def _fresh_throttles():
    """Forget the log-throttle counts so every test sees its warning."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


def _fake_cupy(monkeypatch, get_device_count=None):
    """Install a stand-in ``cupy`` module (an ``ndarray`` class and an optional ``getDeviceCount``) for the test."""
    cp = types.ModuleType("cupy")
    cp.ndarray = type("ndarray", (), {})
    runtime = types.SimpleNamespace(getDeviceCount=get_device_count)
    cp.cuda = types.SimpleNamespace(runtime=runtime)
    monkeypatch.setitem(sys.modules, "cupy", cp)
    return cp


def _warned(caplog, text, level=logging.WARNING):
    """Whether a record at ``level`` or above carries ``text``."""
    return any(r.levelno >= level and text in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------- _orth_mi_backends


def test_gpu_block_mi_failure_is_warned(monkeypatch, caplog):
    """A failing GPU column block falls to the CPU dispatch with a WARNING, not a debug line."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_mi_backends as be

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    y = rng.integers(0, 2, 200).astype(np.int64)
    mis = np.zeros(3)

    def boom(*a, **k):
        """A GPU block MI that always faults."""
        raise RuntimeError("device fault")

    monkeypatch.setattr(be, "_streamed_block_mi", lambda *a, **k: None)
    monkeypatch.setattr(be, "_block_mi_gpu_fn", lambda: boom)
    with caplog.at_level(logging.WARNING, logger=be.logger.name):
        be._retry_mi_column_blocks(X, y, 5, np.arange(3), mis, 200, np.float32, RuntimeError("oom"))
    assert _warned(caplog, "GPU block MI failed")
    assert np.isfinite(mis).all()


def test_retry_after_gpu_batch_failure_keeps_the_criterion_dtype(monkeypatch):
    """The OOM retry plans its column blocks at the criterion dtype, not at the dtype the failed GPU batch used."""
    from mlframe.feature_selection.filters import _fe_gpu_batch
    from mlframe.feature_selection.filters._fe_gpu_batch import _devices as devs
    from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_mi_backends as be

    crit = np.dtype(_crit_np_dtype())
    monkeypatch.setattr(be, "_orth_mi_gpu_enabled", lambda **kw: True)
    monkeypatch.setattr(be, "_mi_path_for", lambda *a, **k: "resident")
    monkeypatch.setattr(devs, "fe_gpu_f32_enabled", lambda: crit != np.dtype(np.float32))
    monkeypatch.setattr(_fe_gpu_batch, "multi_gpu_fe_batch_mi", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fault")))
    seen = {}
    monkeypatch.setattr(be, "_retry_mi_column_blocks", lambda X, y, nb, cols, mis, n, dt, e: seen.setdefault("dt", np.dtype(dt)))
    rng = np.random.default_rng(1)
    be._mi_classif_batch_numba(rng.standard_normal((100, 2)), rng.integers(0, 2, 100), nbins=5)
    assert seen["dt"] == crit


def test_class_balanced_mi_failure_is_warned(monkeypatch, caplog):
    """The class-balanced MI that fails returns None for the plain-MI fallback and says so at WARNING."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _imbalance_mi
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_mi_backends as be

    monkeypatch.setattr(_imbalance_mi, "_class_balanced_mi_batch_njit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kernel")))
    rng = np.random.default_rng(2)
    with caplog.at_level(logging.WARNING, logger=be.logger.name):
        out = be._mi_classif_batch_balanced(rng.standard_normal((100, 2)), rng.integers(0, 2, 100), np.array([1.0, 1.0]))
    assert out is None
    assert _warned(caplog, "Class-balanced batch MI failed")


def test_strict_resident_mi_device_fault_is_warned(monkeypatch, caplog):
    """A device fault in the STRICT resident MI hands the call to the CPU batch with a WARNING."""
    from mlframe.feature_selection.filters import _fe_gpu_strict
    from mlframe.feature_selection.filters import _device_upload_f64
    from mlframe.feature_selection.filters.hermite_fe import shared  # noqa: F401  (imported before the stand-in cupy is installed)
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_mi_backends as be

    _fake_cupy(monkeypatch)
    monkeypatch.setattr(_fe_gpu_strict, "fe_gpu_strict_enabled", lambda *a, **k: True)
    monkeypatch.setattr(_device_upload_f64, "upload_as_float64", lambda cp, X: (_ for _ in ()).throw(RuntimeError("upload fault")))
    monkeypatch.setattr(be, "_device_error_classes", lambda: (RuntimeError,))
    with caplog.at_level(logging.WARNING, logger=be.logger.name):
        out = be._strict_resident_mi(np.ones((50, 2)), np.zeros(50, dtype=np.int64), 5, False)
    assert out is None
    assert _warned(caplog, "STRICT resident MI failed")


# ---------------------------------------------------------------- _orth_pair_cross_fe


def test_pair_cross_scoring_bins_a_continuous_target(monkeypatch):
    """A float target in [0, 1) reaches the MI scorer as several classes, not as one truncated class."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_pair_cross_fe as pc

    rng = np.random.default_rng(3)
    raw = pd.DataFrame({"a": rng.standard_normal(600), "b": rng.standard_normal(600)})
    y = np.clip(0.5 + 0.2 * raw["a"].to_numpy() * raw["b"].to_numpy(), 0.0, 0.999)
    eng = pc.generate_pair_cross_basis_features(raw, [("a", "b")], basis="hermite")
    seen = []
    real = pc._mi_classif_batch

    def spy(X, yy, **kw):
        """Record the target the scorer receives."""
        seen.append(np.asarray(yy))
        return real(X, yy, **kw)

    monkeypatch.setattr(pc, "_mi_classif_batch", spy)
    pc.score_pair_cross_basis_by_mi_uplift(raw, eng, y, basis="hermite")
    assert seen and np.unique(seen[0]).size > 1


def test_pair_recipe_freezes_params_of_the_mean_filled_operand(monkeypatch):
    """The recipe's frozen preprocess params come from the same mean-filled operand the generator fitted, so a NaN in the source gives finite params."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_pair_cross_fe as pc

    rng = np.random.default_rng(4)
    a = rng.standard_normal(300)
    a[::17] = np.nan
    X = pd.DataFrame({"a": a, "b": rng.standard_normal(300)})
    X_aug = X.copy()
    X_aug["a*b__He1_He1"] = 0.0
    monkeypatch.setattr(pc, "hybrid_orth_mi_pair_fe", lambda *a_, **k: (X_aug, pd.DataFrame(), pd.DataFrame()))
    _, _, _, recipes = pc.hybrid_orth_mi_pair_fe_with_recipes(X, np.zeros(300, dtype=np.int64), basis="hermite")
    assert len(recipes) == 1
    params = recipes[0].extra["preprocess_params_i"]
    assert np.isfinite(params["mean"]) and np.isfinite(params["std"])


def test_pair_cross_device_born_probe_failure_is_warned(monkeypatch, caplog):
    """A failing device-born probe keeps the host path and logs at WARNING."""
    from mlframe.feature_selection.filters import _gpu_strict_fe
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_pair_cross_fe as pc

    monkeypatch.setattr(_gpu_strict_fe, "fe_gpu_device_born_crossbasis_enabled", lambda: (_ for _ in ()).throw(RuntimeError("probe")))
    with caplog.at_level(logging.WARNING, logger=pc.logger.name):
        assert pc._crossbasis_device_born_on() is False
    assert _warned(caplog, "fe_gpu_device_born_crossbasis_enabled() check failed")


def test_pair_cross_device_born_decline_is_logged(monkeypatch, caplog):
    """A device-born scorer that declines sends the batch to the host scorer with an INFO line."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_pair_cross_fe as pc

    rng = np.random.default_rng(5)
    raw = pd.DataFrame({"a": rng.standard_normal(200), "b": rng.standard_normal(200)})
    eng = pc.generate_pair_cross_basis_features(raw, [("a", "b")], basis="hermite")
    monkeypatch.setattr(pc, "_crossbasis_device_born_on", lambda: True)
    monkeypatch.setattr(pc, "_resident_pair_cross_mi", lambda *a, **k: None)
    with caplog.at_level(logging.INFO, logger=pc.logger.name):
        pc.score_pair_cross_basis_by_mi_uplift(raw, eng, rng.integers(0, 2, 200), basis="hermite")
    assert _warned(caplog, "device-born scoring declined", logging.INFO)


# ---------------------------------------------------------------- _orth_scoring_memo


def test_column_hash_separates_columns_that_differ_below_float32_precision():
    """The memo key is built at double precision: two columns that agree in float32 but differ in float64 get different keys."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_scoring_memo import _col_hash

    base = np.linspace(1.0, 2.0, 500).astype(np.float32).astype(np.float64)
    a = pd.DataFrame({"x": base})
    b = pd.DataFrame({"x": base + 1e-9})
    assert _col_hash(a, "x") != _col_hash(b, "x")


def test_dense_corr_memo_returns_the_names_of_the_asked_frame():
    """Two frames with equal column contents under different names must not share one cached name list."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_scoring_memo import cached_dense_finite_corr_matrix, orth_scoring_memo_scope

    rng = np.random.default_rng(6)
    first = pd.DataFrame(rng.standard_normal((200, 3)), columns=["a", "b", "c"])
    second = first.set_axis(["p", "q", "r"], axis=1)
    with orth_scoring_memo_scope():
        names1, _ = cached_dense_finite_corr_matrix(first, None, dtype=np.float64)
        names2, _ = cached_dense_finite_corr_matrix(second, None, dtype=np.float64)
    assert names1 == ["a", "b", "c"]
    assert names2 == ["p", "q", "r"]


# ---------------------------------------------------------------- _uplift_univariate_resident


def test_uplift_resident_gpu_failure_is_warned(monkeypatch, caplog):
    """A device fault in the resident uplift scorer returns None for the host scorer and logs at WARNING."""
    from mlframe.feature_selection.filters import _gpu_strict_fe
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _gpu_resident_cross_basis as xb
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _uplift_univariate_resident as ur

    monkeypatch.setattr(_gpu_strict_fe, "fe_gpu_device_born_uplift_univariate_enabled", lambda: True)
    monkeypatch.setattr(xb, "crossbasis_device_chosen", lambda n: True)
    monkeypatch.setattr(xb, "_product_mi_blocked", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("device fault")))
    _fake_cupy(monkeypatch)
    raw = pd.DataFrame({"a": np.arange(50, dtype=np.float64)})
    eng = pd.DataFrame({"a__He2": np.arange(50, dtype=np.float64)})
    with caplog.at_level(logging.WARNING, logger=ur.logger.name):
        out = ur.uplift_univariate_eng_mi_resident(raw, eng, np.zeros(50, dtype=np.int64), nbins=5)
    assert out is None
    assert _warned(caplog, "GPU path failed")


# ---------------------------------------------------------------- _streamed_gpu_mi


def test_plan_streamed_mi_pinned_cap_is_closed_form():
    """Capping the block width by the pinned budget is arithmetic, not a decrement loop over a width of hundreds of millions."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._streamed_gpu_mi import plan_streamed_mi

    plan = plan_streamed_mi(1, [(0, 1 << 40)], 1 << 20, np.float32)
    assert plan["width"] == int((1 << 20) * 0.2) // (2 * 4)


def test_streamed_devices_fault_is_logged(monkeypatch, caplog):
    """A CUDA runtime fault while listing devices (not a missing cupy) is logged and yields no devices."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _streamed_gpu_mi as sg

    def boom():
        """A device count query that faults."""
        raise RuntimeError("driver fault")

    _fake_cupy(monkeypatch, get_device_count=boom)
    with caplog.at_level(logging.INFO, logger=sg.logger.name):
        assert sg._devices() == []
    assert _warned(caplog, "CUDA devices unavailable", logging.INFO)


# ---------------------------------------------------------------- kernel-tuning lookups


class _BrokenSpec:
    """A kernel-tuning spec whose lookup raises."""

    def choose(self, **kw):
        """Always fail."""
        raise RuntimeError("cache broken")


def test_xi_use_device_lookup_failure_is_warned(monkeypatch, caplog):
    """A failing xi kernel-tuning lookup falls back to the host and logs at WARNING."""
    from mlframe.feature_selection.filters import _orthogonal_xi_gpu as xg

    monkeypatch.setattr(xg, "_SPEC", _BrokenSpec())
    with caplog.at_level(logging.WARNING, logger=xg.logger.name):
        assert xg.use_device(1000) is False
    assert _warned(caplog, "KTC choose() failed")


def test_pair_residual_lookup_failure_is_warned(monkeypatch, caplog):
    """A failing pair-residual kernel-tuning lookup falls back to the floor and logs at WARNING."""
    from mlframe.feature_selection.filters import _pair_residual_gpu as pg

    monkeypatch.setattr(pg, "_SPEC", _BrokenSpec())
    with caplog.at_level(logging.WARNING, logger=pg.logger.name):
        assert pg.pair_device_chosen(1000, 10) is False
    assert _warned(caplog, "KTC choose() failed")


def test_pair_floor_lookup_failure_is_warned(monkeypatch, caplog):
    """A failing pair-floor kernel-tuning lookup falls back to the fixed rule and logs at WARNING."""
    from mlframe.feature_selection.filters import _pair_mi_perm_cuda as pm

    monkeypatch.setattr(pm, "_FLOOR_SPEC", _BrokenSpec())
    with caplog.at_level(logging.WARNING, logger=pm.logger.name):
        assert pm.floor_backend_choice(1000, 10) == "cpu"
    assert _warned(caplog, "choose() failed")


# ---------------------------------------------------------------- pair kernels' inputs


def test_pair_mi_perm_rejects_a_pair_index_outside_the_columns():
    """A pair index beyond the factor columns would read past the uploaded byte matrix: it is rejected before any device work."""
    from mlframe.feature_selection.filters._pair_mi_perm_cuda import pair_mi_perm_matrix_cuda

    data = np.zeros((20, 3), dtype=np.int8)
    with pytest.raises(ValueError, match="outside"):
        pair_mi_perm_matrix_cuda(data, np.array([0, 5]), np.array([1, 2]), np.full(3, 2), np.zeros((2, 20), dtype=np.int64), np.array([1.0]), 1000)


def test_fill_pair_bias_rejects_a_short_bias_vector():
    """A bias vector shorter than the pair arrays must raise, not silently leave pairs out of the dict."""
    from mlframe.feature_selection.filters._pair_operand_arrays import fill_pair_bias

    with pytest.raises(ValueError, match="equal lengths"):
        fill_pair_bias({}, np.array([0, 1, 2]), np.array([1, 2, 3]), np.array([0.1, 0.2]))


def test_pair_usability_table_over_cap_is_logged(monkeypatch, caplog):
    """A table too large to keep is dropped with an INFO line, not silently."""
    from mlframe.feature_selection.filters import _pair_usability_table as put

    put.clear()
    monkeypatch.setattr(put, "_MAX_PAIRS_PER_TABLE", 1)
    with caplog.at_level(logging.INFO, logger=put.logger.name):
        put._store(("k",), np.array([1, 2], dtype=np.int64), np.zeros((2, 1)))
    assert _warned(caplog, "not stored", logging.INFO)
    put.clear()


# ---------------------------------------------------------------- pair residual screen


def _screen_frame(n=4000, seed=7):
    """Five continuous columns whose first two interact (target ``c0 * c1`` plus noise)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"c{i}" for i in range(5)])
    y = X["c0"].to_numpy() * X["c1"].to_numpy() + 0.3 * rng.standard_normal(n)
    return X, y


def test_pair_screen_ignores_rows_with_a_missing_target():
    """A few NaN targets must not blank the whole screen: the interaction is still found on the finite rows."""
    from mlframe.feature_selection.filters._pair_residual_screen import pair_interaction_screen

    X, y = _screen_frame()
    found = [(a, b) for a, b, _t in pair_interaction_screen(X, y)]
    assert ("c0", "c1") in found
    y_nan = y.copy()
    y_nan[::97] = np.nan
    found_nan = [(a, b) for a, b, _t in pair_interaction_screen(X, y_nan)]
    assert ("c0", "c1") in found_nan


def test_pair_screen_picks_its_columns_on_the_even_rows_only(monkeypatch):
    """The columns that enter the additive model are chosen without the odd rows that later score the pairs."""
    from mlframe.feature_selection.filters import _pair_residual_gpu as pg
    from mlframe.feature_selection.filters._pair_residual_screen import pair_interaction_screen

    rng = np.random.default_rng(8)
    n = 2000
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=["c0", "c1", "c2", "c3"])
    even = np.arange(n) % 2 == 0
    y = np.where(even, X["c0"] + X["c1"], 3.0 * X["c2"] + 3.0 * X["c3"]) + 0.1 * rng.standard_normal(n)
    monkeypatch.setattr(pg, "score_pairs_dispatch", lambda host, Q, resid, ev, pairs, nb, sh: np.full(pairs.shape[0], 100.0))
    found = pair_interaction_screen(X, y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y), max_cols=2)
    assert len(found) == 1
    assert {found[0][0], found[0][1]} == {"c0", "c1"}
