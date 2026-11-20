"""Regression tests for the chunk-43 audit: recipe replay caches / fallback logs, CPT conditioning, nsb_mi validation."""
from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pandas as pd
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.fixture(autouse=True)
def _reset_throttles():
    """Forget the log-throttle counts so every test sees a fresh budget."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


def test_pair_cross_basis_cache_not_shared_with_univariate_f64_entry(monkeypatch):
    """A univariate basis entry (float64 input) must not be served to a pair-cross operand evaluated from the working-dtype column."""
    from mlframe.feature_selection.filters import _fe_usability_signal
    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
    from mlframe.feature_selection.filters.engineered_recipes._orth_basis_recipes import build_orth_pair_cross_recipe, build_orth_univariate_recipe

    monkeypatch.setattr(_fe_usability_signal, "_crit_np_dtype", lambda: np.float32)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    uni = build_orth_univariate_recipe(name="u", src_name="a", basis="hermite", degree=3)
    pair = build_orth_pair_cross_recipe(name="p", src_a_name="a", src_b_name="b", basis_i="hermite", basis_j="hermite", deg_a=3, deg_b=2)
    alone = apply_recipe(pair, X)
    cache: dict = {}
    apply_recipe(uni, X, basis_cache=cache)
    shared = apply_recipe(pair, X, basis_cache=cache)
    np.testing.assert_array_equal(alone, shared)


def test_coerce_to_int_maps_infinite_floats_like_nan():
    """+-inf in a float code column resolves to the top bin exactly like NaN instead of an undefined int cast."""
    from mlframe.feature_selection.filters.engineered_recipes._recipe_extract import _coerce_to_int_with_nan_handling

    out = _coerce_to_int_with_nan_handling(np.array([1.0, np.inf, -np.inf, np.nan]), 5, "r", "c", "clip")
    assert out.tolist() == [1, 4, 4, 4]


def test_coerce_to_int_raise_strategy_rejects_infinite_floats():
    """unknown_strategy='raise' also rejects infinite codes."""
    from mlframe.feature_selection.filters.engineered_recipes._recipe_extract import _coerce_to_int_with_nan_handling

    with pytest.raises(ValueError):
        _coerce_to_int_with_nan_handling(np.array([1.0, np.inf]), 5, "r", "c", "raise")


def test_unary_binary_gpu_replay_fallback_warns(monkeypatch, caplog):
    """A failing GPU replay falls back to numpy and says so at WARNING."""
    from mlframe.feature_selection.filters import _gpu_strict_fe
    from mlframe.feature_selection.filters.engineered_recipes import _recipe_unary_binary_gpu, apply_recipe
    from mlframe.feature_selection.filters.engineered_recipes._recipe_unary_binary import build_unary_binary_recipe

    def _boom(*a, **k):
        """Simulated cupy runtime failure."""
        raise RuntimeError("cuda fault")

    monkeypatch.setattr(_gpu_strict_fe, "fe_gpu_strict_resident_enabled", lambda: True)
    monkeypatch.setattr(_recipe_unary_binary_gpu, "apply_unary_binary_gpu", _boom)
    rec = build_unary_binary_recipe(
        name="m", src_a_name="a", src_b_name="b", unary_a_name="identity", unary_b_name="identity", binary_name="mul",
        unary_preset="minimal", binary_preset="minimal", quantization_nbins=None, quantization_method=None, quantization_dtype=np.int8,
    )
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with caplog.at_level(logging.WARNING):
        out = apply_recipe(rec, X)
    np.testing.assert_allclose(out, [3.0, 8.0])
    assert any("fell back to numpy" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_hermite_pair_missing_source_warns(caplog):
    """A hermite_pair recipe whose source column is absent returns NaN and logs a WARNING."""
    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
    from mlframe.feature_selection.filters.engineered_recipes._recipe_core import EngineeredRecipe

    rec = EngineeredRecipe(
        name="h", kind="hermite_pair", src_names=("a", "b"),
        extra={"coef_a": np.ones(2), "coef_b": np.ones(2), "basis": "hermite", "bin_func_name": "mul", "preprocess_a": {}, "preprocess_b": {}},
    )
    X = pd.DataFrame({"zzz": [1.0, 2.0]})
    with caplog.at_level(logging.WARNING):
        out = apply_recipe(rec, X)
    assert np.isnan(out).all()
    assert any("source column missing" in r.getMessage() for r in caplog.records)


def _su_boom(*a, **k):
    """Simulated merge_vars failure."""
    raise RuntimeError("merge_vars broke")


def test_su_normalization_failure_warns(monkeypatch, caplog):
    """A failing SU denominator is logged at WARNING, not debug, and the raw gain is kept."""
    from mlframe.feature_selection.filters import evaluation

    monkeypatch.setattr(evaluation, "use_su_normalization", lambda: True)
    monkeypatch.setattr(evaluation, "merge_vars", _su_boom)
    with caplog.at_level(logging.WARNING):
        out = evaluation._su_normalize_relevance(0.5, [0], [1], np.zeros((4, 2), dtype=np.int32), np.array([2, 2]), np.int32)
    assert out == 0.5
    assert any(r.levelno == logging.WARNING and "SU normalization failed" in r.getMessage() for r in caplog.records)


def test_bur_failure_log_is_throttled(monkeypatch, caplog):
    """Repeated BUR failures across candidates are throttled, not one WARNING per candidate."""
    from mlframe.feature_selection.filters import evaluation

    monkeypatch.setattr(evaluation, "get_bur_lambda", lambda: 1.0)
    monkeypatch.setattr(evaluation, "_evaluate_candidate_selected_vars", _su_boom)
    with caplog.at_level(logging.WARNING):
        for i in range(30):
            evaluation._apply_bur_bonus(0.1, 0.5, [1], None, [0], None, np.int32, i, {}, np.zeros(40))
    assert 1 <= sum("BUR bonus" in r.getMessage() for r in caplog.records) <= 5


def test_pid_failure_log_is_throttled(monkeypatch, caplog):
    """Repeated PID-bonus failures across candidates are throttled."""
    from mlframe.feature_selection.filters import evaluation

    monkeypatch.setattr(evaluation, "get_pid_synergy_bonus", lambda: 1.0)

    def _cols():
        """Raise like a failing column materialiser."""
        raise RuntimeError("no cols")

    with caplog.at_level(logging.WARNING):
        for i in range(30):
            evaluation._apply_pid_bonus(0.1, i, [1], {}, np.zeros(40), _cols)
    assert 1 <= sum("PID synergy" in r.getMessage() for r in caplog.records) <= 5


def test_gpu_relevance_fallback_log_is_throttled(monkeypatch, caplog):
    """The per-candidate GPU relevance fallback warning is throttled, and the CPU result is still used."""
    from mlframe.feature_selection.filters import evaluation

    def _gpu(*a, **k):
        """Simulated GPU failure."""
        raise RuntimeError("oom")

    monkeypatch.setattr(evaluation, "mi_direct_gpu", _gpu)
    monkeypatch.setattr(evaluation, "mi_direct", lambda *a, **k: (0.5, None, 0.1, 0.0))
    monkeypatch.setattr(evaluation, "get_group_mi", lambda: None)
    monkeypatch.setattr(evaluation, "use_su_normalization", lambda: False)
    with caplog.at_level(logging.WARNING):
        for i in range(30):
            gain = evaluation._evaluate_candidate_candidate_any_good_target(
                (i,), {}, (99,), np.zeros((4, 3), dtype=np.int32), np.array([2, 2, 2]), np.int32, None, {}, None, 2, 1, i, True, None, None,
            )
            assert gain == 0.5
    assert 1 <= sum("GPU relevance MI failed" in r.getMessage() for r in caplog.records) <= 5


def test_cpt_test_keeps_all_conditioning_variables_past_cardinality_cap(monkeypatch):
    """When the conditioning cardinality passes 1e6 the later selected variables must still reach the test, not be dropped."""
    from mlframe.feature_selection.filters import _conditional_permutation, evaluation

    seen = {}

    def _fake(x, y, z, nbins_x, nbins_y, nbins_z, n_permutations, seed):
        """Capture the conditioning vector handed to the test."""
        seen["z"] = np.asarray(z)
        seen["k"] = nbins_z
        return 0.1, 0.0

    monkeypatch.setattr(_conditional_permutation, "conditional_permutation_test", _fake)
    n = 200
    zeros = np.zeros(n, dtype=np.int64)
    last = (np.arange(n) % 2).astype(np.int64)
    cols = [zeros, zeros, zeros, last]
    nbins = [1000, 1000, 1000, 2]
    knob = lambda: ((zeros, 2), (zeros, 2), (cols, nbins))
    evaluation._evaluate_candidate_mi_distinguishable_its_conditional(True, [0, 1, 2, 3], 0.5, knob, 10, 1, 0, {}, np.zeros(1))
    assert len(np.unique(seen["z"])) == 2
    assert int(seen["z"].max()) < seen["k"]


def _fake_ndd(monkeypatch):
    """Install a stand-in ``ndd`` module so ``nsb_mi`` can run without the optional dependency."""
    mod = types.ModuleType("ndd")
    mod.mutual_information = lambda joint: float(np.asarray(joint).sum())
    monkeypatch.setitem(sys.modules, "ndd", mod)


def test_nsb_mi_rejects_negative_class_codes(monkeypatch):
    """A -1 (NaN sentinel) code must raise instead of silently wrapping into the last bin."""
    from mlframe.feature_selection.filters.estimators import nsb_mi

    _fake_ndd(monkeypatch)
    with pytest.raises(ValueError):
        nsb_mi(np.array([0, 1, -1]), np.array([0, 1, 1]), nbins_x=2, nbins_y=2)


def test_nsb_mi_joint_histogram_matches_pairs(monkeypatch):
    """The vectorised joint histogram counts every (x, y) pair once."""
    from mlframe.feature_selection.filters.estimators import nsb_mi

    _fake_ndd(monkeypatch)
    mod = sys.modules["ndd"]
    captured = {}
    mod.mutual_information = lambda joint: captured.setdefault("j", np.asarray(joint)) is not None and 0.0
    nsb_mi(np.array([0, 1, 1, 1]), np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(captured["j"], [[1, 0], [1, 2]])
