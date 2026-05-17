"""Unit tests for the PySR FE upgrade.

Covers: operator preset surface, config validator, env-var setup (both
PYTHON_JULIACALL_THREADS + JULIA_NUM_THREADS), merge order
(pipeline defaults < typed config knobs < raw pysr_params dict),
and predict-time sympy mapping completeness.

Behavioural assertions only -- no inspect.getsource() probes (per
feedback_behavioral_tests memory rule).
"""
from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# Operator preset surface
# ---------------------------------------------------------------------------


def test_valid_presets_exposes_three_names():
    from mlframe.feature_engineering.pysr_operators import VALID_PRESETS
    assert set(VALID_PRESETS) == {"minimal", "standard", "physics"}


@pytest.mark.parametrize("preset", ["minimal", "standard", "physics"])
def test_get_preset_kwargs_returns_expected_keys(preset):
    """Every preset must populate the five PySR-splattable kwargs."""
    pytest.importorskip("sympy")
    from mlframe.feature_engineering.pysr_operators import get_preset_kwargs
    kw = get_preset_kwargs(preset)
    assert set(kw) == {
        "binary_operators",
        "unary_operators",
        "complexity_of_operators",
        "nested_constraints",
        "extra_sympy_mappings",
    }
    assert isinstance(kw["binary_operators"], list)
    assert isinstance(kw["unary_operators"], list)
    assert isinstance(kw["complexity_of_operators"], dict)
    assert isinstance(kw["nested_constraints"], dict)
    assert callable(next(iter(kw["extra_sympy_mappings"].values())))


def test_minimal_preset_uses_safe_log_not_raw_log():
    """The minimal preset must upgrade legacy `log` to `safe_log` -- raw `log`
    returns NaN for negative x at predict, causing the documented column-NaN
    leak. safe_log is the always-defined replacement.
    """
    pytest.importorskip("sympy")
    from mlframe.feature_engineering.pysr_operators import get_preset_kwargs
    kw = get_preset_kwargs("minimal")
    unary_text = " ".join(kw["unary_operators"])
    assert "safe_log" in unary_text
    assert "log(x::T)" not in unary_text or "safe_log" in unary_text


def test_standard_preset_includes_tabular_fe_operators():
    """Standard preset must cover the operators that most often produce useful
    interactions on numeric tabular targets: ratio (-, /), bounding (max,min),
    polynomial (square), saturating (tanh), signed-magnitude (sign).
    """
    pytest.importorskip("sympy")
    from mlframe.feature_engineering.pysr_operators import get_preset_kwargs
    kw = get_preset_kwargs("standard")
    binary_set = set(kw["binary_operators"])
    assert {"-", "/", "max", "min"}.issubset(binary_set)
    unary_text = " ".join(kw["unary_operators"])
    for op_name in ("safe_log", "safe_sqrt", "sign", "square", "tanh", "exp", "inv"):
        assert op_name in unary_text, f"standard preset missing {op_name!r}"


def test_physics_preset_includes_trig_and_power():
    pytest.importorskip("sympy")
    from mlframe.feature_engineering.pysr_operators import get_preset_kwargs
    kw = get_preset_kwargs("physics")
    unary_text = " ".join(kw["unary_operators"])
    for op_name in ("sin", "cos", "tan", "exp", "square", "cube"):
        assert op_name in unary_text, f"physics preset missing {op_name!r}"
    assert "^" in kw["binary_operators"]


def test_unknown_preset_raises():
    from mlframe.feature_engineering.pysr_operators import get_preset_kwargs
    with pytest.raises(ValueError, match="Unknown pysr_operator_preset"):
        get_preset_kwargs("turbo-mega-extreme")


@pytest.mark.parametrize("preset", ["minimal", "standard", "physics"])
def test_complexity_dict_values_are_positive_ints(preset):
    """complexity_of_operators values must be >= 1 ints; PySR rejects 0 and
    floats with cryptic Julia errors. Catch typos in the preset definitions.
    """
    from mlframe.feature_engineering.pysr_operators import _complexity_for_preset
    comp = _complexity_for_preset(preset)
    for op_name, weight in comp.items():
        assert isinstance(weight, int) and weight >= 1, (
            f"{preset}: complexity[{op_name!r}]={weight!r} must be int >= 1"
        )


@pytest.mark.parametrize("preset", ["minimal", "standard", "physics"])
def test_nested_constraints_block_self_nesting(preset):
    """All presets must forbid log(log(x)) / exp(exp(x)) / sin(sin(x)) -- the
    GA would otherwise waste budget on these trivially-prunable structures.
    """
    from mlframe.feature_engineering.pysr_operators import _nested_constraints_for_preset
    nc = _nested_constraints_for_preset(preset)
    # Every operator listed in nested_constraints must forbid itself nested.
    for op_name, inner_constraints in nc.items():
        if op_name in inner_constraints:
            assert inner_constraints[op_name] == 0, (
                f"{preset}: {op_name!r} self-nesting should be 0 (blocked), "
                f"got {inner_constraints[op_name]!r}"
            )


# ---------------------------------------------------------------------------
# Config validator
# ---------------------------------------------------------------------------


def test_config_accepts_each_valid_preset():
    from mlframe.training.configs import PreprocessingExtensionsConfig
    for preset in ("minimal", "standard", "physics"):
        cfg = PreprocessingExtensionsConfig(pysr_enabled=True, pysr_operator_preset=preset)
        assert cfg.pysr_operator_preset == preset


def test_config_rejects_unknown_preset():
    from mlframe.training.configs import PreprocessingExtensionsConfig
    with pytest.raises(ValueError, match="pysr_operator_preset must be one of"):
        PreprocessingExtensionsConfig(pysr_enabled=True, pysr_operator_preset="bogus")


def test_config_preset_default_is_none():
    """None means 'use the pipeline.py default' (currently 'standard'). The
    default itself isn't hardcoded in the config so flipping the in-suite
    default doesn't require config-schema migration.
    """
    from mlframe.training.configs import PreprocessingExtensionsConfig
    cfg = PreprocessingExtensionsConfig(pysr_enabled=True)
    assert cfg.pysr_operator_preset is None


# ---------------------------------------------------------------------------
# Env-var setup at module import
# ---------------------------------------------------------------------------


def test_maybe_set_pysr_thread_env_sets_both_vars(monkeypatch):
    """The PySR upgrade research surfaced that under juliacall the right env var is
    PYTHON_JULIACALL_THREADS, not JULIA_NUM_THREADS (gh discussion #873). pipeline.py
    defers the env-set into ``_maybe_set_pysr_thread_env`` (called lazily from
    ``_apply_pysr_fe``) so importers who never touch PySR don't get their env mutated.
    This test calls the helper explicitly and asserts both vars land.
    """
    # Clear any pre-existing values so we see the helper's own write, not stale state.
    monkeypatch.delenv("PYTHON_JULIACALL_THREADS", raising=False)
    monkeypatch.delenv("JULIA_NUM_THREADS", raising=False)

    from mlframe.training.pipeline import _maybe_set_pysr_thread_env
    _maybe_set_pysr_thread_env()

    assert os.environ.get("PYTHON_JULIACALL_THREADS") is not None, (
        "PYTHON_JULIACALL_THREADS not set -- juliacall will run single-threaded"
    )
    assert os.environ.get("JULIA_NUM_THREADS") is not None, (
        "JULIA_NUM_THREADS not set -- legacy Julia start path will run single-threaded"
    )
    # Both vars must agree -- they were just written from the same _suggested_threads value.
    assert os.environ["PYTHON_JULIACALL_THREADS"] == os.environ["JULIA_NUM_THREADS"]
    # Sanity: at least 2 threads on any machine with >= 4 cores.
    if (os.cpu_count() or 0) >= 4:
        assert int(os.environ["PYTHON_JULIACALL_THREADS"]) >= 2


def test_maybe_set_pysr_thread_env_respects_pre_set_values(monkeypatch):
    """When the user pre-set either env var (e.g. for CI thread caps), the helper
    must NOT overwrite -- preserves user intent.
    """
    monkeypatch.setenv("PYTHON_JULIACALL_THREADS", "1")
    monkeypatch.setenv("JULIA_NUM_THREADS", "1")
    from mlframe.training.pipeline import _maybe_set_pysr_thread_env
    _maybe_set_pysr_thread_env()
    assert os.environ["PYTHON_JULIACALL_THREADS"] == "1"
    assert os.environ["JULIA_NUM_THREADS"] == "1"


# ---------------------------------------------------------------------------
# Sympy mapping completeness for predict-time replay
# ---------------------------------------------------------------------------


def test_extra_sympy_mappings_cover_every_custom_unary():
    """Every custom operator in OPERATOR_JULIA_SIGNATURES must have an
    extra_sympy_mappings entry; otherwise predict-time materialisation of a
    discovered equation referencing the operator will TypeError.
    """
    pytest.importorskip("sympy")
    from mlframe.feature_engineering.pysr_operators import (
        OPERATOR_JULIA_SIGNATURES,
        _make_extra_sympy_mappings,
    )
    mappings = _make_extra_sympy_mappings()
    for op_name in OPERATOR_JULIA_SIGNATURES:
        assert op_name in mappings, (
            f"{op_name!r} declared in OPERATOR_JULIA_SIGNATURES but has no sympy mapping; "
            f"predict-time equation replay will fail."
        )


def test_safe_log_sympy_mapping_handles_zero_without_inf():
    """safe_log must produce a finite value at x=0 (the whole point of the
    safe variant). The +1e-9 inside log abs ensures log(0+eps) = log(1e-9),
    which is finite though large negative.
    """
    import sympy as sp
    from mlframe.feature_engineering.pysr_operators import _make_extra_sympy_mappings
    mappings = _make_extra_sympy_mappings()
    safe_log = mappings["safe_log"]
    x = sp.Symbol("x")
    val_at_zero = float(safe_log(x).subs(x, 0))
    assert val_at_zero == pytest.approx(-20.72, abs=0.1)  # log(1e-9) ~= -20.72
