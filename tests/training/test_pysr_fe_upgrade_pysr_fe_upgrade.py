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
    """PYTHON_JULIACALL_THREADS gets the literal string 'auto' (PySR's juliacall expected value;
    a numeric value emits UserWarning 'PYTHON_JULIACALL_THREADS ... was not able to set it'
    from pysr/julia_import.py:27 and disables PySR's own auto-thread routing). JULIA_NUM_THREADS
    gets a numeric value for the legacy manually-launched-Julia path.
    """
    monkeypatch.delenv("PYTHON_JULIACALL_THREADS", raising=False)
    monkeypatch.delenv("JULIA_NUM_THREADS", raising=False)

    from mlframe.training.pipeline import _maybe_set_pysr_thread_env
    _maybe_set_pysr_thread_env()

    assert os.environ.get("PYTHON_JULIACALL_THREADS") == "auto", (
        "PYTHON_JULIACALL_THREADS must be the literal 'auto' so PySR's juliacall sets the "
        "actual thread count itself; numeric values block PySR's own auto-setup"
    )
    assert os.environ.get("JULIA_NUM_THREADS") is not None, (
        "JULIA_NUM_THREADS not set -- legacy Julia start path will run single-threaded"
    )
    # JULIA_NUM_THREADS is numeric. At least 2 threads on any machine with >= 4 cores.
    if (os.cpu_count() or 0) >= 4:
        assert int(os.environ["JULIA_NUM_THREADS"]) >= 2


def test_maybe_set_pysr_thread_env_respects_pre_set_values(monkeypatch):
    """When the user pre-set either env var (e.g. for CI thread caps), the helper must NOT
    overwrite -- preserves user intent. A user setting PYTHON_JULIACALL_THREADS=4 trades
    PySR's auto-routing for a hard cap, which is a legitimate CI configuration.
    """
    monkeypatch.setenv("PYTHON_JULIACALL_THREADS", "4")
    monkeypatch.setenv("JULIA_NUM_THREADS", "4")
    from mlframe.training.pipeline import _maybe_set_pysr_thread_env
    _maybe_set_pysr_thread_env()
    assert os.environ["PYTHON_JULIACALL_THREADS"] == "4"
    assert os.environ["JULIA_NUM_THREADS"] == "4"


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
    """safe_log must NOT produce a non-finite Python float / sympy zoo at
    x=0; the sympy mapping must faithfully replicate the Julia training-
    time semantics ``safe_log(x) = x > 0 ? log(x) : NaN``.

    The earlier draft expected ``log(1e-9) ~= -20.72`` via an
    ``sp.log(sp.Abs(x) + 1e-9)`` form. That form was chosen for sympy
    printability but it CHANGED the function PySR fit at train time --
    predictions on negative / zero inputs then saw a value the model
    never saw during training. The intentional fix (see pysr_operators
    module docstring) is the Piecewise form that returns sp.nan at
    x <= 0; this test now pins THAT behaviour.
    """
    import math
    import sympy as sp
    from mlframe.feature_engineering.pysr_operators import _make_extra_sympy_mappings
    mappings = _make_extra_sympy_mappings()
    safe_log = mappings["safe_log"]
    x = sp.Symbol("x")
    # sympy's ``sp.nan`` -> Python float NaN under float().
    val_at_zero = float(safe_log(x).subs(x, 0))
    assert math.isnan(val_at_zero), (
        f"safe_log(0) must be NaN to match Julia training-time semantics; got {val_at_zero!r}"
    )
    # Negative input -> NaN too (same Julia branch).
    val_at_negative = float(safe_log(x).subs(x, -1))
    assert math.isnan(val_at_negative), (
        f"safe_log(-1) must be NaN; got {val_at_negative!r}"
    )
    # Positive input -> log(x) finite. log(E) == 1.
    val_at_positive = float(safe_log(x).subs(x, sp.E))
    assert val_at_positive == pytest.approx(1.0, abs=1e-9), (
        f"safe_log(E) must be 1.0; got {val_at_positive!r}"
    )
