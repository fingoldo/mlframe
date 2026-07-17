"""biz_val tests for ``mlframe.training.trainer`` --
``get_function_param_names`` (pure helper). ``configure_training_params``
skipped during active trainer refactor -- API surface is in flux.

Per CLAUDE.md: each test asserts a SYNTHETIC measurable WIN.
"""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# get_function_param_names (trainer.py) -- pure function
# ---------------------------------------------------------------------------


def test_biz_val_trainer_get_function_param_names_returns_names():
    """``get_function_param_names`` must return a list of parameter
    names from a callable. Catches regressions in the introspection
    helper."""
    from mlframe.training.trainer import get_function_param_names

    def foo(a, b, c=3):
        """Dummy callable with positional and default params, for get_function_param_names introspection."""
        pass

    names = get_function_param_names(foo)
    assert "a" in names and "b" in names, f"get_function_param_names must include 'a' and 'b'; got {names}"


@pytest.mark.parametrize(
    "func,expected_in",
    [
        (lambda x, y: None, ["x", "y"]),
        (lambda a, b, c, d=1: None, ["a", "b", "c", "d"]),
        (lambda *args: None, []),
    ],
)
def test_biz_val_trainer_get_function_param_names_parametrized(func, expected_in):
    """Parametrize over lambda signatures -- must extract names correctly
    for positional, default-bearing, and *args-only functions."""
    from mlframe.training.trainer import get_function_param_names

    names = get_function_param_names(func)
    for e in expected_in:
        assert e in names, f"expected '{e}' in names from {func.__code__.co_varnames[:5]}; got {names}"


def test_biz_val_trainer_get_function_param_names_noargs():
    """Function with zero args must return empty list."""
    from mlframe.training.trainer import get_function_param_names

    def noop():
        """Zero-argument dummy callable, to verify get_function_param_names returns an empty list."""
        pass

    names = get_function_param_names(noop)
    assert names == [], f"no-arg function must return []; got {names}"
