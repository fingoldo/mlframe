"""Batch 5: a failed freeze of the fit-time basis preprocess_params logs at WARNING.

Each orthogonal hybrid entry point freezes the basis preprocessing of its selected columns so replay does not refit it.
When the freeze fails, replay refits on the replay data instead, so the recipe is degraded. The test makes only the
``return_params=True`` call of the basis evaluator raise, leaving the fit itself intact, and asserts the WARNING record.
At DEBUG the same record is dropped by default logging, so these tests fail on the old code.
"""

from __future__ import annotations

import importlib
import logging

import numpy as np
import pandas as pd
import pytest

_HANDLERS = [
    ("_orthogonal_copula_mi_fe", "hybrid_orth_mi_copula_fe_with_recipes"),
    ("_orthogonal_dcor_fe", "hybrid_orth_mi_dcor_fe_with_recipes"),
    ("_orthogonal_hsic_fe", "hybrid_orth_mi_hsic_fe_with_recipes"),
    ("_orthogonal_jmim_fe", "hybrid_orth_mi_jmim_fe_with_recipes"),
    ("_orthogonal_total_correlation_fe", "hybrid_orth_mi_tc_fe_with_recipes"),
    ("_orthogonal_cmim_fe", "hybrid_orth_mi_cmim_fe_with_recipes"),
]


def _warnings_mentioning(caplog, needle: str) -> list[logging.LogRecord]:
    """Return WARNING-or-higher records whose message contains needle."""
    return [r for r in caplog.records if r.levelno >= logging.WARNING and needle in r.getMessage()]


def _frame() -> tuple[pd.DataFrame, np.ndarray]:
    """Return a frame whose x0 carries a quadratic signal, so the orthogonal basis passes its uplift gate for these entry points."""
    rng = np.random.default_rng(0)
    n = 400
    x0 = rng.normal(size=n)
    X = pd.DataFrame({"x0": x0, "x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    y = (x0**2 + 0.1 * rng.normal(size=n)).astype(np.float64)
    return X, y


@pytest.mark.parametrize(("module_name", "function_name"), _HANDLERS, ids=[h[0] for h in _HANDLERS])
def test_freeze_failure_warns_and_degrades_the_recipe(monkeypatch, caplog, module_name, function_name):
    """A failed fit-time basis freeze falls back to refit-at-replay and logs a degraded-result warning."""
    mod = importlib.import_module(f"mlframe.feature_selection.filters.{module_name}")
    basis_pkg = importlib.import_module("mlframe.feature_selection.filters._orthogonal_univariate_fe")
    original = basis_pkg._evaluate_basis_column

    def _fail_only_when_freezing(*args, **kwargs):
        """Raise on the freeze call (return_params=True) and delegate every other call."""
        if kwargs.get("return_params"):
            raise RuntimeError("freeze broken")
        return original(*args, **kwargs)

    # Some modules import the evaluator at call time from the package, others bind it at module import; patch both.
    monkeypatch.setattr(basis_pkg, "_evaluate_basis_column", _fail_only_when_freezing)
    if hasattr(mod, "_evaluate_basis_column"):
        monkeypatch.setattr(mod, "_evaluate_basis_column", _fail_only_when_freezing)
    X, y = _frame()
    entry = getattr(mod, function_name)
    with caplog.at_level(logging.DEBUG):
        result = entry(X, y)
    assert len(result) == 3
    assert _warnings_mentioning(caplog, "failed to freeze fit-time basis preprocess_params")
