"""Cluster 29 regression: a benchmark sweep with no finite score warns that it returns None (audit 2026-11-20, finding 29-F31).

When every combo's objective is missing, the decorated function returns None; a WARNING now names the function. A sweep with a finite score
returns the best output with no warning.
"""

import logging
import os

import numpy as np

from mlframe.utils._param_oracle import ParamOracle


def _store(tmp_path, name="oracle.parquet"):
    """Path to a store file inside the test's temporary directory."""
    return os.path.join(str(tmp_path), name)


def _toy(x, alpha=1, beta=1):
    """Return a value that depends on the combo."""
    return float((x * alpha + beta).sum())


def test_no_finite_score_warns_and_returns_none(tmp_path, caplog):
    """An objective with no usable score makes the decorated benchmark return None with a WARNING (29-F31)."""
    X = np.ones((4, 2))
    space = {"alpha": [1, 2], "beta": [0, 10]}
    oracle = ParamOracle(_store(tmp_path), param_space=space, mode="benchmark", minimize="elapsed_s",
                         objective_fn=lambda out, elapsed, *a: {"elapsed_s": float("nan")})
    with caplog.at_level(logging.WARNING):
        out = oracle(_toy)(X)
    assert out is None
    assert any("no combo gave a finite score" in r.getMessage() for r in caplog.records)


def test_finite_sweep_returns_the_output_silently(tmp_path, caplog):
    """Control: with a usable score the best output is returned and no warning is logged (29-F31)."""
    X = np.ones((4, 2))
    space = {"alpha": [1, 2], "beta": [0, 10]}
    oracle = ParamOracle(_store(tmp_path), param_space=space, mode="benchmark", minimize="elapsed_s")
    with caplog.at_level(logging.WARNING):
        out = oracle(_toy)(X)
    assert out is not None
    assert not any("no combo gave a finite score" in r.getMessage() for r in caplog.records)
