"""The collinearity pass streams (no Gram matrix) above its budgets and finds the pairs the Gram pass finds."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _mrmr_degenerate as dg


def _frame(seed: int, n: int = 500, p: int = 80) -> pd.DataFrame:
    """Independent columns plus +-linear copies (scaled, shifted, sign-flipped), NaN-sprinkled columns and a near-collinear (not perfect) column."""
    rng = np.random.default_rng(seed)
    cols = {f"c{j}": rng.standard_normal(n) for j in range(p)}
    cols["scaled"] = 3.0 * cols["c0"] + 7.0
    cols["flipped"] = -2.0 * cols["c1"] + 0.5
    cols["flipped_copy_of_a_copy"] = -1.0 * cols["scaled"]
    nan_col = cols["c2"].copy()
    nan_col[rng.random(n) < 0.1] = np.nan
    cols["nan_base"] = nan_col
    cols["nan_scaled"] = nan_col * 5.0 - 1.0
    cols["near"] = cols["c3"] + 0.3 * rng.standard_normal(n)
    return pd.DataFrame(cols)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_streaming_pass_equals_the_gram_pass(seed):
    """Forcing the streaming branch (a column limit below the width) gives the same verdicts as the Gram branch, sign flips and NaN fills included."""
    X = _frame(seed)
    gram = dg.audit_degenerate_columns(X)
    streamed = dg.audit_degenerate_columns(X, max_collinearity_cols=3)
    assert streamed == gram
    assert gram["scaled"] == "collinear_with:c0" and gram["flipped"] == "collinear_with:c1"
    assert gram["flipped_copy_of_a_copy"] == "collinear_with:c0"  # the earliest reference, not the intermediate copy that is itself flagged
    assert gram["nan_scaled"] == "collinear_with:nan_base"
    assert "near" not in gram


def test_streaming_branch_is_logged_and_not_skipped(caplog):
    """Above the budget the pass says it streams, and the collinear verdicts are still reported (the byte budget goes the same way)."""
    X = _frame(0)
    with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.filters._mrmr_degenerate"):
        by_cols = dg.audit_degenerate_columns(X, max_collinearity_cols=3)
        by_bytes = dg.audit_degenerate_columns(X, max_collinearity_bytes=1024)
    assert by_cols["scaled"] == by_bytes["scaled"] == "collinear_with:c0"
    assert sum("the collinearity pass streams" in r.getMessage() for r in caplog.records) == 2


def test_streaming_holds_only_a_few_vectors():
    """The streamed pass allocates no (p, n) matrix: its traced peak stays far below the frame size."""
    import tracemalloc

    rng = np.random.default_rng(5)
    n, p = 20_000, 200
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"c{j}" for j in range(p)])
    X["copy"] = 2.0 * X["c7"] - 1.0
    dg.audit_degenerate_columns(X.iloc[:300], max_collinearity_cols=3)  # warm
    tracemalloc.start()
    try:
        reasons = dg.audit_degenerate_columns(X, max_collinearity_cols=3)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert reasons["copy"] == "collinear_with:c7"
    assert peak < 0.35 * n * p * 8, f"streamed audit peak {peak / (n * p * 8):.2f} frame copies"
