"""Regression: cat-num residual FE skips a duplicated column name instead of crashing on `.dtype`.

When X has a duplicated column name, `X[c]` returns a DataFrame (no `.dtype`), so the numeric-column filter raised
`AttributeError: 'DataFrame' object has no attribute 'dtype'`, losing the whole FE pass (fuzz c0034). The ambiguous
duplicated column is now skipped (a single numeric dtype can't be determined for it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._count_freq_interaction_fe import cat_num_interaction_with_recipes


def test_cat_num_interaction_skips_duplicate_named_column():
    """A duplicated column name (X[c] returns a DataFrame, no .dtype) is skipped rather than crashing the whole FE pass."""
    rng = np.random.default_rng(0)
    n = 50
    X = pd.DataFrame(
        np.column_stack([rng.normal(size=n), rng.normal(size=n), rng.integers(0, 3, n).astype(float)]),
        columns=["dup", "dup", "cat"],  # duplicated 'dup' -> X['dup'] is a DataFrame
    )
    y = rng.normal(size=n)
    # Pre-fix: X['dup'].dtype -> AttributeError. Must not raise now (the ambiguous dup column is skipped).
    _out, appended, _recipes = cat_num_interaction_with_recipes(X, y, cat_cols=["cat"], num_cols=["dup"])
    assert appended == []
