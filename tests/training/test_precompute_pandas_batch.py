"""Regression sensor for S06: pandas ``get_trainset_features_stats`` must use the dtype-declared
domain for ``pd.CategoricalDtype`` columns instead of a per-row data scan via ``.unique()``.

The fast path matters when many cat columns survive the post-polars fixes (Enum/Categorical cast).
On a 1M-row x 30 Categorical-col fixture the old code ran ~200ms; the fast path drops to ~1ms.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mlframe.training._precompute import get_trainset_features_stats


def _make_cat_frame(n: int = 1000, n_cols: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(n_cols):
        cats = [f"k_{i}_{k}" for k in range(20)]
        data[f"cc_{i}"] = pd.Categorical(rng.choice(cats, size=n))
    # one object-dtype cat for the slow-path branch coverage
    data["sc_0"] = pd.Series(rng.choice(["a", "b", "c"], size=n), dtype="object")
    return pd.DataFrame(data)


def test_S06_categorical_dtype_uses_dtype_categories_not_unique_scan():
    """``.unique()`` MUST NOT be invoked on ``pd.CategoricalDtype`` columns; the dtype-declared
    categories are the same domain in O(1)."""
    df = _make_cat_frame(n=2000, n_cols=4)
    cat_cols = [c for c in df.columns if c.startswith("cc_")]

    # Patch Series.unique so we can count invocations. We cannot patch pd.unique directly because
    # pandas internals call into a different code path; patching the Series method captures the
    # caller-facing surface that the loop hits.
    call_log: list[str] = []
    orig_unique = pd.Series.unique

    def _counting_unique(self, *args, **kwargs):
        call_log.append(self.name)
        return orig_unique(self, *args, **kwargs)

    with patch.object(pd.Series, "unique", _counting_unique):
        res = get_trainset_features_stats(df)

    # cat_vals populated for every cat col (cc_* + sc_0 if string detection enabled).
    assert "cat_vals" in res
    for c in cat_cols:
        assert c in res["cat_vals"], f"{c} missing from cat_vals"

    # No .unique() call on any pd.CategoricalDtype column. (sc_0 is object dtype; would only show
    # up here if get_categorical_columns include_string=False still picked it -- it doesn't, so
    # call_log should be empty entirely for this fixture.)
    cat_calls = [c for c in call_log if c in cat_cols]
    assert cat_calls == [], f".unique() was called on Categorical-dtype columns {cat_calls}; the fast path must read .dtype.categories instead."


def test_S06_categorical_dtype_returns_declared_domain():
    """``cat_vals`` for a Categorical column must equal the dtype-declared categories (the
    full domain), not only the observed values. Behavioural correctness post-fastpath."""
    # Build a frame where one category is declared but absent from data.
    cats = ["alpha", "beta", "gamma", "delta"]
    n = 100
    rng = np.random.default_rng(7)
    # Only observe alpha + beta; gamma + delta are declared but absent.
    observed = rng.choice(["alpha", "beta"], size=n)
    df = pd.DataFrame(
        {
            "cc_0": pd.Categorical(observed, categories=cats),
        }
    )

    res = get_trainset_features_stats(df)
    got = set(res["cat_vals"]["cc_0"])
    assert got == set(cats), f"cat_vals for declared Categorical must contain full dtype domain {set(cats)}, got {got}."


def test_S06_max_ncats_filter_still_applies():
    """The max_ncats_to_track filter must still drop over-cardinality columns even on the fast path."""
    n = 200
    big_cats = [f"k_{i}" for i in range(150)]
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        {
            "cc_big": pd.Categorical(rng.choice(big_cats, size=n)),
            "cc_small": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
        }
    )
    res = get_trainset_features_stats(df, max_ncats_to_track=100)
    assert "cc_small" in res["cat_vals"]
    assert "cc_big" not in res["cat_vals"], "cc_big has 150 declared categories > 100; should be excluded by max_ncats_to_track."
