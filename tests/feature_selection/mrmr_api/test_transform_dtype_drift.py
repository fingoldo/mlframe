"""transform() warns when a column changes dtype kind since fit or arrives entirely missing.

No fit-time dtypes were stored, so an int column arriving as float, a categorical arriving as string, or an all-NaN selected column passed
transform silently: wrong values, not a crash.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _fitted():
    """A light fit selecting an int column ``k`` and a float column ``x``."""
    rng = np.random.default_rng(0)
    n = 800
    k = rng.integers(0, 5, size=n).astype(np.int64)
    x = rng.normal(size=n)
    X = pd.DataFrame({"k": k, "x": x, "noise": rng.normal(size=n)})
    y = ((x + 0.8 * (k - 2)) > 0).astype(np.int64)
    MRMR._FIT_CACHE.clear()
    m = MRMR(random_seed=0, n_jobs=1, verbose=0, fe_max_steps=0, full_npermutations=3, baseline_npermutations=2).fit(X, y)
    names = set(map(str, m.get_feature_names_out()))
    assert {"k", "x"} <= names, f"fixture precondition: k and x must be selected, got {names}"
    return m, X


def _warnings(caplog, needle):
    """WARNING records whose message contains ``needle``."""
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and needle in r.getMessage()]


def test_numeric_column_arriving_as_string_warns(caplog):
    """The numeric column k arriving as strings at transform changes the logical kind and is named in a dtype-drift warning."""
    m, X = _fitted()
    with caplog.at_level(logging.WARNING):
        m.transform(X.assign(k=X["k"].astype(str)))
    msgs = _warnings(caplog, "changed dtype kind")
    assert msgs and "k" in msgs[0], f"no dtype-drift warning naming k: {[r.getMessage() for r in caplog.records]}"


def test_int_column_arriving_as_float_is_benign_and_replays_identically(caplog):
    """Int <-> float is one logical kind (a nullable int is float64+NaN in a pandas view): no warning, and the replayed values are equal.

    Formerly asserted a warning here; the values an int column holds are the same floats, so the replay cannot differ.
    """
    m, X = _fitted()
    want = m.transform(X)
    with caplog.at_level(logging.WARNING):
        got = m.transform(X.assign(k=X["k"].astype(np.float64)))
    assert not _warnings(caplog, "changed dtype kind"), [r.getMessage() for r in caplog.records]
    assert list(got.columns) == list(want.columns)
    np.testing.assert_array_equal(got.to_numpy(dtype=np.float64), want.to_numpy(dtype=np.float64))


def _fitted_nullable_int_with_engineered_recipe():
    """Fit on the pandas view of a nullable-int column (float64 + NaN) whose hinge makes MRMR replay an engineered recipe on it."""
    import polars as pl

    rng = np.random.default_rng(0)
    n = 2500
    w = rng.integers(1, 60, size=n).astype(np.float64)
    w[rng.random(n) < 0.15] = np.nan
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    y = ((np.maximum(np.nan_to_num(w, nan=30.0) - 30, 0) / 15 + 0.3 * a + rng.standard_normal(n) * 0.3) > 0.6).astype(np.int64)
    pdf = pd.DataFrame({"w": w, "a": a, "b": b})
    pldf = pl.DataFrame({"w": pl.Series("w", [None if np.isnan(v) else int(v) for v in w], dtype=pl.Int16), "a": a, "b": b})
    MRMR._FIT_CACHE.clear()
    m = MRMR(random_seed=0, n_jobs=1, n_workers=1, verbose=0, fe_max_steps=1, use_simple_mode=True, quantization_nbins=8, full_npermutations=5,
             max_consec_unconfirmed=3, min_nonzero_confidence=0.9, max_runtime_mins=1).fit(pdf, y)
    return m, pdf, pldf


def test_nullable_int_fit_as_float_replays_identically_on_polars_int16_with_nulls(caplog):
    """The suite fits on the pandas view (Int16 with nulls -> float64 NaN) and transforms a native polars Int16: no drift warning, same output.

    Covers the engineered column built on the nullable column too, so the NaN / null handling of the replay is pinned, not just the passthrough.
    """
    m, pdf, pldf = _fitted_nullable_int_with_engineered_recipe()
    assert any("w" in getattr(r, "src_names", ()) for r in m._engineered_recipes_), "fixture precondition: a recipe must read the nullable column"
    want = m.transform(pdf)
    with caplog.at_level(logging.WARNING):
        got = m.transform(pldf).to_pandas()
    assert not _warnings(caplog, "changed dtype kind"), [r.getMessage() for r in caplog.records]
    assert list(got.columns) == list(want.columns)
    np.testing.assert_array_equal(got.to_numpy(dtype=np.float64), want.to_numpy(dtype=np.float64))


def test_all_missing_selected_column_warns(caplog):
    """A selected column that is entirely NaN at transform is named in a warning."""
    m, X = _fitted()
    with caplog.at_level(logging.WARNING):
        m.transform(X.assign(x=np.nan))
    msgs = _warnings(caplog, "entirely missing")
    assert msgs and "x" in msgs[0]


def test_unchanged_input_does_not_warn(caplog):
    """Control: transforming the fit frame raises neither warning."""
    m, X = _fitted()
    with caplog.at_level(logging.WARNING):
        m.transform(X)
    assert not _warnings(caplog, "changed dtype kind") and not _warnings(caplog, "entirely missing")
