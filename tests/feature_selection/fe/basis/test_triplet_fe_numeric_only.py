"""Regression: the hybrid_orth triplet (polynomial cross-basis) FE stage must skip string / categorical columns.

Pre-fix the triplet seed pool was every column (``[c for c in X.columns ...]``), so an object/categorical column (values like ``'a_1'``) was fed to the
Hermite/Legendre basis transform, which raised ``ValueError: could not convert string to float: 'a_1'``. The broad ``except Exception`` guard around the stage then
swallowed it and silently dropped the ENTIRE triplet stage ("continuing without triplet-FE columns"). Post-fix the seed pool is restricted to numeric columns, so a
genuine numeric 3-way interaction is still discovered while the string column is left to the dedicated categorical-encoding stages.
"""

import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def test_triplet_fe_skips_string_columns_no_convert_error(caplog):
    """The triplet cross-basis FE seed pool excludes string/categorical columns, so a genuine numeric 3-way interaction survives instead of the whole stage being silently dropped."""
    rng = np.random.default_rng(0)
    n = 1500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    y = (np.sign(x1 * x2 * x3) > 0).astype(int)
    scat = np.array([f"a_{i % 5}" for i in range(n)], dtype=object)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "scat": scat})
    y_ser = pd.Series(y)

    sel = MRMR(
        fe_hybrid_orth_triplet_enable=True,
        fe_hybrid_orth_triplet_max_degree=1,
        verbose=0,
        random_seed=0,
    )
    with caplog.at_level(logging.WARNING):
        sel.fit(X, y_ser)  # must not lose the triplet stage to a string->float error

    assert "could not convert string to float" not in caplog.text, "the triplet stage fed a string column into the polynomial basis: " + caplog.text
    assert "continuing without triplet-FE columns" not in caplog.text, (
        "the triplet stage was silently dropped (string column reached the basis): " + caplog.text
    )
    # Any emitted triplet column (3 legs joined by '*') must reference only numeric sources, never the string 'scat'.
    triplets = [c for c in (getattr(sel, "hybrid_orth_features_", None) or []) if c.split("__", 1)[0].count("*") == 2]
    assert all("scat" not in c for c in triplets), f"string column leaked into a triplet FE column: {triplets}"


def test_quadruplet_fe_skips_string_columns_no_convert_error(caplog):
    """Same numeric-only contract for the 4-way quadruplet cross-basis stage (identical seed-pool selection, identical polynomial basis)."""
    rng = np.random.default_rng(1)
    n = 1500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    x4 = rng.standard_normal(n)
    y = (np.sign(x1 * x2 * x3 * x4) > 0).astype(int)
    scat = np.array([f"a_{i % 5}" for i in range(n)], dtype=object)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "scat": scat})

    sel = MRMR(
        fe_hybrid_orth_quadruplet_enable=True,
        fe_hybrid_orth_quadruplet_max_degree=1,
        verbose=0,
        random_seed=1,
    )
    with caplog.at_level(logging.WARNING):
        sel.fit(X, pd.Series(y))

    assert "could not convert string to float" not in caplog.text, "the quadruplet stage fed a string column into the polynomial basis: " + caplog.text
    quads = [c for c in (getattr(sel, "hybrid_orth_features_", None) or []) if c.split("__", 1)[0].count("*") == 3]
    assert all("scat" not in c for c in quads), f"string column leaked into a quadruplet FE column: {quads}"
