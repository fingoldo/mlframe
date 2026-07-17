"""Regression sensor (bug-hunt c0004): MRMR's hinge support-protection gate must not crash when a RAW categorical column is in the selected set.

When the suite skips categorical encoding, a raw pandas Categorical/string column reaches MRMR and can be selected on its marginal MI. The hinge
change-point protection block then builds a numeric incremental-R^2 baseline over the selected columns; ``np.asarray(cat_values, dtype=float64)``
on the string column raised ``ValueError: could not convert string to float`` and aborted the whole fit. A categorical is simply not a numeric
linear-regression baseline regressor, so the gate must skip it (and still complete the fit + deliver the hinge protection for the numeric source).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def test_hinge_gate_skips_raw_categorical_selected_column():
    """A raw categorical column reaching the hinge support-protection gate is skipped (not cast to float) so the fit survives and still protects the numeric kink signal."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 1500
    x = rng.uniform(-3.0, 3.0, n)
    tau = 0.5
    # A genuine slope change on x (the hinge FE fires + the protection gate runs) PLUS an independent categorical signal strong enough that the
    # MI screen selects the raw categorical column too -- so the gate iterates a non-numeric selected column (the exact c0004 crash path).
    cats = rng.choice(["alpha", "beta", "gamma", "delta"], size=n)
    cat_effect = np.array([{"alpha": 0.0, "beta": 4.0, "gamma": -4.0, "delta": 8.0}[c] for c in cats])
    y = 0.5 * x + 2.5 * np.maximum(x - tau, 0.0) + cat_effect + rng.normal(0.0, 0.1, n)
    X = pd.DataFrame({"x_kink": x, "c_cat": pd.Categorical(cats), "noise": rng.normal(0.0, 1.0, n)})

    fs = MRMR(verbose=0, random_seed=0).fit(X, y)  # pre-fix: ValueError "could not convert string to float" inside the hinge protection gate
    names = list(fs.get_feature_names_out())
    assert names, "MRMR selected nothing"
    # The categorical carried the dominant signal, so it must be in support (proving the gate iterated a non-numeric selected column and survived).
    assert any("c_cat" in nm for nm in names), f"expected the categorical signal in support; got {names}"
