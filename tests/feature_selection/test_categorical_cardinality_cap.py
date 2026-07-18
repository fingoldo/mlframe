"""``max_categorical_cardinality``: fold the rare-category tail of a high-cardinality categorical into one "other"
bucket. Two wins: (1) the codes matrix stays a narrow int (int8 for cap<=127 -> 4x smaller compact-codes storage even
when legit high-card categoricals exist), (2) DENSER contingency cells -> the plug-in MI/CMI on the categorical becomes
reliable (a high-cardinality categorical's cells are otherwise too sparse for a trustworthy MI estimate).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.discretization import cap_categorical_cardinality
from mlframe.feature_selection.filters.mrmr import MRMR


def test_cap_is_noop_below_cap_and_bounds_above():
    """Columns at/below the cap are byte-identical; a high-card column is bounded to fit the cap (int8 for cap<=127) with
    the NaN sentinel preserved."""
    rng = np.random.default_rng(0)
    low = rng.integers(0, 8, size=2000).astype(np.float64)
    high = rng.integers(0, 400, size=2000).astype(np.float64)
    high[::50] = -1.0  # NaN sentinel
    X = np.column_stack([low, high])
    capped = cap_categorical_cardinality(X, 127)
    assert np.array_equal(capped[:, 0], low), "low-card column must be unchanged"
    assert int(capped[:, 1].max()) <= 126, "high-card must be bounded to fit int8"
    assert np.array_equal(capped[:, 1] < 0, high < 0), "NaN sentinel preserved"
    assert capped.min() >= -128 and capped.max() <= 127, "fits int8"
    assert cap_categorical_cardinality(X, None) is X, "None -> no-op (same object)"


def test_cap_preserves_most_frequent_categories():
    """The (cap-1) MOST FREQUENT codes stay distinct; only the rare tail collapses into the 'other' bucket."""
    codes = np.array([[0]] * 100 + [[1]] * 50 + [[2]] * 5 + [[3]] * 2 + [[4]] * 1, dtype=np.float64)
    capped = cap_categorical_cardinality(codes, 3)  # keep 2 most frequent (0,1), fold {2,3,4} -> bucket 2
    col = capped[:, 0]
    assert int(col.max()) == 2
    assert (col[:100] == 0).all() and (col[100:150] == 1).all()  # top-2 kept distinct
    assert (col[150:] == 2).all()  # rare tail -> other bucket


def _sel(X, y, cap):
    """Helper that sel."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(max_runtime_mins=1, fe_max_steps=0, verbose=0, max_categorical_cardinality=cap)
        m.fit(X, y)
    return list(m.get_feature_names_out())


def test_numeric_mdlp_bins_bounded_by_max_depth():
    """The NUMERIC side of the cap: supervised MDLP can emit up to 2**max_depth intervals (which would exceed int8), so
    when max_categorical_cardinality is set the fit lowers max_depth to floor(log2(cap)). Verify the mechanism: a highly
    informative numeric column gets many bins at the default depth but is bounded when max_depth is lowered."""
    from mlframe.feature_selection.filters.discretization import categorize_dataset

    rng = np.random.default_rng(0)
    n = 20000
    X = pd.DataFrame({"smooth": np.linspace(0, 10, n) + rng.normal(0, 0.05, n)})
    y = (X["smooth"] * 2 + 0.3 * rng.normal(size=n)).values
    _, _, nb_deep = categorize_dataset(
        df=X, method="quantile", n_bins=10, nbins_strategy="mdlp", nbins_strategy_kwargs={"max_depth": 8}, y_for_strategy=y, dtype=np.int32
    )
    _, _, nb_shallow = categorize_dataset(
        df=X, method="quantile", n_bins=10, nbins_strategy="mdlp", nbins_strategy_kwargs={"max_depth": 3}, y_for_strategy=y, dtype=np.int32
    )
    assert int(nb_deep[0]) > 16, "informative numeric should get many MDLP bins at depth 8"
    assert int(nb_shallow[0]) <= 8, "max_depth=3 must bound numeric bins to <= 2**3"


def test_biz_val_cap_recovers_high_cardinality_signal():
    """biz_value: a high-cardinality categorical carrying a genuine signal (cat % 3) has cells too sparse to detect
    uncapped -> NOT selected; capping densifies the cells so its MI becomes reliable -> IS selected. Capping a high-card
    categorical is thus selection-IMPROVING here, not just a memory lever."""
    rng = np.random.default_rng(0)
    n = 6000
    cat = rng.integers(0, 400, size=n)
    X = pd.DataFrame({"x0": rng.normal(size=n), "x1": rng.normal(size=n), "hc": pd.Categorical(cat.astype(str))})
    y = X["x0"] * 1.5 + (cat % 3) + 0.1 * rng.normal(size=n)
    uncapped = _sel(X, y, None)
    capped = _sel(X, y, 127)
    assert "hc" not in uncapped, f"uncapped should miss the sparse high-card signal; got {uncapped}"
    assert "hc" in capped, f"capping should recover the densified high-card signal; got {capped}"
