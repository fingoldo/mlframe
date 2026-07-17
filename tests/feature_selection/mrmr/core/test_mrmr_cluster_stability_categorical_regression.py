"""Regression sensor (bug-hunt c0015): MRMR's cluster / complementary-pairs stability selection must not crash on a raw categorical column.

The Pearson-clustering helpers blanket-coerced the whole feature matrix to float64, raising ``could not convert string to float`` on a raw
categorical column (reaching the selector under skip_categorical_encoding, e.g. a ``''`` level) -- which ``_stability_outer_fit`` caught and
silently fell back to classic fit, DISABLING cluster stability on any data carrying a categorical (a band-aid masking the crash). The fix clusters
only the numeric columns (a categorical cannot enter a Pearson graph, so it becomes its own singleton cluster) and hands the bootstrap selector
dtype-preserved rows, so the method RUNS on mixed numeric+categorical data -- and the categorical stays selectable as a singleton.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _frame_with_categorical(n=300, seed=0):
    """Frame with categorical."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "num_0": rng.normal(size=n),
            "num_1": rng.normal(size=n),
            "num_2": rng.normal(size=n),
            "cat": pd.Categorical(rng.choice(["a", "b", "", "c"], size=n)),  # raw categorical incl the '' level (the c0015 value)
        }
    )
    y = (df["num_0"] + df["num_1"] > 0).astype(int).to_numpy()
    return df, y


def test_stability_helpers_handle_raw_categorical_column():
    """Stability helpers handle raw categorical column."""
    from mlframe.feature_selection.filters._stability_cluster import (
        cluster_stability_selection,
        complementary_pairs_stability,
    )

    df, y = _frame_with_categorical()

    def _sel(X_sub, y_sub):
        # X_sub is a dtype-preserved frame row-subset now; the selector picks the 2 numeric signals + the categorical (index 3) so the
        # categorical's singleton cluster is exercised end-to-end.
        """Helper that sel."""
        assert hasattr(X_sub, "iloc")  # the de-mask hands the selector a frame, not a float-coerced array
        return np.array([0, 1, 3], dtype=np.int64)

    sel_c, _freq_c, info_c = cluster_stability_selection(df, y, _sel, n_bootstrap=6, rng_seed=0)
    assert info_c.get("n_failed", 0) == 0, "the bootstrap selector crashed on a categorical-bearing frame (pre-fix float64 coercion)"
    assert 3 in set(int(i) for i in sel_c), "the categorical column (singleton cluster) must remain selectable"

    sel_p, _freq_p, info_p = complementary_pairs_stability(df, y, _sel, n_pairs=6, rng_seed=0)
    assert info_p.get("n_failed", 0) == 0
    assert 3 in set(int(i) for i in sel_p)


def test_mrmr_cluster_stability_runs_on_categorical_without_fallback():
    """End-to-end: ``MRMR(stability_selection_method='cluster')`` fits on a categorical-bearing frame via the cluster path (pre-fix it raised
    inside and silently fell back to classic). The fit completes with a populated support_."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, y = _frame_with_categorical(n=200)
    fs = MRMR(verbose=0, random_seed=0, stability_selection_method="cluster", stability_n_bootstrap=4).fit(df, pd.Series(y, name="y"))
    assert getattr(fs, "support_", None) is not None
