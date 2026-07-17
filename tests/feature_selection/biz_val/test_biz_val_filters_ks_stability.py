"""biz_value test for ``feature_selection.filters.ks_stability_filter``.

The win: a feature whose train/test distributions genuinely shifted is correctly flagged unstable (KS
p-value <= 0.05), while a feature with matching train/test distributions is correctly left alone -- and
dropping the flagged feature avoids feeding a model information it would apply inconsistently between train
and serving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._ks_stability import ks_stability_filter


def test_biz_val_ks_stability_filter_flags_shifted_feature_not_stable_one():
    """Biz val ks stability filter flags shifted feature not stable one."""
    rng = np.random.default_rng(0)
    n = 2000
    train_df = pd.DataFrame(
        {
            "shifted": rng.normal(0, 1, n),
            "stable": rng.normal(0, 1, n),
        }
    )
    test_df = pd.DataFrame(
        {
            "shifted": rng.normal(3, 1, n),  # genuine distributional shift
            "stable": rng.normal(0, 1, n),  # same distribution
        }
    )

    result = ks_stability_filter(train_df, test_df, p_value_threshold=0.05)
    by_col = {row["column"]: row for _, row in result.iterrows()}

    assert by_col["shifted"]["stable"] is False
    assert by_col["shifted"]["p_value"] < 0.05
    assert by_col["stable"]["stable"] is True
    assert by_col["stable"]["p_value"] > 0.05
    assert by_col["shifted"]["p_value"] < by_col["stable"]["p_value"]


def test_ks_stability_filter_all_nan_column_treated_as_stable():
    """Ks stability filter all nan column treated as stable."""
    train_df = pd.DataFrame({"x": [np.nan, np.nan]})
    test_df = pd.DataFrame({"x": [1.0, 2.0]})
    result = ks_stability_filter(train_df, test_df, feature_cols=["x"])
    assert bool(result.iloc[0]["stable"]) is True
