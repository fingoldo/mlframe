"""Regression test for the pre-split numeric null-fill bias.

Pre-fix: ``preprocess_dataframe`` filled numeric nulls with ``config.fillna_value`` (default 0.0 in the suite-level config) on the full pre-split frame. A downstream
``SimpleImputer.fit`` then learned mean/median over the zero-padded rows, producing a biased statistic (e.g. mean ~= 70 on the construction below instead of the true 100).

Post-fix: pre-split fill is removed, the imputer sees real NaN distribution and learns the unbiased statistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from mlframe.training.configs import PreprocessingConfig
from mlframe.training.preprocessing import preprocess_dataframe


def _make_biased_frame() -> pd.DataFrame:
    """70 rows with value 100.0 plus 30 null rows in column ``x``.

    True non-null mean is exactly 100.0. A zero-fill before imputation yields 70.0; this is the bias the fix removes.
    """
    values = np.concatenate([np.full(70, 100.0, dtype=np.float64), np.full(30, np.nan, dtype=np.float64)])
    return pd.DataFrame({"x": values})


def test_preprocess_does_not_pre_fill_nulls_so_imputer_mean_is_unbiased():
    """preprocess_dataframe must leave numeric nulls untouched pre-split, or SimpleImputer learns a mean biased toward zero-padding."""
    df = _make_biased_frame()
    assert df["x"].isna().sum() == 30, "fixture invariant: 30 null rows in column x"

    # ``fillna_value=0.0`` mirrors the suite-level default (mlframe.training.__init__.py and core/main.py both wire PreprocessingConfig(fillna_value=0.0)). The
    # value is intentionally kept in the config surface; what changes is that preprocess_dataframe no longer applies it pre-split.
    config = PreprocessingConfig(fillna_value=0.0, fix_infinities=True, ensure_float32_dtypes=False, remove_constant_columns=False)
    processed = preprocess_dataframe(df, config, verbose=0)

    assert processed["x"].isna().sum() == 30, (
        "preprocess_dataframe must NOT pre-fill numeric nulls; otherwise the downstream SimpleImputer.fit learns a biased mean over zero-padded rows"
    )

    imputer = SimpleImputer()
    imputer.fit(processed[["x"]])
    learned_mean = float(imputer.statistics_[0])
    assert learned_mean == pytest.approx(100.0, abs=1e-6), (
        f"SimpleImputer must learn the true non-null mean (100.0); got {learned_mean!r}. Pre-fix value was ~70.0 (mean over 70x100 + 30x0)."
    )


def test_preprocess_normalises_inf_to_nan_for_downstream_imputer():
    """preprocess_dataframe converts +/-inf to NaN so the downstream imputer treats them as missing, not as outlier values."""
    # Mixing inf with nulls: post-fix inf becomes NaN so the imputer treats both as missing and the non-inf/non-null distribution drives the learned statistic.
    values = np.concatenate(
        [
            np.full(70, 100.0, dtype=np.float64),
            np.full(20, np.nan, dtype=np.float64),
            np.array([np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf], dtype=np.float64),
        ]
    )
    df = pd.DataFrame({"x": values})
    config = PreprocessingConfig(fillna_value=0.0, fix_infinities=True, ensure_float32_dtypes=False, remove_constant_columns=False)
    processed = preprocess_dataframe(df, config, verbose=0)

    col = processed["x"].to_numpy()
    assert not np.isinf(col).any(), "inf must be normalised away so the imputer does not propagate it"
    assert int(np.isnan(col).sum()) == 30, "all 20 original nulls plus 10 inf entries must end up as NaN"

    imputer = SimpleImputer()
    imputer.fit(processed[["x"]])
    assert float(imputer.statistics_[0]) == pytest.approx(100.0, abs=1e-6)
