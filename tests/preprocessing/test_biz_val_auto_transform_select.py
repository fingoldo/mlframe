"""biz_value test for ``preprocessing.select_column_transforms``.

The win: a feature with real signal but a heavy-tailed outlier contamination badly confuses a regularized
linear probe fit on the RAW scale (outlier rows dominate the gradient, actually flipping/destroying the
learned decision boundary's rank order below chance), while a rank-based transform (RankGauss) is immune to
outlier magnitude by construction. ``select_column_transforms`` should detect this and recommend RankGauss
(or another outlier-robust transform) over identity for such a column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.preprocessing.auto_transform_select import select_column_transforms


def test_biz_val_auto_transform_select_prefers_robust_transform_for_outlier_contaminated_column():
    rng = np.random.default_rng(0)
    n = 3000
    z = rng.normal(0, 1, n)
    y = (z + rng.normal(0, 0.5, n) > 0).astype(int)

    outlier_mask = rng.random(n) < 0.01
    x_outliers = z.copy()
    x_outliers[outlier_mask] += rng.normal(0, 1e8, int(outlier_mask.sum()))

    df = pd.DataFrame({"outlier_col": x_outliers})
    result = select_column_transforms(df, y, task="classification", n_splits=3, random_state=0)

    assert "outlier_col" in result
    col_result = result["outlier_col"]
    assert col_result["best_transform"] != "identity", (
        f"identity should not be picked for a heavy-outlier column: {col_result['all_scores']}"
    )
    assert col_result["all_scores"]["identity"] < col_result["best_score"] - 0.1, (
        f"the selected transform should clearly beat identity's collapsed AUC: {col_result['all_scores']}"
    )
    assert col_result["best_score"] > 0.8


def test_auto_transform_select_clean_column_picks_a_high_scoring_transform():
    rng = np.random.default_rng(1)
    n = 1500
    z = rng.normal(0, 1, n)
    y = (z + rng.normal(0, 0.3, n) > 0).astype(int)
    df = pd.DataFrame({"clean_col": z})

    result = select_column_transforms(df, y, task="classification", n_splits=3, random_state=1)

    assert "clean_col" in result
    assert result["clean_col"]["best_score"] > 0.85


def test_auto_transform_select_regression_task_runs_and_scores():
    rng = np.random.default_rng(2)
    n = 1000
    x = rng.normal(0, 1, n)
    y = 2.0 * x + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"x": x})

    result = select_column_transforms(df, y, task="regression", n_splits=3, random_state=2)
    assert "x" in result
    assert result["x"]["best_score"] > -1.0  # negative RMSE; loose sanity bound, not a tuned threshold
