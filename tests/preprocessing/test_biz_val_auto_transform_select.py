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
    """Auto transform select prefers robust transform for outlier contaminated column."""
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
    assert col_result["best_transform"] != "identity", f"identity should not be picked for a heavy-outlier column: {col_result['all_scores']}"
    assert (
        col_result["all_scores"]["identity"] < col_result["best_score"] - 0.1
    ), f"the selected transform should clearly beat identity's collapsed AUC: {col_result['all_scores']}"
    assert col_result["best_score"] > 0.8


def test_auto_transform_select_clean_column_picks_a_high_scoring_transform():
    """Auto transform select clean column picks a high scoring transform."""
    rng = np.random.default_rng(1)
    n = 1500
    z = rng.normal(0, 1, n)
    y = (z + rng.normal(0, 0.3, n) > 0).astype(int)
    df = pd.DataFrame({"clean_col": z})

    result = select_column_transforms(df, y, task="classification", n_splits=3, random_state=1)

    assert "clean_col" in result
    assert result["clean_col"]["best_score"] > 0.85


def test_biz_val_select_column_transforms_multivariate_probe_catches_interaction_only_signal():
    """The win: a column carries signal ONLY as an interaction with a correlated context column
    (``y`` depends on ``sign(x * context)``), plus ``x`` itself is heavily outlier-contaminated. A plain
    univariate probe (fit on the transformed column alone) sees no marginal signal for ANY candidate
    transform -- AUC stays at chance regardless of transform choice, so the column looks worthless and every
    transform ties. The opt-in multivariate probe (small GBM on transformed column + context + their
    product) recovers the true interaction signal and correctly recommends a strong transform.
    """
    rng = np.random.default_rng(0)
    n = 4000
    context = rng.normal(0, 1, n)
    x = rng.normal(0, 1, n)
    outlier_mask = rng.random(n) < 0.02
    x_outliers = x.copy()
    x_outliers[outlier_mask] += rng.normal(0, 1e6, int(outlier_mask.sum()))

    interaction_logit = x * context
    y = (interaction_logit + rng.normal(0, 0.3, n) > 0).astype(int)

    df = pd.DataFrame({"x": x_outliers, "context": context})

    uni_result = select_column_transforms(df, y, columns=["x"], task="classification", n_splits=3, random_state=0)
    mv_result = select_column_transforms(df, y, columns=["x"], task="classification", n_splits=3, random_state=0, multivariate_probe=True, n_context_features=1)

    assert "x" in uni_result and "x" in mv_result
    uni_best = uni_result["x"]["best_score"]
    mv_best = mv_result["x"]["best_score"]

    assert "probe_mode" not in uni_result["x"], "default univariate output must not carry the new opt-in keys"
    assert mv_result["x"]["probe_mode"] == "multivariate"
    assert mv_result["x"]["context_columns"] == ["context"]

    assert uni_best < 0.6, f"univariate probe should see near-chance AUC for a pure interaction signal: {uni_result['x']['all_scores']}"
    assert mv_best > 0.85, f"multivariate probe should recover the interaction signal: {mv_result['x']['all_scores']}"
    assert mv_best - uni_best > 0.3, "multivariate probe should clearly beat the univariate probe on an interaction-only column"


def test_select_column_transforms_multivariate_probe_default_off_is_bit_identical():
    """Omitting the new params must reproduce the exact pre-extension univariate output (opt-in guarantee)."""
    rng = np.random.default_rng(3)
    n = 800
    z = rng.normal(0, 1, n)
    y = (z + rng.normal(0, 0.4, n) > 0).astype(int)
    df = pd.DataFrame({"a": z, "b": rng.normal(0, 1, n)})

    baseline = select_column_transforms(df, y, task="classification", n_splits=3, random_state=7)
    explicit_default = select_column_transforms(df, y, task="classification", n_splits=3, random_state=7, multivariate_probe=False)

    assert baseline == explicit_default, "omitting the new multivariate params must reproduce bit-identical output"
    for col in baseline:
        assert set(baseline[col].keys()) == {"best_transform", "best_score", "all_scores"}, "no new keys leak into the default (univariate-only) output"


def test_auto_transform_select_regression_task_runs_and_scores():
    """Auto transform select regression task runs and scores."""
    rng = np.random.default_rng(2)
    n = 1000
    x = rng.normal(0, 1, n)
    y = 2.0 * x + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"x": x})

    result = select_column_transforms(df, y, task="regression", n_splits=3, random_state=2)
    assert "x" in result
    assert result["x"]["best_score"] > -1.0  # negative RMSE; loose sanity bound, not a tuned threshold
