"""The missingness FE stage must not write into the caller's DataFrame.

Before deriving missingness encodings the stage reinstates the fit-entry NaN positions of columns an earlier stage imputed. It did so with
``X[col] = restored_float64``, so whenever that branch ran the caller's frame came back with a changed column: values replaced and an integer
column coerced to float64.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _frame():
    """Numeric columns with NaNs at a few percent, an informative indicator pattern, and a categorical column."""
    rng = np.random.default_rng(0)
    n = 1200
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    miss = rng.random(n) < 0.08
    y = ((a + 0.8 * miss) > 0.4).astype(np.int64)
    a_nan = a.copy()
    a_nan[miss] = np.nan
    X = pd.DataFrame(
        {
            "a": a_nan,
            "b": b,
            "k": rng.integers(0, 4, size=n).astype(np.int64),
            "cat": pd.Categorical(rng.choice(["u", "v", "w"], size=n)),
        }
    )
    return X, y


def test_missingness_stage_does_not_mutate_caller_frame():
    """After a fit with every missingness family on, the caller's frame is value- and dtype-identical to a pre-fit snapshot."""
    X, y = _frame()
    snapshot = X.copy(deep=True)
    MRMR._FIT_CACHE.clear()
    MRMR(
        random_seed=0,
        n_jobs=1,
        verbose=0,
        fe_max_steps=1,
        full_npermutations=3,
        baseline_npermutations=2,
        fe_missingness_indicator_enable=True,
        fe_missingness_count_enable=True,
        fe_missingness_pattern_enable=True,
    ).fit(X, y)
    assert list(X.columns) == list(snapshot.columns)
    assert X.dtypes.equals(snapshot.dtypes), f"dtypes changed: {X.dtypes.to_dict()} vs {snapshot.dtypes.to_dict()}"
    pd.testing.assert_frame_equal(X, snapshot)


def test_restore_branch_does_not_write_into_caller(monkeypatch):
    """Drive the restore branch directly: a fit-entry NaN mask for a column that is NaN-free now must not change the caller's column."""
    from mlframe.feature_selection.filters._mrmr_fit_impl import _fe_stage_cascade_early_b as stage

    X, y = _frame()
    X["a"] = X["a"].fillna(0.0)  # simulate the earlier in-place imputation having erased the NaNs
    mask = np.zeros(len(X), dtype=bool)
    mask[:50] = True
    k_before = X["k"].copy()
    a_before = X["a"].copy()

    seen = {}

    def _spy_indicator(frame, cols, **kw):
        """Record the NaN count the family sees in the restored columns, then append nothing."""
        seen["a_nan"] = int(frame["a"].isna().sum())
        seen["k_nan"] = int(frame["k"].isna().sum())
        return frame, [], []

    import mlframe.feature_selection.filters._missingness_fe as mfe

    monkeypatch.setattr(mfe, "missing_indicator_with_recipes", _spy_indicator)
    est = MRMR(fe_max_steps=0, fe_missingness_indicator_enable=True)
    est.hybrid_orth_features_ = []
    est.mi_greedy_features_ = []
    recipe_dicts = {
        k: {}
        for k in (
            "_kfold_te_pre_recipes", "_binned_agg_pre_recipes", "_count_enc_pre_recipes", "_freq_enc_pre_recipes", "_cat_num_pre_recipes",
            "_miss_ind_pre_recipes", "_miss_cnt_pre_recipes", "_miss_pat_pre_recipes",
            "_ratio_pre_recipes", "_log_ratio_pre_recipes", "_grouped_delta_pre_recipes", "_lagged_diff_pre_recipes",
        )
    }
    out = stage._fe_stage_cascade_early_b(
        est,
        X=X,
        y=y,
        verbose=0,
        fe_max_steps=0,
        _y_np=np.asarray(y),
        _fe_family_on=lambda *a, **k: False,
        _fit_entry_nan_mask={"a": mask, "k": mask},
        _raw_input_cols_pre_fe=list(X.columns),
        **recipe_dicts,
    )
    assert seen.get("a_nan") == 50 and seen.get("k_nan") == 50, f"the missingness family must still see the fit-entry NaNs: {seen}"
    assert out is not None
    pd.testing.assert_series_equal(X["a"], a_before)
    pd.testing.assert_series_equal(X["k"], k_before)
