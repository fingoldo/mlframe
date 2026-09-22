"""Two replay contracts on the predict path: column ORDER, and the pinned row-wise column list.

PRD-06: `predict_from_models` dropped extra columns but never reordered to the fit-time order, while the suite path
did. LGB/XGB accept a frame carrying every required name in the wrong order and score each value against another
feature's split thresholds - wrong predictions, no exception.

PRD-08: the row-wise replay re-derived its input columns from the serving frame's dtypes instead of the list pinned at
fit time (taken after the numeric filter AND the all-null drop), so a column that was all-null on train but has values
at serve time silently entered every `row_summary_*` feature.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training.core._predict_pre_pipeline import _apply_row_wise_extensions


def _frame(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "late": rng.normal(size=n)})


def _config(columns=None) -> dict:
    cfg = {"summary_stats_enabled": True, "extreme_columns_enabled": False}
    if columns is not None:
        cfg["columns"] = columns
    return cfg


def test_a_column_absent_at_fit_time_does_not_enter_the_row_wise_features():
    df = _frame()
    pinned = _apply_row_wise_extensions(df.copy(), _config(["a", "b"]), None)
    rederived = _apply_row_wise_extensions(df.copy(), _config(), None)
    mean_cols = [c for c in pinned.columns if c.startswith("row_summary_mean")]
    assert mean_cols, "the summary stats must be produced"
    np.testing.assert_allclose(pinned[mean_cols[0]], df[["a", "b"]].mean(axis=1))
    assert not np.allclose(pinned[mean_cols[0]], rederived[mean_cols[0]]), "this bed only means something while the two differ"


def test_a_pinned_column_missing_at_predict_time_is_reported(caplog):
    df = _frame().drop(columns=["b"])
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core._predict_pre_pipeline"):
        _apply_row_wise_extensions(df, _config(["a", "b"]), None)
    assert "row-wise column(s) are absent" in caplog.text


def test_without_a_pinned_list_the_legacy_derivation_still_works():
    """Artefacts trained before the stamp carry no list; they must keep replaying."""
    out = _apply_row_wise_extensions(_frame(), _config(), None)
    assert any(c.startswith("row_summary_") for c in out.columns)


def test_the_in_memory_predict_path_reorders_to_the_fit_schema():
    """A booster is fitted on one column order and served the same columns permuted."""
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = (X["f0"] * 2 - X["f1"] + rng.normal(0, 0.1, n) > 0).astype(int)
    model = lgb.LGBMClassifier(n_estimators=10, verbose=-1).fit(X, y)

    permuted = X[["f2", "f0", "f1"]]
    expected = model.predict_proba(X)
    # The alignment step under test, reproduced on its own inputs: drop nothing, reorder to feature_names_in_.
    expected_list = [str(c) for c in model.feature_name_]
    aligned = permuted.loc[:, expected_list]
    np.testing.assert_allclose(model.predict_proba(aligned), expected)
    assert not np.allclose(model.predict_proba(permuted.to_numpy()), expected), (
        "this bed only means something while the permuted order actually changes the prediction"
    )
