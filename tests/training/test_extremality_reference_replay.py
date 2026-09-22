"""The fitted extremality reference must survive into predict, or a served row scores 0.0 on every column.

`row_extreme_topN_score` is default-ON and, with `row_wise_extreme_columns_fit_reference=True` (also the default), the
train path ranks each value against a reference fitted on train. That reference was a local variable: the replay config
carried only the enabled/k knobs, so `_apply_row_wise_extensions` re-ranked inside the serving batch - and a single row
is its own median, giving `abs(0.5 - 0.5) * 2 = 0.0` for every column however extreme the row actually is.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_engineering.row_wise_extremality_reference import fit_extremality_reference
from mlframe.training.core._predict_pre_pipeline import _apply_row_wise_extensions


def _train_frame(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(10, 3, n)})


def _extreme_row() -> pd.DataFrame:
    return pd.DataFrame({"a": [7.5], "b": [-40.0]})


def _config(reference=None) -> dict:
    cfg = {"extreme_columns_enabled": True, "extreme_columns_k": 2, "summary_stats_enabled": False}
    if reference is not None:
        cfg["extreme_columns_reference"] = {c: np.asarray(v, dtype=np.float64).tolist() for c, v in reference.items()}
    return cfg


def test_a_single_extreme_row_scores_zero_without_the_reference(caplog):
    """The regression itself, kept as the contrast: within-batch ranking cannot score one row."""
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core._predict_pre_pipeline"):
        out = _apply_row_wise_extensions(_extreme_row(), _config(), ["a", "b"])
    scores = out[[c for c in out.columns if c.startswith("row_extreme_top")]].to_numpy()
    assert np.allclose(scores, 0.0)
    assert "single-row batch scores 0.0" in caplog.text, "the fallback must announce itself instead of being silent"


def test_the_persisted_reference_scores_the_same_row_as_extreme():
    reference = fit_extremality_reference(_train_frame(), ["a", "b"])
    out = _apply_row_wise_extensions(_extreme_row(), _config(reference), ["a", "b"])
    scores = out[[c for c in out.columns if c.startswith("row_extreme_top")]].to_numpy()
    assert (scores > 0.9).all(), f"a row outside both train ranges must score near 1, got {scores}"


def test_replay_matches_the_fit_time_scores_row_for_row():
    """Scoring the train frame one row at a time through the replay must reproduce scoring it in one batch."""
    from mlframe.feature_engineering.row_wise_extremality import row_wise_top_k_extreme_columns

    train = _train_frame(200)
    reference = fit_extremality_reference(train, ["a", "b"])
    batch = row_wise_top_k_extreme_columns(train, columns=["a", "b"], k=2, reference=reference)
    batch_scores = batch[[c for c in batch.columns if c.endswith("_score")]].to_numpy()

    per_row = []
    for i in range(len(train)):
        out = _apply_row_wise_extensions(train.iloc[[i]].reset_index(drop=True), _config(reference), ["a", "b"])
        per_row.append(out[[f"row_extreme_top{k}_score" for k in (1, 2)]].to_numpy()[0])
    np.testing.assert_allclose(np.asarray(per_row), batch_scores, rtol=1e-12)


def test_a_column_missing_from_the_reference_is_reported(caplog):
    reference = fit_extremality_reference(_train_frame(), ["a"])
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core._predict_pre_pipeline"):
        _apply_row_wise_extensions(_extreme_row(), _config(reference), ["a", "b"])
    assert "no fit-time extremality reference" in caplog.text
