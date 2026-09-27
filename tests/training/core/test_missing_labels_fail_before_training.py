"""Missing target labels are reported once, before any target trains, naming every affected target.

A regression target with NaN used to fail deep inside its first model fit, after every earlier target had trained for
nothing; a classification target failed in the extractor with a different message.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.core._target_labels import label_mask, missing_label_counts


@pytest.mark.parametrize(
    "values",
    [
        np.array([1.0, np.nan, 2.0]),
        pd.Series([1, None, 2], dtype="Int64"),
        pl.Series([1, None, 2]),
        pl.Series([1.0, float("nan"), 2.0]),
        np.array([["a"], [None], ["b"]], dtype=object),
    ],
)
def test_every_kind_of_missing_label_is_seen(values):
    """NaN, None and pandas NA all count as missing labels."""
    assert label_mask(values).tolist() == [True, False, True]


def test_fully_labelled_targets_cost_nothing_and_return_none():
    """A target with no missing labels returns None, so callers skip the mask."""
    assert label_mask(np.arange(5)) is None
    assert label_mask(np.array([0.5, 1.5])) is None


def test_a_2d_row_is_labelled_only_when_every_column_is():
    """A multi-output row counts as labelled only when none of its columns is missing."""
    assert label_mask(np.array([[1.0, 2.0], [1.0, np.nan]])).tolist() == [True, False]


def test_counts_name_each_target():
    """Missing-label counts are keyed by target type and name, and only targets with missing labels appear."""
    counts = missing_label_counts({"regression": {"a": np.array([1.0, np.nan]), "b": np.array([1.0, 2.0])}})
    assert counts == {("regression", "a"): (1, 2)}


def test_the_suite_fails_before_training_any_target(tmp_path, monkeypatch):
    """A suite whose target has missing labels fails before any model trains, not partway through."""
    from mlframe.training.configs import OutputConfig
    from mlframe.training.core import _phase_runners
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f": rng.normal(size=200), "clean": rng.normal(size=200), "gappy": rng.normal(size=200)})
    df.loc[:9, "gappy"] = np.nan
    monkeypatch.setattr(_phase_runners, "_train_one_target", lambda *a, **k: pytest.fail("a target trained before the check"))
    with pytest.raises(ValueError, match=r"regression/gappy: target contains 10 NaN"):
        train_mlframe_models_suite(
            df=df, target_name="t", model_name="m",
            features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(regression_targets=["clean", "gappy"]),
            mlframe_models=["ridge"], use_ordinary_models=True, use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"), verbose=0,
        )
