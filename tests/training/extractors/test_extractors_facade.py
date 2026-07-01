"""Wave 10e monolith-split sensor for ``mlframe.training.extractors``."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def parent_module():
    from mlframe.training import extractors
    return extractors


@pytest.fixture(scope="module")
def siblings():
    from mlframe.training.extractors import (
        _extractors_dtype_helpers,
        _extractors_showcase,
        _extractors_simple,
    )
    return {
        "dtype": _extractors_dtype_helpers,
        "showcase": _extractors_showcase,
        "simple": _extractors_simple,
    }


def test_dtype_helpers_identity(parent_module, siblings):
    dh = siblings["dtype"]
    assert parent_module.get_dataframe_info is dh.get_dataframe_info
    assert parent_module.intize_targets is dh.intize_targets
    assert parent_module.get_sample_weights_by_recency is dh.get_sample_weights_by_recency


def test_showcase_identity(parent_module, siblings):
    assert parent_module.showcase_features_and_targets is siblings["showcase"].showcase_features_and_targets


def test_simple_extractor_identity(parent_module, siblings):
    assert parent_module.SimpleFeaturesAndTargetsExtractor is siblings["simple"].SimpleFeaturesAndTargetsExtractor


def test_subclass_isinstance_preserved(parent_module):
    e = parent_module.SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    assert isinstance(e, parent_module.FeaturesAndTargetsExtractor)


def test_facade_loc_budget(parent_module):
    path = Path(parent_module.__file__)
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    assert n_lines <= 600, f"extractors.py facade is {n_lines} LOC, expected <= 600"


def test_smoke_intize_targets_round_trip(parent_module):
    import numpy as np
    targets = {"y": np.array([0, 1, 2, 200], dtype=np.int64)}
    parent_module.intize_targets(targets)
    # Range 0..200 -> requires int16 (since int8 max is 127); test the no-wrap promotion.
    assert targets["y"].dtype == np.dtype(np.int16)
    assert int(targets["y"].max()) == 200


def test_smoke_recency_weights_finite(parent_module):
    import numpy as np
    import pandas as pd
    dates = pd.Series(pd.date_range("2020-01-01", periods=10, freq="D"))
    w = parent_module.get_sample_weights_by_recency(dates)
    assert np.isfinite(w).all()
    assert len(w) == 10
