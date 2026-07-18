"""Error-path coverage for train_mlframe_models_suite() entry validation.

Exercises the four explicit raise branches in src/mlframe/training/core/main.py
that previously had no test (E-P1.3). All four cases short-circuit before any
expensive setup, so the tests are sub-second.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlframe.training.core.main import train_mlframe_models_suite


class _DummyExtractor:
    """Minimal stand-in for the FeaturesAndTargetsExtractor protocol.

    Validation raises before the extractor is touched, so we never need a
    real one - we just need *something* non-None where applicable.
    """

    def build_targets(self, *_args, **_kwargs):
        """Build targets."""
        raise AssertionError("validation should fire before extractor is used")


@pytest.mark.fast
def test_raises_typeerror_for_invalid_df_type():
    """df must be pandas/polars DataFrame or .parquet path string."""
    with pytest.raises(TypeError, match="df must be"):
        train_mlframe_models_suite(
            df=[1, 2, 3],  # bare list - not a supported type
            target_name="t",
            model_name="m",
            features_and_targets_extractor=_DummyExtractor(),
        )


@pytest.mark.fast
def test_raises_valueerror_for_non_parquet_path():
    """String paths must end with .parquet."""
    with pytest.raises(ValueError, match=r"\.parquet"):
        train_mlframe_models_suite(
            df="data.csv",
            target_name="t",
            model_name="m",
            features_and_targets_extractor=_DummyExtractor(),
        )


@pytest.mark.fast
def test_raises_valueerror_for_empty_target_name():
    """target_name="" must be rejected."""
    with pytest.raises(ValueError, match="target_name"):
        train_mlframe_models_suite(
            df=pd.DataFrame({"a": [1, 2, 3]}),
            target_name="",
            model_name="m",
            features_and_targets_extractor=_DummyExtractor(),
        )


@pytest.mark.fast
def test_raises_valueerror_for_empty_model_name():
    """model_name="" must be rejected."""
    with pytest.raises(ValueError, match="model_name"):
        train_mlframe_models_suite(
            df=pd.DataFrame({"a": [1, 2, 3]}),
            target_name="t",
            model_name="",
            features_and_targets_extractor=_DummyExtractor(),
        )


@pytest.mark.fast
def test_raises_valueerror_for_missing_extractor():
    """features_and_targets_extractor=None must be rejected."""
    with pytest.raises(ValueError, match="features_and_targets_extractor"):
        train_mlframe_models_suite(
            df=pd.DataFrame({"a": [1, 2, 3]}),
            target_name="t",
            model_name="m",
            features_and_targets_extractor=None,
        )


# E-P1.3 extension: 4 additional error-path cases.


@pytest.mark.fast
def test_raises_for_whitespace_only_target_name():
    """A whitespace-only target_name should be rejected just like ""."""
    with pytest.raises(ValueError):
        train_mlframe_models_suite(
            df=pd.DataFrame({"a": [1, 2, 3]}),
            target_name="   ",
            model_name="m",
            features_and_targets_extractor=_DummyExtractor(),
        )


@pytest.mark.fast
def test_raises_for_none_target_name():
    """target_name=None must raise rather than KeyError downstream."""
    with pytest.raises((TypeError, ValueError)):
        train_mlframe_models_suite(
            df=pd.DataFrame({"a": [1, 2, 3]}),
            target_name=None,
            model_name="m",
            features_and_targets_extractor=_DummyExtractor(),
        )


@pytest.mark.fast
def test_raises_for_nonstring_model_name():
    """model_name must be str, not int/list."""
    with pytest.raises((TypeError, ValueError)):
        train_mlframe_models_suite(
            df=pd.DataFrame({"a": [1, 2, 3]}),
            target_name="t",
            model_name=42,
            features_and_targets_extractor=_DummyExtractor(),
        )


@pytest.mark.fast
def test_raises_for_dict_as_df():
    """A dict (mistaken for a DataFrame) must be rejected with TypeError."""
    with pytest.raises(TypeError):
        train_mlframe_models_suite(
            df={"a": [1, 2, 3]},
            target_name="t",
            model_name="m",
            features_and_targets_extractor=_DummyExtractor(),
        )
