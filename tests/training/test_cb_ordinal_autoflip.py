"""Regression test for the iter#36 auto-flip:
when CatBoost is in mlframe_models AND categorical_encoding='ordinal'
AND the input frame carries categorical/string columns, the pipeline must
auto-flip skip_categorical_encoding=True so CB keeps native cat-handling
and text_features stay string-typed.

Pre-fix the same configuration silently let the ordinal encoder turn
text columns into ints, then CatBoost rejected them at fit time with
``Invalid type for text_feature ... must have string type``.

The fix lives in mlframe.training.core._phase_helpers._phase_fit_pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.configs import (
    PreprocessingBackendConfig,
    PreprocessingConfig,
    FeatureTypesConfig,
)


def test_cb_ordinal_autoflip_with_string_categoricals():
    """Build a tiny pandas frame with a string-categorical column, pass it
    through ``_phase_fit_pipeline`` with CB in models + categorical_encoding
    'ordinal', and verify the returned pipeline_config has
    ``skip_categorical_encoding=True``."""
    from mlframe.training.core._phase_helpers import _phase_fit_pipeline

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "x0": rng.normal(size=n).astype("float32"),
            "x1": rng.normal(size=n).astype("float32"),
            "cat_low": np.array(["A", "B", "C", "D", "E"], dtype=object)[rng.integers(0, 5, n)],
        }
    )

    pipeline_config = PreprocessingBackendConfig(
        categorical_encoding="ordinal",
        skip_categorical_encoding=False,
    )

    _result = _phase_fit_pipeline(
        train_df=df.copy(),
        val_df=None,
        test_df=None,
        mlframe_models=["cb"],
        pipeline_config=pipeline_config,
        preprocessing_config=PreprocessingConfig(),
        feature_types_config=FeatureTypesConfig(),
        preprocessing_extensions=None,
        metadata={},
        verbose=False,
        target_by_type=None,
    )
    pipeline_config_out = _result[13]

    assert pipeline_config_out.skip_categorical_encoding is True, "CB + ordinal + string-categoricals must auto-flip skip_categorical_encoding"


def test_no_autoflip_when_no_cb():
    """Same setup but no CB in models -> the flip should NOT happen (linear
    models do need encoded inputs)."""
    from mlframe.training.core._phase_helpers import _phase_fit_pipeline

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "x0": rng.normal(size=n).astype("float32"),
            "cat_low": np.array(["A", "B", "C"], dtype=object)[rng.integers(0, 3, n)],
        }
    )

    pipeline_config = PreprocessingBackendConfig(
        categorical_encoding="ordinal",
        skip_categorical_encoding=False,
    )

    _result = _phase_fit_pipeline(
        train_df=df.copy(),
        val_df=None,
        test_df=None,
        mlframe_models=["linear", "ridge"],
        pipeline_config=pipeline_config,
        preprocessing_config=PreprocessingConfig(),
        feature_types_config=FeatureTypesConfig(),
        preprocessing_extensions=None,
        metadata={},
        verbose=False,
        target_by_type=None,
    )
    pipeline_config_out = _result[13]

    assert pipeline_config_out.skip_categorical_encoding is False, "Without CB in models, sklearn linear models need encoded cats - no autoflip expected"


def test_no_autoflip_when_no_cats():
    """CB present but no categorical columns -> no auto-flip needed."""
    from mlframe.training.core._phase_helpers import _phase_fit_pipeline

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "x0": rng.normal(size=n).astype("float32"),
            "x1": rng.normal(size=n).astype("float32"),
        }
    )

    pipeline_config = PreprocessingBackendConfig(
        categorical_encoding="ordinal",
        skip_categorical_encoding=False,
    )

    _result = _phase_fit_pipeline(
        train_df=df.copy(),
        val_df=None,
        test_df=None,
        mlframe_models=["cb"],
        pipeline_config=pipeline_config,
        preprocessing_config=PreprocessingConfig(),
        feature_types_config=FeatureTypesConfig(),
        preprocessing_extensions=None,
        metadata={},
        verbose=False,
        target_by_type=None,
    )
    pipeline_config_out = _result[13]

    assert pipeline_config_out.skip_categorical_encoding is False
