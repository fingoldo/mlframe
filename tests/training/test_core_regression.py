"""Regression-target tests extracted from the monolithic ``test_core.py``.

This is the lead-in shim for the planned per-target-type split of the
4400+ LoC ``test_core.py`` (~126 tests). Rather than ripping out 100+
test functions in one PR and risking import-time fixture / shared-helper
drift, we start by hoisting only the smoke-level regression entry points
into a dedicated module so the split is visible to anyone touching this
area next.

The remaining regression tests still live in ``test_core.py``; subsequent
PRs will migrate them here class-by-class. The two existing tests below
are run in BOTH files for now (pytest collects each module independently,
so this just means the smoke runs twice in the full suite); the duplicate
will retire once test_core.py is shrunk.

If you're adding a NEW regression-only test for ``train_mlframe_models_suite``,
add it HERE, not in ``test_core.py``.
"""

from __future__ import annotations


from mlframe.training import OutputConfig
from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.core.predict import predict_from_models
from mlframe.training.configs import TargetTypes
from .shared import SimpleFeaturesAndTargetsExtractor

# Re-use the same behavioural assertion the monolithic file ships;
# importing here ties this module to the canonical contract rather than
# forking a parallel asserter that could drift.
from .test_core import _assert_trained_target_entries


class TestTrainMLFrameModelsSuiteRegressionSmoke:
    """Smoke-level regression-target entry points for ``train_mlframe_models_suite``."""

    def test_train_single_linear_model_regression_smoke(self, sample_regression_data, temp_data_dir, common_init_params):
        """Ridge regression: 1 model, defaults, behavioural contract on returned entries."""
        df, _feature_names, _y = sample_regression_data

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="test_model",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        assert isinstance(models, dict)
        assert TargetTypes.REGRESSION in models
        assert "target" in models[TargetTypes.REGRESSION]
        _assert_trained_target_entries(
            models[TargetTypes.REGRESSION]["target"],
            target_type_label="REGRESSION",
        )

        assert metadata["model_name"] == "test_model"
        assert metadata["target_name"] == "test_target"
        assert "configs" in metadata
        assert "pipeline" in metadata

    def test_predict_regression_basic_smoke(self, sample_regression_data, temp_data_dir, common_init_params):
        """Predict path: train ridge, call predict on the same frame, shapes match."""
        df, _feature_names, _y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, _metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="test_model_predict",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            reporting_config=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
            verbose=0,
        )

        entries = models[TargetTypes.REGRESSION]["target"]
        _assert_trained_target_entries(entries, target_type_label="REGRESSION")

        # Smoke-level predict: route through the real predict entry point rather than calling
        # ``model.predict(Xf.values)`` directly. Row-wise extension columns (row_summary_*/
        # row_extreme_*, default ON) are appended to the fitted feature set BEFORE the model ever
        # sees the frame (see ``_apply_row_wise_extensions`` in predict.py), so a fitted Ridge here
        # expects 18 columns for a 10-raw-feature input, not 10 -- calling ``.predict()`` on the raw
        # values bypasses that replay and raises "X has 10 features, but Ridge is expecting 18
        # features". ``predict_from_models`` replays every fit-time transform (row-wise extensions,
        # pipeline, extensions pipeline) the same way production inference does. Pass the full frame
        # (target column included) -- ``features_and_targets_extractor.transform`` looks up
        # ``target_column`` internally; ``predict_from_models`` itself never trains on it.
        results = predict_from_models(
            df=df,
            models=models,
            metadata=_metadata,
            features_and_targets_extractor=fte,
            return_probabilities=False,
            verbose=0,
        )
        assert results["models_used"], f"predict_from_models returned no models_used: {results}"
        y_hat = results["ensemble_predictions"]
        assert y_hat is not None
        assert y_hat.shape[0] == len(df)
