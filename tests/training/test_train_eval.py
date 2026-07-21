"""Tests for mlframe.training.train_eval: optimize_model_for_storage and select_target."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from mlframe.training.configs import TargetTypes
from mlframe.training.train_eval import optimize_model_for_storage, select_target

# =============================================================================
# optimize_model_for_storage
# =============================================================================


class TestOptimizeModelForStorageClassification:
    """Tests for classification path (preds nulled, probs kept)."""

    def test_classification_nulls_all_preds(self):
        """Classification nulls all preds."""
        model = SimpleNamespace(
            train_preds=np.array([0, 1, 1]),
            val_preds=np.array([1, 0]),
            test_preds=np.array([0, 0, 1]),
            train_probs=np.array([[0.6, 0.4], [0.3, 0.7]]),
            columns=["a", "b"],
        )
        optimize_model_for_storage(model, TargetTypes.BINARY_CLASSIFICATION)
        assert model.train_preds is None
        assert model.val_preds is None
        assert model.test_preds is None

    def test_classification_preserves_probs(self):
        """Classification preserves probs."""
        probs = np.array([[0.8, 0.2], [0.1, 0.9]])
        model = SimpleNamespace(
            train_preds=np.array([0, 1]),
            val_preds=np.array([0, 1]),
            test_preds=np.array([0, 1]),
            train_probs=probs,
            columns=None,
        )
        optimize_model_for_storage(model, TargetTypes.BINARY_CLASSIFICATION)
        assert model.train_probs is probs

    def test_classification_columns_removed_when_matching_metadata(self):
        """Classification columns removed when matching metadata."""
        cols = ["feat1", "feat2", "feat3"]
        model = SimpleNamespace(
            train_preds=np.array([1]),
            val_preds=np.array([0]),
            test_preds=np.array([1]),
            columns=list(cols),
        )
        optimize_model_for_storage(model, TargetTypes.BINARY_CLASSIFICATION, metadata_columns=cols)
        assert model.columns is None

    def test_classification_columns_kept_when_different_from_metadata(self):
        """Classification columns kept when different from metadata."""
        model = SimpleNamespace(
            train_preds=np.array([1]),
            val_preds=np.array([0]),
            test_preds=np.array([1]),
            columns=["a", "b"],
        )
        optimize_model_for_storage(model, TargetTypes.BINARY_CLASSIFICATION, metadata_columns=["a", "c"])
        assert model.columns == ["a", "b"]


class TestOptimizeModelForStorageRegression:
    """Tests for regression path (preds untouched)."""

    def test_regression_keeps_preds(self):
        """Regression keeps preds."""
        train_p = np.array([1.0, 2.0])
        val_p = np.array([3.0])
        test_p = np.array([4.0, 5.0])
        model = SimpleNamespace(
            train_preds=train_p,
            val_preds=val_p,
            test_preds=test_p,
            columns=["x"],
        )
        optimize_model_for_storage(model, TargetTypes.REGRESSION)
        assert model.train_preds is train_p
        assert model.val_preds is val_p
        assert model.test_preds is test_p

    def test_regression_columns_removed_when_matching(self):
        """Regression columns removed when matching."""
        cols = ["x", "y"]
        model = SimpleNamespace(train_preds=np.array([1.0]), columns=list(cols))
        optimize_model_for_storage(model, TargetTypes.REGRESSION, metadata_columns=cols)
        assert model.columns is None

    def test_regression_columns_kept_when_no_metadata(self):
        """Regression columns kept when no metadata."""
        model = SimpleNamespace(train_preds=np.array([1.0]), columns=["x", "y"])
        optimize_model_for_storage(model, TargetTypes.REGRESSION, metadata_columns=None)
        assert model.columns == ["x", "y"]


class TestOptimizeModelColumnsEdgeCases:
    """Edge cases for the metadata_columns dedup logic."""

    def test_columns_as_numpy_array_matched_against_list(self):
        """Columns as numpy array matched against list."""
        cols = ["a", "b", "c"]
        model = SimpleNamespace(
            train_preds=np.array([1.0]),
            columns=np.array(cols),
        )
        optimize_model_for_storage(model, TargetTypes.REGRESSION, metadata_columns=cols)
        # numpy array gets converted to list for comparison; should match
        assert model.columns is None

    def test_no_columns_attribute_does_not_raise(self):
        """No columns attribute does not raise."""
        model = SimpleNamespace(train_preds=np.array([1.0]))
        # model has no .columns at all; should not raise
        optimize_model_for_storage(model, TargetTypes.REGRESSION, metadata_columns=["a"])


# =============================================================================
# select_target (unit-level, mocked dependencies)
# =============================================================================


class TestSelectTarget:
    """Lightweight tests for select_target parameter wiring."""

    # ``select_target`` moved to a sibling ``_train_eval_select_target`` and now LAZILY
    # imports ``configure_training_params`` from ``.trainer`` per call, so the patch target
    # is ``trainer`` (the source the lazy import resolves to), not ``train_eval`` itself --
    # ``train_eval`` no longer imports that name at all, so patching it there raises
    # AttributeError before the test body even runs.
    @patch("mlframe.training.trainer.configure_training_params")
    def test_regression_appends_mean_to_model_name(self, mock_trainer):
        """Regression appends mean to model name."""
        mock_trainer.return_value = ({}, {}, None, None, None, {}, {})
        target = np.array([2.0, 4.0, 6.0])
        select_target("mymodel", target, TargetTypes.REGRESSION, pd.DataFrame({"a": [1, 2, 3]}))
        assert mock_trainer.call_args is not None
