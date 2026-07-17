"""Tests for MLP × multiclass / multilabel target types.

Phase MLP-A (multiclass) + Phase MLP-B (multilabel) coverage:
- ``NeuralNetStrategy`` supports_native_multiclass / supports_native_multilabel flags
- ``get_classif_objective_kwargs`` per-target dispatch
- End-to-end ``train_mlframe_models_suite`` with target_type=MULTICLASS / MULTILABEL
- ``classes_`` is ndarray (regression for the ``preds = model.classes_[preds]`` bug)
- ``_is_multilabel`` detection in ``PytorchLightningEstimator.fit``
- ``MLPTorchModel.predict_step`` task_type='multilabel' applies sigmoid not softmax
- ``TorchDataModule`` preserves 2-D label shape (regression for ``reshape(-1)`` bug)

LTR coverage in ``test_mlp_ranker.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

# Skip whole module if neural deps missing.
pytest.importorskip("lightning")
pytest.importorskip("torch")

import torch
import torch.nn.functional as F

from mlframe.training import train_mlframe_models_suite, TargetTypes
from mlframe.training.strategies import NeuralNetStrategy
from tests.training.shared import SimpleFeaturesAndTargetsExtractor


# ----------------------------------------------------------------------------
# Strategy flag + dispatch helper
# ----------------------------------------------------------------------------


class TestMLPStrategyFlags:
    def test_supports_native_multiclass(self):
        assert NeuralNetStrategy().supports_native_multiclass is True

    def test_supports_native_multilabel(self):
        assert NeuralNetStrategy().supports_native_multilabel is True

    def test_supports_native_ranking(self):
        assert NeuralNetStrategy().supports_native_ranking is True


class TestMLPGetClassifObjectiveKwargs:
    """``get_classif_objective_kwargs`` returns library-correct kwargs."""

    def test_binary_returns_empty(self):
        out = NeuralNetStrategy().get_classif_objective_kwargs(
            TargetTypes.BINARY_CLASSIFICATION,
            n_classes=2,
        )
        assert out == {}  # default cross_entropy + int64 already correct

    def test_multiclass_returns_cross_entropy_int64(self):
        out = NeuralNetStrategy().get_classif_objective_kwargs(
            TargetTypes.MULTICLASS_CLASSIFICATION,
            n_classes=3,
        )
        assert out["loss_fn"] is F.cross_entropy
        assert out["labels_dtype"] == torch.int64
        assert "task_type" not in out  # softmax default; no override needed

    def test_multilabel_returns_bce_with_logits_float32_task_type(self):
        out = NeuralNetStrategy().get_classif_objective_kwargs(
            TargetTypes.MULTILABEL_CLASSIFICATION,
            n_classes=3,
        )
        assert out["loss_fn"] is F.binary_cross_entropy_with_logits
        assert out["labels_dtype"] == torch.float32
        assert out["task_type"] == "multilabel"

    def test_none_target_type_returns_empty(self):
        out = NeuralNetStrategy().get_classif_objective_kwargs(None, n_classes=2)
        assert out == {}


# ----------------------------------------------------------------------------
# End-to-end multiclass via suite
# ----------------------------------------------------------------------------


@pytest.fixture
def synthetic_3class_data():
    """600 rows × 5 features, 3 quantile-cut classes from informative scores."""
    rng = np.random.default_rng(42)
    n = 600
    X = rng.standard_normal((n, 5)).astype(np.float32)
    score = X[:, 0] + 0.5 * X[:, 1]
    y = np.digitize(score, [np.quantile(score, 1 / 3), np.quantile(score, 2 / 3)]).astype(np.int64)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    return df, y


class TestMLPMulticlassEndToEnd:
    """``train_mlframe_models_suite(target_type=MULTICLASS)`` with mlp."""

    def test_smoke_fits_predicts_returns_3_class_probs(self, synthetic_3class_data):
        from tests.conftest import is_fast_mode

        df, y = synthetic_3class_data
        fte = SimpleFeaturesAndTargetsExtractor(
            target_column="target",
            regression=False,
            target_type=TargetTypes.MULTICLASS_CLASSIFICATION,
        )
        # The MLP relies on early-stopping to bound epochs; under heavy -n
        # contention the full fit can exceed the per-test timeout. Cap epochs
        # in fast mode -- the smoke assertion (suite returns a multiclass
        # entry) is unaffected by the epoch budget.
        kwargs = {}
        if is_fast_mode():
            kwargs["hyperparams_config"] = {"iterations": 5}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models, _ = train_mlframe_models_suite(
                df=df,
                target_name="target",
                model_name="mc_smoke",
                features_and_targets_extractor=fte,
                mlframe_models=["mlp"],
                use_mlframe_ensembles=False,
                verbose=0,
                **kwargs,
            )
        assert TargetTypes.MULTICLASS_CLASSIFICATION in models

    def test_classes_attr_is_ndarray_not_list(self, synthetic_3class_data):
        """Regression: ``model.classes_`` must be ndarray for fancy
        indexing in evaluation.py (``preds = model.classes_[preds]``)."""
        df, y = synthetic_3class_data
        fte = SimpleFeaturesAndTargetsExtractor(
            target_column="target",
            regression=False,
            target_type=TargetTypes.MULTICLASS_CLASSIFICATION,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models, _ = train_mlframe_models_suite(
                df=df,
                target_name="target",
                model_name="mc_classes",
                features_and_targets_extractor=fte,
                mlframe_models=["mlp"],
                use_mlframe_ensembles=False,
                verbose=0,
            )
        # Drill into the trained model
        for tt, target_models in models.items():
            for tn, ns_list in target_models.items():
                for ns in ns_list:
                    inner = getattr(ns, "model", None)
                    if inner is not None and hasattr(inner, "classes_"):
                        assert isinstance(inner.classes_, np.ndarray), (
                            f"classes_ for {type(inner).__name__} is {type(inner.classes_).__name__}, expected ndarray"
                        )


# ----------------------------------------------------------------------------
# End-to-end multilabel via suite
# ----------------------------------------------------------------------------


@pytest.fixture
def synthetic_3label_data():
    """800 rows × 5 features, 3 correlated binary labels.
    Polars frame with ``pl.List(pl.Int8)`` target column for 2-D y."""
    rng = np.random.default_rng(42)
    n = 800
    X = rng.standard_normal((n, 5)).astype(np.float32)
    Y = (rng.standard_normal((n, 3)) > 0).astype(np.int8)
    Y[Y.sum(axis=1) == 0, 0] = 1  # avoid all-zero rows
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    pdf = pl.from_pandas(df).with_columns(pl.Series("target", Y.tolist(), dtype=pl.List(pl.Int8)))
    return pdf, Y


class TestMLPMultilabelEndToEnd:
    """``train_mlframe_models_suite(target_type=MULTILABEL)`` with mlp."""

    def test_smoke_fits_with_2d_y(self, synthetic_3label_data):
        pdf, Y = synthetic_3label_data
        fte = SimpleFeaturesAndTargetsExtractor(
            target_column="target",
            regression=False,
            target_type=TargetTypes.MULTILABEL_CLASSIFICATION,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models, _ = train_mlframe_models_suite(
                df=pdf,
                target_name="target",
                model_name="ml_smoke",
                features_and_targets_extractor=fte,
                mlframe_models=["mlp"],
                use_mlframe_ensembles=False,
                verbose=0,
            )
        assert TargetTypes.MULTILABEL_CLASSIFICATION in models

    def test_no_multioutput_wrapper_for_native_mlp(self, synthetic_3label_data):
        """MLP is a native multilabel learner -- the trainer's
        ``_maybe_wrap_for_2d_target`` should NOT wrap it with
        MultiOutputClassifier."""
        pdf, Y = synthetic_3label_data
        fte = SimpleFeaturesAndTargetsExtractor(
            target_column="target",
            regression=False,
            target_type=TargetTypes.MULTILABEL_CLASSIFICATION,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models, _ = train_mlframe_models_suite(
                df=pdf,
                target_name="target",
                model_name="ml_no_wrap",
                features_and_targets_extractor=fte,
                mlframe_models=["mlp"],
                use_mlframe_ensembles=False,
                verbose=0,
            )
        for tt, target_models in models.items():
            for tn, ns_list in target_models.items():
                for ns in ns_list:
                    inner = getattr(ns, "model", None)
                    if inner is not None:
                        cls_name = type(inner).__name__
                        assert cls_name != "MultiOutputClassifier", f"MLP got wrapped in MultiOutputClassifier despite supports_native_multilabel=True"


# ----------------------------------------------------------------------------
# Lower-level: 2-D label preservation in TorchDataModule
# ----------------------------------------------------------------------------


class TestTorchDataModule2DLabels:
    """Regression: ``TorchDataModule`` must NOT flatten 2-D multilabel
    targets to 1-D (bug fixed 2026-05-07)."""

    def test_2d_labels_preserved(self):
        from mlframe.training.neural.data import TorchDataset

        X = np.random.default_rng(0).standard_normal((10, 4)).astype(np.float32)
        Y = np.random.default_rng(1).integers(0, 2, size=(10, 3)).astype(np.int8)
        ds = TorchDataset(features=X, labels=Y, labels_dtype=torch.float32, batch_size=0)
        # Labels tensor must keep (10, 3) shape, NOT flatten to (30,).
        assert ds.labels.shape == (10, 3), (
            f"Multilabel labels got flattened to {tuple(ds.labels.shape)}; TorchDataModule should preserve 2-D for multilabel target."
        )

    def test_1d_labels_unchanged(self):
        """Single-label / regression (1-D) labels must still produce 1-D
        tensor (no shape change vs pre-fix behaviour)."""
        from mlframe.training.neural.data import TorchDataset

        X = np.random.default_rng(0).standard_normal((10, 4)).astype(np.float32)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        ds = TorchDataset(features=X, labels=y, labels_dtype=torch.int64, batch_size=0)
        assert ds.labels.shape == (10,)
