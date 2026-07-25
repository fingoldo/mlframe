"""Regression test for a bug DISCOVERED while running the training_pipeline.md / core_infra_b.md
home-suite verification (not itself part of either audit report): pyutilz's
``store_params_in_object`` changed its default ``postfix`` from ``""`` to ``"_param_"`` (pyutilz
commit 8128f16, part of pyutilz's own audit fix-wave). Every mlframe class that calls
``store_params_in_object(obj=self, params=...)`` WITHOUT pinning ``postfix=""`` explicitly, then
reads its constructor params back by their BARE name (the established mlframe convention -- see
each class's own class-level type annotations), silently lost every such attribute: the params were
still stored, just under ``<name>_param_`` instead of ``<name>``, so ``self.<name>`` raised
``AttributeError`` the first time any method actually read it (often not at construction time, but
on first ``.fit()``/``.predict()`` call -- e.g. ``PytorchLightningRegressor.model_class`` was only
read inside ``predict()``, so this silently broke MLP predict calls without ever raising at
``__init__`` time).

Grepped every ``store_params_in_object(`` call site under ``src/mlframe/`` (excluding the
MRMR/shap_proxied_fs zone, which is out of scope for this fix-wave) and fixed all 8: calibration/post.py
(BinaryPostCalibrator), estimators/early_stopping.py (EarlyStoppingWrapper),
feature_selection/wrappers/rfecv/__init__.py (RFECV), metrics/_ice_metric.py (ICE),
models/_optimization_search.py (MBHOptimizer), training/callbacks/_callbacks.py (UniversalCallback),
training/extractors/__init__.py (FeaturesAndTargetsExtractor),
training/neural/base/__init__.py (PytorchLightningEstimator),
training/neural/_base_callbacks.py (AggregatingValidationCallback).
"""

from __future__ import annotations

import logging

logging.disable(logging.CRITICAL)


def test_binary_post_calibrator_bare_attrs_set():
    """BinaryPostCalibrator stores params bare, no _param_-suffixed shadow attribute."""
    from sklearn.isotonic import IsotonicRegression

    from mlframe.calibration.post import BinaryPostCalibrator

    obj = BinaryPostCalibrator(calibrator=IsotonicRegression(out_of_bounds="clip"), fit_method_name="fit")
    assert obj.fit_method_name == "fit"
    assert obj.transform_method_name == "transform"
    assert not hasattr(obj, "fit_method_name_param_"), "postfix='' must be honored -- no _param_-suffixed shadow attribute"


def test_early_stopping_wrapper_bare_attrs_set():
    """EarlyStoppingWrapper stores params bare, no _param_-suffixed shadow attribute."""
    from sklearn.linear_model import LogisticRegression

    from mlframe.estimators.early_stopping import EarlyStoppingWrapper

    obj = EarlyStoppingWrapper(base_model=LogisticRegression(), tolerance=0.1)
    assert obj.tolerance == 0.1
    assert not hasattr(obj, "tolerance_param_")


def test_rfecv_bare_attrs_set():
    """RFECV stores params bare, no _param_-suffixed shadow attribute."""
    from sklearn.linear_model import LinearRegression

    from mlframe.feature_selection.wrappers.rfecv import RFECV

    obj = RFECV(estimator=LinearRegression())
    assert hasattr(obj, "estimator")
    assert obj.estimator is not None
    assert not hasattr(obj, "estimator_param_")


def test_ice_metric_bare_attrs_set():
    """ICE stores params bare, no _param_-suffixed shadow attribute."""
    from mlframe.metrics._ice_metric import ICE

    obj = ICE(metric=lambda y_true, y_pred: 0.0, higher_is_better=True, max_arr_size=100)
    assert obj.higher_is_better is True
    assert obj.max_arr_size == 100
    assert not hasattr(obj, "max_arr_size_param_")


def test_mbh_optimizer_bare_attrs_set():
    """MBHOptimizer stores params bare, no _param_-suffixed shadow attribute."""
    import numpy as np

    from mlframe.models._optimization_search import MBHOptimizer

    search_space = np.arange(0, 20)
    obj = MBHOptimizer(search_space=search_space, model_name="ETR", model_params={}, init_num_samples=5, random_state=0)
    assert np.array_equal(obj.search_space, search_space)
    assert not hasattr(obj, "search_space_param_")


def test_universal_callback_bare_attrs_set():
    """UniversalCallback stores params bare, no _param_-suffixed shadow attribute."""
    from mlframe.training.callbacks._callbacks import UniversalCallback

    obj = UniversalCallback(patience=5, monitor_dataset="valid_0", monitor_metric="loss", mode="min")
    assert obj.patience == 5
    assert obj.monitor_metric == "loss"
    assert not hasattr(obj, "patience_param_")


def test_features_and_targets_extractor_bare_attrs_set():
    """The exact bug this discovery started from: SimpleFeaturesAndTargetsExtractor's __init__
    reads self.columns_to_drop immediately after construction."""
    from mlframe.training.extractors._extractors_simple import SimpleFeaturesAndTargetsExtractor

    obj = SimpleFeaturesAndTargetsExtractor(columns_to_drop={"a", "b"})
    assert obj.columns_to_drop == {"a", "b"}
    assert not hasattr(obj, "columns_to_drop_param_")


def test_pytorch_lightning_estimator_bare_attrs_set():
    """The other exact bug this discovery started from: PytorchLightningRegressor.predict() reads
    self.model_class."""
    import torch

    from mlframe.training.neural import MLPTorchModel, PytorchLightningRegressor, TorchDataModule

    obj = PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={"loss_fn": torch.nn.MSELoss(), "learning_rate": 1e-2},
        network_params={"nlayers": 1, "first_layer_num_neurons": 8},
        datamodule_class=TorchDataModule,
        datamodule_params={},
        trainer_params={"max_epochs": 1, "accelerator": "cpu", "logger": False},
    )
    assert obj.model_class is MLPTorchModel
    assert not hasattr(obj, "model_class_param_")


def test_aggregating_validation_callback_bare_attrs_set():
    """AggregatingValidationCallback stores params bare, no _param_-suffixed shadow attribute."""
    from mlframe.training.neural._base_callbacks import AggregatingValidationCallback

    obj = AggregatingValidationCallback(metric_name="my_metric", metric_fcn=lambda y_true, y_pred: 0.0)
    assert obj.metric_name == "my_metric"
    assert not hasattr(obj, "metric_name_param_")
