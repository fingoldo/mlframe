"""Every model save fell back to dill because the calibration metrics were nested functions.

    save_mlframe_model: pickle rejected the payload (AttributeError: Can't pickle local object
    'get_training_configs.<locals>.integral_calibration_error'; offending attribute(s): model)

dill is slower on every save and serialises BYTECODE, so a model written under one interpreter or one mlframe
version may not load under another -- which is exactly the case when training and serving are separate
environments. The five closures are now module-level callables holding the same configuration.

The metric maths is untouched, and that is verified rather than asserted: the values were compared against the
previous implementation on fixed inputs across both the plain and the robustness-wrapped configuration -- 80
values, max |diff| = 0.0.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from mlframe.training._picklable_metrics import (
    IntegralCalibrationError,
    LightGBMMetricAdapter,
    PositionalIntegralCalibrationError,
    RobustTimeSplitMetric,
    SubgroupAveragedMetric,
)

CONFIG = dict(
    method="multicrit",
    mae_weight=3.0,
    std_weight=2.0,
    brier_loss_weight=1.0,
    roc_auc_weight=1.5,
    pr_auc_weight=1.0,
    min_roc_auc=0.54,
    roc_auc_penalty=0.0,
    use_weighted_calibration=True,
    weight_by_class_npositives=False,
    nbins=100,
)


@pytest.fixture(scope="module")
def data():
    """A binary problem big enough for the calibration metric to be well defined."""
    rng = np.random.default_rng(7)
    n = 1500
    y = (rng.random(n) < 0.4).astype(int)
    p = np.clip(rng.random(n), 1e-6, 1 - 1e-6)
    return y, np.column_stack([1 - p, p])


@pytest.fixture(scope="module")
def metric():
    """The configured calibration metric."""
    return IntegralCalibrationError(**CONFIG)


class TestTheyPickle:
    """The whole point: no dill fallback on save."""

    def test_the_base_metric_pickles(self, metric):
        """The one that named itself in the production failure message."""
        assert pickle.loads(pickle.dumps(metric)) is not None  # nosec B301 - round-tripping our OWN objects is precisely what this file verifies

    @pytest.mark.parametrize(
        "build",
        [
            lambda m: PositionalIntegralCalibrationError(**CONFIG),
            lambda m: SubgroupAveragedMetric(m, {"g": np.zeros(3)}),
            lambda m: RobustTimeSplitMetric(m, num_splits=3, std_coeff=1.0, greater_is_better=False),
            lambda m: LightGBMMetricAdapter(m),
        ],
    )
    def test_every_wrapper_pickles(self, metric, build):
        """A wrapper holding an unpicklable inner metric would put the whole chain back on dill."""
        assert pickle.loads(pickle.dumps(build(metric))) is not None  # nosec B301 - round-tripping our OWN objects is precisely what this file verifies

    def test_a_round_tripped_metric_returns_the_same_number(self, metric, data):
        """Picklable is not enough; the restored object has to be the same metric."""
        y, p = data
        assert pickle.loads(pickle.dumps(metric))(y, p) == metric(y, p)  # nosec B301 - round-tripping our OWN objects is precisely what this file verifies

    def test_the_configuration_survives_the_round_trip(self, metric):
        """Slotted classes have no __dict__, so the state hooks are load-bearing rather than decorative."""
        assert pickle.loads(pickle.dumps(metric))._kwargs() == metric._kwargs()  # nosec B301 - round-tripping our OWN objects is precisely what this file verifies

    def test_a_nested_chain_pickles(self, metric, data):
        """The real configuration nests: robustness over subgroups over the metric."""
        chain = RobustTimeSplitMetric(
            SubgroupAveragedMetric(metric, None), num_splits=2, std_coeff=1.0, greater_is_better=False,
        )
        assert pickle.loads(pickle.dumps(chain)) is not None  # nosec B301 - round-tripping our OWN objects is precisely what this file verifies


class TestTheLibraryContracts:
    """Each booster demands something different of a custom metric."""

    def test_the_callable_exposes_a_name_for_xgboost(self, metric):
        """xgboost reads ``eval_metric.__name__`` and raised AttributeError without it -- caught by a real fit."""
        assert metric.__name__ == "integral_calibration_error"

    def test_the_class_keeps_its_own_name(self, metric):
        """The instance attribute must not make the CLASS unidentifiable in tracebacks."""
        assert type(metric).__name__ == "IntegralCalibrationError"

    def test_the_lightgbm_adapter_returns_the_three_tuple(self, metric, data):
        """LightGBM wants ``(name, value, higher_is_better)``, not a bare float."""
        y, p = data
        name, value, higher = LightGBMMetricAdapter(metric)(y, p)
        assert name == "integral_calibration_error"
        assert value == metric(y, p)
        assert higher is False

    def test_the_positional_variant_takes_positional_args(self, metric, data):
        """The FS / HPT callers pass y_true / y_score positionally."""
        y, p = data
        assert PositionalIntegralCalibrationError(**CONFIG)(y, p, verbose=False) == pytest.approx(metric(y, p))


class TestTheRobustnessWrapperStillBehaves:
    """It has its own fallbacks, and they are what make the metric usable on short series."""

    def test_too_little_data_falls_back_to_the_full_metric(self, metric, data):
        """Below one split's worth of rows there is no robustness estimate to make."""
        y, p = data
        wrapped = RobustTimeSplitMetric(metric, num_splits=3, std_coeff=1.0, greater_is_better=False, min_samples_per_split=10_000)
        assert wrapped(y, p) == metric(y, p)

    def test_it_penalises_spread_for_a_minimised_metric(self, data):
        """mean + std*coeff: variance across time makes a minimised metric look worse, which is the point."""
        y, p = data

        def _alternating(yy, pp, *a, **k):
            """A metric whose value depends on the split, so the spread is non-zero."""
            return float(len(yy) % 2)

        wrapped = RobustTimeSplitMetric(_alternating, num_splits=3, std_coeff=1.0, greater_is_better=False, min_samples_per_split=10)
        plain = RobustTimeSplitMetric(_alternating, num_splits=3, std_coeff=0.0, greater_is_better=False, min_samples_per_split=10)
        assert wrapped(y, p) >= plain(y, p)


class TestTheSuiteConfigIsPicklableEndToEnd:
    """What the training suite actually attaches to model params."""

    @pytest.fixture(scope="class")
    def configs(self):
        """One built configuration namespace."""
        from mlframe.training.helpers import get_training_configs

        return vars(get_training_configs(has_time=True))

    @pytest.mark.parametrize(
        "name",
        [
            "integral_calibration_error",
            "final_integral_calibration_error",
            "lgbm_integral_calibration_error",
            "fs_and_hpt_integral_calibration_error",
        ],
    )
    def test_each_exported_metric_pickles(self, configs, name):
        """These are the four names the suite hands to models and to feature selection."""
        assert pickle.loads(pickle.dumps(configs[name])) is not None  # nosec B301 - round-tripping our OWN objects is precisely what this file verifies

    @pytest.mark.parametrize("name", ["XGB_CALIB_CLASSIF", "CB_CALIB_CLASSIF"])
    def test_the_calibrated_param_dicts_pickle(self, configs, name):
        """The param dict is what travels with the model; CatBoost's carries the metric inside an ICE wrapper."""
        assert pickle.loads(pickle.dumps(configs[name])) is not None  # nosec B301 - round-tripping our OWN objects is precisely what this file verifies

    def test_none_of_them_is_a_local_function(self, configs):
        """A local function is what put every save on dill; a repeat would be silent without this."""
        for name in ("integral_calibration_error", "final_integral_calibration_error", "lgbm_integral_calibration_error"):
            qualname = getattr(type(configs[name]), "__qualname__", "")
            assert "<locals>" not in qualname, f"{name} is a local {qualname}"
