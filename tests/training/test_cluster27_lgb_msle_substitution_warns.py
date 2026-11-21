"""Audit 2026-11-20 finding 27-F16: LightGBM's MSLE substitution is logged, not silent.

The LightGBM params fall back to rmse for an MSLE regression metric, while CatBoost and XGBoost use their own MSLE. The test builds the
params for MSLE and checks the warning is logged; the RMSE path does not warn.
"""

import logging

from mlframe.training._helpers_training_configs import _build_lgb_params
from mlframe.training.configs import TargetTypes


def _params(metric, caplog):
    """Build the LightGBM params for a regression run with the given metric, capturing WARNING logs."""
    with caplog.at_level(logging.WARNING, logger="mlframe.training.helpers"):
        _build_lgb_params(
            iterations=10,
            learning_rate=0.1,
            has_gpu=False,
            early_stopping_rounds=5,
            random_seed=0,
            lgb_kwargs={},
            def_regr_metric=metric,
            _resolved_tt=TargetTypes.REGRESSION,
            n_classes=1,
        )
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_msle_substitution_is_logged(caplog):
    """MSLE produces a warning that names the rmse fallback."""
    messages = _params("MSLE", caplog)
    assert any("MSLE" in m and "rmse" in m for m in messages)


def test_rmse_does_not_warn(caplog):
    """RMSE maps to l2 with no substitution warning."""
    assert not _params("RMSE", caplog)
