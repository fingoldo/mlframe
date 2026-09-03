"""FE_TRANSFORMER_B-5 regression test: the 5 independent ``_fit_3baselines*`` helpers must log the
LogisticRegression-member-failure fallback at the SAME level.

The bug (fixed): 4 of the 5 duplicated ``_fit_3baselines*`` helpers logged the identical
LogisticRegression-fit-failure condition at ``debug`` (invisible by default), while
``multi_baseline_hard_row.py`` already logged it at ``info`` -- an observable drift from
independently-maintained near-identical code, silently hiding a real fit failure in 4 of the 5 modules.
All 5 now log at ``info``. (The other drift the audit flagged -- ``gradient_direction_agreement``
substituting a zero gradient vs. the other four substituting a constant class-prior prediction -- is not
a bug: a constant function has a zero gradient everywhere, so both are the same degenerate-model
fallback expressed in the shape each call site's own output needs.)
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pytest

pytestmark = pytest.mark.fast


def _fail_logistic_regression():
    """Context manager substitute: patch LogisticRegression.fit to always raise, to force the fallback path."""
    return patch("sklearn.linear_model.LogisticRegression.fit", side_effect=RuntimeError("forced failure"))


@pytest.mark.parametrize(
    "module_name,func_name",
    [
        ("mlframe.feature_engineering.transformer.gradient_direction_agreement", "_fit_3baselines"),
        ("mlframe.feature_engineering.transformer.ib_baseline_codes", "_fit_3baselines_two"),
        ("mlframe.feature_engineering.transformer.nn_oof_target_mean", "_fit_3baselines_predict_two"),
        ("mlframe.feature_engineering.transformer.pairwise_kl_divergence", "_fit_3baselines_with_sigma"),
    ],
)
def test_logistic_regression_failure_logs_at_info(caplog, module_name, func_name):
    """Each duplicated _fit_3baselines* helper logs the LogisticRegression-fit-failure fallback at INFO."""
    import importlib

    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name)
    rng = np.random.default_rng(0)
    Xt = rng.standard_normal((60, 4)).astype(np.float32)
    y_t = rng.integers(0, 2, size=60).astype(np.float32)
    Xq = rng.standard_normal((10, 4)).astype(np.float32)

    with caplog.at_level(logging.INFO, logger=module_name):
        with _fail_logistic_regression():
            fn(Xt, y_t, Xq, task="binary", seed=0) if func_name != "_fit_3baselines" else fn(Xt, y_t, task="binary", seed=0)

    info_records = [r for r in caplog.records if r.levelno == logging.INFO and "LogisticRegression" in r.getMessage()]
    assert info_records, f"{module_name}.{func_name} should log the LogisticRegression fallback at INFO"
