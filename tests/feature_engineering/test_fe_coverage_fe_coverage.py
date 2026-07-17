"""§8.2 FE test coverage gaps -- regression tests for previously uncovered FE code paths.

Each test corresponds to one §8.2 finding. Sibling F1 tests already cover the P0 OOF leakage
(test_leakage_safe_encoder_time_aware_no_shuffle / external splitter) and the P1 WoE-prior /
P1 polynomial-dim findings; this file focuses on the rest.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.conftest import fast_subset


# ---------------------------------------------------------------------------
# §8.2 P1: _phase_train_one_target.py:640 FHC validated only against sorted_models[0]
# ---------------------------------------------------------------------------


def test_fhc_seed_uses_first_sorted_model_kind(monkeypatch):
    """The single-shot FHC seed call resolves ``model_kind`` from ``sorted_mlframe_models[0]``. We
    intercept ``feature_handling_apply`` and assert the call-time ``model_kind`` matches the first
    sorted entry -- a regression sentinel for heterogeneous suites where mixing tree / linear /
    neural families means the chosen model_kind drives the validation contract for ALL family
    members downstream."""
    pl = pytest.importorskip("polars")
    from mlframe.training.core import _phase_train_one_target as mod
    from mlframe.training.feature_handling.config import FeatureHandlingConfig

    train_df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    val_df = pl.DataFrame({"x": [5.0, 6.0]})
    test_df = pl.DataFrame({"x": [7.0, 8.0]})

    captured = {}

    def _fake_apply(*, train_df, val_df, test_df, train_target, fhc, model_kind, **_kw):
        """Helper: Fake apply."""
        captured["model_kind"] = model_kind
        return SimpleNamespace(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            fitted={},
            generated_columns=[],
        )

    fhc = FeatureHandlingConfig()
    ctx = SimpleNamespace(
        feature_handling_config=fhc,
        sorted_mlframe_models=["catboost", "lightgbm", "xgboost"],
        artifacts={},
    )

    # ``_maybe_run_feature_handling_apply`` imports ``feature_handling_apply`` lazily from
    # ``mlframe.training.feature_handling``; patch the lookup site so the fake intercepts.
    import mlframe.training.feature_handling as _fh_pkg

    monkeypatch.setattr(_fh_pkg, "feature_handling_apply", _fake_apply, raising=False)
    mod._maybe_run_feature_handling_apply(
        ctx=ctx,
        cur_target_name="y",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        current_train_target=pd.Series([0, 1, 0, 1]),
    )
    assert captured.get("model_kind") == "catboost", f"FHC seed must use sorted_mlframe_models[0]; got {captured}"


# ---------------------------------------------------------------------------
# §8.2 P2: pipeline.py:202 PySR niterations=5 vs higher convergence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("niterations", fast_subset([5, 25]))
def test_pysr_niterations_round_trips_through_pysr_params(niterations):
    """The PySR stage's ``niterations`` knob is plumbed through ``pysr_params``. Round-tripping the
    value through the config is the cheap construction-time sentry -- a true convergence assertion
    is gated by the PySR/julia runtime which is unsuitable for fast suites."""
    from mlframe.training.configs import PreprocessingExtensionsConfig

    cfg = PreprocessingExtensionsConfig(pysr_enabled=True, pysr_params={"niterations": niterations})
    assert cfg.pysr_params is not None
    assert cfg.pysr_params.get("niterations") == niterations


# ---------------------------------------------------------------------------
# §8.2 P2: custom_handler.py:99 fit(X, y)->TypeError->fit(X) fallback
# ---------------------------------------------------------------------------


def test_custom_handler_falls_back_when_transformer_fit_rejects_y(caplog):
    """A transformer whose ``fit`` signature lacks ``y`` raises TypeError when y is passed; the
    handler must catch this, log a WARN, and retry with just X."""
    from mlframe.training.feature_handling.custom_handler import CustomHandler
    from mlframe.training.feature_handling.handlers import CustomParams

    class _UnsupervisedOnly:
        """Bare ``fit(self, X)`` -- legacy sklearn-ish transformer without y support."""

        def __init__(self):
            """Helper: Init  ."""
            self.x_seen = None

        def fit(self, X):
            """Fit."""
            self.x_seen = np.asarray(X).copy()
            return self

        def transform(self, X):
            """Transform."""
            return np.asarray(X)

    transformer = _UnsupervisedOnly()
    handler = CustomHandler(
        column="x",
        params=CustomParams(transformer=transformer, output_kind="dense"),
    )
    df = pd.DataFrame({"x": [0.1, 0.2, 0.3]})
    y = np.array([0, 1, 0])
    with caplog.at_level(logging.WARNING):
        handler.fit(df, y)
    assert transformer.x_seen is not None, "fit(X) fallback must have run after the TypeError"
    assert any("fit(X) only" in rec.getMessage() for rec in caplog.records), "fallback path must emit a WARN log line"


# ---------------------------------------------------------------------------
# §8.2 Low: polynomial.py:84 projected>5000 WARN message + shape
# ---------------------------------------------------------------------------


def test_polynomial_projected_over_5000_emits_memory_warn(caplog):
    """When ``projected > 5000`` the expander emits a memory-budget WARN with the explicit MB
    estimate (we assert the line is present and the resulting transform shape matches the
    projected count)."""
    from mlframe.training.feature_handling.polynomial import (
        PolynomialFeatureExpander,
        _projected_output_cols,
    )

    n_features = 100
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, n_features)).astype(np.float32)

    projected = _projected_output_cols(n_features, 2, False)
    assert projected > 5000, f"sanity: {projected} should exceed 5000 for n=100 d=2"

    exp = PolynomialFeatureExpander(degree=2, interaction_only=False, max_features_out=None)
    with caplog.at_level(logging.WARNING):
        exp.fit(X)
    out = exp.transform(X)
    assert out.shape[1] == projected, f"transform output cols must match projected count; got {out.shape[1]} vs {projected}"
    warn_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("polynomial expansion" in m and "MB" in m for m in warn_msgs), f"WARN line must include MB cost estimate; saw {warn_msgs}"
