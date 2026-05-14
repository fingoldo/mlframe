"""Regression tests for the strategy-pre_pipeline imputer propagation bug surfaced by the 2026-05-14 prod log (4M-row regression suite crashed on LinearRegression.fit with `ValueError: Input X contains NaN`).

Root cause was in main.py: `_get_pipeline_components` resolved a SimpleImputer default into a local variable but never wrote it back to ctx. ctx.imputer stayed None, LinearModelStrategy.build_pipeline(imputer=None) silently skipped the imputation step, raw NaN reached LinearRegression.fit.

These tests lock in two contracts:

1. LinearModelStrategy.build_pipeline emits a WARN when requires_imputation=True but imputer=None (defence-in-depth: surface silent misconfigurations at the source).
2. _get_pipeline_components defaults imputer to a real SimpleImputer when PreprocessingConfig.imputer is None, so downstream calls into build_pipeline get a working imputer.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from mlframe.training.configs import PreprocessingConfig
from mlframe.training.core._setup_helpers import _get_pipeline_components
from mlframe.training.strategies import LinearModelStrategy


class TestLinearStrategyImputerWarning:
    """LinearModelStrategy.build_pipeline must emit a WARN when requires_imputation=True but imputer=None."""

    def test_imputer_none_emits_warning(self, caplog):
        strat = LinearModelStrategy()
        assert strat.requires_imputation is True, "precondition: LinearModelStrategy must require imputation"

        with caplog.at_level(logging.WARNING, logger="mlframe.training.strategies"):
            pipeline = strat.build_pipeline(
                base_pipeline=None,
                cat_features=[],
                category_encoder=None,
                imputer=None,
                scaler=SimpleImputer(),  # scaler can be anything truthy; we just don't want a scaler-WARN
            )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        imputer_warnings = [r for r in warnings if "requires_imputation=True" in r.getMessage()]
        assert len(imputer_warnings) == 1, (
            f"expected exactly one imputer-skip WARN, got {len(imputer_warnings)}: "
            f"{[r.getMessage() for r in warnings]!r}"
        )

        # Verify the WARN says what a maintainer needs to fix it
        msg = imputer_warnings[0].getMessage()
        assert "imputer is None" in msg
        assert "SimpleImputer" in msg, "WARN should mention the obvious fix"

        # And verify the pipeline really lacks the imp step
        if pipeline is not None:
            step_names = [name for name, _ in pipeline.steps] if hasattr(pipeline, "steps") else []
            assert "imp" not in step_names, "imp step should be skipped when imputer=None"

    def test_imputer_present_no_warning(self, caplog):
        strat = LinearModelStrategy()
        with caplog.at_level(logging.WARNING, logger="mlframe.training.strategies"):
            pipeline = strat.build_pipeline(
                base_pipeline=None,
                cat_features=[],
                category_encoder=None,
                imputer=SimpleImputer(strategy="mean"),
                scaler=None,  # accept the scaler-WARN; we test the imputer path
            )

        imputer_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "requires_imputation=True" in r.getMessage()
        ]
        assert imputer_warnings == [], (
            f"unexpected imputer WARN when imputer was supplied: "
            f"{[r.getMessage() for r in imputer_warnings]!r}"
        )

        # The imp step must actually be in the pipeline
        assert pipeline is not None, "build_pipeline should return a non-None pipeline when steps are added"
        step_names = [name for name, _ in pipeline.steps] if hasattr(pipeline, "steps") else []
        assert "imp" in step_names, f"expected 'imp' step in pipeline, got {step_names!r}"

    def test_imputer_present_actually_imputes_nan(self):
        """End-to-end: the built pipeline must turn NaN-containing X into NaN-free X."""
        strat = LinearModelStrategy()
        pipeline = strat.build_pipeline(
            base_pipeline=None,
            cat_features=[],
            category_encoder=None,
            imputer=SimpleImputer(strategy="mean"),
            scaler=None,
        )
        assert pipeline is not None

        X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [10.0, 20.0, np.nan, 40.0]})
        X_out = pipeline.fit_transform(X)
        assert not np.isnan(np.asarray(X_out)).any(), (
            "SimpleImputer in the strategy pipeline must produce a NaN-free transform; "
            "if this fails the 2026-05-14 LinearRegression.fit NaN crash is back."
        )


class TestGetPipelineComponentsDefaults:
    """_get_pipeline_components must never return None for imputer/scaler when the caller leaves them blank - that was the root cause of the prod crash."""

    def test_none_inputs_resolve_to_real_components(self):
        cfg = PreprocessingConfig()  # all fields default
        category_encoder, imputer, scaler = _get_pipeline_components(cfg, cat_features=[])
        assert imputer is not None, "imputer must default to a real SimpleImputer when PreprocessingConfig.imputer is None"
        assert scaler is not None, "scaler must default to a real StandardScaler when PreprocessingConfig.scaler is None"
        # category_encoder is allowed to stay None when cat_features is empty - it would only fire for cat_features
        assert hasattr(imputer, "fit_transform"), "imputer default must be a sklearn-compatible transformer"
        assert hasattr(scaler, "fit_transform"), "scaler default must be a sklearn-compatible transformer"

    def test_explicit_imputer_passes_through(self):
        my_imputer = SimpleImputer(strategy="median")
        cfg = PreprocessingConfig(imputer=my_imputer)
        _, imputer, _ = _get_pipeline_components(cfg, cat_features=[])
        assert imputer is my_imputer, "explicit PreprocessingConfig.imputer must pass through untouched"
