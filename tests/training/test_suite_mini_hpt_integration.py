"""Suite-level integration coverage for the mini-HPT target distribution analyzer.

The standalone analyzer is exercised by ``test_target_distribution_analyzer.py``;
this file verifies the WIRING into ``train_mlframe_models_suite``: the flag toggles
the call, the report lands in metadata, and recommendations are gap-fill merged
into ``hyperparams_config`` (caller-supplied values win)."""
from __future__ import annotations

import logging
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from .shared import SimpleFeaturesAndTargetsExtractor


def _build_heavy_tail_frame(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 5)).astype(np.float32)
    y = rng.standard_t(df=3, size=n).astype(np.float32)  # heavy tails
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    return df


def _run_suite(df: pd.DataFrame, *, tmp, enable_analyzer: bool, hyperparams: dict | None = None, verbose: int = 1):
    from mlframe.training.core.main import train_mlframe_models_suite
    from mlframe.training import OutputConfig, ReportingConfig

    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)
    hp = {"iterations": 30}
    if hyperparams is not None:
        hp.update(hyperparams)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        models, meta = train_mlframe_models_suite(
            df=df, target_name="target", model_name="mini_hpt_test",
            features_and_targets_extractor=fte,
            mlframe_models=["linear", "lgb"],  # skip MLP to dodge unrelated Lightning issues
            use_ordinary_models=True, use_mlframe_ensembles=False,
            reporting_config=ReportingConfig(show_perf_chart=False, show_fi=False),
            output_config=OutputConfig(data_dir=str(tmp), models_dir="models"),
            verbose=verbose,
            hyperparams_config=hp,
            enable_target_distribution_analyzer=enable_analyzer,
        )
    return models, meta


class TestMiniHPTSuiteWiring:
    def test_flag_default_true_stamps_report_into_metadata(self, tmp_path, caplog):
        df = _build_heavy_tail_frame()
        with caplog.at_level(logging.INFO, logger="mlframe.training.core.main"):
            _models, meta = _run_suite(df, tmp=tmp_path, enable_analyzer=True)
        assert "target_distribution_report" in meta, "Analyzer did not stamp report into metadata"
        rep = meta["target_distribution_report"]
        assert rep["target_type"] == "regression"
        assert any("heavy_tail" in p for p in rep["pathologies"]), (
            f"Heavy-tail target was not detected: {rep['pathologies']}"
        )
        # Log line must surface so operators see what fired.
        assert any("[mini-HPT]" in r.getMessage() for r in caplog.records), (
            "Mini-HPT log line never emitted on a default-on run."
        )

    def test_flag_false_skips_analyzer_entirely(self, tmp_path, caplog):
        df = _build_heavy_tail_frame()
        with caplog.at_level(logging.INFO, logger="mlframe.training.core.main"):
            _models, meta = _run_suite(df, tmp=tmp_path, enable_analyzer=False)
        assert "target_distribution_report" not in meta, (
            "Analyzer must NOT stamp metadata when flag is False."
        )
        assert not any("[mini-HPT]" in r.getMessage() for r in caplog.records), (
            "Mini-HPT log line leaked when flag was False."
        )

    def test_feature_distribution_report_also_stamps_into_metadata(self, tmp_path):
        """The feature-side analyzer runs alongside the target-side one when the flag is on.
        Build a frame with deliberate feature pathologies and assert they're surfaced."""
        rng = np.random.default_rng(7)
        n = 2000
        X = rng.normal(0, 1, (n, 6)).astype(np.float64)
        # f1 = NaN-heavy (60%), f2 = leaks target
        nan_idx = rng.choice(n, size=int(n * 0.6), replace=False)
        X[nan_idx, 1] = np.nan
        y = X[:, 3].astype(np.float32)
        X[:, 2] = y + 0.001 * rng.normal(0, 1, n)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
        df["target"] = y

        _models, meta = _run_suite(df, tmp=tmp_path, enable_analyzer=True, verbose=0)
        fdr = meta.get("feature_distribution_report")
        assert fdr is not None, "Feature-distribution report missing from metadata"
        assert "nan_heavy_features" in " ".join(fdr["pathologies"]), fdr["pathologies"]
        assert "suspected_target_leakage" in " ".join(fdr["pathologies"]), fdr["pathologies"]
        # f1 should be in drop_candidates (NaN-heavy). f2 should be in leakage_candidates.
        assert "f1" in fdr["drop_candidates"]
        assert "f2" in fdr["leakage_candidates"]

    def test_recommendation_merges_into_dict_hyperparams_config(self, tmp_path):
        """Heavy-tail target -> analyzer recommends lgb_kwargs.objective='huber'.
        With dict-form hyperparams that doesn't pre-set objective, the gap-fill
        merge must land it. Caller-supplied hyperparam values take precedence."""
        df = _build_heavy_tail_frame()
        # Caller supplies a different objective; gap-fill must NOT override it.
        _models, meta = _run_suite(
            df, tmp=tmp_path, enable_analyzer=True,
            hyperparams={"lgb_kwargs": {"objective": "regression_l2"}},
        )
        rep = meta["target_distribution_report"]
        assert any("heavy_tail" in p for p in rep["pathologies"])
        # The recommendation table itself must still contain the huber suggestion;
        # the merge preserved user objective but the recommendation is recorded.
        assert rep["knob_overrides"].get("lgb_kwargs", {}).get("objective") == "huber"
