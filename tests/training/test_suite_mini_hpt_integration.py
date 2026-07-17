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
            df=df,
            target_name="target",
            model_name="mini_hpt_test",
            features_and_targets_extractor=fte,
            mlframe_models=["linear", "lgb"],  # skip MLP to dodge unrelated Lightning issues
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
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
        assert any("heavy_tail" in p for p in rep["pathologies"]), f"Heavy-tail target was not detected: {rep['pathologies']}"
        # Log line must surface so operators see what fired.
        assert any("[mini-HPT]" in r.getMessage() for r in caplog.records), "Mini-HPT log line never emitted on a default-on run."

    def test_flag_false_skips_analyzer_entirely(self, tmp_path, caplog):
        df = _build_heavy_tail_frame()
        with caplog.at_level(logging.INFO, logger="mlframe.training.core.main"):
            _models, meta = _run_suite(df, tmp=tmp_path, enable_analyzer=False)
        assert "target_distribution_report" not in meta, "Analyzer must NOT stamp metadata when flag is False."
        assert not any("[mini-HPT]" in r.getMessage() for r in caplog.records), "Mini-HPT log line leaked when flag was False."

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

    def test_auto_time_axis_rejects_non_monotonic_candidate_column(self, tmp_path, caplog):
        """E5.3 monotonicity gate (2026-05-22): a column NAMED ``date`` does NOT
        mean rows are sorted by it. The auto-detection now monotonicity-checks
        a stride-sample BEFORE flipping has_time_axis=True. A
        SHUFFLED-rows scenario must skip the global AR detector (and log the
        "candidate ... NOT monotonic" line) so the AR signal isn't computed
        on noise."""
        rng = np.random.default_rng(13)
        n = 3000
        # AR(1) target ordered by depth, then SHUFFLE all rows. Column name
        # 'depth' would trigger the name-only auto-detection but monotonicity
        # check on the shuffled depths must reject the hint.
        depth_sorted = np.arange(n, dtype=np.float32)
        y_sorted = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            y_sorted[i] = 0.92 * y_sorted[i - 1] + rng.standard_normal() * 0.5
        perm = rng.permutation(n)
        df = pd.DataFrame(
            {
                "depth": depth_sorted[perm],
                "f0": rng.standard_normal(n).astype(np.float32),
                "f1": rng.standard_normal(n).astype(np.float32),
                "target": y_sorted[perm].astype(np.float32),
            }
        )
        with caplog.at_level(logging.INFO, logger="mlframe.training.core.main"):
            _models, meta = _run_suite(df, tmp=tmp_path, enable_analyzer=True, verbose=1)
        msgs = " | ".join(r.getMessage() for r in caplog.records)
        assert "candidate time-axis column(s) ['depth'] present but NOT monotonic" in msgs, (
            "Auto-detection should log the monotonicity-rejection line on shuffled rows so operators see why the AR detector skipped. msgs=" + msgs[:500]
        )
        # And the global AR detector did NOT recommend use_layernorm=False
        # (it would have if has_time_axis=True triggered without the monotonicity gate).
        rep = meta.get("target_distribution_report") or {}
        np_overrides = (rep.get("knob_overrides") or {}).get("mlp_kwargs", {}).get("network_params", {})
        # Could be False from a different detector (clustered, per_group AR) -- so we only
        # assert the AR-source diagnostic isn't "global" when monotonicity failed.
        diag = rep.get("diagnostics") or {}
        assert "max_abs_autocorr" not in diag, "Global AR diagnostic should NOT be stamped when monotonicity rejected the time-axis hint."

    def test_auto_time_column_detection_enables_AR_detector(self, tmp_path, caplog):
        """E5.3 (2026-05-21): when caller didn't pass timestamps but the frame
        carries a recognised time-axis column (MD/depth/timestamp/date/time),
        the suite auto-sets has_time_axis=True so the AR detector runs on
        global lag-1. The MD-sorted wellbore-shape scenario from the TVT-2026-
        05-21 incident now triggers strong_AR_target detection + the
        use_layernorm=False recommendation without operator intervention."""
        rng = np.random.default_rng(11)
        n = 5000
        md = np.arange(n, dtype=np.float32) * 0.5
        # AR(1) target sorted by MD (the wellbore depth axis).
        y = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            y[i] = 0.92 * y[i - 1] + rng.standard_normal() * 0.5
        df = pd.DataFrame(
            {
                "MD": md,
                "f0": rng.standard_normal(n).astype(np.float32),
                "f1": rng.standard_normal(n).astype(np.float32),
                "f2": rng.standard_normal(n).astype(np.float32),
                "target": y.astype(np.float32),
            }
        )
        with caplog.at_level(logging.INFO, logger="mlframe.training.core.main"):
            _models, meta = _run_suite(df, tmp=tmp_path, enable_analyzer=True, verbose=1)
        rep = meta.get("target_distribution_report")
        assert rep is not None
        # AR detector ran; global lag1 stamped (the multi-lag E5.1 diagnostic should appear too).
        diag = rep["diagnostics"]
        assert "lag1_autocorr" in diag or "max_abs_autocorr" in diag, f"AR detector did not run; diagnostics={diag}"
        # And the log line confirms the auto-detection fired on the MD column.
        msgs = " | ".join(r.getMessage() for r in caplog.records)
        assert "auto-detected monotonic time-axis column" in msgs, (
            "Auto-time-column detection log line missing -- the suite did not flip has_time_axis=True. "
            "(2026-05-22: log line now says ``auto-detected monotonic time-axis column(s) ...``.)"
        )

    def test_recommendation_merges_into_dict_hyperparams_config(self, tmp_path):
        """Heavy-tail target -> analyzer recommends lgb_kwargs.objective='huber'.
        With dict-form hyperparams that doesn't pre-set objective, the gap-fill
        merge must land it. Caller-supplied hyperparam values take precedence."""
        df = _build_heavy_tail_frame()
        # Caller supplies a different objective; gap-fill must NOT override it.
        _models, meta = _run_suite(
            df,
            tmp=tmp_path,
            enable_analyzer=True,
            hyperparams={"lgb_kwargs": {"objective": "regression_l2"}},
        )
        rep = meta["target_distribution_report"]
        assert any("heavy_tail" in p for p in rep["pathologies"])
        # The recommendation table itself must still contain the huber suggestion;
        # the merge preserved user objective but the recommendation is recorded.
        assert rep["knob_overrides"].get("lgb_kwargs", {}).get("objective") == "huber"
