"""Tests for ``mlframe.reporting.diagnostics_dispatch`` -- the suite wiring for the
error-analysis + drift diagnostic charts.

Covers: default-ON rendering + file emission + chart accounting for each entry point,
the no-op short-circuits (missing df / plot_outputs / base_path), large-n RAM safety
(bounded row subsample), and biz_value (worst-K localises the injected weak region;
adversarial AUC ~0.5 on identical splits vs > 0.7 on a shifted split).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting.diagnostics_dispatch import (
    DIAG_ROW_CAP,
    render_split_error_diagnostics,
    render_target_dist_overlay,
    render_target_drift_diagnostics,
)


@pytest.fixture
def reg_data():
    """Regression with an injected weak region f0>0.8 & f1<0.2 (predicted +3.0 too high)."""
    rng = np.random.default_rng(0)
    n = 5000
    f0 = rng.uniform(0, 1, n)
    f1 = rng.uniform(0, 1, n)
    df = pd.DataFrame({"f0": f0, "f1": f1, "f2": rng.normal(0, 1, n)})
    y = 2 * f0 + rng.normal(0, 0.1, n)
    yp = y.copy()
    bad = (f0 > 0.8) & (f1 < 0.2)
    yp[bad] += 3.0
    ts = np.sort(rng.uniform(0, 1e9, n)).astype("datetime64[s]").astype("datetime64[ns]")
    return df, y, yp, bad, ts


# ----------------------------------------------------------------------------
# Per-split error diagnostics
# ----------------------------------------------------------------------------


class TestSplitErrorDiagnostics:
    """Groups tests for: TestSplitErrorDiagnostics."""
    def test_renders_all_default_on(self, reg_data, tmp_path):
        """Renders all default on."""
        df, y, yp, _bad, _ts = reg_data
        m: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = render_split_error_diagnostics(
                df=df,
                y_true=y,
                y_pred=yp,
                task="regression",
                plot_outputs="matplotlib[png]",
                base_path=str(tmp_path / "reg"),
                metrics_dict=m,
                feature_names=["f0", "f1", "f2"],
                subgroups={"hi": df["f0"].to_numpy() > 0.5, "lo": df["f0"].to_numpy() <= 0.5},
            )
        assert os.path.exists(tmp_path / "reg_weak_segments.png")
        assert os.path.exists(tmp_path / "reg_error_bias.png")
        assert os.path.exists(tmp_path / "reg_segments.png")
        assert set(m["charts"]["saved"]) >= {"weak_segments", "error_bias", "segments"}
        assert m["charts"]["failed"] == []
        assert len(out["worst_k_indices"]) == 20
        assert len(out["worst_k_table"]) == 20

    def test_worst_k_localises_injected_region(self, reg_data, tmp_path):
        """biz_value: the worst-K rows MUST concentrate in the injected bad region.
        Measured 100%; floor 0.85 absorbs noise."""
        df, y, yp, bad, _ts = reg_data
        out = render_split_error_diagnostics(
            df=df,
            y_true=y,
            y_pred=yp,
            task="regression",
            plot_outputs="matplotlib[png]",
            base_path=str(tmp_path / "reg"),
            metrics_dict={},
            feature_names=["f0", "f1", "f2"],
            worst_k=30,
        )
        frac = bad[out["worst_k_indices"]].mean()
        assert frac >= 0.85, f"worst-K should land in the injected region; got {frac:.3f}"

    def test_noop_when_no_df(self, reg_data, tmp_path):
        """Noop when no df."""
        _df, y, yp, _bad, _ts = reg_data
        m: dict = {}
        out = render_split_error_diagnostics(
            df=None,
            y_true=y,
            y_pred=yp,
            task="regression",
            plot_outputs="matplotlib[png]",
            base_path=str(tmp_path / "x"),
            metrics_dict=m,
        )
        assert out["worst_k_table"] is None
        assert not list(tmp_path.glob("x*"))

    def test_noop_when_no_plot_outputs(self, reg_data, tmp_path):
        """Noop when no plot outputs."""
        df, y, yp, _bad, _ts = reg_data
        out = render_split_error_diagnostics(
            df=df,
            y_true=y,
            y_pred=yp,
            task="regression",
            plot_outputs="",
            base_path=str(tmp_path / "x"),
            metrics_dict={},
        )
        assert out["worst_k_table"] is None
        assert not list(tmp_path.glob("x*"))

    def test_large_n_subsamples_and_maps_indices(self, tmp_path):
        """RAM safety: n > DIAG_ROW_CAP -> feature builders see a bounded subsample, but worst-K indices still map
        onto the FULL array (so the caller can highlight original scatter points)."""
        rng = np.random.default_rng(1)
        n = DIAG_ROW_CAP + 20_000
        df = pd.DataFrame({"f0": rng.uniform(0, 1, n), "f1": rng.uniform(0, 1, n)})
        y = df["f0"].to_numpy()
        yp = y.copy()
        # The single worst row is the last one -- it must survive subsampling and surface in worst-K.
        yp[-1] += 100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = render_split_error_diagnostics(
                df=df,
                y_true=y,
                y_pred=yp,
                task="regression",
                plot_outputs="matplotlib[png]",
                base_path=str(tmp_path / "big"),
                metrics_dict={},
                feature_names=["f0", "f1"],
                worst_k=5,
            )
        assert (n - 1) in set(out["worst_k_indices"].tolist())
        assert (out["worst_k_indices"] < n).all()


# ----------------------------------------------------------------------------
# Per-target drift + adversarial
# ----------------------------------------------------------------------------


class TestTargetDriftDiagnostics:
    """Groups tests for: TestTargetDriftDiagnostics."""
    def test_renders_all_default_on(self, reg_data, tmp_path):
        """Renders all default on."""
        df, y, yp, _bad, ts = reg_data
        m: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_target_drift_diagnostics(
                train_frame=df.iloc[:2500],
                test_frame=df.iloc[2500:],
                y_true=y,
                y_pred=yp,
                timestamps=ts,
                task="regression",
                plot_outputs="matplotlib[png]",
                base_path=str(tmp_path / "drift"),
                metrics_dict=m,
                feature_names=["f0", "f1", "f2"],
            )
        assert os.path.exists(tmp_path / "drift_psi.png")
        assert os.path.exists(tmp_path / "drift_residual_vs_time.png")
        assert os.path.exists(tmp_path / "drift_metric_over_time.png")
        assert os.path.exists(tmp_path / "drift_adversarial.png")
        assert set(m["charts"]["saved"]) >= {"psi_heatmap", "residual_vs_time", "metric_over_time", "adversarial"}

    def test_no_drift_panels_without_timestamps(self, reg_data, tmp_path):
        """No drift panels without timestamps."""
        df, y, yp, _bad, _ts = reg_data
        m: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_target_drift_diagnostics(
                train_frame=df.iloc[:2500],
                test_frame=df.iloc[2500:],
                y_true=y,
                y_pred=yp,
                timestamps=None,
                task="regression",
                plot_outputs="matplotlib[png]",
                base_path=str(tmp_path / "drift"),
                metrics_dict=m,
                feature_names=["f0", "f1", "f2"],
            )
        # Temporal panels gated out; adversarial still fires (frames present).
        assert not (tmp_path / "drift_psi.png").exists()
        assert not (tmp_path / "drift_residual_vs_time.png").exists()
        assert os.path.exists(tmp_path / "drift_adversarial.png")

    def test_adversarial_identical_vs_shifted(self, tmp_path):
        """biz_value: adversarial AUC ~0.5 on identical train/test, > 0.7 when a feature is shifted.
        Floors 0.6 / 0.7 sit below the measured ~0.5 / ~0.95."""
        from mlframe.reporting.charts.drift import adversarial_auc

        rng = np.random.default_rng(2)
        n = 4000
        base = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
        other_same = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
        auc_same, *_ = adversarial_auc(base, other_same, feature_names=["a", "b"])
        assert auc_same <= 0.6, f"identical splits should be indistinguishable; AUC={auc_same:.3f}"
        shifted = pd.DataFrame({"a": rng.normal(4, 1, n), "b": rng.normal(0, 1, n)})
        auc_shift, *_ = adversarial_auc(base, shifted, feature_names=["a", "b"])
        assert auc_shift >= 0.7, f"shifted feature should be detectable; AUC={auc_shift:.3f}"


# ----------------------------------------------------------------------------
# Target distribution overlay
# ----------------------------------------------------------------------------


class TestTargetDistOverlay:
    """Groups tests for: TestTargetDistOverlay."""
    def test_renders_default_on(self, reg_data, tmp_path):
        """Renders default on."""
        _df, y, yp, _bad, _ts = reg_data
        m: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ok = render_target_dist_overlay(
                y_true_by_split={"train": y[:2500], "test": y[2500:]},
                pred_by_split={"train": yp[:2500], "test": yp[2500:]},
                task="regression",
                plot_outputs="matplotlib[png]",
                base_path=str(tmp_path / "dist"),
                metrics_dict=m,
            )
        assert ok
        assert os.path.exists(tmp_path / "dist_target_dist.png")
        assert "target_dist" in m["charts"]["saved"]

    def test_noop_empty_inputs(self, tmp_path):
        """Noop empty inputs."""
        ok = render_target_dist_overlay(
            y_true_by_split={},
            task="regression",
            plot_outputs="matplotlib[png]",
            base_path=str(tmp_path / "x"),
            metrics_dict={},
        )
        assert ok is False
        assert not list(tmp_path.glob("x*"))


class TestNoCircularImport:
    """Regression: importing the CHILD module (_diagnostics_dispatch_extra) FIRST must not raise a
    circular ImportError. Previously its top-level ``from .diagnostics_dispatch import _record, ...``
    re-entered the half-initialised parent, whose bottom re-export then failed to find ``_entry_score``
    (defined further down the child). discover_tuners (refresh-all) imports the child first, so this
    crashed the kernel-tuning CLI. The four parent helpers are now lazy-delegated. Verified in a FRESH
    subprocess so module-cache order from earlier tests cannot mask the cycle."""

    def test_child_first_import_in_subprocess(self):
        """Child first import in subprocess."""
        import subprocess, sys  # nosec B404 -- test-only local trusted subprocess invocation (fixed argv, no shell, no untrusted input)

        for first in ("mlframe.reporting._diagnostics_dispatch_extra", "mlframe.reporting.diagnostics_dispatch"):
            code = (
                f"import {first} as m;"
                "import mlframe.reporting._diagnostics_dispatch_extra as e;"
                "assert hasattr(e, '_entry_score');"
                "import mlframe.reporting.diagnostics_dispatch as d;"
                "assert hasattr(d, '_entry_score');"
                "print('OK')"
            )
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)  # nosec B603 -- fixed local argv (sys.executable/git + literal args), no shell, no untrusted input
            assert r.returncode == 0, f"first={first} stderr=\n{r.stderr}"
            assert "OK" in r.stdout
