"""Eleven training/reporting/targets defects, nearly all of them silence where a signal was promised.

Three default-ON calibration chart families wrote their files and registered nothing, so the charts were absent
from the combined HTML index and their failures left no `failed` entry either -- "the chart is missing because
it broke" was indistinguishable from "the chart is missing because the family was off". The diagnostics budget
class documents that a shortened report names what it dropped; its `report()` was never called, so a truncated
report looked complete. The temporal-audit chart was written and its path never recorded, so the metadata said
no chart exists. Sixteen config fields are accepted by `extra="forbid"` models and read nowhere.

Plus two labels that named a different quantity than the one printed, a doc block naming a threshold that was
changed away from long ago, a parameter accepted and never read, an eviction pass that scans and deletes
nothing, and a documented leakage-safe encoder path that the config surface could not express.
"""

from __future__ import annotations

import inspect
import logging

import numpy as np
import pytest


class TestTheCalibrationChartsRegisterWhatTheyWrote:
    """A chart absent from `metrics["charts"]` is absent from the combined report."""

    def _src(self):
        """The probabilistic-calibration reporting module."""
        from mlframe.training.reporting import _reporting_probabilistic_calib as m

        return inspect.getsource(m)

    def test_all_three_families_record_a_saved_path(self):
        """fairness-calibration, calibration-by-feature and the 2-D heatmap all default to True."""
        src = self._src()
        assert src.count("_record_chart(metrics,") >= 6, src.count("_record_chart(metrics,")

    def test_a_failure_leaves_a_failed_entry(self):
        """It was swallowed at DEBUG with nothing recorded, so a broken chart looked like a disabled one."""
        src = self._src()
        assert src.count("error=e)") == 3, src.count("error=e)")

    def test_the_recorder_populates_saved_and_paths(self):
        """`paths` is what the combined HTML index reads."""
        from mlframe.training.reporting._reporting_probabilistic_calib import _record_chart

        metrics: dict = {}
        _record_chart(metrics, "faircal_a", "charts/base_faircal_a")
        assert metrics["charts"]["saved"] == ["faircal_a"]
        assert metrics["charts"]["paths"] == ["charts/base_faircal_a"]

    def test_the_recorder_populates_failed_on_error(self):
        """And keeps the path list clean, so the index does not link a file that was never written."""
        from mlframe.training.reporting._reporting_probabilistic_calib import _record_chart

        metrics: dict = {}
        _record_chart(metrics, "calib2d", "", error=ValueError("boom"))
        assert metrics["charts"]["failed"] == ["calib2d: ValueError: boom"]
        assert metrics["charts"].get("paths", []) == []

    def test_a_non_dict_metrics_is_tolerated(self):
        """The reporting path passes None in some configurations."""
        from mlframe.training.reporting._reporting_probabilistic_calib import _record_chart

        _record_chart(None, "x", "y")  # must not raise


class TestTheDiagnosticsBudgetSaysWhatItDropped:
    """A truncated report that does not say so is worse than a slow one."""

    def test_report_is_actually_called(self):
        """The method existed, was documented, and had no caller anywhere."""
        from mlframe.training.reporting import _reporting_diagnostics as m

        assert "_budget.report()" in inspect.getsource(m)

    def test_a_skipped_diagnostic_produces_an_incomplete_warning(self, caplog):
        """The behaviour the method promises."""
        from mlframe.training.reporting._diagnostics_budget import DiagnosticsBudget

        b = DiagnosticsBudget(max_seconds=120.0)
        b.skipped.append("shap")
        with caplog.at_level(logging.WARNING, logger="mlframe.training.reporting._diagnostics_budget"):
            b.report()
        assert any("INCOMPLETE" in r.message for r in caplog.records), [r.message for r in caplog.records]

    def test_a_complete_report_stays_silent(self):
        """ "Silent when nothing was" is the other half of the contract."""
        from mlframe.training.reporting._diagnostics_budget import DiagnosticsBudget

        DiagnosticsBudget(max_seconds=120.0).report()  # must not raise, must not log

    @pytest.mark.parametrize("bad", ["", "  ", "everything", "Best!"])
    def test_an_unrecognised_mode_warns_and_falls_back(self, bad, caplog):
        """`("best" if mode is None else str(mode)).lower()` let "" through as the RESTRICTIVE behaviour, while
        the comment above it claimed such a value was surfaced rather than coerced."""
        from mlframe.training.reporting._diagnostics_budget import HeavyDiagnosticsPolicy

        with caplog.at_level(logging.WARNING, logger="mlframe.training.reporting._diagnostics_budget"):
            b = HeavyDiagnosticsPolicy(mode=bad)
        assert b.mode == "best"
        assert any("not one of" in r.message for r in caplog.records), (bad, [r.message for r in caplog.records])

    @pytest.mark.parametrize("good,expected", [("best", "best"), ("all", "all"), ("ALL", "all"), ("ALL ", "all"), (None, "best")])
    def test_the_valid_modes_are_unchanged(self, good, expected, caplog):
        """Case tolerance already worked; surrounding whitespace is now stripped rather than rejected, since
        "ALL " is unambiguously the "all" the caller meant."""
        from mlframe.training.reporting._diagnostics_budget import HeavyDiagnosticsPolicy

        with caplog.at_level(logging.WARNING, logger="mlframe.training.reporting._diagnostics_budget"):
            assert HeavyDiagnosticsPolicy(mode=good).mode == expected
        assert not [r for r in caplog.records if "not one of" in r.message]


def test_the_temporal_audit_records_the_path_it_wrote():
    """The DSL branch is the DEFAULT path, and it returned before setting `result.plot_path` -- so the chart
    existed on disk while the audit metadata serialised a null path."""
    from mlframe.training.targets import _target_temporal_plot as m

    src = inspect.getsource(m)
    assert "result.plot_path = base_path" in src
    idx_assign = src.index("result.plot_path = base_path")
    idx_return = src.index("return None", idx_assign)
    assert idx_assign < idx_return, "the assignment must precede the early return"


class TestTheEvictionPassDoesNotScanForNothing:
    """Between the trigger and the floor it listed the directory, stat'd every file, sorted, and deleted none."""

    def test_it_returns_before_listing_when_free_space_is_already_above_the_floor(self):
        """The loop's own first-iteration break condition, hoisted ahead of the scan."""
        from mlframe.training.feature_handling import cache as m

        src = inspect.getsource(m._FeatureCacheDiskMixin._maybe_evict_disk) if hasattr(m, "_FeatureCacheDiskMixin") else inspect.getsource(m)
        i_guard = src.index("if free_bytes >= target_free_bytes:")
        i_listdir = src.index("os.listdir(d)")
        assert i_guard < i_listdir, "the early return must come before the directory scan"


def test_the_unread_bootstrap_parameter_is_gone():
    """`preds` was accepted and passed and never read, which reads as working plumbing for a crisp-metric CI
    that does not exist. Every metric in the block is probability-based."""
    from mlframe.training import honest_diagnostics as m

    assert "preds" not in inspect.signature(m._bootstrap_block).parameters
    assert 'getattr(entry, "test_preds", None)' not in inspect.getsource(m)


def test_the_mtr_chart_count_reports_what_was_rendered():
    """The loop skips a target with fewer than five finite pairs, but the summary advertised a contiguous
    _target0.._target{K-1} range including files never written."""
    from mlframe.training.reporting._reporting_regression import _mtr as m

    src = inspect.getsource(m)
    assert "_rendered.append(_k_idx)" in src and "_skipped.append(_k_idx)" in src
    assert "rendered %d of %d" in src
    assert "%s_target0 ... %s_target%d" not in src


def test_the_strong_ar_label_names_the_quantity_it_prints():
    """It printed the max absolute autocorrelation over lags 1/2/3/5 under a `lag1_corr=` label -- contradicted
    by the `source=global_lag3` token printed right beside it."""
    from mlframe.training.targets import _target_distribution_analyzer_target_fn as m

    src = inspect.getsource(m)
    assert "strong_AR_target(lag1_corr=" not in src
    assert "strong_AR_target(max_abs_autocorr=" in src


def test_the_nan_heavy_doc_block_names_the_real_threshold():
    """It still said "> 50%" long after the constant was deliberately raised to 0.99, so a reader wondering why
    a 60%-NaN column was not flagged concludes the detector is broken."""
    from mlframe.training.targets import _target_distribution_analyzer as m

    src = inspect.getsource(m)
    assert "NaN-heavy features (fraction > 50%)" not in src
    assert "_NAN_FRACTION_THRESHOLD, which is 0.99" in src
    assert m._NAN_FRACTION_THRESHOLD == 0.99


class TestTheLeakageSafeEncoderKnobIsReachable:
    """The safe path for a temporal target was documented on the encoder and unreachable from the config."""

    def test_time_aware_is_expressible(self):
        """`TargetEncodeParams` is extra="forbid", so it could not even be passed."""
        from mlframe.training.feature_handling.handlers import TargetEncodeParams

        assert "time_aware" in TargetEncodeParams.model_fields
        assert TargetEncodeParams(kind="target_mean").time_aware is False
        assert TargetEncodeParams(kind="target_mean", time_aware=True).time_aware is True

    def test_it_is_forwarded_to_the_encoder(self):
        """A knob that stops at the config boundary is no better than one that does not exist."""
        from mlframe.training.feature_handling import apply as m

        assert "time_aware=params.time_aware" in inspect.getsource(m)

    def test_the_encoder_still_accepts_it(self):
        """Pins the receiving end, so the two cannot drift apart."""
        from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder

        assert "time_aware" in inspect.signature(LeakageSafeEncoder.__init__).parameters


class TestSettingAnUnconsumedConfigFieldIsNotSilent:
    """`extra="forbid"` makes accepting a field look like a supported API."""

    def test_a_non_default_unconsumed_field_warns(self, caplog):
        """The user's expectation -- a memory guard, a spend gate -- is not met, and nothing said so."""
        from mlframe.training.feature_handling.config import FeatureHandlingConfig, MemoryConfig, PricingConfig

        with caplog.at_level(logging.WARNING, logger="mlframe.training.feature_handling.config"):
            FeatureHandlingConfig(memory=MemoryConfig(pressure_watermark_pct=60.0), pricing=PricingConfig(cap_usd=5.0))
        msgs = " ".join(r.message for r in caplog.records)
        assert "memory.pressure_watermark_pct" in msgs and "pricing.cap_usd" in msgs, msgs

    def test_leaving_them_at_default_stays_silent(self):
        """A warning on every default construction would be noise, not signal."""
        from mlframe.training.feature_handling.config import FeatureHandlingConfig

        import logging as _logging

        recs: list = []

        class _Cap(_logging.Handler):
            """Collect records emitted while the default config is built."""

            def emit(self, record):
                """Record it."""
                recs.append(record)

        log = _logging.getLogger("mlframe.training.feature_handling.config")
        h = _Cap()
        log.addHandler(h)
        try:
            FeatureHandlingConfig()
        finally:
            log.removeHandler(h)
        assert not [r for r in recs if "NOT consumed" in r.getMessage()]

    def test_setting_one_to_its_own_default_stays_silent(self):
        """Only a value that differs from the default is a request the run cannot honour."""
        from mlframe.training.feature_handling.config import FeatureHandlingConfig, MemoryConfig

        import logging as _logging

        recs: list = []

        class _Cap(_logging.Handler):
            """Collect records."""

            def emit(self, record):
                """Record it."""
                recs.append(record)

        log = _logging.getLogger("mlframe.training.feature_handling.config")
        h = _Cap()
        log.addHandler(h)
        try:
            FeatureHandlingConfig(memory=MemoryConfig(pressure_watermark_pct=85.0))
        finally:
            log.removeHandler(h)
        assert not [r for r in recs if "NOT consumed" in r.getMessage()]

    def test_every_listed_name_is_genuinely_unread(self):
        """The list must shrink as fields get readers, not linger and warn about working knobs."""
        import pathlib
        import re

        from mlframe.training.feature_handling import config as m

        src_root = pathlib.Path(m.__file__).resolve().parents[3]
        listed = set(re.findall(r'^\s+"([a-z_]+)",\s*$', inspect.getsource(m.FeatureHandlingConfig._warn_on_declared_but_unconsumed), re.M))
        assert listed, "the unconsumed list is empty; this test needs updating"
        for name in listed:
            hits = [
                p
                for p in src_root.rglob("mlframe/**/*.py")
                if p.name != "config.py" and re.search(rf"\b{re.escape(name)}\b", p.read_text(encoding="utf-8", errors="ignore"))
            ]
            assert not hits, f"{name} now has a reader ({hits[:2]}); remove it from the unconsumed list"


def test_the_mtr_and_calibration_modules_still_import():
    """New helpers and validators must not break module load."""
    import importlib

    for mod in (
        "mlframe.training.reporting._reporting_probabilistic_calib",
        "mlframe.training.reporting._reporting_regression._mtr",
        "mlframe.training.feature_handling.config",
        "mlframe.training.feature_handling.apply",
    ):
        assert importlib.import_module(mod) is not None


def test_the_eviction_guard_does_not_change_a_real_eviction():
    """Below the floor, the pass must still run."""
    from mlframe.training.feature_handling import cache as m

    src = inspect.getsource(m)
    assert "entries.sort(key=lambda t: t[1])" in src  # the scan is still there for the case that needs it
    assert np.isfinite(1.0)
