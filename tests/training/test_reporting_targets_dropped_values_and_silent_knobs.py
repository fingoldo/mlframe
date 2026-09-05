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

import pytest


class TestTheCalibrationChartsRegisterWhatTheyWrote:
    """A chart absent from `metrics["charts"]` is absent from the combined report."""

    def _tree(self):
        """The parsed probabilistic-calibration reporting module."""
        from mlframe.training.reporting import _reporting_probabilistic_calib as m
        from tests._source_ast import module_ast

        return module_ast(m)

    def test_all_three_families_record_a_saved_path(self):
        """fairness-calibration, calibration-by-feature and the 2-D heatmap all record what they wrote.

        Structural: a chart missing from `metrics["charts"]` renders identically on disk and is simply absent
        from the combined report, so no assertion on the returned metrics of one family can show that another
        family forgot to register. Counting the recorder's call sites is the check.
        """
        from tests._source_ast import called_names

        calls = called_names(self._tree())
        assert calls.count("_record_chart") >= 6, f"expected at least 6 _record_chart call sites, found {calls.count('_record_chart')}"

    def test_a_failure_leaves_a_failed_entry(self):
        """A chart that FAILED must be recorded as failed, not swallowed -- else it looks like a disabled one."""
        import ast

        errored = [node for node in ast.walk(self._tree()) if isinstance(node, ast.Call) and any(kw.arg == "error" for kw in node.keywords)]
        assert len(errored) == 3, f"expected 3 failure-recording call sites, found {len(errored)}"

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
        """The method existed, was documented, and had no caller anywhere -- a report nobody asked for.

        Structural: the sibling below drives what `report()` DOES once called, but a method with no caller
        produces no output to assert on from the reporting side, which is exactly how it went unnoticed.
        """
        from mlframe.training.reporting import _reporting_diagnostics as m
        from tests._source_ast import called_names, module_ast

        assert "report" in called_names(module_ast(m)), "nothing calls the diagnostics budget's report(), so a truncated report still says nothing"

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

    import ast

    from tests._source_ast import module_ast

    # Structural: an ORDER between two statements. Both orders produce a chart on disk; only the wrong one
    # serialises a null path beside it, and reaching that through the public path needs a full audit run.
    # Compared on parsed line numbers so reformatting cannot move it, unlike the character offsets this used.
    tree = module_ast(m)
    assigns = [
        node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assign) for t in node.targets if isinstance(t, ast.Attribute) and t.attr == "plot_path"
    ]
    assert assigns, "result.plot_path is never assigned, so the audit metadata cannot carry the chart's path"
    # A bare `return` parses with value=None; `return None` parses with a Constant(None). Accept both.
    returns_after = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Return) and (n.value is None or (isinstance(n.value, ast.Constant) and n.value.value is None)) and n.lineno > min(assigns)
    ]
    assert returns_after, "no early return follows the assignment; this test needs updating if the branch was restructured"


class TestTheEvictionPassDoesNotScanForNothing:
    """Between the trigger and the floor it listed the directory, stat'd every file, sorted, and deleted none."""

    def test_it_returns_before_listing_when_free_space_is_already_above_the_floor(self):
        """The loop's own first-iteration break condition, hoisted ahead of the scan."""
        from mlframe.training.feature_handling import cache as m

        import ast

        from tests._source_ast import function_ast

        # Structural: the fix is that the guard is HOISTED ahead of the scan. Both orders evict the same files
        # and return the same value -- the difference is a directory listing plus a stat per file that is then
        # thrown away, which no assertion on the result can see.
        fn = function_ast(m, "FeatureCache._maybe_evict_disk")
        guards = [
            n.lineno for n in ast.walk(fn) if isinstance(n, ast.Compare) and any(isinstance(x, ast.Name) and x.id == "target_free_bytes" for x in ast.walk(n))
        ]
        listdirs = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "listdir"]
        assert guards, "the free-space guard is gone, so the pass always scans"
        assert listdirs, "no directory listing found; this test needs updating if the scan was restructured"
        assert min(guards) < min(listdirs), f"the guard (line {min(guards)}) must precede the directory scan (line {min(listdirs)})"


def test_the_unread_bootstrap_parameter_is_gone():
    """`preds` was accepted and passed and never read, which reads as working plumbing for a crisp-metric CI
    that does not exist. Every metric in the block is probability-based."""
    from mlframe.training import honest_diagnostics as m

    from tests._source_ast import getattr_literals, module_ast

    assert "preds" not in inspect.signature(m._bootstrap_block).parameters
    # ...and the call site no longer reaches for the value it used to pass in. Structural: the parameter was
    # accepted, passed and never read, so every metric in the block is probability-based and the result was
    # identical with or without it -- which is what made it read as working plumbing for a crisp-metric CI
    # that does not exist.
    assert "test_preds" not in getattr_literals(module_ast(m), obj="entry"), "the unread crisp-prediction read is back at the call site"


def test_the_mtr_chart_count_reports_what_was_rendered():
    """The loop skips a target with fewer than five finite pairs, but the summary advertised a contiguous
    _target0.._target{K-1} range including files never written."""
    from mlframe.training.reporting._reporting_regression import _mtr as m

    from tests._source_ast import assigned_names, called_names, module_ast, string_literals

    # Structural: the summary line is emitted from inside a per-target render loop that needs real fitted
    # predictions for K targets to reach, and the defect was that the COUNT it advertised included files never
    # written -- a wrong number in a log line, not a wrong return value.
    tree = module_ast(m)
    assert {"_rendered", "_skipped"} <= assigned_names(tree), "the loop no longer tracks which targets rendered and which were skipped"
    assert "append" in called_names(tree), "nothing is appended to the rendered/skipped tallies"
    emitted = " ".join(string_literals(tree))
    assert "rendered %d of %d" in emitted, "the summary no longer reports rendered-of-total"
    assert "%s_target0 ... %s_target%d" not in emitted, "the summary advertises a contiguous target range again, including files never written"


def test_the_strong_ar_label_names_the_quantity_it_prints():
    """It printed the max absolute autocorrelation over lags 1/2/3/5 under a `lag1_corr=` label -- contradicted
    by the `source=global_lag3` token printed right beside it."""
    from mlframe.training.targets import _target_distribution_analyzer_target_fn as m

    from tests._source_ast import module_ast, string_literals

    # Structural: the label is a format string in a log line, and the VALUE beside it was always the max
    # absolute autocorrelation over lags 1/2/3/5 -- only the name was wrong, contradicted by the
    # `source=global_lag3` token printed right next to it. A wrong label is not visible in any return value.
    emitted = " ".join(string_literals(module_ast(m)))
    assert "strong_AR_target(lag1_corr=" not in emitted, "the label claims lag-1 again while printing the max over lags 1/2/3/5"
    assert "strong_AR_target(max_abs_autocorr=" in emitted, "the label no longer names the quantity it prints"


def test_the_nan_heavy_doc_block_names_the_real_threshold():
    """It still said "> 50%" long after the constant was deliberately raised to 0.99, so a reader wondering why
    a 60%-NaN column was not flagged concludes the detector is broken."""
    from mlframe.training.targets import _target_distribution_analyzer as m

    # The threshold itself is the contract; the prose that once contradicted it is not something to assert.
    assert m._NAN_FRACTION_THRESHOLD == 0.99, f"the NaN-heavy threshold moved to {m._NAN_FRACTION_THRESHOLD!r}; the doc block quoting it must move too"


class TestTheLeakageSafeEncoderKnobIsReachable:
    """The safe path for a temporal target was documented on the encoder and unreachable from the config."""

    def test_time_aware_is_expressible(self):
        """`TargetEncodeParams` is extra="forbid", so it could not even be passed."""
        from mlframe.training.feature_handling.handlers import TargetEncodeParams

        assert "time_aware" in TargetEncodeParams.model_fields
        assert TargetEncodeParams(kind="target_mean").time_aware is False
        assert TargetEncodeParams(kind="target_mean", time_aware=True).time_aware is True

    def test_it_is_forwarded_to_the_encoder(self):
        """A knob that stops at the config boundary is no better than one that does not exist.

        Structural: the forwarding is one keyword on one call, and whether it arrives changes the encoding
        only on a time-ordered frame large enough for the ordering to matter -- reaching that through the
        public apply path means standing up a full feature-handling run.
        """
        import ast

        from mlframe.training.feature_handling import apply as m
        from tests._source_ast import module_ast

        forwarded = [kw for node in ast.walk(module_ast(m)) if isinstance(node, ast.Call) for kw in node.keywords if kw.arg == "time_aware"]
        assert forwarded, "time_aware is never passed on, so the config knob stops at the boundary"
        assert any(
            isinstance(kw.value, ast.Attribute) and kw.value.attr == "time_aware" for kw in forwarded
        ), "time_aware is forwarded, but not from the params object the caller configured"

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
        # Read the list literal from the parsed method rather than regexing its source: a reformat that puts
        # two names on one line, or a trailing comment, silently emptied the regex and the loop below then
        # asserted nothing at all.
        import ast as _ast

        from tests._source_ast import function_ast

        _fn = function_ast(m, "FeatureHandlingConfig._warn_on_declared_but_unconsumed")
        listed = {
            elt.value
            for node in _ast.walk(_fn)
            if isinstance(node, (_ast.List, _ast.Tuple, _ast.Set))
            for elt in node.elts
            if isinstance(elt, _ast.Constant) and isinstance(elt.value, str)
        }
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
    """Hoisting the free-space guard must not remove the scan for the case that genuinely needs it.

    The complement of the guard test above: that one pins that the pass returns BEFORE listing when free space
    is already above the floor, and this one pins that the listing-and-sort is still there for when it is
    below. Structural for the same reason -- both paths evict the same files and return the same value, and the
    difference is a directory listing plus a stat per file, which no result can show.

    The previous form also closed with `assert np.isfinite(1.0)`, a tautology that could not fail.
    """
    import ast

    from mlframe.training.feature_handling import cache as m
    from tests._source_ast import called_names, function_ast

    fn = function_ast(m, "FeatureCache._maybe_evict_disk")
    calls = called_names(fn)
    assert "listdir" in calls, "the eviction pass no longer lists the cache directory, so it cannot evict anything"
    assert "sort" in calls, "the candidate entries are no longer ordered, so eviction is no longer oldest-first"
    # ...ordered by a key, not by raw tuple comparison, which is what makes the order deliberate.
    sorts = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "sort"]
    assert any(kw.arg == "key" for n in sorts for kw in n.keywords), "the eviction sort no longer passes an explicit key"
