"""Tests for opt-in data-aware binary panel emphasis.

Covers the pure selection helper ``select_binary_emphasis_panels`` (back-compat
default, imbalanced vs balanced selection, single-class / tiny-n fallback,
custom-template-not-reordered gate) plus end-to-end emphasis through
``render_multi_target_panels`` and a biz_value assertion that the adaptive
order leads with the right diagnostic for the data skew.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

from mlframe.reporting import render_multi_target_panels
from mlframe.reporting.auto_dispatch import select_binary_emphasis_panels

_DEFAULT = "ROC PR SCORE_DIST KS THRESHOLD GAIN PIT"


def _imbalanced_y(n=5000, rate=0.03, seed=0):
    return (np.random.default_rng(seed).random(n) < rate).astype(int)


def _balanced_y(n=5000, seed=1):
    return (np.random.default_rng(seed).random(n) < 0.5).astype(int)


# ---------------------------------------------------------------------------
# Pure helper: selection logic
# ---------------------------------------------------------------------------


class TestSelectBinaryEmphasisPanels:
    def test_all_mode_is_identity(self):
        """Default emphasis="all" returns the template unchanged (back-compat)."""
        y = _imbalanced_y()
        assert select_binary_emphasis_panels(y, _DEFAULT, emphasis="all") == _DEFAULT

    def test_imbalanced_leads_pr_threshold_and_drops_roc(self):
        y = _imbalanced_y()
        out = select_binary_emphasis_panels(y, _DEFAULT, emphasis="data_aware").split()
        assert out[0] == "PR"
        assert out[1] == "THRESHOLD"
        assert "ROC" not in out
        # Other requested panels are preserved (no silent loss).
        for tok in ("SCORE_DIST", "KS", "GAIN", "PIT"):
            assert tok in out

    def test_high_base_rate_also_imbalanced(self):
        """Base rate > hi threshold is just as imbalanced as < lo."""
        y = (np.random.default_rng(2).random(5000) < 0.97).astype(int)
        out = select_binary_emphasis_panels(y, _DEFAULT, emphasis="data_aware").split()
        assert out[0] == "PR"
        assert "ROC" not in out

    def test_balanced_leads_roc(self):
        y = _balanced_y()
        out = select_binary_emphasis_panels(y, _DEFAULT, emphasis="data_aware").split()
        assert out[0] == "ROC"
        assert "ROC" in out and "PR" in out

    def test_single_class_falls_back(self):
        y = np.ones(5000, dtype=int)
        assert select_binary_emphasis_panels(y, _DEFAULT, emphasis="data_aware") == _DEFAULT
        y0 = np.zeros(5000, dtype=int)
        assert select_binary_emphasis_panels(y0, _DEFAULT, emphasis="data_aware") == _DEFAULT

    def test_tiny_n_falls_back(self):
        y = _imbalanced_y(n=20)
        assert select_binary_emphasis_panels(y, _DEFAULT, emphasis="data_aware") == _DEFAULT

    def test_empty_template_is_returned_as_is(self):
        assert select_binary_emphasis_panels(_imbalanced_y(), "", emphasis="data_aware") == ""

    def test_nan_labels_are_ignored_for_base_rate(self):
        """Float labels with NaN: only finite entries count toward the base rate."""
        y = _imbalanced_y().astype(float)
        y[:100] = np.nan
        out = select_binary_emphasis_panels(y, _DEFAULT, emphasis="data_aware").split()
        assert out[0] == "PR"

    def test_configurable_thresholds(self):
        """A base rate of 0.25 is imbalanced only under a tighter lo threshold."""
        y = (np.random.default_rng(3).random(5000) < 0.25).astype(int)
        # Default lo=0.2 -> 0.25 counts as balanced -> ROC-led.
        assert select_binary_emphasis_panels(y, _DEFAULT, emphasis="data_aware").split()[0] == "ROC"
        # Tighter lo=0.3 -> 0.25 counts as imbalanced -> PR-led.
        assert (
            select_binary_emphasis_panels(
                y,
                _DEFAULT,
                emphasis="data_aware",
                imbalance_lo=0.3,
            ).split()[0]
            == "PR"
        )


# ---------------------------------------------------------------------------
# Dispatcher: emphasis only applied when binary_panels_is_default + data_aware
# ---------------------------------------------------------------------------


def _binary_inputs(y):
    rng = np.random.default_rng(7)
    p = np.clip(y * 0.5 + rng.normal(0, 0.3, len(y)) + 0.25, 0.0, 1.0)
    return np.column_stack([1 - p, p])


class TestDispatcherEmphasis:
    def test_default_mode_unchanged(self, tmp_path):
        """panel_emphasis defaults to "all": dispatcher renders the requested set."""
        y = _imbalanced_y()
        proba = _binary_inputs(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tag = render_multi_target_panels(
                targets=y,
                probs=proba,
                classes=[0, 1],
                plot_outputs="matplotlib[png]",
                binary_panels=_DEFAULT,
                base_path=str(tmp_path / "bin"),
                target_type="binary_classification",
            )
        assert tag == "binary"
        assert os.path.exists(tmp_path / "bin_binary_panels.png")

    def test_data_aware_applies_only_when_default(self, monkeypatch):
        """Emphasis fires only when binary_panels_is_default=True; a custom
        template is passed through untouched even in data_aware mode."""

        captured = {}

        def _spy(yt, ys, *, panels_template, **kw):
            captured["template"] = panels_template
            raise RuntimeError("stop before render")

        import mlframe.reporting.charts.binary as bin_mod

        monkeypatch.setattr(bin_mod, "compose_binary_figure", _spy)

        y = _imbalanced_y()
        proba = _binary_inputs(y)

        # default template + data_aware + is_default -> reordered (PR-led).
        render_multi_target_panels(
            targets=y,
            probs=proba,
            plot_outputs="matplotlib[png]",
            binary_panels=_DEFAULT,
            base_path="x",
            target_type="binary_classification",
            panel_emphasis="data_aware",
            binary_panels_is_default=True,
        )
        assert captured["template"].split()[0] == "PR"

        # custom template + data_aware but is_default=False -> untouched.
        custom = "ROC GAIN"
        render_multi_target_panels(
            targets=y,
            probs=proba,
            plot_outputs="matplotlib[png]",
            binary_panels=custom,
            base_path="x",
            target_type="binary_classification",
            panel_emphasis="data_aware",
            binary_panels_is_default=False,
        )
        assert captured["template"] == custom


# ---------------------------------------------------------------------------
# Suite threading: ReportingConfig.panel_emphasis reaches the dispatcher via
# report_model_perf (the boundary where binary_panels_is_default is computed).
# ---------------------------------------------------------------------------


class TestReportModelPerfThreading:
    """report_model_perf reads panel_emphasis / emphasis_imbalance_* off the
    ReportingConfig and computes binary_panels_is_default by comparing the
    threaded binary_panels to the field default, so data_aware takes effect in
    a real suite run (the dispatcher is otherwise never told to emphasize)."""

    def _run(self, monkeypatch, tmp_path, reporting_config, binary_panels):
        from mlframe.training.configs import ReportingConfig  # noqa: F401  (ensures model_fields cache primes)
        import mlframe.reporting.charts.binary as bin_mod
        from mlframe.training.reporting import _reporting as rep

        captured = {}

        def _spy(yt, ys, *, panels_template, **kw):
            captured["template"] = panels_template
            raise RuntimeError("stop before render")

        monkeypatch.setattr(bin_mod, "compose_binary_figure", _spy)

        y = _imbalanced_y(rate=0.03)
        proba = _binary_inputs(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep.report_model_perf(
                targets=y,
                columns=["f0"],
                model_name="m",
                model=None,
                preds=(proba[:, 1] >= 0.5).astype(int),
                probs=proba,
                classes=[0, 1],
                print_report=False,
                show_perf_chart=False,
                show_fi=False,
                plot_file=str(tmp_path / "bin"),
                plot_outputs="matplotlib[png]",
                target_type="binary_classification",
                binary_panels=binary_panels,
                reporting_config=reporting_config,
            )
        return captured.get("template")

    def test_data_aware_default_template_reorders_pr_led(self, monkeypatch, tmp_path):
        from mlframe.training.configs import ReportingConfig

        cfg = ReportingConfig(panel_emphasis="data_aware")
        tmpl = self._run(monkeypatch, tmp_path, cfg, cfg.binary_panels).split()
        assert tmpl[0] == "PR"
        assert "ROC" not in tmpl

    def test_all_mode_passes_default_through_unchanged(self, monkeypatch, tmp_path):
        from mlframe.training.configs import ReportingConfig

        cfg = ReportingConfig()  # panel_emphasis defaults to "all"
        assert cfg.panel_emphasis == "all"
        tmpl = self._run(monkeypatch, tmp_path, cfg, cfg.binary_panels)
        assert tmpl == _DEFAULT

    def test_custom_template_not_reordered_even_in_data_aware(self, monkeypatch, tmp_path):
        from mlframe.training.configs import ReportingConfig

        custom = "ROC GAIN"
        cfg = ReportingConfig(panel_emphasis="data_aware", binary_panels=custom)
        # binary_panels_is_default is False (custom != field default) -> untouched.
        tmpl = self._run(monkeypatch, tmp_path, cfg, cfg.binary_panels)
        assert tmpl == custom

    def test_single_class_falls_back_to_all(self, monkeypatch, tmp_path):
        from mlframe.training.configs import ReportingConfig
        import mlframe.reporting.charts.binary as bin_mod
        from mlframe.training.reporting import _reporting as rep

        captured = {}

        def _spy(yt, ys, *, panels_template, **kw):
            captured["template"] = panels_template
            raise RuntimeError("stop")

        monkeypatch.setattr(bin_mod, "compose_binary_figure", _spy)
        cfg = ReportingConfig(panel_emphasis="data_aware")
        y = np.ones(5000, dtype=int)
        proba = _binary_inputs(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep.report_model_perf(
                targets=y,
                columns=["f0"],
                model_name="m",
                model=None,
                preds=np.ones(len(y), dtype=int),
                probs=proba,
                classes=[0, 1],
                print_report=False,
                show_perf_chart=False,
                show_fi=False,
                plot_file=str(tmp_path / "bin"),
                plot_outputs="matplotlib[png]",
                target_type="binary_classification",
                binary_panels=cfg.binary_panels,
                reporting_config=cfg,
            )
        assert captured["template"] == _DEFAULT


# ---------------------------------------------------------------------------
# biz_value: adaptive order surfaces the right diagnostic for the skew
# ---------------------------------------------------------------------------


class TestBizValuePanelEmphasis:
    def test_biz_imbalanced_leads_pr_drops_roc(self):
        """On a 0.03 base-rate synthetic, data_aware leads with PR/THRESHOLD and
        excludes ROC (optimistic under imbalance). all-mode keeps ROC first."""
        y = _imbalanced_y(rate=0.03)
        data_aware = select_binary_emphasis_panels(y, _DEFAULT, emphasis="data_aware").split()
        assert data_aware[0] == "PR"
        assert data_aware[1] == "THRESHOLD"
        assert "ROC" not in data_aware
        # Back-compat: all-mode is byte-identical to the requested default.
        assert select_binary_emphasis_panels(y, _DEFAULT, emphasis="all") == _DEFAULT

    def test_biz_balanced_includes_and_leads_roc(self):
        """On a 0.5 base-rate synthetic, data_aware includes and leads with ROC."""
        y = _balanced_y()
        data_aware = select_binary_emphasis_panels(y, _DEFAULT, emphasis="data_aware").split()
        assert data_aware[0] == "ROC"
        assert "ROC" in data_aware

    def test_biz_backcompat_default_identical_to_current(self):
        """The default (emphasis="all") selection is identical to the current
        behavior for BOTH skews -- no surprise change for existing users."""
        for y in (_imbalanced_y(rate=0.03), _balanced_y()):
            assert select_binary_emphasis_panels(y, _DEFAULT, emphasis="all") == _DEFAULT


# ---------------------------------------------------------------------------
# cProfile: emphasis selection is a trivial O(n) pass
# ---------------------------------------------------------------------------


def test_cprofile_emphasis_is_trivial():
    """The base-rate derivation is one O(n) pass; assert it is sub-millisecond
    on a 1M-row label vector so it adds no measurable dispatch overhead."""
    import cProfile
    import pstats
    import io

    y = _imbalanced_y(n=1_000_000)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(20):
        select_binary_emphasis_panels(y, _DEFAULT, emphasis="data_aware")
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(5)
    total = pstats.Stats(pr).total_tt
    # 20 calls over 1M rows -> well under 0.5s even with profiler overhead.
    assert total < 0.5, s.getvalue()
