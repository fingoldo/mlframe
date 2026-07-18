"""Tests for ``mlframe.reporting.auto_dispatch.render_multi_target_panels``.

Covers shape-based dispatch (binary / multiclass / multilabel / LTR / no-op) +
file emission via the matplotlib + plotly backends, and the no-op
short-circuits (empty base_path / empty plot_outputs / empty templates /
regression).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from mlframe.reporting import render_multi_target_panels

# ----------------------------------------------------------------------------
# Synthetic input fixtures (mirror the chart-test fixtures)
# ----------------------------------------------------------------------------


@pytest.fixture
def mc_inputs():
    """Multiclass: 200 rows, 3 classes, planted signal."""
    rng = np.random.default_rng(0)
    n, K = 200, 3
    y = rng.integers(0, K, n)
    proba = rng.dirichlet(alpha=[1] * K, size=n)
    for i, t in enumerate(y):
        proba[i, t] += 0.7
        proba[i] /= proba[i].sum()
    return y, proba, ["cat", "dog", "bird"]


@pytest.fixture
def ml_inputs():
    """Multilabel: 200 rows × 3 labels."""
    rng = np.random.default_rng(0)
    n, K = 200, 3
    y = rng.integers(0, 2, (n, K)).astype(np.int8)
    proba = np.clip(rng.uniform(0, 0.5, (n, K)) + y * 0.4, 0.01, 0.99)
    return y, proba, ["spam", "promo", "social"]


@pytest.fixture
def ltr_inputs():
    """LTR: 30 queries × 4-8 docs each, graded relevance 0..3."""
    rng = np.random.default_rng(0)
    y, score, gid = [], [], []
    for q in range(30):
        sz = int(rng.integers(4, 9))
        rels = rng.integers(0, 4, sz)
        scores = rels.astype(float) + rng.normal(0, 0.5, sz)
        y.extend(rels.tolist())
        score.extend(scores.tolist())
        gid.extend([q] * sz)
    return np.asarray(y), np.asarray(score, dtype=np.float64), np.asarray(gid)


@pytest.fixture
def binary_inputs():
    """Binary: 1000 rows, 2 classes with a planted signal so the curve panels build cleanly."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    p = np.clip(y * 0.5 + rng.normal(0, 0.3, 1000) + 0.25, 0.0, 1.0)
    proba = np.column_stack([1 - p, p])
    return y, proba, [0, 1]


# ----------------------------------------------------------------------------
# Dispatch-by-shape
# ----------------------------------------------------------------------------


class TestDispatch:
    """Groups tests for: TestDispatch."""
    def test_multiclass_dispatch(self, mc_inputs, tmp_path):
        """Multiclass dispatch."""
        y, proba, classes = mc_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tag = render_multi_target_panels(
                targets=y,
                probs=proba,
                classes=classes,
                plot_outputs="matplotlib[png]",
                multiclass_panels="CONFUSION PR_F1",
                base_path=str(tmp_path / "mc"),
            )
        assert tag == "multiclass"
        assert os.path.exists(tmp_path / "mc_multiclass_panels.png")

    def test_multilabel_dispatch(self, ml_inputs, tmp_path):
        """Multilabel dispatch."""
        y, proba, labels = ml_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tag = render_multi_target_panels(
                targets=y,
                probs=proba,
                classes=labels,
                plot_outputs="matplotlib[png]",
                multilabel_panels="PR_F1 COOCCURRENCE",
                base_path=str(tmp_path / "ml"),
            )
        assert tag == "multilabel"
        assert os.path.exists(tmp_path / "ml_multilabel_panels.png")

    def test_ltr_dispatch(self, ltr_inputs, tmp_path):
        """Ltr dispatch."""
        y, score, gid = ltr_inputs
        # LTR: scores arrive as ``preds`` (1-D), no ``probs``.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tag = render_multi_target_panels(
                targets=y,
                preds=score,
                group_ids=gid,
                plot_outputs="matplotlib[png]",
                ltr_panels="NDCG_K MRR_DIST",
                base_path=str(tmp_path / "ltr"),
            )
        assert tag == "ltr"
        assert os.path.exists(tmp_path / "ltr_ltr_panels.png")

    def test_binary_renders_default_on(self, binary_inputs, tmp_path):
        """Binary classification now renders curve panels (ROC/PR/...) by default
        when binary_panels is supplied. Authoritative target_type gate routes it."""
        y, proba, classes = binary_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tag = render_multi_target_panels(
                targets=y,
                probs=proba,
                classes=classes,
                plot_outputs="matplotlib[png]",
                binary_panels="ROC PR SCORE_DIST KS THRESHOLD GAIN",
                base_path=str(tmp_path / "bin"),
                target_type="binary_classification",
            )
        assert tag == "binary"
        assert os.path.exists(tmp_path / "bin_binary_panels.png")

    def test_binary_skipped_when_no_binary_template(self, binary_inputs, tmp_path):
        """Binary opt-out: no binary_panels template -> no binary panels, and
        multiclass/multilabel templates do not misfire on a 2-column proba."""
        y, proba, classes = binary_inputs
        tag = render_multi_target_panels(
            targets=y,
            probs=proba,
            classes=classes,
            plot_outputs="matplotlib[png]",
            multiclass_panels="CONFUSION",
            multilabel_panels="PR_F1",
            base_path=str(tmp_path / "bin"),
            target_type="binary_classification",
        )
        assert tag is None
        assert not list(tmp_path.glob("bin*"))

    def test_panel_render_exception_is_tracked_in_panel_failures(self, mc_inputs, tmp_path, monkeypatch):
        """Regression: a render-time exception inside a target-type branch must be surfaced via ``panel_failures``,
        not just a ``logger.exception`` line a caller can miss in a batch run.

        Pre-fix, the dispatcher swallowed the exception and returned ``None`` -- indistinguishable from "nothing
        matched" (a legitimate no-op). ``panel_failures`` lets a caller aggregating across many reports count how
        many actually dropped a whole panel set.
        """
        import mlframe.reporting.charts.multiclass as multiclass_mod

        def _boom(*a, **kw):
            """Helper: Boom."""
            raise RuntimeError("synthetic composer failure")

        monkeypatch.setattr(multiclass_mod, "compose_multiclass_figure", _boom)
        y, proba, classes = mc_inputs
        failures: list = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tag = render_multi_target_panels(
                targets=y,
                probs=proba,
                classes=classes,
                plot_outputs="matplotlib[png]",
                multiclass_panels="CONFUSION PR_F1",
                base_path=str(tmp_path / "mc_boom"),
                panel_failures=failures,
            )
        assert tag is None
        assert failures == ["multiclass"]

    def test_panel_failures_stays_empty_on_legitimate_noop(self, tmp_path):
        """Panel failures stays empty on legitimate noop."""
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 100)
        preds = y + rng.normal(0, 0.1, 100)
        failures: list = []
        tag = render_multi_target_panels(
            targets=y,
            preds=preds,
            probs=None,
            plot_outputs="matplotlib[png]",
            multiclass_panels="CONFUSION",
            multilabel_panels="PR_F1",
            base_path=str(tmp_path / "reg"),
            panel_failures=failures,
        )
        assert tag is None
        assert failures == []

    def test_regression_is_skipped(self, tmp_path):
        # Regression: probs is None.
        """Regression is skipped."""
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 100)
        preds = y + rng.normal(0, 0.1, 100)
        tag = render_multi_target_panels(
            targets=y,
            preds=preds,
            probs=None,
            plot_outputs="matplotlib[png]",
            multiclass_panels="CONFUSION",
            multilabel_panels="PR_F1",
            base_path=str(tmp_path / "reg"),
        )
        assert tag is None


# ----------------------------------------------------------------------------
# No-op short-circuits
# ----------------------------------------------------------------------------


class TestShortCircuits:
    """Groups tests for: TestShortCircuits."""
    def test_empty_base_path_is_noop(self, mc_inputs):
        """Empty base path is noop."""
        y, proba, classes = mc_inputs
        tag = render_multi_target_panels(
            targets=y,
            probs=proba,
            classes=classes,
            plot_outputs="matplotlib[png]",
            multiclass_panels="CONFUSION",
            base_path="",
        )
        assert tag is None

    def test_empty_plot_outputs_is_noop(self, mc_inputs, tmp_path):
        """Empty plot outputs is noop."""
        y, proba, classes = mc_inputs
        tag = render_multi_target_panels(
            targets=y,
            probs=proba,
            classes=classes,
            plot_outputs="",
            multiclass_panels="CONFUSION",
            base_path=str(tmp_path / "x"),
        )
        assert tag is None

    def test_empty_multiclass_template_is_noop(self, mc_inputs, tmp_path):
        """Empty multiclass template is noop."""
        y, proba, classes = mc_inputs
        # Multiclass shape -> dispatcher matches multiclass branch, but
        # the template is empty -> falls through to None.
        tag = render_multi_target_panels(
            targets=y,
            probs=proba,
            classes=classes,
            plot_outputs="matplotlib[png]",
            multiclass_panels="",  # explicit empty
            base_path=str(tmp_path / "x"),
        )
        assert tag is None

    def test_empty_multilabel_template_is_noop(self, ml_inputs, tmp_path):
        """Empty multilabel template is noop."""
        y, proba, labels = ml_inputs
        tag = render_multi_target_panels(
            targets=y,
            probs=proba,
            classes=labels,
            plot_outputs="matplotlib[png]",
            multilabel_panels="",
            base_path=str(tmp_path / "x"),
        )
        assert tag is None

    def test_empty_ltr_template_is_noop(self, ltr_inputs, tmp_path):
        """Empty ltr template is noop."""
        y, score, gid = ltr_inputs
        tag = render_multi_target_panels(
            targets=y,
            preds=score,
            group_ids=gid,
            plot_outputs="matplotlib[png]",
            ltr_panels="",
            base_path=str(tmp_path / "x"),
        )
        assert tag is None


# ----------------------------------------------------------------------------
# Multi-backend emission
# ----------------------------------------------------------------------------


class TestMultiBackend:
    """Groups tests for: TestMultiBackend."""
    def test_multiclass_emits_both_backends(self, mc_inputs, tmp_path):
        """Multiclass emits both backends."""
        y, proba, classes = mc_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_multi_target_panels(
                targets=y,
                probs=proba,
                classes=classes,
                plot_outputs="matplotlib[png] + plotly[html]",
                multiclass_panels="CONFUSION PR_F1",
                base_path=str(tmp_path / "mc"),
            )
        assert os.path.exists(tmp_path / "mc_multiclass_panels.matplotlib.png")
        assert os.path.exists(tmp_path / "mc_multiclass_panels.plotly.html")


# ----------------------------------------------------------------------------
# Dispatch precedence (when multiple branches could plausibly match)
# ----------------------------------------------------------------------------


class TestDispatchPrecedence:
    """Groups tests for: TestDispatchPrecedence."""
    def test_2d_probs_with_group_ids_does_not_misfire_ltr(self, mc_inputs, tmp_path):
        """Caller has multiclass probs + group_ids + ltr_panels (e.g. an
        LTR run that also happens to have probs on a model that's NOT a
        ranker). LTR scores MUST be 1-D; the dispatcher should refuse to
        coerce probs into ranker-scores and skip LTR. With multiclass_panels
        also set, multiclass should win.
        """
        y, proba, classes = mc_inputs
        # group_ids same length as y; arbitrary partition.
        gid = np.repeat(np.arange(20), 10)[: len(y)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tag = render_multi_target_panels(
                targets=y,
                probs=proba,
                preds=None,
                classes=classes,
                group_ids=gid,
                plot_outputs="matplotlib[png]",
                multiclass_panels="CONFUSION",
                ltr_panels="NDCG_K",
                base_path=str(tmp_path / "x"),
            )
        # LTR has priority but bails on 2-D scores -> multiclass takes over.
        assert tag == "multiclass"
        assert os.path.exists(tmp_path / "x_multiclass_panels.png")
        # No LTR file leaked.
        assert not (tmp_path / "x_ltr_panels.png").exists()

    def test_ltr_wins_when_1d_score_present(self, ltr_inputs, tmp_path):
        """When 1-D ranker scores are passed via ``preds`` AND group_ids
        AND ltr_panels, LTR fires even if multilabel/multiclass templates
        are also set."""
        y, score, gid = ltr_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tag = render_multi_target_panels(
                targets=y,
                preds=score,
                probs=None,
                group_ids=gid,
                plot_outputs="matplotlib[png]",
                multiclass_panels="CONFUSION",
                multilabel_panels="PR_F1",
                ltr_panels="NDCG_K",
                base_path=str(tmp_path / "x"),
            )
        assert tag == "ltr"
        assert os.path.exists(tmp_path / "x_ltr_panels.png")


# ----------------------------------------------------------------------------
# Resilience: composer failure must not raise out of the dispatcher
# ----------------------------------------------------------------------------


class TestFailureSwallowing:
    """Groups tests for: TestFailureSwallowing."""
    def test_degenerate_input_returns_none_no_raise(self, tmp_path, caplog):
        """1-row multiclass input -- some sklearn paths inside composers
        crash on degenerate slices. The dispatcher MUST swallow the
        exception (panels are additive; the rest of report_model_perf
        must continue) and return None.
        """
        # Force a guaranteed-broken scenario: probs shape doesn't match
        # targets. compose_multiclass_figure will hit a sklearn-side error.
        y_bad = np.array([0])  # 1 row
        proba_bad = np.array([[]]).reshape(1, 0)  # 0 columns -> bogus
        # ndim==2, shape[1]==0 -- doesn't match the multiclass shape gate
        # (shape[1] >= 3), so dispatcher returns None cleanly without
        # entering the composer at all. Verify no raise + no file.
        result = render_multi_target_panels(
            targets=y_bad,
            probs=proba_bad,
            plot_outputs="matplotlib[png]",
            multiclass_panels="CONFUSION",
            base_path=str(tmp_path / "x"),
        )
        assert result is None
        assert not list(tmp_path.glob("x*"))

    def test_composer_exception_is_swallowed(self, tmp_path, monkeypatch, caplog):
        """If the composer itself raises (e.g. shape OK but downstream
        sklearn metric crashes on a degenerate split), the dispatcher
        must log + return None, not propagate."""
        import logging
        from mlframe.reporting import auto_dispatch

        def _raising_composer(*args, **kwargs):
            """Helper: Raising composer."""
            raise RuntimeError("synthetic composer failure")

        # Patch the composer used by the multiclass branch.
        import mlframe.reporting.charts.multiclass as mc_mod

        monkeypatch.setattr(mc_mod, "compose_multiclass_figure", _raising_composer)

        # Re-import inside dispatcher path: dispatcher does
        # ``from mlframe.reporting.charts.multiclass import compose_multiclass_figure``
        # at call time, so monkeypatching the module attribute is enough.
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 50)
        proba = rng.dirichlet([1, 1, 1], size=50)

        with caplog.at_level(logging.ERROR, logger=auto_dispatch.logger.name):
            tag = render_multi_target_panels(
                targets=y,
                probs=proba,
                classes=[0, 1, 2],
                plot_outputs="matplotlib[png]",
                multiclass_panels="CONFUSION",
                base_path=str(tmp_path / "x"),
            )
        assert tag is None
        # No file written because composer raised before render.
        assert not list(tmp_path.glob("x*"))
        # Caller saw a logged exception.
        assert any("Multiclass panel rendering failed" in r.getMessage() for r in caplog.records)


# ----------------------------------------------------------------------------
# Integration: dispatcher fires from the real report_model_perf hot path
# ----------------------------------------------------------------------------


class TestReportModelPerfIntegration:
    """Smoke that the new ``plot_outputs`` / ``*_panels`` kwargs threaded
    into ``report_model_perf`` actually reach the dispatcher and emit
    files. Uses ``model=None`` + pre-computed probs (just_evaluate path)
    so we don't need to fit anything."""

    def test_multiclass_via_report_model_perf(self, mc_inputs, tmp_path):
        """Multiclass via report model perf."""
        from mlframe.training.evaluation import report_model_perf

        y, proba, classes = mc_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _preds, _probs = report_model_perf(
                targets=y,
                columns=[],
                model_name="testmc",
                model=None,
                preds=np.argmax(proba, axis=1),
                probs=proba,
                classes=classes,
                plot_file=str(tmp_path / "smoke"),
                plot_outputs="matplotlib[png]",
                multiclass_panels="CONFUSION PR_F1",
                show_perf_chart=False,  # skip the legacy per-class calib plot
                show_fi=False,
                print_report=False,
            )
        # Multi-target panel file emitted by the dispatcher.
        assert os.path.exists(tmp_path / "smoke_multiclass_panels.png")

    def test_multilabel_via_report_model_perf(self, ml_inputs, tmp_path):
        """Multilabel via report model perf."""
        from mlframe.training.evaluation import report_model_perf

        y, proba, labels = ml_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _preds, _probs = report_model_perf(
                targets=y,
                columns=[],
                model_name="testml",
                model=None,
                preds=(proba >= 0.5).astype(np.int8),
                probs=proba,
                classes=labels,
                plot_file=str(tmp_path / "smoke"),
                plot_outputs="matplotlib[png]",
                multilabel_panels="PR_F1 COOCCURRENCE",
                show_perf_chart=False,
                show_fi=False,
                print_report=False,
            )
        assert os.path.exists(tmp_path / "smoke_multilabel_panels.png")

    def test_binary_via_report_model_perf(self, binary_inputs, tmp_path):
        """Binary curve panels render default-ON through report_model_perf, the
        decile table lands in the metrics dict, and metrics['charts'] is recorded."""
        from mlframe.training.evaluation import report_model_perf

        y, proba, classes = binary_inputs
        metrics: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report_model_perf(
                targets=y,
                columns=[],
                model_name="testbin",
                model=None,
                preds=(proba[:, 1] > 0.5).astype(int),
                probs=proba,
                classes=classes,
                plot_file=str(tmp_path / "smoke"),
                plot_outputs="matplotlib[png]",
                binary_panels="ROC PR SCORE_DIST KS THRESHOLD GAIN",
                target_type="binary_classification",
                metrics=metrics,
                show_perf_chart=False,
                show_fi=False,
                print_report=False,
            )
        assert os.path.exists(tmp_path / "smoke_binary_panels.png")
        assert metrics["charts"]["saved"] == ["binary_panels"]
        assert metrics["charts"]["failed"] == []
        dec = metrics["binary_decile_table"]
        assert dec["decile"].tolist() == list(range(1, 11))
        # Terminal cumulative gain is 1.0 by construction (all positives captured).
        assert abs(float(dec["gain"][-1]) - 1.0) < 1e-9
        # Planted signal: top decile lift clearly above the no-skill 1.0.
        assert float(dec["lift"][0]) >= 1.3

    def test_no_panels_kwargs_means_no_file(self, mc_inputs, tmp_path):
        """When the caller doesn't opt in (templates remain None),
        report_model_perf must NOT write any *_panels file -- legacy
        behaviour preserved."""
        from mlframe.training.evaluation import report_model_perf

        y, proba, classes = mc_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report_model_perf(
                targets=y,
                columns=[],
                model_name="testmc",
                model=None,
                preds=np.argmax(proba, axis=1),
                probs=proba,
                classes=classes,
                plot_file=str(tmp_path / "smoke"),
                # plot_outputs / multiclass_panels NOT supplied -> no-op
                show_perf_chart=False,
                show_fi=False,
                print_report=False,
            )
        assert not list(tmp_path.glob("*_panels.png"))
