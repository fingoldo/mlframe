"""W3-F regression-reporting tests: subpackage split smoke + INV-6 / INV-15 fixes.

- Module-split smoke: the carved siblings (``_sensors`` / ``_mtr``) import, expose their public symbols, and a call
  into each MOVED body runs without a NameError (the sibling-split regression class). Re-export identity preserved.
- INV-6: the legacy regression chart path saved with NO file extension; verify an extension-less ``plot_file`` now
  yields ``<plot_file>.png``, and the audit-failure (audit is None) case routes through the spec path (no crash).
- INV-15 / PERF-20: the plotting subsample now force-includes the MaxError point via subsample_preserving_extremes.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Module-split smoke (sibling-split NameError gate)
# ---------------------------------------------------------------------------


def test_subpackage_exports_and_reexport_identity():
    """Subpackage exports and reexport identity."""
    import mlframe.training.reporting._reporting_regression as pkg
    from mlframe.training.reporting._reporting_regression import _mtr, _sensors
    from mlframe.training.reporting._reporting import report_regression_model_perf

    assert hasattr(pkg, "report_regression_model_perf")
    assert hasattr(_sensors, "apply_prediction_envelope_clip")
    assert hasattr(_sensors, "run_collapse_sensor")
    assert hasattr(_mtr, "render_mtr_report")
    # Re-export through _reporting resolves to the SAME function object (no duplicate identity).
    assert report_regression_model_perf is pkg.report_regression_model_perf


def test_apply_prediction_envelope_clip_moved_body_runs():
    """Apply prediction envelope clip moved body runs."""
    from mlframe.training.reporting._reporting_regression._sensors import apply_prediction_envelope_clip

    rng = np.random.default_rng(0)
    yt = rng.normal(0.0, 1.0, 200)
    yp = yt + rng.normal(0.0, 0.2, 200)
    # No envelope stats -> eval-fallback path; train stats -> train path. Both must run + return same shape.
    out_none = apply_prediction_envelope_clip(yp.copy(), yt, y_train_min=None, y_train_max=None, y_train_std=None, model_name="M", report_title="t")
    out_train = apply_prediction_envelope_clip(yp.copy(), yt, y_train_min=-5.0, y_train_max=5.0, y_train_std=1.0, model_name="Ridge", report_title="split")
    assert out_none.shape == yp.shape and out_train.shape == yp.shape
    # A wildly out-of-range prediction must be clipped toward the envelope.
    yp_bad = yp.copy()
    yp_bad[0] = 1e6
    clipped = apply_prediction_envelope_clip(yp_bad, yt, y_train_min=-5.0, y_train_max=5.0, y_train_std=1.0, model_name="Ridge", report_title="split")
    assert clipped[0] < 1e6


def test_run_collapse_sensor_moved_body_runs_warning_branch(caplog):
    """Run collapse sensor moved body runs warning branch."""
    from mlframe.training.reporting._reporting_regression._sensors import run_collapse_sensor

    logging.disable(logging.NOTSET)
    rng = np.random.default_rng(0)
    yt = rng.normal(0.0, 1.0, 200)
    collapsed = np.full(200, float(yt.mean()))  # near-constant pred + R2<0 -> std-collapse branch
    with caplog.at_level(logging.WARNING):
        run_collapse_sensor(collapsed, yt, R2=-2.0, model_name="MLP", y_train_min=-5, y_train_max=5, y_train_std=1.0)
        run_collapse_sensor(collapsed, yt, R2=-2.0, model_name="Ridge")  # non-neural branch
    logging.disable(logging.CRITICAL)
    assert any("regression-collapse-sensor" in r.message for r in caplog.records)


def test_render_mtr_report_moved_body_runs():
    """Render mtr report moved body runs."""
    from mlframe.training.reporting._reporting_regression._mtr import render_mtr_report

    rng = np.random.default_rng(0)
    ytk = rng.normal(0.0, 1.0, (150, 2))
    ypk = ytk + rng.normal(0.0, 0.2, (150, 2))
    m: dict = {}
    _preds, probs = render_mtr_report(
        ytk,
        ypk,
        "M",
        metrics=m,
        print_report=True,
        plot_outputs=None,
        plot_file="",
        figsize=(8, 4),
        plot_sample_size=500,
        plot_dpi=None,
        report_title="t",
        verbose=True,
    )
    assert probs is None
    assert any(k.endswith("_macro") for k in m)  # aggregated MTR metrics stamped


# ---------------------------------------------------------------------------
# INV-6 / INV-15 bug fixes
# ---------------------------------------------------------------------------


def _report(**kw):
    """Report."""
    from mlframe.training.reporting._reporting import report_regression_model_perf

    return report_regression_model_perf(**kw)


def test_inv6_legacy_path_adds_png_extension(tmp_path):
    """Legacy (plot_outputs=None) chart save with an extension-less plot_file MUST write <plot_file>.png."""
    rng = np.random.default_rng(0)
    n = 6000
    yt = rng.normal(0.0, 1.0, n)
    yp = yt + rng.normal(0.0, 0.3, n)
    base = os.path.join(str(tmp_path), "reg_no_ext")  # deliberately no extension
    _report(
        targets=yt,
        columns=["f"],
        model_name="M",
        model=None,
        preds=yp,
        plot_file=base,
        plot_outputs=None,
        metrics={},
        print_report=False,
        show_perf_chart=False,
        plot_sample_size=5000,
    )
    assert os.path.exists(base + ".png"), "legacy savefig must default the extension to .png (INV-6)"
    # And it must NOT have written the extension-less file matplotlib can't infer a format for.
    assert not os.path.exists(base)


def test_inv6_audit_none_routes_through_spec_path(tmp_path, monkeypatch):
    """When the residual audit raises, the chart still renders via the spec path (no extension-less downgrade)."""
    import mlframe.training.targets as targets_mod

    def _boom(*a, **k):
        """Boom."""
        raise RuntimeError("audit deliberately broken")

    # Break the audit so _residual_audit becomes None upstream AND the chart re-attempt also fails.
    monkeypatch.setattr(targets_mod, "audit_residuals", _boom, raising=False)
    rng = np.random.default_rng(1)
    n = 6000
    yt = rng.normal(0.0, 1.0, n)
    yp = yt + rng.normal(0.0, 0.3, n)
    base = os.path.join(str(tmp_path), "reg_audit_none")
    # plot_outputs set -> spec path. Must not crash and must produce a chart even with a broken audit.
    _report(
        targets=yt,
        columns=["f"],
        model_name="M",
        model=None,
        preds=yp,
        plot_file=base,
        plot_outputs="matplotlib[png]",
        metrics={},
        print_report=False,
        show_perf_chart=False,
        plot_sample_size=5000,
    )
    assert any(f.startswith("reg_audit_none") and f.endswith(".png") for f in os.listdir(str(tmp_path)))


def test_inv15_plot_subsample_preserves_max_error():
    """The legacy-path plotting subsample must include the MaxError point (extremes-preserving, PERF-20 / INV-15).

    Verifies at the index level: subsample_preserving_extremes(preds, targets, extreme_values=resid) keeps the row
    whose |residual| is the global max. A uniform rng.choice would drop it with probability (1 - 5000/N).
    """
    from mlframe.reporting.charts._sampling import subsample_preserving_extremes

    rng = np.random.default_rng(2)
    n = 50_000
    yt = rng.normal(0.0, 1.0, n)
    yp = yt + rng.normal(0.0, 0.3, n)
    yt[12345] = 999.0  # the MaxError row
    resid = yt - yp
    worst = int(np.argmax(np.abs(resid)))
    idx = subsample_preserving_extremes(yp, yt, sample_size=5000, extreme_values=resid, rng=42)
    assert worst in set(idx.tolist()), "extremes-preserving subsample must retain the MaxError row"
