"""Regression: the chart-render kill-switch must fire regardless of verbose.

INV-5 bug class: the block that clears ``output_config.plot_file`` when the run
is non-interactive and ``save_charts=False`` lived inside ``if verbose:``. A
``verbose=0`` run with a stale non-empty ``plot_file`` would then render 100+
charts to a temp dir nobody sees (~60s wasted on 1M-row multiclass). The gating
computation + mutation are now hoisted out of the verbose block; only the INFO
log lines stay verbose-gated.
"""

from __future__ import annotations

from mlframe.training._training_runtime_configs import OutputConfig
from mlframe.training.core._phase_config_setup import setup_configuration


def _setup(output_config, verbose):
    return setup_configuration(
        preprocessing_config=None,
        pipeline_config=None,
        feature_types_config=None,
        split_config=None,
        hyperparams_config=None,
        behavior_config=None,
        reporting_config=None,
        output_config=output_config,
        outlier_detection_config=None,
        feature_selection_config=None,
        confidence_analysis_config=None,
        baseline_diagnostics_config=None,
        dummy_baselines_config=None,
        quantile_regression_config=None,
        composite_target_discovery_config=None,
        feature_handling_config=None,
        linear_model_config=None,
        multilabel_dispatch_config=None,
        model_name="sc_test",
        target_name="sc_test",
        mlframe_models=None,
        verbose=verbose,
    )


def test_short_circuit_clears_plot_file_when_verbose_zero():
    """verbose=0 + save_charts=False must still clear plot_file (pre-fix: did not)."""
    oc = OutputConfig(data_dir="", plot_file="C:/tmp/stale_plot", save_charts=False)
    ctx = _setup(oc, verbose=0)
    assert ctx.output_config.plot_file == "", (
        "plot_file should be cleared by the short-circuit on a non-interactive "
        "verbose=0 + save_charts=False run; got "
        f"{ctx.output_config.plot_file!r}. Pre-fix this gating lived under "
        "`if verbose:` so a quiet run still rendered every chart."
    )


def test_short_circuit_also_clears_when_verbose_nonzero():
    """The fix must not regress the original verbose>0 behavior."""
    oc = OutputConfig(data_dir="", plot_file="C:/tmp/stale_plot", save_charts=False)
    ctx = _setup(oc, verbose=1)
    assert ctx.output_config.plot_file == ""


def test_short_circuit_inactive_keeps_plot_file_when_saving():
    """save_charts=True (with a data_dir) must NOT clear plot_file at any verbosity."""
    oc = OutputConfig(data_dir="C:/tmp/charts", plot_file="C:/tmp/keep_plot", save_charts=True)
    ctx = _setup(oc, verbose=0)
    assert ctx.output_config.plot_file == "C:/tmp/keep_plot"
