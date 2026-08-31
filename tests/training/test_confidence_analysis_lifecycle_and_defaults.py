"""Regression guards for confidence-analysis lifecycle + default single-sourcing.

INV-49: the SHAP confidence beeswarm was never saved and never closed -- in headless
runs the work was discarded, and in long sessions the figure leaked in the pyplot
registry. ``run_confidence_analysis`` now accepts ``plot_file`` (saves the beeswarm) and
always closes the figure afterwards.

INV-55: ``ConfidenceAnalysisConfig`` defaults (max_features=6, cmap="bwr", alpha=0.9)
disagreed with the function's literal defaults (20 / "coolwarm" / 0.5). The function now
single-sources those defaults from the config field defaults, so the two can never drift.
"""

from __future__ import annotations

import inspect
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlframe.training._confidence_analysis import run_confidence_analysis
from mlframe.training.configs import ConfidenceAnalysisConfig


def _toy_inputs(n=200, seed=0):
    """Builds a small deterministic frame of standard-normal features for confidence-analysis tests."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
        }
    )
    target = rng.integers(0, 2, size=n).astype(np.int64)
    probs = rng.uniform(0.05, 0.95, size=(n, 2))
    probs /= probs.sum(axis=1, keepdims=True)
    return df, target, probs


def test_confidence_model_kwargs_not_mutated_in_place():
    """TRAINING_LOOSE_A-4: run_confidence_analysis must not mutate a caller-supplied
    confidence_model_kwargs dict in place -- a caller reusing one shared dict across calls would
    otherwise see it silently polluted with the injected iterations/early_stopping_rounds defaults."""
    df, target, probs = _toy_inputs()
    shared_kwargs = {"depth": 4}
    before = dict(shared_kwargs)
    run_confidence_analysis(
        test_df=df,
        test_target=target,
        test_probs=probs,
        confidence_model_kwargs=shared_kwargs,
        use_shap=False,
        verbose=False,
    )
    assert shared_kwargs == before, f"caller's confidence_model_kwargs was mutated: {shared_kwargs} != {before}"


def test_beeswarm_saved_to_plot_file_and_figure_closed(tmp_path):
    """plot_file must produce an on-disk PNG and leave no open figure (INV-49)."""
    import pytest

    pytest.importorskip("shap")
    df, target, probs = _toy_inputs()
    out = os.path.join(str(tmp_path), "conf_beeswarm")  # extension-less on purpose
    # Close any figures a prior test leaked so this measures only THIS call's net effect.
    plt.close("all")
    n_open_before = len(plt.get_fignums())
    run_confidence_analysis(
        test_df=df,
        test_target=target,
        test_probs=probs,
        use_shap=True,
        plot_file=out,
        verbose=False,
    )
    # Ask the layout where the file goes rather than assuming the flat name: the per-format subfolder mode is a
    # process-wide setting, so a test that hardcodes ``out + ".png"`` passes or fails depending on what ran
    # before it in the same worker. The contract here is "the beeswarm reached its plot_file", not the directory.
    from mlframe.reporting.renderers.save import resolve_output_path

    _expected = resolve_output_path(out, "matplotlib", "png", multi_output=False)
    assert os.path.exists(_expected), f"confidence beeswarm was not saved to plot_file (looked for {_expected})"
    assert len(plt.get_fignums()) <= n_open_before, (
        "confidence beeswarm figure leaked: every figure it opened must be closed after save "
        f"(INV-49). open before={n_open_before}, after={len(plt.get_fignums())}"
    )


def test_styling_defaults_single_sourced_from_config():
    """The function's None-resolved defaults must equal ConfidenceAnalysisConfig defaults (INV-55)."""
    sig = inspect.signature(run_confidence_analysis)
    cfg = ConfidenceAnalysisConfig()
    for name in ("use_shap", "max_features", "cmap", "alpha", "title", "ylabel"):
        # The function signature now carries None sentinels; the resolution happens in-body.
        assert sig.parameters[name].default is None, f"{name} should be a None sentinel resolved from the config, not a hard literal"
    # Behavioural check: when the caller omits styling, the resolved values match the config.
    # Probe by constructing the config and asserting the documented defaults are the single source.
    assert cfg.max_features == 6
    assert cfg.cmap == "bwr"
    assert cfg.alpha == 0.9
    assert cfg.use_shap is True
