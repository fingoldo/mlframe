"""A frame that does not match the target must not reach the frame-paired diagnostics.

The confidence-filtered ensemble reports a COV=10% subset: the target becomes 24,243 rows while ``df`` stays the
full 242,426-row split. In one production run that crashed the separability panel eight times with
``length mismatch X=242426 y=24243`` -- and the crash was the lucky outcome. A diagnostic that does NOT
length-check would have computed on mismatched rows and produced a plausible-looking figure about nothing.

The rows were selected by an index this function never receives, so the frame cannot be realigned here. Dropping
it degrades the frame-paired diagnostics to "skipped", which is honest, and leaves the rest running.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlframe.training.reporting._reporting_diagnostics import _render_post_fit_diagnostics

LOGGER = "mlframe.training.reporting._reporting_diagnostics"


def _cfg():
    """A reporting config with every diagnostic off, so the test exercises the gate and nothing else."""
    return SimpleNamespace(
        pdp_ice=False, slice_finder=False, decision_curve=False, decile_table=False, shap=False,
        shap_interactions=False, shap_per_instance=False, risk_coverage_charts=False, model_card=False,
        class_structure=False, category_discriminability=False, engineered_separability=False,
        interaction_strength=False, pdp_2d=False, learning_curve=False, combined_html_report=False,
        diagnostics_max_seconds=0.0,
    )


def _call(n_rows: int, n_targets: int, caplog, tmp_path):
    """Run the diagnostics entry point with a deliberately mismatched frame and target."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=n_rows), "b": rng.normal(size=n_rows)})
    y = (rng.random(n_targets) < 0.4).astype(int)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _render_post_fit_diagnostics(
            targets=y, model=None, df=df, columns=["a", "b"], preds=None, probs=None,
            target_type="binary_classification", plot_file=str(tmp_path / "m"), plot_outputs="matplotlib[png]",
            metrics={}, reporting_config=_cfg(), model_name="m",
        )
    return " ".join(r.getMessage() for r in caplog.records)


class TestTheAlignmentGate:
    """It must fire on a mismatch and stay silent otherwise."""

    def test_mismatch_is_reported_with_both_counts(self, caplog, tmp_path):
        """The operator needs to see WHICH two numbers disagreed, not a traceback repeated eight times."""
        text = _call(1000, 100, caplog, tmp_path)
        assert "1,000 rows" in text and "100" in text
        assert "skipped rather than computed on mismatched rows" in text

    def test_matching_lengths_pass_silently(self, caplog, tmp_path):
        """A normal report must not carry a warning about alignment."""
        assert "not filtered alongside" not in _call(500, 500, caplog, tmp_path)

    @pytest.mark.parametrize("n_rows, n_targets", [(1000, 100), (100, 1000)])
    def test_either_direction_is_caught(self, n_rows, n_targets, caplog, tmp_path):
        """A shorter frame is just as unusable as a longer one."""
        assert "not filtered alongside" in _call(n_rows, n_targets, caplog, tmp_path)

    def test_no_frame_is_not_a_mismatch(self, caplog, tmp_path):
        """Diagnostics that need no frame must still run when none was supplied."""
        rng = np.random.default_rng(0)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            _render_post_fit_diagnostics(
                targets=(rng.random(50) < 0.4).astype(int), model=None, df=None, columns=["a"],
                preds=None, probs=None, target_type="binary_classification", plot_file="",
                plot_outputs="matplotlib[png]", metrics={}, reporting_config=_cfg(), model_name="m",
            )
        assert "not filtered alongside" not in " ".join(r.getMessage() for r in caplog.records)

    def test_no_target_is_not_a_mismatch(self, caplog, tmp_path):
        """Nothing to compare against; the gate must not fire on an absent target."""
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            _render_post_fit_diagnostics(
                targets=None, model=None, df=df, columns=["a"], preds=None, probs=None,
                target_type="binary_classification", plot_file=str(tmp_path / "m"), plot_outputs="matplotlib[png]",
                metrics={}, reporting_config=_cfg(), model_name="m",
            )
        assert "not filtered alongside" not in " ".join(r.getMessage() for r in caplog.records)
