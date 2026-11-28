"""Cluster 37 regression: the suite's matplotlib / plotly style overrides can be undone (audit 2026-11-20, finding 37-F42)."""

import matplotlib

matplotlib.use("Agg")

from mlframe.training.core._phase_helpers import _apply_plot_style_overrides
from mlframe.training.core._process_flag_scope import restore_process_flags


def test_rcparams_override_is_applied_and_restored():
    """An rcParams override takes effect, and the returned restore callable puts the previous value back."""
    before = matplotlib.rcParams["font.size"]
    restore = _apply_plot_style_overrides(matplotlib_rcparams={"font.size": before + 7})
    assert matplotlib.rcParams["font.size"] == before + 7
    restore()
    assert matplotlib.rcParams["font.size"] == before


def test_nothing_overridden_returns_none_and_the_scope_helper_runs_the_stashed_restore():
    """No override gives None; a restore stashed in the artifacts is invoked (and removed) by restore_process_flags."""
    assert _apply_plot_style_overrides() is None
    before = matplotlib.rcParams["font.size"]
    restore = _apply_plot_style_overrides(matplotlib_rcparams={"font.size": before + 3})
    artifacts = {"_process_flag_prior_plot_style_restore": restore}
    restore_process_flags(artifacts)
    assert matplotlib.rcParams["font.size"] == before and not artifacts
