"""Cluster 37 regression: the plot_outputs read has no dead NameError guard (audit 2026-11-20, finding 37-F40).

``getattr(config, "plot_outputs", "")`` with a default cannot raise NameError, so the try/except around it was dead. The read is now a plain
expression with the same value.
"""

from pathlib import Path

from mlframe.training.core import _phase_config_setup as setup


def test_plot_outputs_read_is_not_wrapped_in_a_dead_try():
    """The getattr read is no longer inside a NameError try block (37-F40)."""
    src = Path(setup.__file__).resolve().read_text(encoding="utf-8")
    assert "try:\n            _po = getattr(reporting_config" not in src
    assert '_po = getattr(reporting_config, "plot_outputs", "") or ""' in src
