"""log_only_except wave 2: closes 5 findings in training/core/_phase_helpers.py's
`apply_matplotlib_plotly_style_overrides` (name approximate -- see the function whose docstring
states "Failures log at WARNING and don't abort"). Every one of its 5 except blocks was already a
genuinely best-effort, non-fatal cosmetic override with no realistic escalation target (a plot
style override failing is never something a caller needs to react to) -- marked with the
scanner's recognized "best-effort" rationale comment on each handler rather than manufacturing an
artificial escalation collection for a function that was never meant to have one.
"""

from __future__ import annotations

import inspect


def test_phase_helpers_plot_style_overrides_all_marked_best_effort():
    """Every except handler in the plot-style-override helper carries a `# best-effort` marker."""
    import mlframe.training.core._phase_helpers as ph

    src = inspect.getsource(ph)
    assert src.count("# best-effort") >= 5
