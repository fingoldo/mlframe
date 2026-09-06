"""A label whose own panel title refers to its suffix must not have that suffix truncated away.

``slice_finder`` and ``category_discriminability`` append the payload their titles tell the reader to look
for -- "(n=12_345, 2.31x)", "(n=248, p=0.35)". The cap kept the head and appended an ellipsis, so on a
two-feature slice (which slice_finder produces by default, and whose bounds string alone runs past the cap)
the chart documented a field it had just deleted.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting.renderers._shared_helpers import _BAR_LABEL_MAXLEN, truncate_bar_label
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import BarPanelSpec

_PAYLOAD = re.compile(r"\(n=[\d_]+,")


def test_the_helper_keeps_the_tail_when_asked():
    """Head, ellipsis, tail -- and never longer than the cap allows."""
    label = "feature_with_a_very_long_generated_name_that_runs_well_past_the_cap  (n=12_345, 2.31x)"
    out = truncate_bar_label(label, maxlen=60, keep_tail=20)
    assert out.endswith(label[-20:]), f"tail was not preserved: {out!r}"
    assert "..." in out, "no ellipsis marks the elision"
    assert len(out) <= 60 + 2, f"result is {len(out)} chars, past the cap"


def test_the_default_stays_head_preserving():
    """Unflagged labels keep the existing shape; this is an opt-in, not a global change."""
    out = truncate_bar_label("x" * 200, maxlen=60)
    assert out.endswith("..."), f"default truncation changed shape: {out!r}"


def test_a_short_label_is_untouched_either_way():
    """Guard: the mode must not rewrite labels that fit."""
    assert truncate_bar_label("short  (n=5)", maxlen=60, keep_tail=20) == "short  (n=5)"


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_slice_finder_labels_still_carry_their_support_and_ratio(backend):
    """End to end, on the default two-feature slices whose bounds alone exceed the cap."""
    from mlframe.reporting.charts.slice_finder import find_weak_slices

    rng = np.random.default_rng(0)
    n, p = 8000, 10
    names = [f"job_posted_at_day_of_year_component_{i}_cos" for i in range(p)]
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=names)
    y_true = rng.normal(size=n)
    y_pred = y_true + rng.normal(0, 1, n) + (X.iloc[:, 0] > 1.2) * 3.0

    spec = find_weak_slices(X, y_true, y_pred).figure
    panel = next(pnl for row in spec.panels for pnl in row if isinstance(pnl, BarPanelSpec))
    assert panel.label_keep_tail > 0, "slice_finder no longer asks for its payload to be preserved"
    assert any(len(c) > _BAR_LABEL_MAXLEN for c in panel.categories), "fixture no longer produces labels past the cap"

    if backend == "matplotlib":
        fig = MatplotlibRenderer().render(spec)
        try:
            drawn = [t.get_text() for t in fig.axes[0].get_yticklabels() if t.get_text()]
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)
    else:
        fig = PlotlyRenderer().render(spec)
        drawn = [t for t in (fig.layout.yaxis.ticktext or []) if t]

    assert drawn, f"{backend} drew no slice labels"
    without_payload = [d for d in drawn if not _PAYLOAD.search(d)]
    assert not without_payload, f"{backend}: the title promises the support and ratio, but these labels lost them: {without_payload[:3]}"
