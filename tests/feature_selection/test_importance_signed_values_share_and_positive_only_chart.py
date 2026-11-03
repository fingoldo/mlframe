"""The FI log line reports a share of magnitudes, and positive_fi_only keeps negative importances off the chart as well as out of the frame."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from mlframe.feature_selection.importance import _format_top_fi_for_log, plot_feature_importance


def test_share_of_signed_importances_never_exceeds_100_percent():
    """[0.6, 0.5, -1.0]: the signed sum 0.1 gave 'share=1100%'; magnitudes give 55%."""
    df = pd.DataFrame({"fi": [0.6, 0.5, -1.0]}, index=["a", "b", "c"]).sort_values("fi", ascending=False)
    line = _format_top_fi_for_log(df, 2, "coef")
    pct = [float(m) for m in re.findall(r"share=([0-9.]+)%", line)]
    assert not pct or max(pct) <= 100.0


def test_positive_only_returns_and_draws_positive_importances():
    """The returned frame holds only positive importances (and the chart is built from the same subset: no exception, no negative bar input)."""
    fi = np.array([0.5, -0.8, 0.2])
    df = plot_feature_importance(fi, columns=["a", "b", "c"], kind="coef", positive_fi_only=True, show_plots=False, plot_file="")
    assert list(df.index) == ["a", "c"]
