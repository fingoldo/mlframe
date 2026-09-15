"""explain_selection's survivor section counts only the features that reach the output.

``fe_provenance_`` also lists produced engineered columns the screen dropped (NaN gain, support_rank -1). The section reported
``len(fe_provenance_)`` as "selected", over-counting the selection and its engineered/raw split by every screened-out candidate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_explain import _survivor_section


class _Stub:
    """A fitted-looking estimator: two outputs, one screened-out engineered column, and a fallback raw column with rank -1."""

    fe_provenance_ = pd.DataFrame(
        {
            "feature_name": ["a", "mul(a,b)", "sqr(b)", "c"],
            "origin": ["raw", "unary_binary_pair", "unary_binary_pair", "raw"],
            "mechanism_details": ["{}"] * 4,
            "mrmr_gain": [0.30, 0.20, np.nan, np.nan],
            "support_rank": [0, 1, -1, -1],
        }
    )

    def get_feature_names_out(self):
        """The actual selection: the fallback raw column c is selected although its rank is -1; sqr(b) was screened out."""
        return np.array(["a", "mul(a,b)", "c"])


def test_survivor_section_counts_only_the_selected_features():
    """3 selected (1 engineered, 2 raw), with the screened-out column reported separately and absent from the roster."""
    text = _survivor_section(_Stub())
    assert "Surviving features: 3 selected (1 engineered, 2 raw)" in text, text
    assert "1 produced column(s) were screened out" in text
    roster = [ln for ln in text.splitlines() if "by MI/gain attribution" in ln]
    assert roster and "sqr(b)" not in roster[0]
    assert "c[raw]" in roster[0], "a selected fallback raw column with rank -1 must still be listed"


def test_survivor_section_labels_the_gain_as_in_screen():
    """The attribution line must not present the in-screen greedy gain as an unqualified attribution."""
    assert "in-screen greedy gain" in _survivor_section(_Stub())
