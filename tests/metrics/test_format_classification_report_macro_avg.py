"""METRICS-5: format_classification_report's macro-avg docstring must accurately describe the actual
behavior -- it inherits fast_classification_report's default macro_over_present=True (present-classes-
only), which is what matches sklearn's classification_report, not the deflating
macro_over_present=False legacy convention the docstring previously claimed."""

from __future__ import annotations

import re

import numpy as np
from sklearn.metrics import classification_report

from mlframe.metrics.core import format_classification_report


def test_macro_avg_matches_sklearn_present_classes_only():
    """A 3-class target where one class never appears in y_true: macro avg must match sklearn's
    present-classes-only convention, not the deflating divide-by-nclasses legacy behavior."""
    # class 2 never appears in y_true -- exactly the scenario the docstring caveat is about.
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])

    ours = format_classification_report(y_true, y_pred, nclasses=3, digits=3)
    # labels=None (sklearn's default): the label set is inferred from y_true/y_pred, so an absent
    # class 2 is excluded from the report entirely -- the genuine present-classes-only macro avg,
    # matching this function's documented macro_over_present=True behavior.
    sk = classification_report(y_true, y_pred, digits=3, zero_division=0)

    our_macro_f1 = float(re.search(r"macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)", ours).group(1))
    sk_macro_f1 = float(re.search(r"macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)", sk).group(1))

    assert abs(our_macro_f1 - sk_macro_f1) < 1e-3, f"expected present-classes-only macro F1 to match sklearn's, got ours={our_macro_f1} sklearn={sk_macro_f1}"


def test_docstring_describes_macro_over_present_true_not_false():
    """The MACRO-AVG CAVEAT docstring must describe the actual default (macro_over_present=True,
    present-classes-only) rather than the deflating legacy False behavior it previously claimed."""
    doc = format_classification_report.__doc__
    assert doc is not None
    assert "macro_over_present=True" in doc
    normalized = " ".join(doc.split())
    assert "does NOT deflate" in normalized
