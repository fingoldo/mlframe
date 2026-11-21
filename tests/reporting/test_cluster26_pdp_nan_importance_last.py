"""Audit 2026-11-20 finding 26-F23: a NaN importance is ranked last in the PDP panels, not reported as the most important feature.

The helper orders importances from largest to smallest with NaN at the end. The source check guards the three PDP sites from
going back to a reversed argsort, which put NaN first.
"""

import re
from pathlib import Path

import numpy as np

import mlframe.reporting._diagnostics_pdp as pdp


def test_nan_importance_sorts_last():
    """The NaN feature is at the end; the others are in descending order."""
    order = pdp._descending_importance_order([0.2, np.nan, 0.9, 0.5])
    assert order.tolist() == [2, 3, 0, 1]


def test_no_reversed_argsort_left_in_the_pdp_module():
    """No ``argsort(...)[::-1]`` remains in the module: the NaN-first ordering cannot come back."""
    source = Path(pdp.__file__).read_text(encoding="utf-8")
    assert re.search(r"argsort\([^\n]*\)\[::-1\]", source) is None
