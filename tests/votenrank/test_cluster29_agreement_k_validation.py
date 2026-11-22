"""Audit 2026-11-20 finding 29-F21: agreement_rate refuses k below 1, instead of reporting rates above 1.

With k=0 and top_k=False the slice is the whole subset while the denominator stayed 1. The test checks the refusal, and that an ordinary k
still gives a rate between 0 and 1.
"""

import pandas as pd
import pytest

from mlframe.votenrank.utils import agreement_rate


def _frame():
    """Two methods' rankings as the 'rank: name' strings agreement_rate reads."""
    return pd.DataFrame({"AM": ["1: a", "2: b", "3: c"], "BM": ["1: a", "2: c", "3: b"]})


def test_k_zero_is_refused():
    """k=0 with top_k=False raises instead of returning rates above 1."""
    with pytest.raises(ValueError, match="at least 1"):
        agreement_rate(_frame(), k=0, top_k=False)


def test_ordinary_k_gives_a_rate_within_zero_and_one():
    """k=2 gives a rate between 0 and 1."""
    out = agreement_rate(_frame(), k=2, top_k=True)
    assert all(0.0 <= v <= 1.0 for v in out.values())
