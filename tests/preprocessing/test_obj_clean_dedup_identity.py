"""Identity regression for the dedup-aware object-column cleaning map.

``map_elementwise_dedup`` must produce output bit-identical to ``Series.map(fcn)`` for a pure elementwise
``fcn`` across every cardinality regime — including the gated all-distinct fallback and the stride-probe
adversarial case (head-clustered duplication, distinct tail) — and must preserve None/NaN positions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.preprocessing.cleaning_helpers import map_elementwise_dedup


def _clean(s):
    return s.strip().lower() if isinstance(s, str) else s


@pytest.mark.parametrize(
    "label, builder",
    [
        ("low-card", lambda rng, n: pd.Series(rng.choice([f" V{i} " for i in range(20)], size=n), dtype=object)),
        ("mid-card", lambda rng, n: pd.Series(rng.choice([f" S{i} " for i in range(500)], size=n), dtype=object)),
        ("all-unique", lambda rng, n: pd.Series([f" U{i} " for i in range(n)], dtype=object)),
        ("with-none", lambda rng, n: pd.Series(rng.choice([" A ", " B ", None, " C "], size=n), dtype=object)),
        ("head-dup-tail-unique", lambda rng, n: pd.Series([" DUP "] * (n // 2) + [f" T{i} " for i in range(n - n // 2)], dtype=object)),
        ("empty", lambda rng, n: pd.Series([], dtype=object)),
        ("tiny", lambda rng, n: pd.Series([" a ", None, " A "], dtype=object)),
    ],
)
def test_dedup_map_is_identical_to_plain_map(label, builder):
    rng = np.random.default_rng(0)
    n = 120_000  # above the 4*sample gate so the stride-probe path is exercised
    col = builder(rng, n)
    expected = col.map(_clean)
    got = map_elementwise_dedup(col, _clean)
    assert got.equals(expected), f"{label}: dedup map diverged from plain map"


def test_dedup_map_preserves_index():
    s = pd.Series([" a ", " B ", " a "], index=[10, 20, 30], dtype=object)
    out = map_elementwise_dedup(s, _clean)
    assert list(out.index) == [10, 20, 30]
    assert out.tolist() == ["a", "b", "a"]
