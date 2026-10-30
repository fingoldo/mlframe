"""``_factorize_dense_njit`` numbers the distinct values by first appearance whichever table it uses (direct for a narrow id range, hashed for a wide one or negative ids)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._mi_greedy_cmi_fe_binning import _factorize_dense_njit


def _reference(v: np.ndarray) -> tuple[np.ndarray, int]:
    """First-seen ids with a Python dict."""
    seen: dict = {}
    out = np.empty(v.shape[0], dtype=np.int64)
    for i, x in enumerate(v.tolist()):
        out[i] = seen.setdefault(x, len(seen))
    return out, len(seen)


@pytest.mark.parametrize("kind", ["narrow", "wide", "huge", "negative", "constant", "all_distinct", "adversarial"])
@pytest.mark.parametrize("n", [1, 7, 1000, 50000])
def test_ids_are_first_seen_and_equal_the_dict_reference(kind, n):
    """Narrow ranges use the direct table, wide and negative ones the hash table; both give the dict reference."""
    rng = np.random.default_rng(n)
    v = {
        "narrow": rng.integers(0, 40, n),
        "wide": rng.integers(0, 10**7, n),
        "huge": rng.integers(0, 2**62, n),
        "negative": rng.integers(-(10**6), 10**6, n),
        "constant": np.full(n, 12345678901),
        "all_distinct": rng.permutation(n) * 1_000_003,
        "adversarial": (np.arange(n) % 97) * (2**40),
    }[kind].astype(np.int64)
    got, k = _factorize_dense_njit(v)
    want, kw = _reference(v)
    assert k == kw and np.array_equal(got, want)
