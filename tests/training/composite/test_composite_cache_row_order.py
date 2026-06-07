"""Regression test for composite_cache row-order signature.

Pre-fix ``data_signature`` used a seeded-RNG row-sample, which produces identical samples for the
original frame AND for any row-permutation of it. A shuffled frame therefore hit the cache and
replayed a stale spec computed against the unshuffled order.

Post-fix ``data_signature`` folds a cheap O(1) first-and-last row fingerprint into the hash
(``_row_order_fingerprint``). Shuffled rows => different fingerprint => different signature => no
stale cache hit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.cache import (
    _row_order_fingerprint,
    data_signature,
)


def _make_frame(seed: int = 0, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "y": rng.integers(0, 2, size=n),
    })


def test_row_order_fingerprint_changes_on_shuffle_pandas():
    df = _make_frame()
    shuffled = df.sample(frac=1.0, random_state=999).reset_index(drop=True)
    fp_orig = _row_order_fingerprint(df)
    fp_shuf = _row_order_fingerprint(shuffled)
    assert fp_orig and fp_shuf
    assert fp_orig != fp_shuf


def test_row_order_fingerprint_changes_on_shuffle_polars():
    pl = pytest.importorskip("polars")
    pdf = _make_frame()
    df = pl.from_pandas(pdf)
    shuffled_pdf = pdf.sample(frac=1.0, random_state=999).reset_index(drop=True)
    shuffled = pl.from_pandas(shuffled_pdf)
    fp_orig = _row_order_fingerprint(df)
    fp_shuf = _row_order_fingerprint(shuffled)
    assert fp_orig and fp_shuf
    assert fp_orig != fp_shuf


def test_data_signature_changes_on_row_swap_outside_sample_pandas():
    """The seeded RNG sample only inspects ``sample_n`` row positions; pre-fix, swapping two rows
    OUTSIDE those positions left the signature unchanged (because the per-column min/max/null stats
    are permutation-invariant and the sample missed the swap). Post-fix the row-order fingerprint
    on the first/last edge rows changes when the swap touches the head/tail region.
    """
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "y": rng.integers(0, 2, size=n),
    })
    # Reproduce the sample_idx the function will draw, then pick swap rows in the head region but
    # outside the sample so pre-fix sees identical sampled values + identical stats.
    rng_probe = np.random.default_rng(0)
    sample_idx = set(np.sort(rng_probe.choice(n, size=5, replace=False)).tolist())
    # Walk forward from 0 / 1 until we find two indices not in the sampled set; both lie in the
    # first n_edge=8 rows so the row-order fingerprint differs after the swap.
    swap_a, swap_b = None, None
    for i in range(8):
        if i not in sample_idx:
            if swap_a is None:
                swap_a = i
            elif swap_b is None and i != swap_a:
                swap_b = i
                break
    assert swap_a is not None and swap_b is not None, "test harness: must find two unsampled head rows"
    df_swapped = df.copy()
    df_swapped.iloc[swap_a], df_swapped.iloc[swap_b] = (
        df.iloc[swap_b].copy(), df.iloc[swap_a].copy(),
    )
    s_orig = data_signature(df, "y", ["a", "b"], sample_n=5, random_state=0)
    s_swap = data_signature(df_swapped, "y", ["a", "b"], sample_n=5, random_state=0)
    assert s_orig != s_swap, (
        "row swap inside the head region must produce a distinct discovery signature so a stale "
        "spec is not replayed from the disk cache (pre-fix the seeded sample missed the swap and "
        "the column min/max/null stats are permutation-invariant)"
    )


def test_data_signature_still_stable_for_identical_frame():
    """Sanity: the row-order hash must not introduce flakiness for identical inputs."""
    df = _make_frame()
    s1 = data_signature(df, "y", ["a", "b"], sample_n=200, random_state=0)
    s2 = data_signature(df, "y", ["a", "b"], sample_n=200, random_state=0)
    assert s1 == s2
