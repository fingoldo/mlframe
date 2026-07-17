"""Regression test for MRMR._FIT_CACHE y-collision.

Pre-fix, the cache key was ``(_content_array_signature(X), _content_array_signature(y), params_sig)``
where ``_content_array_signature`` samples y at 10 evenly-spaced positions. Two targets whose 10
sampled cells happen to coincide (e.g. both balanced binary in those positions) collide on the y
signature and the second fit incorrectly replays the first fit's ``support_``.

Post-fix the cache key folds in (1) the y column name / Series.name and (2) a full blake2b over the
y content, so distinct targets always produce distinct cache keys.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import (
    MRMR,
    _content_array_signature,
    _full_y_content_hash,
    _target_name_signature,
)


def _sample_positions(n: int, n_samples: int = 1024) -> list[int]:
    """Mirror ``_content_array_signature``'s 1024-sample positional stride."""
    if n >= n_samples:
        return [int(i * (n - 1) / (n_samples - 1)) for i in range(n_samples)]
    return list(range(n))


def _build_collision_targets(n: int = 4096, seed: int = 0):
    """Return (y_a, y_b) whose ``_content_array_signature`` strided samples are byte-identical.

    The signature samples 1024 evenly-spaced positions. We pin those cells to the same alternating 0/1 pattern in both targets, and randomise the rest differently so the underlying class distributions diverge. We pick n>1024 so unsampled positions exist and remain distinct between the two targets.
    """
    rng = np.random.default_rng(seed)
    sample_idx = _sample_positions(n)
    n_samples = len(sample_idx)
    # Position-shared cells: alternating 0/1.
    pinned = np.array([i % 2 for i in range(n_samples)], dtype=np.int64)

    y_a = rng.integers(0, 2, size=n).astype(np.int64)
    # Different randomness for b's bulk -- distinct class distribution.
    y_b = rng.integers(0, 2, size=n).astype(np.int64)
    # Force b's bulk to be the inverse of a's outside the sampled positions, so the supports diverge.
    mask = np.ones(n, dtype=bool)
    mask[sample_idx] = False
    y_b[mask] = 1 - y_a[mask]
    for k, pos in enumerate(sample_idx):
        y_a[pos] = pinned[k]
        y_b[pos] = pinned[k]
    return y_a, y_b


def test_content_array_signature_collides_on_pinned_samples():
    """Sanity: the 10-cell sample is identical for the engineered targets."""
    y_a, y_b = _build_collision_targets()
    sig_a = _content_array_signature(pd.Series(y_a))
    sig_b = _content_array_signature(pd.Series(y_b))
    # shape, dtype, sampled bytes -- all identical. col_names is () for unnamed Series.
    assert sig_a == sig_b, "test harness invalid: 10-cell samples must collide for this test"


def test_full_y_content_hash_distinguishes_collision_targets():
    """The full content hash must differ for the two distinct y vectors even when 10-cell samples match."""
    y_a, y_b = _build_collision_targets()
    h_a = _full_y_content_hash(pd.Series(y_a))
    h_b = _full_y_content_hash(pd.Series(y_b))
    assert h_a and h_b, "hash helper must succeed on plain int64 Series"
    assert h_a != h_b, "blake2b over y content must distinguish distinct targets"


def test_target_name_signature_uses_series_name():
    s = pd.Series([0, 1, 0], name="my_target")
    assert _target_name_signature(s) == ("my_target",)
    assert _target_name_signature(pd.Series([0, 1, 0])) == ()
    df = pd.DataFrame({"a": [0, 1], "b": [1, 0]})
    assert _target_name_signature(df) == ("a", "b")


def test_mrmr_cache_does_not_collide_on_distinct_targets_with_shared_samples():
    """End-to-end: two MRMR fits on the SAME X but DIFFERENT y must produce different support_,
    even when the y's 10-cell content signature is identical. Pre-fix the second fit replayed the
    first fit's ``support_`` from the cache; post-fix the cache key carries the full y hash so the
    second fit recomputes from scratch.
    """
    MRMR._FIT_CACHE.clear()
    try:
        rng = np.random.default_rng(42)
        n = 4096
        # Build X with features whose informativeness for y_a vs y_b will differ.
        X = pd.DataFrame(
            {
                "f0": rng.normal(size=n),
                "f1": rng.normal(size=n),
                "f2": rng.normal(size=n),
                "f3": rng.normal(size=n),
            }
        )
        y_a_arr, y_b_arr = _build_collision_targets(n=n, seed=0)
        # Make y_a strongly correlated with f0, y_b strongly correlated with f2 -- different bulk content AND different supports while still preserving the strided-sample collision.
        sample_idx = _sample_positions(n)
        mask = np.ones(n, dtype=bool)
        mask[sample_idx] = False
        y_a_arr[mask] = (X["f0"].to_numpy()[mask] > 0).astype(np.int64)
        y_b_arr[mask] = (X["f2"].to_numpy()[mask] > 0).astype(np.int64)
        # Sanity re-check after rebinding bulk: strided sample still matches.
        assert _content_array_signature(pd.Series(y_a_arr)) == _content_array_signature(pd.Series(y_b_arr))
        y_a = pd.Series(y_a_arr, name="target_a")
        y_b = pd.Series(y_b_arr, name="target_b")

        kw = dict(
            full_npermutations=2,
            baseline_npermutations=2,
            fe_max_steps=0,
            verbose=0,
            n_jobs=1,
        )
        m_a = MRMR(**kw)
        m_b = MRMR(**kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_a.fit(X, y_a)
            m_b.fit(X, y_b)

        # Two distinct cache entries (one per target hash). Pre-fix this was 1 (second fit hit the cache).
        assert len(MRMR._FIT_CACHE) == 2, f"cache must hold two entries for two distinct targets; got {len(MRMR._FIT_CACHE)}"

        # Supports must differ: f0 informs y_a, f2 informs y_b.
        sup_a = set(m_a.support_) if hasattr(m_a.support_, "__iter__") else {m_a.support_}
        sup_b = set(m_b.support_) if hasattr(m_b.support_, "__iter__") else {m_b.support_}
        assert sup_a != sup_b, f"distinct targets must yield distinct supports; got identical {sup_a}"
    finally:
        MRMR._FIT_CACHE.clear()
