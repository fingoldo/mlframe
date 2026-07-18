"""Regression + correctness tests for ``dispatch_batch_pair_mi_chunked`` (the RAM-bounded
row-block chunking wrapper that replaced the flat ``_MRMR_BATCH_PRECOMPUTE_MAX_K=200``
pool-size cap, 2026-07-09).

Prior behavior (the bug this closes): any FE candidate pool wider than 200 columns fell back
onto a ``~35s/pair`` legacy joblib loop instead of the fast batched path, turning a realistic
several-hundred-column production pool into a multi-hour cliff. The fix enumerates the
``C(k, 2)`` pair space in RAM-bounded row-block chunks (never materialising the full pair-index
arrays) so the batched path is now used at any pool width the front-end SIS gate lets through.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest


def _build_factor_data(n_samples: int, n_cols: int, nbins: int, seed: int):
    """Build factor data."""
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, nbins, size=n_samples) for _ in range(n_cols)]
    data = np.column_stack(cols).astype(np.int32)
    return data, np.full(n_cols, nbins, dtype=np.int32)


def test_batch_precompute_max_k_cap_is_removed():
    """Pins the removal: the old flat pool-size cap must no longer exist anywhere on the
    public/semi-public mrmr surface. Fails loudly (ImportError) if a future change
    reintroduces it under the same name without updating this sensor."""
    from mlframe.feature_selection.filters import mrmr as mrmr_pkg

    assert not hasattr(mrmr_pkg, "_MRMR_BATCH_PRECOMPUTE_MAX_K")
    assert "_MRMR_BATCH_PRECOMPUTE_MAX_K" not in mrmr_pkg.__all__


def test_iter_upper_triangle_pair_chunks_covers_every_pair_exactly_once():
    """Iter upper triangle pair chunks covers every pair exactly once."""
    from mlframe.feature_selection.filters.batch_pair_mi_gpu import _iter_upper_triangle_pair_chunks

    for k, chunk_pairs in [(2, 1), (5, 1), (5, 3), (20, 7), (37, 5), (100, 1)]:
        seen = set()
        total = 0
        for a_pos, b_pos in _iter_upper_triangle_pair_chunks(k, chunk_pairs):
            assert a_pos.shape == b_pos.shape
            assert np.all(a_pos < b_pos)
            total += a_pos.shape[0]
            for a, b in zip(a_pos.tolist(), b_pos.tolist()):
                assert (a, b) not in seen, f"pair ({a},{b}) yielded twice at k={k}, chunk_pairs={chunk_pairs}"
                seen.add((a, b))
        expected = k * (k - 1) // 2
        assert total == expected, f"k={k}, chunk_pairs={chunk_pairs}: got {total} pairs, expected {expected}"
        assert seen == set(itertools.combinations(range(k), 2))


def test_iter_upper_triangle_pair_chunks_empty_for_trivial_k():
    """Iter upper triangle pair chunks empty for trivial k."""
    from mlframe.feature_selection.filters.batch_pair_mi_gpu import _iter_upper_triangle_pair_chunks

    assert list(_iter_upper_triangle_pair_chunks(0, 100)) == []
    assert list(_iter_upper_triangle_pair_chunks(1, 100)) == []


@pytest.mark.parametrize("max_pairs_per_chunk", [1, 5, 50, 10_000])
def test_dispatch_batch_pair_mi_chunked_matches_unchunked_dispatch(max_pairs_per_chunk):
    """At every chunk size (including chunk_size=1, forcing maximal fragmentation), the chunked
    dispatcher must reproduce the SAME per-pair MI values as one unchunked ``dispatch_batch_pair_mi``
    call over the full ``np.triu_indices`` pair set -- chunking is purely an enumeration/memory
    strategy and must not change any numerics."""
    from mlframe.feature_selection.filters.batch_pair_mi_gpu import (
        dispatch_batch_pair_mi,
        dispatch_batch_pair_mi_chunked,
    )

    n_cols = 12
    data, nbins = _build_factor_data(n_samples=800, n_cols=n_cols, nbins=5, seed=3)
    rng = np.random.default_rng(11)
    y = rng.integers(0, 3, size=800).astype(np.int32)
    freqs_y = np.bincount(y, minlength=3).astype(np.float64) / 800

    ids = np.arange(n_cols, dtype=np.int64)
    ia, ib = np.triu_indices(n_cols, k=1)
    mi_ref, _ = dispatch_batch_pair_mi(
        factors_data=data,
        pair_a=ids[ia],
        pair_b=ids[ib],
        nbins=nbins,
        classes_y=y,
        freqs_y=freqs_y,
        force_backend="njit",
    )
    ref_map = {(int(ids[ia[i]]), int(ids[ib[i]])): float(mi_ref[i]) for i in range(ia.shape[0])}

    a_out, b_out, mi_out, backend_counts = dispatch_batch_pair_mi_chunked(
        factors_data=data,
        ids=ids,
        nbins=nbins,
        classes_y=y,
        freqs_y=freqs_y,
        force_backend="njit",
        max_pairs_per_chunk=max_pairs_per_chunk,
    )

    assert a_out.shape[0] == n_cols * (n_cols - 1) // 2
    assert sum(backend_counts.values()) >= 1
    for a, b, mi in zip(a_out.tolist(), b_out.tolist(), mi_out.tolist()):
        assert mi == pytest.approx(ref_map[(a, b)], abs=1e-9)


def test_dispatch_batch_pair_mi_chunked_handles_pool_width_above_old_cap():
    """Direct regression sensor for the reported bug: a pool width of 300 (above the removed
    ``_MRMR_BATCH_PRECOMPUTE_MAX_K=200``) must complete via the batched chunked path, not silently
    degrade. Uses a small, cheap chunk size to force real chunking at this width."""
    from mlframe.feature_selection.filters.batch_pair_mi_gpu import dispatch_batch_pair_mi_chunked

    n_cols = 300
    data, nbins = _build_factor_data(n_samples=300, n_cols=n_cols, nbins=4, seed=5)
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, size=300).astype(np.int32)
    freqs_y = np.bincount(y, minlength=2).astype(np.float64) / 300

    ids = np.arange(n_cols, dtype=np.int64)
    a_out, _b_out, mi_out, backend_counts = dispatch_batch_pair_mi_chunked(
        factors_data=data,
        ids=ids,
        nbins=nbins,
        classes_y=y,
        freqs_y=freqs_y,
        force_backend="njit",
        max_pairs_per_chunk=2_000,
    )
    expected_pairs = n_cols * (n_cols - 1) // 2
    assert a_out.shape[0] == expected_pairs
    assert mi_out.shape[0] == expected_pairs
    assert np.all(np.isfinite(mi_out))
    assert np.all(mi_out >= -1e-9)  # MI is non-negative up to fp noise
    assert backend_counts.get("njit", 0) >= 1


def test_dispatch_batch_pair_mi_chunked_empty_and_singleton():
    """Dispatch batch pair mi chunked empty and singleton."""
    from mlframe.feature_selection.filters.batch_pair_mi_gpu import dispatch_batch_pair_mi_chunked

    data, nbins = _build_factor_data(n_samples=50, n_cols=1, nbins=3, seed=1)
    y = np.zeros(50, dtype=np.int32)
    freqs_y = np.array([1.0])

    a_out, b_out, mi_out, backend_counts = dispatch_batch_pair_mi_chunked(
        factors_data=data,
        ids=np.array([0], dtype=np.int64),
        nbins=nbins,
        classes_y=y,
        freqs_y=freqs_y,
    )
    assert a_out.shape == (0,)
    assert b_out.shape == (0,)
    assert mi_out.shape == (0,)
    assert backend_counts == {}
