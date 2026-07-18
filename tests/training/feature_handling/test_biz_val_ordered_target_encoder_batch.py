"""biz_value test for ``ordered_target_encode_batch``.

The win: ``categorical_powerset_concat``'s ``prune_against_target`` (and any other caller scoring many
composite/candidate categorical columns against the SAME ``(y, order)`` pair) previously called
``ordered_target_encode`` once per column, redoing the causal ``argsort`` of ``order`` and the global-prior
reduction every time even though both are column-independent. ``ordered_target_encode_batch`` computes that
shared work exactly once and reuses it across every column, which should measurably speed up the many-columns
case while producing bit-identical per-column results (when ``noise_std == 0.0``, the default).
"""

from __future__ import annotations

import time

import numpy as np

from mlframe.training.feature_handling.ordered_target_encoder import ordered_target_encode, ordered_target_encode_batch


def _make_columns(n_rows: int, n_cols: int, n_cats: int, seed: int):
    """Build n_cols random categorical columns sharing one (y, order) pair, the shape batch-encoding is meant for."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    y = rng.integers(0, 2, n_rows).astype(np.float64)
    columns = {f"c{i}": rng.integers(0, n_cats, n_rows) for i in range(n_cols)}
    return columns, y, order


def test_biz_val_ordered_target_encode_batch_speeds_up_many_shared_columns():
    """Batched encoding of 25 shared-order columns beats 25 separate calls by >=15% wall time, bit-identical output."""
    n_rows, n_cols, n_cats = 40_000, 25, 2_000
    columns, y, order = _make_columns(n_rows, n_cols, n_cats, seed=0)

    # warm up (JIT/pandas caches, page faults) before timing either path.
    ordered_target_encode(next(iter(columns.values())), y, order=order, smoothing=1.0)
    ordered_target_encode_batch({k: v for k, v in list(columns.items())[:2]}, y, order=order, smoothing=1.0)

    t0 = time.perf_counter()
    separate = {name: ordered_target_encode(cats, y, order=order, smoothing=1.0) for name, cats in columns.items()}
    separate_wall = time.perf_counter() - t0

    t0 = time.perf_counter()
    batched = ordered_target_encode_batch(columns, y, order=order, smoothing=1.0)
    batched_wall = time.perf_counter() - t0

    for name in columns:
        np.testing.assert_array_equal(separate[name], batched[name])

    speedup = separate_wall / batched_wall
    assert (
        speedup >= 1.15
    ), f"expected ordered_target_encode_batch to beat N separate calls by >=15% on {n_cols} shared-order columns, got speedup={speedup:.3f} (separate={separate_wall * 1000:.1f}ms batched={batched_wall * 1000:.1f}ms)"


def test_ordered_target_encode_batch_matches_separate_calls_exactly():
    """Batch encoding with smoothing+prior overrides matches per-column ordered_target_encode calls exactly."""
    columns, y, order = _make_columns(n_rows=500, n_cols=6, n_cats=40, seed=1)
    separate = {name: ordered_target_encode(cats, y, order=order, smoothing=2.5, prior=0.3) for name, cats in columns.items()}
    batched = ordered_target_encode_batch(columns, y, order=order, smoothing=2.5, prior=0.3)
    assert set(batched.keys()) == set(separate.keys())
    for name in columns:
        np.testing.assert_array_equal(separate[name], batched[name])


def test_ordered_target_encode_batch_empty_columns_returns_empty_dict():
    """An empty columns dict returns {} rather than raising on the shared-order precompute."""
    y = np.array([1.0, 0.0, 1.0])
    assert ordered_target_encode_batch({}, y) == {}


def test_ordered_target_encode_batch_noise_columns_are_independent_and_reproducible():
    """Same random_state reproduces identical batch output, but distinct columns still get distinct noise draws."""
    columns, y, order = _make_columns(n_rows=300, n_cols=4, n_cats=20, seed=2)
    batch_a = ordered_target_encode_batch(columns, y, order=order, smoothing=1.0, noise_std=0.4, random_state=7)
    batch_b = ordered_target_encode_batch(columns, y, order=order, smoothing=1.0, noise_std=0.4, random_state=7)
    for name in columns:
        np.testing.assert_array_equal(batch_a[name], batch_b[name])
    # different columns must not receive identical noise draws (would defeat the point of per-column independence).
    names = list(columns.keys())
    assert not np.array_equal(batch_a[names[0]], batch_a[names[1]])
