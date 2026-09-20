"""``time_ordering`` must reach the splits that claim to be forward-walks.

The MI screen sorts its own sample by time and sets a flag. Consumers then read that flag and switch to a
``TimeSeriesSplit`` or a first-half/second-half comparison -- but the tiny rerank draws its OWN sample and the drift gate
slices the caller's ``train_idx`` by position, so both walked row order while reporting themselves as time-aware.
Leakage protection was therefore absent exactly when the caller asked for it: a frame whose row order differs from time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.composite.discovery._fit_temporal import order_rows_by_time


def _shuffled_time_frame(n: int = 300, seed: int = 0):
    """Rows in scrambled order carrying an explicit time key, i.e. the case the key exists for."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    perm = rng.permutation(n)
    return t[perm], perm


def test_order_rows_by_time_sorts_a_consumers_own_sample():
    """The permutation puts the caller's rows in time order, whatever order the sampler produced."""
    time_key, perm = _shuffled_time_frame()
    rows = np.arange(perm.size)
    order = order_rows_by_time(rows, time_key)
    assert order is not None
    ordered_time = time_key[rows[order]]
    assert np.all(np.diff(ordered_time) >= 0), "rows must come out non-decreasing in the time key"


def test_order_rows_by_time_is_stable_for_repeated_keys():
    """Equal timestamps keep their original relative order, so the split stays deterministic."""
    key = np.array([5, 5, 5, 1, 1])
    order = order_rows_by_time(np.arange(5), key)
    assert list(order) == [3, 4, 0, 1, 2]


def test_order_rows_by_time_returns_none_when_the_key_cannot_be_applied():
    """No key, an empty sample or a key shorter than the row indices leaves the caller on its current order."""
    assert order_rows_by_time(np.arange(3), None) is None
    assert order_rows_by_time(np.array([], dtype=int), np.arange(3)) is None
    assert order_rows_by_time(np.array([7]), np.arange(3)) is None


def test_the_drift_gate_halves_are_the_time_halves():
    """The alpha-drift gate compares earlier rows against later ones, not two arbitrary partitions.

    The spec's alpha is 1.0 on the first half of TIME and 5.0 on the second. Under row order the two halves mix both
    regimes and the z-score sees no break; ordered by time the break is there to find.
    """
    from mlframe.training.composite.discovery._eval_stats import apply_alpha_drift_gate

    n = 400
    rng = np.random.default_rng(3)
    time_key = np.arange(n)
    base = rng.normal(loc=10.0, scale=2.0, size=n)
    alpha = np.where(time_key < n // 2, 1.0, 5.0)
    y = alpha * base + rng.normal(scale=0.05, size=n)

    perm = rng.permutation(n)  # the frame arrives scrambled
    df = pd.DataFrame({"base": base[perm], "y": y[perm]})
    train_idx = np.arange(n)

    class _Spec:
        transform_name = "linear_residual"
        base_column = "base"
        name = "spec"

    class _Disc:
        config = type("C", (), {"detect_linear_residual_alpha_drift": True, "alpha_drift_z_threshold": 3.0, "reject_on_alpha_drift": False})()
        _auto_base_pool: dict = {}
        _time_ordering_ = time_key[perm]

    disc = _Disc()
    apply_alpha_drift_gate(
        disc, [_Spec()], df=df, train_idx=train_idx, y_full=df["y"].to_numpy(),
        extract_column_array=lambda frame, col: frame[col].to_numpy(),
    )
    z_ordered = float(disc._alpha_drift_flags["spec"]["z_score"])

    disc_unordered = _Disc()
    disc_unordered._time_ordering_ = None  # what every consumer effectively had before
    apply_alpha_drift_gate(
        disc_unordered, [_Spec()], df=df, train_idx=train_idx, y_full=df["y"].to_numpy(),
        extract_column_array=lambda frame, col: frame[col].to_numpy(),
    )
    z_row_order = float(disc_unordered._alpha_drift_flags["spec"]["z_score"])

    assert z_ordered > z_row_order, f"the time-ordered halves must expose the break the row order hides: {z_ordered:.2f} vs {z_row_order:.2f}"
    assert z_ordered > 3.0, f"a 1.0 -> 5.0 slope break must exceed the default z threshold; got {z_ordered:.2f}"
