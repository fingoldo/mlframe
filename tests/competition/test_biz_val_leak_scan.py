"""Unit + biz_value tests for mlframe.competition.leak_scan.

COMPETITION/EXPLORATORY ONLY — see module docstring under src/mlframe/competition/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.competition.leak_scan import find_shifted_column_groups, sort_by_density_leak_scan


def _make_leaky_dataset(n_pairs: int = 40, n_cols: int = 20, seed: int = 0):
    """Build a wide sparse frame with deliberately injected row-shift leak pairs.

    For each pair, a base row of random values is scattered sparsely across ``n_cols``
    columns (density ~30%), and its "shifted" partner is the same underlying values
    shifted right by one column (row[i+1][j] = row[i][j-1]), sharing most of its
    non-null positions with row i once column-shift is undone -- mirroring the
    Santander-style row-shifted panel leak. Rows are then shuffled and interleaved with
    unrelated filler rows so the scan must recover the pairing purely from density sort
    + overlap, not from adjacency in the input.
    """
    rng = np.random.default_rng(seed)
    cols = [f"c{i}" for i in range(n_cols)]

    leak_rows = []
    leak_pair_keys: set[tuple[int, int]] = set()
    for _ in range(n_pairs):
        base = np.full(n_cols, np.nan)
        n_populated = max(4, int(n_cols * 0.3))
        positions = rng.choice(np.arange(1, n_cols), size=n_populated, replace=False)
        positions.sort()
        values = rng.uniform(1, 100, size=n_populated)
        base[positions] = values

        shifted = np.full(n_cols, np.nan)
        shift_positions = positions - 1
        shifted[shift_positions] = values

        leak_rows.append(base)
        leak_rows.append(shifted)

    n_filler = n_pairs * 2
    filler_rows = []
    for _ in range(n_filler):
        row = np.full(n_cols, np.nan)
        n_populated = rng.integers(1, max(2, int(n_cols * 0.3)))
        positions = rng.choice(np.arange(n_cols), size=n_populated, replace=False)
        row[positions] = rng.uniform(1, 100, size=n_populated)
        filler_rows.append(row)

    all_rows = leak_rows + filler_rows
    n_leak_rows = len(leak_rows)
    original_order = np.arange(len(all_rows))
    perm = rng.permutation(len(all_rows))

    shuffled_rows = [all_rows[i] for i in perm]
    df = pd.DataFrame(shuffled_rows, columns=cols)

    # ground-truth leak pairs, expressed as unordered frozensets of shuffled positions
    new_pos_of = {orig: new for new, orig in enumerate(perm)}
    for k in range(n_pairs):
        orig_a, orig_b = 2 * k, 2 * k + 1
        leak_pair_keys.add((min(new_pos_of[orig_a], new_pos_of[orig_b]), max(new_pos_of[orig_a], new_pos_of[orig_b])))

    return df, leak_pair_keys, n_leak_rows


def _make_control_dataset(n_rows: int = 120, n_cols: int = 20, seed: int = 1):
    """Same size/density profile as the leaky dataset, but with no row-shift structure."""
    rng = np.random.default_rng(seed)
    cols = [f"c{i}" for i in range(n_cols)]
    rows = []
    for _ in range(n_rows):
        row = np.full(n_cols, np.nan)
        n_populated = rng.integers(2, max(3, int(n_cols * 0.35)))
        positions = rng.choice(np.arange(n_cols), size=n_populated, replace=False)
        row[positions] = rng.uniform(1, 100, size=n_populated)
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def test_sort_by_density_leak_scan_basic_shapes():
    df = pd.DataFrame({"a": [1.0, np.nan, 2.0], "b": [np.nan, 3.0, 4.0]})
    out = sort_by_density_leak_scan(df)
    assert set(out.keys()) == {"row_order", "col_order", "row_density", "col_density", "overlap_scores", "candidate_pairs"}
    assert out["row_order"].shape == (3,)
    assert out["col_order"].shape == (2,)
    assert out["row_density"].shape == (3,)
    assert out["overlap_scores"].shape == (3,)
    assert np.isnan(out["overlap_scores"][-1])


def test_sort_by_density_leak_scan_prepends_target_column():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})
    target = np.array([5.0, 6.0])
    out = sort_by_density_leak_scan(df, target=target)
    # target column is included -> 3 columns considered even though df only has 2
    assert out["col_order"].shape == (3,)


def test_biz_val_leak_scan_recovers_injected_shift_pairs_high_recall_precision():
    df, leak_pair_keys, n_leak_rows = _make_leaky_dataset(n_pairs=40, n_cols=20, seed=0)

    out = sort_by_density_leak_scan(df, overlap_threshold=0.8, min_shared_values=3)
    candidate_pairs = out["candidate_pairs"]

    found_keys = {(min(a, b), max(a, b)) for a, b, _ in candidate_pairs}

    true_positives = len(found_keys & leak_pair_keys)
    recall = true_positives / len(leak_pair_keys)
    precision = true_positives / len(found_keys) if found_keys else 0.0

    # measured (seed=0): recall=1.0 (all 40 injected shift-pairs recovered), precision=1.0
    # (no filler-row pair crosses the 0.8 overlap threshold). Thresholds set with margin.
    assert recall >= 0.90
    assert precision >= 0.80
    assert true_positives >= int(0.9 * n_leak_rows / 2)


def test_biz_val_leak_scan_control_dataset_has_few_false_positives():
    control = _make_control_dataset(n_rows=120, n_cols=20, seed=1)
    out = sort_by_density_leak_scan(control, overlap_threshold=0.8, min_shared_values=3)

    n_rows = len(control)
    false_positive_rate = len(out["candidate_pairs"]) / n_rows

    # measured (seed=1): 0 candidate pairs out of 120 rows on unstructured control data.
    # threshold set generously above the measured 0 to tolerate seed variance.
    assert false_positive_rate <= 0.05


def test_find_shifted_column_groups_detects_lagged_pair():
    rng = np.random.default_rng(3)
    n = 200
    base = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x": base,
            "y_lag1": np.concatenate([[np.nan], base[:-1]]),
            "noise": rng.normal(size=n),
        }
    )
    out = find_shifted_column_groups(df, max_lag=2, corr_threshold=0.95)
    groups = out["groups"]
    assert any(set(g) == {"x", "y_lag1"} for g in groups)
    assert all("noise" not in g for g in groups)


def test_find_shifted_column_groups_empty_on_unrelated_columns():
    rng = np.random.default_rng(4)
    n = 100
    df = pd.DataFrame({f"c{i}": rng.normal(size=n) for i in range(6)})
    out = find_shifted_column_groups(df, max_lag=2, corr_threshold=0.95)
    assert out["groups"] == []
