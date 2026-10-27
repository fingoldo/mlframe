"""The gate's best-existing-operator floor is the same with a shared raw-column MI memo as without, and the memo is filled once per column."""
import numpy as np

from mlframe.feature_selection.filters._conditional_gate_fe import best_existing_op_mi


def _arrs(n=3000, seed=0):
    """Six operand columns and a binary target driven by one product."""
    rng = np.random.default_rng(seed)
    arrs = {f"x{i}": rng.normal(size=n) for i in range(6)}
    y = ((arrs["x0"] * arrs["x1"] + 0.5 * rng.normal(size=n)) > 0).astype(np.int64)
    return arrs, y


def test_floor_equals_the_memoless_floor_and_the_memo_holds_each_raw_once():
    """Every pair and triple of the six columns: the floor with the memo equals the floor without it; the memo ends with one entry per column that appeared."""
    arrs, y = _arrs()
    names = list(arrs)
    memo: dict = {}
    sets = [(names[i], names[j]) for i in range(6) for j in range(i + 1, 6)] + [(names[0], names[1], names[2]), (names[2], names[4], names[5])]
    for operands in sets:
        with_memo = best_existing_op_mi(arrs, operands, y, 12, memo)
        without = best_existing_op_mi(arrs, operands, y, 12)
        assert with_memo == without
    assert set(memo) == set(names)
    assert all(np.isfinite(v) for v in memo.values())
