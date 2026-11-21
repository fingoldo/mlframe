"""The many-set resident floor builds all candidate blocks in place in one joined buffer; values equal the per-set scores."""
import numpy as np
import pytest


def test_many_matches_one_by_one_inplace_join():
    """best_existing_op_mi_resident_many gives the same floors as best_existing_op_mi_resident for each operand set."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from mlframe.feature_selection.filters._resident_candidate_mi import best_existing_op_mi_resident, best_existing_op_mi_resident_many

    rng = np.random.default_rng(3)
    n = 4000
    arrs = {c: rng.standard_normal(n) for c in "abcde"}
    yi = (arrs["a"] * arrs["b"] + 0.5 * rng.standard_normal(n) > 0).astype(np.int64)
    sets = [["a", "b"], ["b", "c", "d"], ["a", "e"], ["c", "d", "e"]]
    many = best_existing_op_mi_resident_many(arrs, sets, yi, 8, col_budget=20)
    one = [best_existing_op_mi_resident(arrs, s, yi, 8) for s in sets]
    assert many is not None
    np.testing.assert_allclose(many, one, rtol=1e-6)
