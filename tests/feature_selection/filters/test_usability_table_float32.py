"""The float32 pair-combo table of the usability pool agrees with the float64 one."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _table(dtype, n=3000, seed=0):
    """The pair-combo MI table of four operand pairs in the given value dtype."""
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _quantile_bin, precompute_marginal_y_terms
    from mlframe.feature_selection.filters._usability_njit_pool import njit_binary_codes_or_none, njit_unary_codes_or_none
    from mlframe.feature_selection.filters._usability_pool_resident import score_pair_combos_table_resident
    from mlframe.feature_selection.filters.feature_engineering import create_binary_transformations, create_unary_transformations

    rng = np.random.default_rng(seed)
    ops = [(rng.uniform(0.1, 1.1, n), rng.uniform(0.1, 1.1, n)) for _ in range(3)] + [(rng.standard_normal(n), rng.lognormal(0, 1, n))]
    y = ops[0][0] ** 2 / ops[0][1] + np.log(ops[1][0]) * np.sin(ops[2][1]) + 0.1 * rng.standard_normal(n)
    yc = _quantile_bin(y, 10, host_only=True)
    uc = njit_unary_codes_or_none(list(create_unary_transformations("medium")))
    bc = njit_binary_codes_or_none(list(create_binary_transformations("minimal")))
    return score_pair_combos_table_resident(ops, yc, precompute_marginal_y_terms(yc), 10, uc, uc, bc, value_dtype=dtype)


def test_float32_table_keeps_the_ranking_of_the_float64_table():
    """The same live combos, the same top-12 set and the same best combo per pair; the MI differs by float32 rounding only."""
    a, b = _table(np.float64), _table(np.float32)
    assert a is not None and b is not None and a.shape == b.shape
    assert np.array_equal(a < 0, b < 0)
    assert float(np.abs(a - b)[(a >= 0) & (b >= 0)].max()) < 1e-3
    assert a.shape[0] > 0, "no pairs in the table: the loop below would pass vacuously"
    for p in range(a.shape[0]):
        assert set(np.argsort(-a[p], kind="stable")[:12]) == set(np.argsort(-b[p], kind="stable")[:12])
        assert np.argmax(a[p]) == np.argmax(b[p])


def test_float32_is_the_default_value_dtype():
    """The table function defaults to float32 and the float64 path stays selectable."""
    import inspect

    from mlframe.feature_selection.filters._usability_pool_resident import score_pair_combos_table_resident

    assert inspect.signature(score_pair_combos_table_resident).parameters["value_dtype"].default is np.float32
