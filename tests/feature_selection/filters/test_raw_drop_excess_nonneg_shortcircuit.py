"""Pin for the 2026-07-03 raw-drop near-unique-conditioning short-circuit.

``drop_redundant_raw_operands`` takes the MIN debiased ``excess`` across a raw operand's conditionings
(base clean-subexpr, full-composite fallback, sibling-augmented) and refines it ONLY on a strict
``_excess_f < excess`` update. The optimisation skips the expensive full-composite / sibling perm-nulls -- which
land on a near-unique HIGH-cardinality support (kz~n, the degenerate df<=0 case, ~0.8s of the fit) -- as soon as
a cheaper conditioning has driven ``excess`` to 0. That skip is BIT-IDENTICAL to the full computation ONLY
because ``_excess_and_floor`` clamps ``excess = max(0, cmi - null_mean) >= 0``: with excess already 0 and every
candidate ``_excess_f >= 0``, the strict ``_excess_f < excess`` update can never fire. This pin guards that
load-bearing invariant -- if the clamp is ever removed (excess could go negative), the short-circuit would
silently start changing verdicts, so this test must fail first.
"""

from __future__ import annotations

import numpy as np
import pytest


def _codes(rng, n, k):
    return rng.integers(0, k, n).astype(np.int64)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_excess_and_floor_never_returns_negative_excess(seed):
    """Across marginal (z=None), dense-conditional, and near-unique high-cardinality conditionings the returned
    excess must be >= 0 -- the invariant the MIN-excess short-circuit relies on for bit-identicality."""
    from mlframe.feature_selection.filters._fe_raw_redundancy_helpers import _excess_and_floor

    rng = np.random.default_rng(seed)
    n = 60_000
    y = _codes(rng, n, 6)
    cand = _codes(rng, n, 6)
    cases = {
        "marginal": None,
        "dense_cond": _codes(rng, n, 5),  # ~n/5 rows/stratum: dense
        "near_unique": _codes(rng, n, n // 2),  # ~2 rows/stratum: the degenerate df<=0 regime
        "fully_unique": np.arange(n).astype(np.int64),  # 1 row/stratum: maximally degenerate
    }
    for name, z in cases.items():
        cmi, floor, excess = _excess_and_floor(cand, y, z, seed=seed)
        assert excess >= 0.0, f"[{name}] _excess_and_floor returned NEGATIVE excess {excess} -> short-circuit unsafe"
        assert np.isfinite(cmi) and np.isfinite(floor) and np.isfinite(excess)
