"""Wave 11 (Category 3) M6: ``_pairs_score.py``'s degenerate-pair veto and ``_pairs_emit.py``'s
diverse-multi-emit dedup both computed a manual ``np.corrcoef`` despite ``_abs_corr_finite_njit`` (masked)
already being used elsewhere in ``_pairs_core.py`` for the identical shape. Both call sites actually needed
the ZERO-FILL semantics of the pre-fix ``np.corrcoef(np.nan_to_num(a), np.nan_to_num(b))`` (not the masked
semantics ``_abs_corr_finite_njit`` provides), so a new sibling kernel ``_abs_corr_zerofill_njit`` was added
and both call sites were swapped to it. This pins the new kernel against the exact pre-fix
nan_to_num-then-corrcoef reference, plus the ``min_n`` extension to ``_abs_corr_finite_njit`` used by the H2
fix in ``_ratio_delta_fe.py``.
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core import (
    _abs_corr_finite_njit,
    _abs_corr_zerofill_njit,
)


def _zerofill_corrcoef_ref(a, b):
    r = np.corrcoef(
        np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(np.asarray(b, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0),
    )[0, 1]
    return abs(float(r)) if np.isfinite(r) else 0.0


def test_abs_corr_zerofill_matches_nan_to_num_corrcoef_reference():
    rng = np.random.default_rng(7)
    max_diff = 0.0
    n_cases = 0
    for trial in range(300):
        n = int(rng.integers(20, 2000))
        kind = trial % 5
        a = rng.normal(size=n) * rng.choice([1.0, 100.0, 0.001])
        if kind == 0:
            b = a * 3.0 + rng.normal(scale=1e-3, size=n)
        elif kind == 1:
            b = rng.normal(size=n)
        elif kind == 2:
            a2 = a.copy()
            a2[rng.choice(n, size=n // 4, replace=False)] = np.nan
            b = rng.normal(size=n)
            a = a2
        elif kind == 3:
            b = a.copy()
            b[rng.choice(n, size=n // 3, replace=False)] = np.inf
        else:
            a = rng.normal(size=n)
            b = rng.normal(size=n)
            a[rng.choice(n, size=n // 5, replace=False)] = -np.inf
            b[rng.choice(n, size=n // 6, replace=False)] = np.nan

        r_ref = _zerofill_corrcoef_ref(a, b)
        r_new = float(_abs_corr_zerofill_njit(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)))
        d = abs(r_ref - r_new)
        max_diff = max(max_diff, d)
        n_cases += 1
        assert d < 1e-9, f"trial={trial} kind={kind} n={n} ref={r_ref} new={r_new} diff={d}"
    assert n_cases == 300
    assert max_diff < 1e-9


def test_abs_corr_zerofill_floors_constant_columns_to_zero():
    a_const = np.full(500, 3.0)
    rng = np.random.default_rng(8)
    b_rand = rng.normal(size=500)
    assert _zerofill_corrcoef_ref(a_const, b_rand) == 0.0
    assert float(_abs_corr_zerofill_njit(a_const, b_rand)) == 0.0


def test_abs_corr_finite_njit_min_n_matches_masked_corrcoef_at_thin_overlap():
    """The ``min_n`` param H2 relies on: a masked corrcoef defined from as few as 2 jointly-finite rows
    must be reproduced exactly with ``min_n=2``, while the default ``min_n=8`` still floors thin overlaps
    to 0.0 for the small-sample-noise-protected call sites."""
    rng = np.random.default_rng(9)
    n = 40
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    mask = np.zeros(n, dtype=bool)
    mask[:3] = True
    rng.shuffle(mask)
    b_e = b.copy()
    b_e[~mask] = np.nan
    b_fin = np.isfinite(b_e)

    c2 = a[mask]
    s2 = b_e[mask]
    ref = abs(float(np.corrcoef(c2, s2)[0, 1])) if c2.std() > 1e-12 and s2.std() > 1e-12 else 0.0

    r_min2 = float(_abs_corr_finite_njit(a, b_e, b_fin, 2))
    r_default = float(_abs_corr_finite_njit(a, b_e, b_fin))  # default min_n=8, mask.sum()==3 < 8

    assert abs(r_min2 - ref) < 1e-9
    assert r_default == 0.0
