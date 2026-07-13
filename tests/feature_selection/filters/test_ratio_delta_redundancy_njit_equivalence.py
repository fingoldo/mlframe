"""Wave 11 (Category 3) H2: ``_passes_redundancy`` (``_ratio_delta_fe.py``) was rewritten from a masked
``np.corrcoef`` per (candidate, source) check to the shared ``_abs_corr_finite_njit`` kernel (with
``min_n=2`` so the small-overlap floor matches what a masked ``np.corrcoef`` implicitly allows). Pins the
new implementation against a frozen copy of the pre-fix masked-corrcoef reference, including small
joint-finite-overlap edge cases (the reason ``min_n=2`` rather than the kernel's default 8 was required).
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._ratio_delta_fe import _passes_redundancy


def _passes_redundancy_ref(candidate, a_vals, b_vals, threshold):
    cand = candidate
    if cand.std() <= 1e-12:
        return False
    for src in (a_vals, b_vals):
        if src.std() <= 1e-12:
            continue
        mask = np.isfinite(cand) & np.isfinite(src)
        if not mask.any():
            continue
        c2 = cand[mask]
        s2 = src[mask]
        if c2.std() <= 1e-12 or s2.std() <= 1e-12:
            continue
        rho = float(np.corrcoef(c2, s2)[0, 1])
        if abs(rho) > float(threshold):
            return False
    return True


def test_passes_redundancy_matches_reference_across_random_configs():
    rng = np.random.default_rng(1)
    n_checks = 0
    for trial in range(400):
        n = int(rng.integers(20, 300))
        kind = trial % 6
        a = rng.normal(size=n)
        if kind == 0:
            b = a * 2.0 + rng.normal(scale=0.001, size=n)
        elif kind == 1:
            b = rng.normal(size=n)
        elif kind == 2:
            b = np.full(n, 3.0)
        elif kind == 3:
            a = np.full(n, 5.0)
            b = rng.normal(size=n)
        elif kind == 4:
            a = rng.normal(size=n)
            b = a.copy()
            b[rng.choice(n, size=max(1, n // 5), replace=False)] = np.nan
        else:
            a = rng.normal(size=n)
            b = rng.normal(size=n)
            a[rng.choice(n, size=max(1, n // 3), replace=False)] = np.nan

        cand = np.nan_to_num(a / (b + 1e-9), nan=0.0, posinf=0.0, neginf=0.0)
        threshold = float(rng.choice([0.5, 0.9, 0.99]))
        ref = _passes_redundancy_ref(cand, a, b, threshold)
        new = _passes_redundancy(cand.copy(), a.copy(), b.copy(), threshold)
        n_checks += 1
        assert ref == new, f"trial={trial} kind={kind} n={n} threshold={threshold} ref={ref} new={new}"
    assert n_checks == 400


def test_passes_redundancy_matches_reference_on_thin_joint_overlap():
    """Edge case that motivated ``min_n=2``: a joint-finite overlap of 1-9 rows (below the kernel's
    small-sample-noise default floor of 8) must still reproduce the masked-corrcoef verdict exactly."""
    rng = np.random.default_rng(2)
    for trial in range(80):
        n = 40
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        n_finite = int(rng.integers(1, 10))
        mask = np.zeros(n, dtype=bool)
        mask[:n_finite] = True
        rng.shuffle(mask)
        b_e = b.copy()
        b_e[~mask] = np.nan
        cand = np.nan_to_num(a / (b_e + 1e-9), nan=0.0, posinf=0.0, neginf=0.0)
        threshold = 0.5
        ref = _passes_redundancy_ref(cand, a, b_e, threshold)
        new = _passes_redundancy(cand.copy(), a.copy(), b_e.copy(), threshold)
        assert ref == new, f"trial={trial} n_finite={n_finite} ref={ref} new={new}"
