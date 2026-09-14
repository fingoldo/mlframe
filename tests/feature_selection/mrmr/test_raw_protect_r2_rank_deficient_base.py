"""The raw-feature floor-drop protection must score candidates correctly on a rank-deficient base design (mrmr_audit_2026-09-14 RO-3).

The protection re-adds a raw column when it lifts the held-out R^2 of a least-squares fit over the already-selected columns by at least
0.005. A perf fix replaced the per-candidate ``lstsq`` with one unpivoted QR of the base plus a one-column ``qr_insert`` per candidate.
That is exact on a full-rank base, but the base is built from every selected column, where collinear twins routinely survive together.
On a rank-deficient base the triangular factor has a near-zero diagonal, ``solve_triangular`` returns huge coefficients without raising,
and the R^2 (and so the admit decision) is noise; an exactly zero diagonal instead raised and scored every candidate ``-inf``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._mrmr_fit_impl._friend_graph_and_redundancy._raw_protect_r2 import heldout_r2_scorer


def _split(n, seed=0):
    """One third held out, as the protection does."""
    perm = np.random.default_rng(seed).permutation(n)
    va = np.zeros(n, dtype=bool)
    va[perm[: n // 3]] = True
    return ~va, va


def _lstsq_r2(A, y, tr, va):
    """Reference: the rank-revealing, minimum-norm least-squares held-out R^2."""
    coef = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
    yv = y[va]
    return 1.0 - float(np.sum((yv - A[va] @ coef) ** 2)) / float(np.sum((yv - yv.mean()) ** 2))


def _data(duplicate_base: bool, n=3000, seed=0):
    """y = 2a + 3c + noise; the base holds an intercept, a, (optionally a scaled twin of a), and b; the candidate is c."""
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(size=(3, n))
    y = 2.0 * a + 3.0 * c + 0.5 * rng.normal(size=n)
    cols = [np.ones(n), a] + ([2.0 * a] if duplicate_base else []) + [b]
    return np.column_stack(cols), c, y


@pytest.mark.parametrize("duplicate_base", [True, False], ids=["rank_deficient_base", "full_rank_base"])
def test_heldout_r2_matches_lstsq(duplicate_base):
    """Base-only and base+candidate R^2 must both match the lstsq reference, whether or not the base is rank deficient."""
    base, cand, y = _data(duplicate_base)
    tr, va = _split(len(y))
    r2 = heldout_r2_scorer(base, y, tr, va)
    ref_base = _lstsq_r2(base, y, tr, va)
    ref_cand = _lstsq_r2(np.column_stack((base, cand)), y, tr, va)
    assert r2() == pytest.approx(ref_base, abs=1e-8)
    assert r2(cand) == pytest.approx(ref_cand, abs=1e-8)
    assert r2(cand) - r2() > 0.005, "a genuinely informative candidate must clear the protection's R^2 increment bar"


def test_candidate_collinear_with_the_base_adds_nothing():
    """A candidate that duplicates a base column carries no new information: its increment must be ~0, not noise."""
    base, _, y = _data(duplicate_base=False)
    tr, va = _split(len(y))
    r2 = heldout_r2_scorer(base, y, tr, va)
    twin = 3.0 * base[:, 1]
    assert abs(r2(twin) - r2()) < 1e-8
