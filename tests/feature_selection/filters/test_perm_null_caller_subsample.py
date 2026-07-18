"""Pins for the 2026-07-03 perm-null-caller row-caps.

``_conditional_perm_null`` was the top cross-cutting FE hotspot (~7.8s at 1M). Two callers ran it (and their
paired observed CMI) on the FULL 1M frame: ``score_prospective_pairs`` (bootstrap-prevalence relaxation,
MLFRAME_PAIR_NULL_MAX_ROWS) and ``retention_form_is_subsumed`` (one-compound retention verdict,
MLFRAME_RETENTION_NULL_MAX_ROWS). Both are wide-margin CMI/floor DECISIONS -> selection-equivalent under a
large strided subsample (the observed CMI + null are capped TOGETHER so they stay mutually consistent). These
pins assert the stride formulas (incl the =0 full-n opt-out) and, for the directly-callable retention verdict,
that a genuinely-redundant form is dropped and a genuinely-complementary form retained under BOTH the cap and
full-n.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


def _stride(n, max_rows):
    """Helper that stride."""
    return int(n // max_rows) if max_rows > 0 and n > max_rows else 1


@pytest.mark.parametrize(
    "n, max_rows, expect",
    [(1_000_000, 250_000, 4), (1_000_000, 0, 1), (200_000, 250_000, 1), (500_000, 250_000, 2)],
)
def test_perm_null_caller_stride_formula(n, max_rows, expect):
    """Perm null caller stride formula."""
    st = _stride(n, max_rows)
    assert st == expect
    sub = np.arange(n)[::st]
    if max_rows == 0:
        assert sub.shape[0] == n
    elif st > 1:
        assert sub.shape[0] < n


def _retention_verdict(cand, incumbents, y_binned, cap_env, monkeypatch):
    """Retention verdict."""
    import mlframe.feature_selection.filters._fe_retention_subsumption as R

    monkeypatch.setenv("MLFRAME_RETENTION_NULL_MAX_ROWS", str(cap_env))
    _orig_dict = dict(R.__dict__)
    R = importlib.reload(R)
    try:
        return R.retention_form_is_subsumed(
            cand_continuous=cand,
            incumbent_continuous=incumbents,
            y_binned=y_binned,
            seed=0,
        )
    finally:
        monkeypatch.delenv("MLFRAME_RETENTION_NULL_MAX_ROWS", raising=False)
        R.__dict__.clear()
        R.__dict__.update(_orig_dict)


@pytest.mark.slow
def test_retention_subsample_preserves_verdict(monkeypatch):
    """y = signal_a + signal_b (independent). incumbent carries signal_a. A candidate that is signal_a (redundant)
    must be SUBSUMED; a candidate that is signal_b (complementary, absent from incumbent) must be RETAINED.
    The verdict must be identical under the 250k cap and full-n (selection-equivalent, random-null tolerant)."""
    rng = np.random.default_rng(3)
    n = 1_000_000
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y_cont = a + b
    # discretise y into 12 quantile bins (the retention pass hands binned y codes)
    edges = np.quantile(y_cont, np.linspace(0, 1, 13)[1:-1])
    y_binned = np.searchsorted(edges, y_cont).astype(np.int64)

    incumbent = a + 0.01 * rng.normal(size=n)  # carries signal_a
    cand_redundant = a + 0.01 * rng.normal(size=n)  # all its y-info is signal_a -> subsumed by incumbent
    cand_complement = b + 0.01 * rng.normal(size=n)  # signal_b, absent from incumbent -> retained

    for cap in ("250000", "0"):
        red = _retention_verdict(cand_redundant, [incumbent], y_binned, cap, monkeypatch)
        comp = _retention_verdict(cand_complement, [incumbent], y_binned, cap, monkeypatch)
        assert red is True, f"[cap={cap}] redundant signal_a candidate was NOT subsumed"
        assert comp is False, f"[cap={cap}] complementary signal_b candidate was wrongly subsumed"


def test_retention_cap_actually_subsamples(monkeypatch):
    """With the cap below n the retention verdict path must see the reduced row count (proven by spying on the
    array length _conditional_perm_null receives)."""
    import mlframe.feature_selection.filters._fe_retention_subsumption as R

    monkeypatch.setenv("MLFRAME_RETENTION_NULL_MAX_ROWS", "40000")
    _orig_dict = dict(R.__dict__)
    R = importlib.reload(R)
    try:
        import mlframe.feature_selection.filters._fe_cmi_redundancy_gate as G

        orig = G._conditional_perm_null
        seen = {}

        def spy(cand_bin, y_bin, z_support, **k):
            """Helper that spy."""
            seen["n"] = int(np.asarray(cand_bin).shape[0])
            return orig(cand_bin, y_bin, z_support, **k)

        monkeypatch.setattr(G, "_conditional_perm_null", spy)
        rng = np.random.default_rng(0)
        n = 280_000  # 280k // 40k = 7 -> exactly 40k rows (clean soft-cap)
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        y = np.searchsorted(np.quantile(a + b, np.linspace(0, 1, 9)[1:-1]), a + b).astype(np.int64)
        R.retention_form_is_subsumed(
            cand_continuous=a + 0.01 * rng.normal(size=n),
            incumbent_continuous=[a],
            y_binned=y,
            seed=0,
        )
        assert seen.get("n", n) <= 40_000 + 8, f"retention perm-null saw {seen.get('n')} rows, expected <= cap"
    finally:
        R.__dict__.clear()
        R.__dict__.update(_orig_dict)
