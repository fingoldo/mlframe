"""Pins for the 2026-07-03 raw-drop CMI/perm-null row-cap (``MLFRAME_CMI_NULL_MAX_ROWS``).

The raw-operand redundancy ``_excess_and_floor`` estimates the observed CMI AND the conditional-permutation
null on a strided subsample of cand/y/z together, so the returned (cmi, floor, excess) stay mutually
consistent while the ~25-perm within-stratum null on a full-1M raw stops dominating. These pins assert the
cap is applied, is opt-out-able, and yields a decision consistent with the full-n estimate (the perm-null is
an explicitly random null -> selection-equivalence, not byte-identity).
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

MOD = "mlframe.feature_selection.filters._fe_raw_redundancy_helpers"


def _codes(rng, n, k):
    return rng.integers(0, k, n).astype(np.int64)


def test_excess_and_floor_subsample_keeps_drop_decision():
    """A genuinely REDUNDANT raw (cand independent of y given z) stays below its floor, and a genuinely
    RELEVANT raw (cand carries private signal) clears it -- under BOTH the default cap and full-n."""
    m = importlib.import_module(MOD)
    rng = np.random.default_rng(5)
    n = 1_000_000
    y = _codes(rng, n, 4)
    z = _codes(rng, n, 4)
    redundant = _codes(rng, n, 4)                       # independent of y -> excess ~ 0, below floor
    relevant = y.copy()                                 # perfectly informative -> excess >> floor
    for max_rows in ("250000", "0"):                    # capped, then full-n
        import os
        os.environ["MLFRAME_CMI_NULL_MAX_ROWS"] = max_rows
        mm = importlib.reload(m)
        try:
            _cmi_r, _fl_r, exc_r = mm._excess_and_floor(redundant, y, z, seed=1)
            _cmi_g, _fl_g, exc_g = mm._excess_and_floor(relevant, y, z, seed=1)
            assert exc_g > exc_r, f"[cap={max_rows}] relevant excess {exc_g:.4f} !> redundant {exc_r:.4f}"
            assert _cmi_g > _fl_g, f"[cap={max_rows}] relevant CMI did not clear its floor"
        finally:
            os.environ.pop("MLFRAME_CMI_NULL_MAX_ROWS", None)
            importlib.reload(m)


def test_excess_and_floor_cap_actually_subsamples(monkeypatch):
    """With the cap set below n, the perm-null path must see the reduced row count (proven by capturing the
    array length _conditional_perm_null receives)."""
    m = importlib.import_module(MOD)
    monkeypatch.setenv("MLFRAME_CMI_NULL_MAX_ROWS", "50000")
    mm = importlib.reload(m)
    try:
        seen = {}
        import mlframe.feature_selection.filters._fe_cmi_redundancy_gate as G
        orig = G._conditional_perm_null

        def spy(cand_bin, y_bin, z_support, **k):
            seen["n"] = int(np.asarray(cand_bin).shape[0])
            return orig(cand_bin, y_bin, z_support, **k)

        monkeypatch.setattr(G, "_conditional_perm_null", spy)
        rng = np.random.default_rng(0)
        n = 400_000
        y = _codes(rng, n, 3)
        z = _codes(rng, n, 3)
        cand = _codes(rng, n, 3)
        mm._excess_and_floor(cand, y, z, seed=0)
        assert seen.get("n", n) <= 50_000 + 8, f"perm-null saw {seen.get('n')} rows, expected <= cap"
    finally:
        importlib.reload(m)
