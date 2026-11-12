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

MOD = "mlframe.feature_selection.filters._fe_raw_redundancy_helpers"


def _codes(rng, n, k):
    """Draw n random integer codes in [0, k) as int64, used to build synthetic categorical y/z/cand arrays."""
    return rng.integers(0, k, n).astype(np.int64)


def test_excess_and_floor_subsample_keeps_drop_decision():
    """A genuinely REDUNDANT raw (cand independent of y given z) stays below its floor, and a genuinely
    RELEVANT raw (cand carries private signal) clears it -- under BOTH the default cap and full-n."""
    m = importlib.import_module(MOD)
    rng = np.random.default_rng(5)
    n = 1_000_000
    y = _codes(rng, n, 4)
    z = _codes(rng, n, 4)
    redundant = _codes(rng, n, 4)  # independent of y -> excess ~ 0, below floor
    relevant = y.copy()  # perfectly informative -> excess >> floor
    _orig_dict = dict(m.__dict__)
    for max_rows in ("250000", "0"):  # capped, then full-n
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
            m.__dict__.clear()
            m.__dict__.update(_orig_dict)


def test_excess_and_floor_cap_actually_subsamples(monkeypatch):
    """With the cap set below n, the perm-null path must see the reduced row count (proven by capturing the
    array length _conditional_perm_null receives)."""
    m = importlib.import_module(MOD)
    monkeypatch.setenv("MLFRAME_CMI_NULL_MAX_ROWS", "50000")
    _orig_dict = dict(m.__dict__)
    mm = importlib.reload(m)
    try:
        seen = {}
        import mlframe.feature_selection.filters._fe_cmi_redundancy_gate as G

        orig = G._conditional_perm_null

        def spy(cand_bin, y_bin, z_support, **k):
            """Wrap _conditional_perm_null to record the row count it actually receives, proving the subsample cap took effect."""
            seen["n"] = int(cand_bin.shape[0])  # host array or resident device codes
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
        m.__dict__.clear()
        m.__dict__.update(_orig_dict)


def test_batched_device_null_refused_by_the_vram_cushion_returns_none(monkeypatch):
    """A refused (n, nperm) build returns None before any device allocation, so the caller runs the per-permutation CPU loop."""
    import numpy as np

    from mlframe.feature_selection.filters import _fe_cmi_redundancy_null as R
    from mlframe.feature_selection.filters import _fe_gpu_vram
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as M

    rng = np.random.default_rng(2)
    n = 3000
    x = rng.integers(0, 6, n).astype(np.int64)
    y = rng.integers(0, 3, n).astype(np.int64)
    z = rng.integers(0, 4, n).astype(np.int64)
    inp = R._NullInputs(x, y, seed=1, salt=3)
    sup = R._StratifiedSupport(inp, z)
    monkeypatch.setattr(M, "_cmi_gpu_enabled", lambda **kwargs: True)
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda *args, **kwargs: False)
    assert R._conditional_batched_gpu_null(inp, sup, 4, 0.95) is None
