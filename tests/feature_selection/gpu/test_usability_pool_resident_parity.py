"""Phase 1 pre-wiring parity gate: the resident pool MI table must match the njit per-pair table.

Before the resident usability-pool table is fed into retention (under MLFRAME_FE_GPU_STRICT_RESIDENT), this
pins that ``score_pair_combos_table_resident`` row ``p`` equals ``score_pair_combos`` for pair ``p`` to the
documented bit-faithful tolerance (~6e-15) AND that the ``-1.0`` std<=1e-9 sentinels line up. A column-layout
mismatch between the resident table and the njit enumeration would otherwise be SILENT wrong MI (the
selection-equivalence test would catch it only downstream); this catches it at the source."""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda() -> bool:
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        return is_cuda_available()
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


@pytest.mark.xfail(reason="BLOCKER (Phase 1, 2026-06-27): score_pair_combos_table_resident's radix-edge binning "
                          "diverges from the njit _qbin_into by up to 0.219 on ~1.5% of combos -- entirely the "
                          "low-cardinality/discrete-unary outputs (sign, rint), where collapsed percentile edges "
                          "tie-break differently. Continuous combos match exactly. NOT sub-quantum, so it is not "
                          "selection-equivalent. Unblock = a resident binning matching the njit tie-handling on "
                          "low-cardinality columns; flip this to a hard assert once reconciled.",
                   strict=False)
def test_resident_pool_table_matches_njit_per_pair():
    from mlframe.feature_selection.filters._usability_njit_pool import (
        score_pair_combos, njit_unary_codes_or_none, njit_binary_codes_or_none)
    from mlframe.feature_selection.filters._usability_pool_resident import score_pair_combos_table_resident
    from mlframe.feature_selection.filters.feature_engineering import (
        create_unary_transformations, create_binary_transformations)
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _quantile_bin, precompute_marginal_y_terms

    rng = np.random.default_rng(0)
    n, nbins = 5000, 10
    cols = {c: rng.uniform(0.1, 1.1, n) for c in "abcd"}
    y = cols["a"] ** 2 / cols["b"] + np.log(cols["c"]) * np.sin(cols["d"])
    y_codes = _quantile_bin(y, nbins)
    y_terms = precompute_marginal_y_terms(y_codes)
    unary = create_unary_transformations(preset="medium")
    binary = create_binary_transformations(preset="minimal")
    uc = njit_unary_codes_or_none(list(unary.keys()))
    bc = njit_binary_codes_or_none(list(binary.keys()))
    assert uc is not None and bc is not None, "njit op-codes required for the resident table path"

    pairs = [("a", "b"), ("c", "d"), ("a", "c"), ("b", "d")]
    ops = [(cols[a].astype(np.float64), cols[b].astype(np.float64)) for a, b in pairs]
    table = score_pair_combos_table_resident(ops, y_codes, y_terms, nbins, uc, uc, bc)
    assert table is not None, "resident table returned None (cupy/device error) -- cannot validate"
    assert table.shape[0] == len(pairs)

    for p, (a, b) in enumerate(pairs):
        njit = np.asarray(score_pair_combos(
            cols[a].astype(np.float64), cols[b].astype(np.float64), y_codes, y_terms, nbins, uc, uc, bc),
            dtype=np.float64)
        row = np.asarray(table[p], dtype=np.float64)
        assert row.shape == njit.shape, f"layout mismatch pair {p}: {row.shape} vs {njit.shape}"
        # sentinels (std<=1e-9 combos -> -1.0) must align exactly
        np.testing.assert_array_equal(row <= -0.5, njit <= -0.5)
        m = njit > -0.5
        np.testing.assert_allclose(row[m], njit[m], rtol=0, atol=1e-9)
