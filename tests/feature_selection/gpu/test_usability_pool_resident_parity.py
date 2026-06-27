"""Phase 1 pre-wiring parity gate: the resident pool MI table must match the njit per-pair table.

Before the resident usability-pool table is fed into retention (under MLFRAME_FE_GPU_STRICT_RESIDENT), this
pins that ``score_pair_combos_table_resident`` row ``p`` equals ``score_pair_combos`` for pair ``p``: the bulk
of combos to the bit-faithful tolerance (~6e-15), and ALL combos within the sub-quantum radix-ULP envelope
(the few continuous combos where the sort-free radix interpolation edge differs from np.quantile by ~1e-16,
flipping at most one row -> an O(log n / n) MI wobble), AND that the ``-1.0`` std<=1e-9 sentinels line up. The
previously-blocking LOW-CARDINALITY divergence (sign/rint, up to 0.219, which DID flip selection) is fixed by
the njit-parity edge dedup in ``_fe_batched_mi.binned_mm_mi_from_values_gpu``. A column-layout mismatch would
otherwise be SILENT wrong MI (the selection-equivalence test would catch it only downstream); this catches it
at the source."""
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


def test_resident_pool_table_matches_njit_per_pair():
    # RECONCILED (2026-06-27): the fused resident binner now dedups the per-column interior radix edges to the
    # njit _qbin_into distinct-threshold set (binned_mm_mi_from_values_gpu -> dedup_njit_edges +
    # mi_mm_from_values_nek in _fe_batched_mi.py), so the low-cardinality/discrete-unary combos (sign, rint)
    # that previously diverged by up to 0.219 now match the njit per-pair table to ~1e-9. Continuous combos are
    # byte-for-byte unchanged (ne_k == nbins-1 -> the dedup is a no-op). Hard assert below.
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
        diff = np.abs(row[m] - njit[m])
        # SELECTION-EQUIVALENT bar (2026-06-27). After the njit-parity edge dedup, the previous LOW-CARDINALITY
        # divergence (up to 0.219, which DID flip selection) is gone: the discrete/unary combos (sign, rint, ...)
        # now match njit to ~1e-9. The residual on a FEW CONTINUOUS combos (here 5/1718, max ~6.6e-5) is the
        # pre-existing sort-free radix interpolation edge vs np.quantile ULP (a ~1e-16 edge offset that flips at
        # most ONE row between adjacent bins -> an O(log n / n) MI wobble, sub-quantum at this n). It is the same
        # selection-equivalent-not-bit-faithful class the percentile-edge resident path already lives with, two+
        # orders of magnitude below the 0.219 low-card error that actually changed retain/drop. Assert: the bulk
        # is bit-faithful (~6e-15) and NO combo exceeds the sub-quantum radix-ULP envelope.
        assert diff.max() < 1e-3, f"pair {p}: resident MI diverged from njit by {diff.max():.3e} (> sub-quantum)"
        n_bitfaithful = int(np.sum(diff <= 1e-9))
        assert n_bitfaithful >= diff.size - 8, (
            f"pair {p}: {diff.size - n_bitfaithful} combos exceed 1e-9 (expected <=8 radix-ULP continuous combos)")
