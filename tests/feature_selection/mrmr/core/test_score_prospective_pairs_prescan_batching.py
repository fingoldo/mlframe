"""Regression test for the batched dominant-pair prescan in ``score_prospective_pairs``
(``_step_pairs_rank.py``, 2026-07-11 perf fix).

The tail-concentration first-sweep prevalence-relaxation prescan used to find the pool's dominant pair (by
best PAIR-form ``|corr|``) via a serial Python loop calling ``usability_form_corrs`` once per candidate pair
-- measured 4.2-4.9x SLOWER at prescan-representative scale (2k-30k pairs) than collecting the pool first
and dispatching ONE batched call (``batch_pair_usability_corr_gpu``'s ``njit(parallel=True)`` backend).

This test pins the CONTRACT the fix depends on: replicating BOTH the original per-pair serial logic and the
new batched logic (as standalone helpers mirroring the real code paths) on synthetic data, and asserting
they pick the IDENTICAL dominant pair -- including the exact tie-break rule (numpy ``argmax``'s
first-occurrence-of-the-max matching the original loop's strict ``>`` comparison), the row-subsample
consistency (the batched path must apply the SAME ``_corr_stride`` the per-pair path applies internally, or
results silently diverge), AND dtype consistency (``usability_form_corrs`` casts its inputs to
``_crit_np_dtype()`` -- f32 by default -- internally; a batched caller that skips this and passes raw f64
computes every form at strictly higher precision than the reference, a real ~1e-9 divergence found via
direct A/B during development, not assumed)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_usability_signal import _corr_stride, _crit_np_dtype, usability_form_corrs
from mlframe.feature_selection.filters.batch_pair_usability_corr_gpu import (
    ALL_PAIR_FORM_IDS,
    dispatch_batch_pair_usability_corr,
)


def _serial_dominant_pair(y, operands, pair_keys):
    """Mirrors the ORIGINAL per-pair serial loop exactly."""
    dom_p0, dom_p1, dom_cp, dom_pk = None, None, -1.0, None
    for pk in pair_keys:
        p0, p1 = operands[pk[0]], operands[pk[1]]
        cp_pre, _ = usability_form_corrs(y, p0, p1)
        if cp_pre > dom_cp:
            dom_cp, dom_p0, dom_p1, dom_pk = cp_pre, p0, p1, pk
    return dom_pk, dom_cp, dom_p0, dom_p1


def _batched_dominant_pair(y, operands, pair_keys):
    """Mirrors the NEW batched logic exactly (same stride handling, same dtype cast, same argmax tie-break)."""
    stride = _corr_stride(y.shape[0])
    dtype = _crit_np_dtype()
    y_batch = np.asarray(y[::stride] if stride > 1 else y, dtype=dtype)

    op_row_of: dict = {}
    op_rows: list = []
    for pk in pair_keys:
        for idx in (pk[0], pk[1]):
            if idx not in op_row_of:
                op_row_of[idx] = len(op_rows)
                full = operands[idx]
                sub = full[::stride] if stride > 1 else full
                op_rows.append(np.asarray(sub, dtype=dtype))
    operand_matrix = np.vstack(op_rows)
    pair_a = np.array([op_row_of[pk[0]] for pk in pair_keys], dtype=np.int64)
    pair_b = np.array([op_row_of[pk[1]] for pk in pair_keys], dtype=np.int64)

    corrs, _ = dispatch_batch_pair_usability_corr(y_batch, operand_matrix, pair_a, pair_b, form_ids=ALL_PAIR_FORM_IDS)
    cp_all = corrs.max(axis=1)
    best_i = int(np.argmax(cp_all))
    dom_pk = pair_keys[best_i]
    dom_cp = float(cp_all[best_i])
    return dom_pk, dom_cp, operands[dom_pk[0]], operands[dom_pk[1]]


def test_batched_and_serial_pick_identical_dominant_pair_below_subsample_cap():
    """n below _ABS_PEARSON_MAX_ROWS -- no subsampling involved, isolates the batching logic itself."""
    rng = np.random.default_rng(30)
    n = 4000
    y = rng.standard_normal(n)
    operands = {i: rng.standard_normal(n) + i * 0.3 for i in range(8)}
    pair_keys = [(0, 1), (2, 3), (1, 4), (5, 6), (0, 6), (3, 7)]

    serial_pk, serial_cp, _, _ = _serial_dominant_pair(y, operands, pair_keys)
    batched_pk, batched_cp, _, _ = _batched_dominant_pair(y, operands, pair_keys)

    assert batched_pk == serial_pk
    assert batched_cp == pytest.approx(serial_cp, abs=1e-6)


def test_batched_and_serial_agree_above_subsample_cap():
    """n ABOVE _ABS_PEARSON_MAX_ROWS (30000) -- exercises the stride-consistency fix directly; a batched
    path that forgets to subsample would silently diverge from the serial reference here."""
    rng = np.random.default_rng(31)
    n = 45_000  # > _ABS_PEARSON_MAX_ROWS default (30_000) -- triggers a real stride > 1
    y = rng.standard_normal(n)
    operands = {i: rng.standard_normal(n) + i * 0.25 for i in range(6)}
    pair_keys = [(0, 1), (2, 3), (1, 5), (4, 0), (3, 5)]

    serial_pk, serial_cp, _, _ = _serial_dominant_pair(y, operands, pair_keys)
    batched_pk, batched_cp, _, _ = _batched_dominant_pair(y, operands, pair_keys)

    assert batched_pk == serial_pk, (batched_pk, serial_pk)
    assert batched_cp == pytest.approx(serial_cp, abs=1e-6)


def test_tie_break_matches_strict_greater_than_semantics():
    """Two pairs with near-tied (by construction, IDENTICAL) dominant |corr| -- the FIRST-scanned pair must
    win, matching the original loop's strict `>` (a later equal value never replaces the incumbent) and
    numpy argmax's first-occurrence-of-the-max behavior."""
    rng = np.random.default_rng(32)
    n = 3000
    y = rng.standard_normal(n)
    x_shared = rng.standard_normal(n)
    x_other = rng.standard_normal(n)
    x_noise = rng.standard_normal(n)
    # pair (0,1) and pair (2,1) reduce to the EXACT same underlying values (operand 2 is a copy of operand
    # 0) -> identical |corr|; operand 3 is unrelated noise so pair (3,1) is not a contender for the max.
    operands = {0: x_shared, 1: x_other, 2: x_shared.copy(), 3: x_noise}
    pair_keys = [(0, 1), (2, 1), (3, 1)]

    serial_pk, _serial_cp, _, _ = _serial_dominant_pair(y, operands, pair_keys)
    batched_pk, _batched_cp, _, _ = _batched_dominant_pair(y, operands, pair_keys)

    assert batched_pk == serial_pk == (0, 1), "first-scanned pair among exact ties must win in BOTH paths"


def test_single_pair_pool():
    """Single pair pool."""
    rng = np.random.default_rng(33)
    n = 2000
    y = rng.standard_normal(n)
    operands = {0: rng.standard_normal(n), 1: rng.standard_normal(n)}
    pair_keys = [(0, 1)]

    serial_pk, serial_cp, _, _ = _serial_dominant_pair(y, operands, pair_keys)
    batched_pk, batched_cp, _, _ = _batched_dominant_pair(y, operands, pair_keys)
    assert batched_pk == serial_pk
    assert batched_cp == pytest.approx(serial_cp, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
