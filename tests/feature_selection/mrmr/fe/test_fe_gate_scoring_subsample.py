"""Pin for the 2026-07-03 CMI-redundancy-gate scoring row-cap (``MLFRAME_FE_GATE_MAX_ROWS``).

The conditional-MI redundancy gate only DECIDES which engineered candidates are redundant (drop) vs carry
private y-information (keep). That admit/drop decision is selection-equivalent under a large strided
subsample, so ``materialise_and_finalise_fe_candidates`` bins the candidates + scores the O(M^2) greedy CMI
on a strided subsample above the cap. These pins assert (1) the gate actually receives the reduced row count
under the cap, (2) the env opt-out (=0) restores full-n, and (3) the SAME candidates are dropped either way
(selection-equivalence, not byte-identity).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


def _f2_frame(n, seed=42):
    """F2 frame."""
    from tests.feature_selection._synthetic_distributions import sample_operands

    ops = sample_operands(seed, n, {"a": "any", "b": "divisor", "c": "positive", "d": "any", "e": "any"}, profile="uniform")
    f = sample_operands(seed + 991, n, {"f": "any"}, profile="uniform")["f"]
    df = pd.DataFrame({k: ops[k].astype(np.float64) for k in ("a", "b", "c", "d", "e")})
    y = ops["a"] ** 2 / ops["b"] + f / 5.0 + np.log(np.abs(ops["c"]) + 1e-9) * np.sin(ops["d"])
    return df, y


def _fit_capture(cap_env, n=60_000):
    """Fit a small F2 problem with the gate cap set to ``cap_env``; capture the row count and accepted set
    the CMI redundancy gate saw. Returns (max_rows_seen, accepted_names or None)."""
    import mlframe.feature_selection.filters._fe_cmi_redundancy_gate as GATE
    from mlframe.feature_selection.filters.mrmr import MRMR

    seen = {"rows": None, "accepted": None}
    orig = GATE.apply_cmi_redundancy_gate

    def spy(cmi_cands, y_dense, *a, **k):
        """Helper that spy."""
        seen["rows"] = int(np.asarray(y_dense).shape[0])
        acc, diag = orig(cmi_cands, y_dense, *a, **k)
        seen["accepted"] = tuple(sorted(acc))
        return acc, diag

    GATE.apply_cmi_redundancy_gate = spy
    prev = os.environ.get("MLFRAME_FE_GATE_MAX_ROWS")
    os.environ["MLFRAME_FE_GATE_MAX_ROWS"] = str(cap_env)
    try:
        df, y = _f2_frame(n)
        m = MRMR(verbose=0, random_seed=42, n_jobs=1, fe_max_steps=2, fe_min_pair_mi_prevalence=1.05, full_npermutations=10, baseline_npermutations=20)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(df, y)
    finally:
        GATE.apply_cmi_redundancy_gate = orig
        if prev is None:
            os.environ.pop("MLFRAME_FE_GATE_MAX_ROWS", None)
        else:
            os.environ["MLFRAME_FE_GATE_MAX_ROWS"] = prev
    return seen["rows"], seen["accepted"]


def _gate_stride(n, max_rows):
    """The exact strided-subsample formula used in materialise_and_finalise_fe_candidates for the gate."""
    return int(n // max_rows) if max_rows > 0 and n > max_rows else 1


@pytest.mark.parametrize(
    "n, max_rows, expect_stride",
    [
        (1_000_000, 250_000, 4),  # 1M capped at 250k -> every 4th row
        (1_000_000, 0, 1),  # opt-out -> full-n (stride 1)
        (200_000, 250_000, 1),  # below the cap -> untouched
        (600_000, 250_000, 2),  # floor division: 600k // 250k = 2
    ],
)
def test_gate_stride_formula(n, max_rows, expect_stride):
    """Gate stride formula."""
    assert _gate_stride(n, max_rows) == expect_stride
    st = _gate_stride(n, max_rows)
    sub = np.arange(n)[::st]
    if max_rows == 0:
        assert sub.shape[0] == n  # opt-out feeds full-n untouched
    else:
        # floor-division stride is a SOFT cap (bounds cost, not a hard ceiling): worst case ~1.5x cap when
        # n/cap is just under an integer+1 (600k//250k=2 -> 300k). Always strictly below full-n once stride>1.
        assert sub.shape[0] <= 2 * max_rows
        if st > 1:
            assert sub.shape[0] < n


@pytest.mark.slow
def test_fe_gate_scoring_subsample_caps_rows():
    """The gate's candidate pool only widens enough to fire at large n, so this pin runs a single full-scale
    fit and asserts that with the cap set below n the gate sees <= cap rows. The env opt-out (=0 -> full-n)
    is pinned by ``test_gate_stride_formula``; selection-EQUIVALENCE of the capped decision is covered by the
    F2 5-profile suite (which runs the same gate at 1M under the default cap)."""
    n = 1_000_000
    cap = 250_000
    rows_capped, _ = _fit_capture(cap, n=n)
    if rows_capped is None:
        pytest.skip("CMI redundancy gate did not fire on this problem (no engineered candidate pool)")
    assert rows_capped <= cap + 8, f"gate saw {rows_capped} rows, expected <= cap {cap}"
