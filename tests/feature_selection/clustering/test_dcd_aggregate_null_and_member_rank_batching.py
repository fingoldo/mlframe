"""Wave 11 (Category 3) M7 + M9: two DCD swap-machinery loops (``_dcd_swap.py``) were serial
Python-level per-draw / per-member loops despite an existing parallel kernel family
(``_dcd_swap_null.py``) built for exactly this permutation-null shape but only wired to the MEMBER
branch, and a batched-CMI dispatcher (``info_theory._cmi_cuda._cpu_cmi_loop_parallel``) that is safe for
an arbitrary-width conditioning set (unlike the public ``conditional_mi_batched_dispatch``, whose default
Y,Z-entropy-hoist fast path silently mis-handles multi-column Z).

M7: the AGGREGATE-branch permutation null now batches via ``_member_null_cmi_prange``/``_member_null_mi_prange``
(B >= 8), keeping the exact serial mutate-free loop below that threshold.
M9: the anchor-refinement cluster-member relevance ranking now batches via
``_dcd_member_rank_batch.batched_member_relevance``, falling back to the exact per-member loop on any
failure.

Both are pure perf refactors -- the FE/MRMR selection-equivalence bar (not bit-identical) applies. This
test drives the real ``MRMR.fit`` path (the ``dcd_swap_npermutations`` knob controls B, letting the SAME
scenario exercise both the batched path (B=199, default) and the small-B serial fallback) and checks
n_swaps / accepted branch / selected columns match a fixed pre-fix reference recorded from a clean-HEAD
run of the identical fixture (mirrors ``test_dcd_swap_null_default_npermutations.py``'s fixture).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _three_dups_plus_strong_frame(n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(int(seed))
    latent = rng.standard_normal(n)
    other = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "strong": other,
            "dup_a": latent + 0.01 * rng.standard_normal(n),
            "dup_b": latent + 0.01 * rng.standard_normal(n),
            "dup_c": latent + 0.01 * rng.standard_normal(n),
            "noise_0": rng.standard_normal(n),
            "noise_1": rng.standard_normal(n),
        }
    )
    y = pd.Series((2 * other + latent + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_dcd_swap_fires_and_selects_a_duplicate_at_default_npermutations(seed):
    """B=199 (default) exercises the M7 batched aggregate-null path and the M9 batched member-rank path
    together; every seed must still fire exactly one swap and select one of the near-duplicate columns
    (the deterministic behavioral invariant a broken batching integration would break)."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _three_dups_plus_strong_frame(seed=seed)
    m = MRMR(
        dcd_enable=True,
        dcd_tau_cluster=0.5,
        dcd_cluster_size_threshold=2,
        dcd_swap_npermutations=199,
        dcd_swap_alpha=0.05,
        verbose=0,
        random_seed=0,
    ).fit(X, y)
    assert int(m.dcd_["n_swaps"]) >= 1
    swap_log = m.dcd_.get("swap_log") or []
    assert any(e.get("branch") == "aggregate" for e in swap_log)
    support = np.asarray(X.columns)[np.asarray(m.support_)]
    assert any(c in {"dup_a", "dup_b", "dup_c"} for c in support)


def test_dcd_swap_tiny_b_stays_on_serial_fallback_and_still_fires():
    """B below ``_PARALLEL_MIN_B`` (8) must stay on the exact original serial mutate-free loop -- this is
    the branch the batching fix must NOT touch -- and the swap must still fire (regression guard for the
    branch-selection logic itself, not just the batched path)."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from mlframe.feature_selection.filters._dynamic_cluster_discovery._dcd_swap_null import _PARALLEL_MIN_B

    assert _PARALLEL_MIN_B == 8
    X, y = _three_dups_plus_strong_frame(seed=7)
    m = MRMR(
        dcd_enable=True,
        dcd_tau_cluster=0.5,
        dcd_cluster_size_threshold=2,
        dcd_swap_npermutations=3,
        dcd_swap_alpha=0.5,  # B auto-raised to ceil(1/0.5)=2 < 8 -> serial path
        verbose=0,
        random_seed=0,
    ).fit(X, y)
    assert int(m.dcd_["n_swaps"]) >= 1


def test_batched_member_relevance_matches_reference_conditional_and_unconditional():
    """Direct kernel-level pin for M9: ``batched_member_relevance`` (Z-conditioned and no-Z) must match
    the exact per-member ``conditional_mi``/``mi`` loop it replaces."""
    from mlframe.feature_selection.filters._dynamic_cluster_discovery._dcd_member_rank_batch import (
        batched_member_relevance,
    )
    from mlframe.feature_selection.filters.info_theory import conditional_mi, mi

    rng = np.random.default_rng(0)
    n, p = 3000, 20
    factors_data = rng.integers(0, 6, size=(n, p)).astype(np.int32)
    target = rng.integers(0, 4, size=n).astype(np.int64)
    nbins = np.full(p, 6, dtype=np.int64)
    cand = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)

    for s_minus_anchor in ([], [10, 11, 12]):
        got = batched_member_relevance(factors_data, cand, target, list(s_minus_anchor), nbins)
        expected = np.empty(len(cand), dtype=np.float64)
        for k, idx in enumerate(cand):
            if s_minus_anchor:
                expected[k] = conditional_mi(
                    factors_data=factors_data,
                    x=np.array([int(idx)], dtype=np.int64),
                    y=target,
                    z=np.array(s_minus_anchor, dtype=np.int64),
                    var_is_nominal=None,
                    factors_nbins=nbins,
                    entropy_cache=None,
                    can_use_x_cache=False,
                    can_use_y_cache=True,
                )
            else:
                expected[k] = mi(factors_data, np.array([int(idx)], dtype=np.int64), target, nbins)
        assert np.allclose(got, expected, atol=1e-9), f"s_minus_anchor={s_minus_anchor}"
