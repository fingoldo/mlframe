"""Wave 60 (2026-05-20): negative-index slice OOB silently wrapping.

Audit class: `arr[-N:]` / `arr[:-N]` / `df.tail(N)` where N > len(arr) silently
returns the WHOLE array instead of just the last N elements; downstream code
that assumes "exactly N" produces biased windowed stats.

Result: 1 P2 fix. mlframe is hardened against this bug class -- wave 39 (empty-
input edges) already pushed authors to guard the common patterns. Of 30+
candidates audited, only 1 cold-path leaderboard helper had a real bug.

  1. votenrank/utils.py:29 (agreement_rate)
     `iloc[-k:]` on a subset shorter than k silently returned the whole
     subset; the downstream `len(intersection) / k` divided by the original
     k anyway, inflating the agreement-rate. Fix: clamp k to actual subset
     size and use that as the denominator.

Verified clean (do not refactor):
  - composite_estimator.py:482,489 -- explicit `W < len(train_y)` and
    `min(len(train_y), 10_000)` guards.
  - feature_engineering/categorical.py:72,75 -- explicit nan-pad branch.
  - feature_engineering/transformer/hard_row_attention.py:125 -- branches
    on `k_eff < n_hard`.
  - feature_engineering/transformer/spectral_attention.py:96 -- k clamped
    to `min(n_eigvecs + 1, n - 1)`.
  - composite_cache.py / preprocessing.py / extractors.py -- tail() for
    display/hashing, caller doesn't assume exact N.
  - target_temporal_audit.py:733 -- constant `[:-1]`, author-controlled.
"""

from __future__ import annotations


def test_votenrank_agreement_rate_clamps_k_to_subset_size() -> None:
    """agreement_rate must clamp k to len(subset) before dividing: a 3-row leaderboard asked for k=10 divides by 3, not 10."""
    import pandas as pd

    from mlframe.votenrank.utils import agreement_rate

    df = pd.DataFrame(
        {
            "AM": ["1: a", "2: b", "3: c"],
            "same": ["1: a", "2: b", "3: c"],
            "partial": ["1: a", "2: d", "3: e"],
        }
    )
    assert agreement_rate(df, 10) == {"same": 1.0, "partial": 0.33}
    assert agreement_rate(df, 10, top_k=False) == {"same": 1.0, "partial": 0.33}
    # Within the subset size the divisor is k itself.
    assert agreement_rate(df, 2) == {"same": 1.0, "partial": 0.5}


def test_negative_index_slice_wraps_on_short_array_documents_invariant() -> None:
    """Document the bug-class invariant: arr[-N:] when N > len(arr) silently
    returns the whole array. Sensor here makes the contract visible for any
    future code reviewer."""
    arr = [1, 2, 3]
    assert arr[-100:] == [1, 2, 3], (
        "Python's negative-index slice returns the WHOLE array when |N| > len(arr); callers that assume `arr[-N:]` returns exactly N items must guard."
    )
    assert arr[:-100] == [], "Conversely, `arr[:-N]` returns empty when |N| >= len(arr)."
