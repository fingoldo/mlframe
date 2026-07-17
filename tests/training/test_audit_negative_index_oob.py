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

from pathlib import Path



MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def _read(rel: str) -> str:
    """Reads an mlframe source file's text for source-level assertions."""
    return (MLFRAME_ROOT / rel).read_text(encoding="utf-8")


def test_votenrank_agreement_rate_clamps_k_to_subset_size() -> None:
    """agreement_rate must clamp k to len(subset) before dividing, so a small subset can't produce a negative index."""
    src = _read("votenrank/utils.py")
    # The fix introduces _k_eff = min(_k_eff, len(subset)) per iter and uses
    # _denom = max(1, _k_eff) as the divisor.
    assert "_k_eff = min(_k_eff, len(subset))" in src
    assert "_denom = max(1, _k_eff)" in src
    # The pre-fix `len(...) / k` divisor must be gone.
    assert 'intersection(set(res_d["AM"]))) / k' not in src


def test_negative_index_slice_wraps_on_short_array_documents_invariant() -> None:
    """Document the bug-class invariant: arr[-N:] when N > len(arr) silently
    returns the whole array. Sensor here makes the contract visible for any
    future code reviewer."""
    arr = [1, 2, 3]
    assert arr[-100:] == [1, 2, 3], (
        "Python's negative-index slice returns the WHOLE array when |N| > len(arr); callers that assume `arr[-N:]` returns exactly N items must guard."
    )
    assert arr[:-100] == [], "Conversely, `arr[:-N]` returns empty when |N| >= len(arr)."
