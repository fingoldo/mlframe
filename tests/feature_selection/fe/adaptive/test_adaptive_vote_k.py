"""Guarded-hybrid resolution of fe_stability_vote_k (hardcoded-threshold conversion).

See MRMR_HARDCODED_THRESHOLDS_BENCH.md. An explicit int K is honoured verbatim (default 5 ->
byte-identical to pre-2026-06-13); the opt-in "auto" sentinel adapts ONLY downward for tiny n that
cannot sustain 5 reliable folds, and equals 5 for n >= 500 (so it can never inject the noise the
bench saw when K was lowered on data that could sustain 5 folds).
"""

from __future__ import annotations

import pytest

from mlframe.feature_selection.filters._fe_stability_vote import resolve_adaptive_vote_k


@pytest.mark.parametrize("k_int", [2, 3, 5, 7, 10])
@pytest.mark.parametrize("n", [50, 500, 5000, 100000])
def test_explicit_int_is_honoured_verbatim(k_int, n):
    # An explicit int must pass through unchanged at every n -> default 5 stays byte-identical.
    """Explicit int is honoured verbatim."""
    assert resolve_adaptive_vote_k(k_int, n) == k_int


@pytest.mark.parametrize(
    "n,expected",
    [
        (100000, 5),
        (5000, 5),
        (500, 5),  # n >= 500 -> 5 (== legacy default, no behaviour change)
        (499, 4),
        (300, 3),
        (200, 2),
        (100, 2),
        (50, 2),  # tiny n -> guarded downward, floored at 2
    ],
)
def test_auto_is_guarded_n_floored(n, expected):
    """Auto is guarded n floored."""
    assert resolve_adaptive_vote_k("auto", n) == expected


def test_auto_never_exceeds_five_and_never_below_two():
    """Auto never exceeds five and never below two."""
    for n in (1, 10, 250, 600, 10_000, 10_000_000):
        k = resolve_adaptive_vote_k("auto", n)
        assert 2 <= k <= 5


def test_auto_case_insensitive_and_whitespace_tolerant():
    """Auto case insensitive and whitespace tolerant."""
    assert resolve_adaptive_vote_k("AUTO", 5000) == 5
    assert resolve_adaptive_vote_k(" auto ", 300) == 3


def test_min_rows_per_fold_override():
    # a stricter per-fold floor reduces K sooner (needs more rows to sustain 5 folds).
    """Min rows per fold override."""
    assert resolve_adaptive_vote_k("auto", 1000, min_rows_per_fold=300) == 3
    assert resolve_adaptive_vote_k("auto", 2000, min_rows_per_fold=300) == 5


def test_min_rows_per_fold_zero_does_not_divide_by_zero():
    # degenerate public-param value must not raise (audit P2): the floor is clamped to >=1.
    """Min rows per fold zero does not divide by zero."""
    assert resolve_adaptive_vote_k("auto", 5000, min_rows_per_fold=0) == 5
    assert resolve_adaptive_vote_k("auto", 3, min_rows_per_fold=0) == 3
    assert resolve_adaptive_vote_k(5, 5000, min_rows_per_fold=0) == 5  # explicit int unaffected
