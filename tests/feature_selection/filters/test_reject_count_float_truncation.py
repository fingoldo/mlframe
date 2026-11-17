"""The permutation noise gate's reject count must not lose one to float truncation of ``1 - confidence``."""

from __future__ import annotations

import pytest

from mlframe.feature_selection.filters.info_theory._reject_count import reject_count


@pytest.mark.parametrize(
    ("n_perm", "confidence", "expected"),
    [(10, 0.8, 2), (100, 0.9, 10), (20, 0.95, 1), (200, 0.95, 10), (100, 0.99, 1), (1000, 0.99, 10), (30, 0.9, 3)],
)
def test_reject_count_is_exact_on_representable_ratios(n_perm: int, confidence: float, expected: int) -> None:
    """Values whose exact answer is an integer must not round down."""
    assert reject_count(n_perm, confidence) == expected


def test_reject_count_still_floors_non_integer_products() -> None:
    """A genuinely fractional product keeps flooring, the slack only absorbs float error."""
    assert reject_count(7, 0.9) == 0
    assert reject_count(15, 0.9) == 1
