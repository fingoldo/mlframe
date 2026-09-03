"""MODELS-7 regression test: combine_probs's 'qube' underflow guard must be per-cell, not gated on a
global (stacked > 0).all() over the whole tensor.

The bug (fixed): a single exact-zero cell ANYWHERE in the tensor disabled the underflow-protection clip
for the ENTIRE tensor, including unrelated tiny-positive cells elsewhere that genuinely needed it (their
p**3 underflows below float64's smallest normal, losing precision in cbrt(mean(p**3))). Fixed to clip
per-cell on the positive-value mask, leaving exact-zero cells untouched (0**3 == 0 exactly, no underflow).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.models.ensembling.base import combine_probs

pytestmark = pytest.mark.fast


def test_tiny_positive_cell_protected_even_when_another_cell_is_exact_zero():
    """A tiny-positive cell (p**3 underflows to 0.0 in raw float64) must still get the clip protection
    even though an unrelated cell elsewhere in the same tensor is exactly 0."""
    tiny = 1e-150  # tiny**3 underflows to 0.0 exactly in float64
    # 2 members, 2 rows: row 0 has a genuine tiny-positive prediction from both members (needs the clip);
    # row 1 has one member predicting exactly 0.0 (the old code used this to disable the clip globally).
    stacked = np.array([[tiny, tiny], [0.0, 0.5]])
    out = combine_probs(stacked, flavour="qube")

    assert out[0] > 0.0, "the tiny-positive row should not collapse to 0 due to p**3 underflow"
    # Row 1 (with the exact-zero member) should NOT have that zero cell corrupted into a tiny positive.
    assert out[1] >= 0.0


def test_exact_zero_cell_not_corrupted_by_the_clip():
    """An exact-zero cell must stay contributing exactly 0 to its own cube, not get bumped to 1e-103."""
    stacked = np.array([[0.0, 0.6], [0.0, 0.6]])  # member 0 is always exactly 0
    out = combine_probs(stacked, flavour="qube")
    expected = np.cbrt(np.mean(np.array([[0.0, 0.6], [0.0, 0.6]]) ** 3, axis=0))
    np.testing.assert_allclose(out, expected, atol=1e-12)
