"""Matrix-based Rényi-α MI must not depend on an additive offset of its inputs (mrmr_audit_2026-09-14 NUM-3).

The RBF Gram matrix was built with ``d2 = |a|^2 + |b|^2 - 2 a.b`` on raw columns. That identity's rounding error scales with the offset
squared while the true squared distance scales with the spread squared, and the bandwidth is spread-sized, so on a price- or epoch-like
column the kernel matrix, its eigenvalues, and the MI were corrupted. An RBF kernel depends only on differences, so an offset must change
nothing; ``np.maximum(d2, 0)`` existed only to hide the negative distances this form produced.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._renyi_alpha import renyi_alpha_cmi, renyi_alpha_mi


def _xy(n=600, seed=0):
    """A column that shares signal with y, plus an unrelated conditioning column."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = x + 0.5 * rng.normal(size=n)
    z = rng.normal(size=n)
    return x, y, z


@pytest.mark.parametrize("offset", [1e4, 1e8])
def test_mi_is_invariant_to_an_offset_on_x(offset):
    """Shifting x must leave the MI unchanged."""
    x, y, _ = _xy()
    base = renyi_alpha_mi(x, y)
    assert base > 0.05, "fixture precondition: the pair must carry measurable MI"
    assert renyi_alpha_mi(x + offset, y) == pytest.approx(base, abs=1e-9)


def test_cmi_is_invariant_to_offsets_on_every_argument():
    """Shifting x, y and z together must leave the conditional MI unchanged."""
    x, y, z = _xy()
    base = renyi_alpha_cmi(x, y, z)
    assert renyi_alpha_cmi(x + 1e8, y - 3e7, z + 5e6) == pytest.approx(base, abs=1e-9)
