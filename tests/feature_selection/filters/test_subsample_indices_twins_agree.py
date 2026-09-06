"""The two `_subsample_indices` copies must produce identical index sets.

`_orthogonal_hsic_fe._subsample_indices` is a deliberate inline copy of the `_orthogonal_dcor_fe` one --
its docstring says so: "inlined here to keep the sibling module dependency surface tight (no cross-layer
import)". The bodies are byte-identical after AST normalisation today, but nothing imports across the two
and no test asserted they agree, so a seeding change or an edit to the `n <= n_sample` short-circuit applied
to one would be invisible in the other and the two modules would silently subsample different rows for the
same `random_state`.

If the no-cross-layer-import constraint is ever dropped, replace both with one shared helper and delete this
file; until then this is what makes a one-sided edit fail loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_dcor_fe import _subsample_indices as _dcor_subsample
from mlframe.feature_selection.filters._orthogonal_hsic_fe import _subsample_indices as _hsic_subsample


@pytest.mark.parametrize("random_state", [0, 1, 7, 12345])
@pytest.mark.parametrize(
    "n,n_sample",
    [
        (1000, 200),  # ordinary subsample
        (200, 1000),  # n <= n_sample -> the arange short-circuit
        (500, 500),  # exactly equal, the boundary of that short-circuit
        (500, 499),  # one below it
        (1, 1),  # degenerate
        (1000, 0),  # n_sample <= 0 -> also the short-circuit
        (1000, -5),  # negative, same branch
    ],
)
def test_the_two_copies_return_the_same_indices(n, n_sample, random_state):
    """Same inputs, same output -- including which branch is taken and the sort order."""
    a = _dcor_subsample(n, n_sample, random_state)
    b = _hsic_subsample(n, n_sample, random_state)
    np.testing.assert_array_equal(
        a,
        b,
        err_msg=f"the dcor and hsic _subsample_indices copies diverged at n={n}, n_sample={n_sample}, random_state={random_state}",
    )


@pytest.mark.parametrize("random_state", [0, 3])
def test_the_subsample_is_sorted_and_unique(random_state):
    """The property both docstrings claim: a SORTED sample without replacement.

    Without this, the equality above would still hold if both copies were changed the same wrong way.
    """
    for fn, name in ((_dcor_subsample, "dcor"), (_hsic_subsample, "hsic")):
        idx = fn(1000, 200, random_state)
        assert idx.size == 200, f"{name}: expected 200 indices, got {idx.size}"
        assert np.array_equal(idx, np.sort(idx)), f"{name}: indices are not sorted, so paired calls lose their relative ordering"
        assert np.unique(idx).size == idx.size, f"{name}: sampled with replacement"
        assert idx.min() >= 0 and idx.max() < 1000, f"{name}: index out of range"


def test_the_short_circuit_returns_every_row():
    """`n <= n_sample` must return arange(n), not a permutation of it."""
    for fn, name in ((_dcor_subsample, "dcor"), (_hsic_subsample, "hsic")):
        np.testing.assert_array_equal(fn(50, 500, 0), np.arange(50), err_msg=f"{name}: the no-work branch no longer returns arange(n)")


def test_a_different_seed_gives_a_different_sample():
    """Guards the equality tests above against a copy that ignores random_state entirely."""
    for fn, name in ((_dcor_subsample, "dcor"), (_hsic_subsample, "hsic")):
        assert not np.array_equal(fn(1000, 200, 0), fn(1000, 200, 1)), f"{name}: random_state has no effect on the sample"
