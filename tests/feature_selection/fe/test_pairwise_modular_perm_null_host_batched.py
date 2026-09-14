"""The pairwise-modular host permutation null must stay equal to the per-permutation loop (mrmr_audit_2026-09-14 PERF-1, rejected).

PERF-1 proposed batching ``_perm_null_hi``'s 12 host MI calls into one, stacking ``r[inv_perm_i]`` as columns, via
``MI(r; y[perm]) == MI(r[inv_perm]; y)``. That identity holds for the true MI but not for this estimator: rank binning splits TIED values by
row order. The loop keeps the residue in its original order and permutes y, so the residue gets the same bin codes every time; each
re-ordered column in a batch splits its ties differently. Residues ``c mod k`` are tied by construction, and the batched null band moved by
up to 31% on these fixtures, so the batching was not shipped. This pins the loop's result so a future re-batching cannot slip in unnoticed.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlframe.feature_selection.filters._pairwise_modular_fe as pm
import mlframe.feature_selection.filters._pairwise_modular_resident as pmr
from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi


@pytest.fixture(autouse=True)
def _no_resident_path(monkeypatch):
    """Force the default-configuration host path: the resident batcher declines, as it does without cupy + STRICT."""
    monkeypatch.setattr(pmr, "perm_null_residue_mis_resident", lambda *a, **k: None)


def _inputs(seed, n=5000):
    """An integer-valued combination column and a 3-class target."""
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 50, size=n).astype(np.float64)
    y = rng.integers(0, 3, size=n)
    return c, y


def _reference(c, y, k, nbins, n_perm, seed, z):
    """The per-permutation loop, written out with the module's own single-column MI."""
    r = np.mod(c, k).astype(np.float64)
    rng = np.random.default_rng(seed)
    yi = encode_y_for_classif_mi(y)
    perms = [rng.permutation(yi.size) for _ in range(n_perm)]
    vals = np.array([pm._mi(r, yi[p], nbins=max(nbins, k)) for p in perms], dtype=np.float64)
    return float(vals.mean() + z * vals.std())


@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("k", [3, 7])
def test_host_null_band_is_bit_identical_to_the_per_permutation_loop(seed, k):
    """Same draws, same estimator, same binning: the host null band must equal the loop exactly, on tied residues."""
    c, y = _inputs(seed)
    assert pm._perm_null_hi(c, y, k, nbins=12, seed=seed) == _reference(c, y, k, 12, 12, seed, 3.0)
