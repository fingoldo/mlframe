"""Region-adaptive fitting must fit only each region's winner on the full region, and produce the same spec.

Every candidate fitted the full region after its OOF folds and every non-winning fit was discarded; the default
``linear_residual`` was also fitted up front. The candidates now return only their OOF score and the winner is fitted
once per region.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from mlframe.training.composite.discovery import _region_adaptive as ra
from mlframe.training.composite.discovery._region_adaptive import fit_region_adaptive


def _data(n: int = 600, seed: int = 0):
    """A base with a regime change: linear in one half, curved in the other."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 10.0, n)
    y = np.where(base < 5.0, 2.0 * base, 0.3 * base**2) + rng.normal(scale=0.2, size=n)
    return y, base


def test_each_region_fits_its_winner_once(monkeypatch):
    """Full-region fits: one per region, not one per candidate plus a seeding fit."""
    y, base = _data()
    full_fits = {"n": 0}
    sizes = set()
    patched = {}
    for name, tr in ra._TRANSFORMS_REGISTRY.items():

        def counting(yy, bb, *a, _real=tr.fit, **k):
            """Count fits over a whole region (the OOF fold fits are on fewer rows)."""
            if len(yy) in sizes:
                full_fits["n"] += 1
            return _real(yy, bb, *a, **k)

        patched[name] = dataclasses.replace(tr, fit=counting)
    monkeypatch.setattr(ra, "_TRANSFORMS_REGISTRY", {**ra._TRANSFORMS_REGISTRY, **patched})
    edges = ra._quantile_edges(base, 4)
    regions = ra.assign_regions(base, edges)
    sizes.update(int((regions == k).sum()) for k in range(len(edges) + 1))
    spec = fit_region_adaptive(y, base, k=4, n_folds=3, random_state=0)
    assert full_fits["n"] == len(spec.region_transforms), f"{full_fits['n']} full-region fits for {len(spec.region_transforms)} regions"


def test_each_region_stores_a_full_region_fit_of_its_winner():
    """The stored params are exactly the winner's fit on the region's rows, and the spec round-trips y."""
    y, base = _data()
    spec = fit_region_adaptive(y, base, k=4, n_folds=3, random_state=0)
    regions = ra.assign_regions(base, np.asarray(spec.edges))
    for k, (name, params) in enumerate(zip(spec.region_transforms, spec.region_params)):
        m = regions == k
        expected = ra._TRANSFORMS_REGISTRY[name].fit(y[m], base[m])
        assert params.keys() == expected.keys()
        for key in params:
            np.testing.assert_array_equal(np.asarray(params[key]), np.asarray(expected[key]))
    t = spec.forward(y, base)
    np.testing.assert_allclose(spec.inverse(t, base), y, rtol=1e-9, atol=1e-9)
