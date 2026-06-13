"""Bench: Bernstein/Bezier + Jacobi/Gegenbauer vs the best EXISTING mlframe basis on their natural targets.

The existing catalog already ships Hermite/Legendre/Chebyshev/Laguerre (hermite_fe) + Fourier/RBF/Sigmoid/Pade (bases.py)
+ B-spline/Haar-wavelet (extra-basis path). The question is whether Bernstein or Jacobi/Gegenbauer add anything those cannot.

For each target: fit each candidate + a panel of existing bases by least-squares and compare fitted-column MI vs y.
GENUINE EDGE := candidate_mi / best_existing_mi >= 1.20. REDUNDANT otherwise (the existing catalog already spans it).

Run: CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 MLFRAME_DISABLE_GPU=1 PYTHONPATH=src python -m \
     mlframe.feature_selection._benchmarks.bench_extra_basis_fe
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from numpy.polynomial.legendre import legvander

from mlframe.feature_selection.filters._extra_basis_fe_proto import (
    bernstein_design,
    fit_basis_mi,
    gegenbauer_design,
    jacobi_design,
)

_RESULTS = Path(__file__).parent / "_results"
_NBINS = 12
_DEG = 8


def _to_pm1(x):
    c = x.astype(np.float64)
    return 2 * (c - c.min()) / (np.ptp(c) + 1e-12) - 1


def _rbf_mi(x, y, n_centres=9):
    """Surrogate for the shipped RBF basis: Gaussian bumps at quantile centres, Silverman bandwidth, lstsq fit."""
    from mlframe.feature_selection.filters._extra_basis_fe_proto import fit_basis_mi as _f
    c = x.astype(np.float64)
    centres = np.quantile(c, np.linspace(0.1, 0.9, n_centres))
    bw = 1.06 * c.std() * len(c) ** (-1 / 5) + 1e-9
    design = np.column_stack([np.exp(-(((c - ce) / bw) ** 2)) for ce in centres] + [np.ones_like(c)])
    return _f(design, y, nbins=_NBINS)


def _spline_mi(x, y, n_knots=6):
    """Surrogate for the shipped cubic-B-spline basis via a truncated-power piecewise basis at quantile knots."""
    from mlframe.feature_selection.filters._extra_basis_fe_proto import fit_basis_mi as _f
    c = x.astype(np.float64)
    knots = np.quantile(c, np.linspace(0.1, 0.9, n_knots))
    cols = [np.ones_like(c), c, c**2, c**3] + [np.clip(c - k, 0, None) ** 3 for k in knots]
    return _f(np.column_stack(cols), y, nbins=_NBINS)


def _best_existing_mi(x, y):
    cheb = fit_basis_mi(chebvander(_to_pm1(x), _DEG), y, nbins=_NBINS)
    leg = fit_basis_mi(legvander(_to_pm1(x), _DEG), y, nbins=_NBINS)
    return max(cheb, leg, _rbf_mi(x, y), _spline_mi(x, y))


def _make_targets(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3, 3, n))
    out = {}
    # Bounded saturating sigmoid (CDF-like plateau) -- Bernstein's variation-diminishing home turf.
    out["bounded_saturation"] = (x, (1.0 / (1.0 + np.exp(-3.0 * x)) > 0.5).astype(int))
    # Smooth global polynomial -- Jacobi/Gegenbauer's claimed home, but Legendre/Chebyshev already span it (redundancy probe).
    out["smooth_poly"] = (x, ((x**3 - 2 * x) > 0).astype(int))
    # Endpoint-heavy monotone (steep tails) -- where global Chebyshev rings; Bernstein partition-of-unity stays bounded.
    out["endpoint_monotone"] = (x, (np.tanh(2 * x) > 0).astype(int))
    return out


def main():
    targets = _make_targets()
    rows = []
    t0 = time.perf_counter()
    for name, (x, y) in targets.items():
        y = np.asarray(y).astype(np.int64)
        best_existing = _best_existing_mi(x, y)
        bern = fit_basis_mi(bernstein_design(x, _DEG), y, nbins=_NBINS)
        # Jacobi/Gegenbauer: sweep a small alpha/beta (lambda) grid, take the best -- the most generous redundancy check.
        jac = max(fit_basis_mi(jacobi_design(x, _DEG, a, b), y, nbins=_NBINS)
                  for a in (-0.5, 0.0, 1.0, 2.0) for b in (-0.5, 0.0, 1.0, 2.0))
        geg = max(fit_basis_mi(gegenbauer_design(x, _DEG, lam), y, nbins=_NBINS) for lam in (0.25, 0.5, 1.0, 2.0))
        for fam, mi in (("bernstein", bern), ("jacobi", jac), ("gegenbauer", geg)):
            lift = mi / max(best_existing, 1e-6)
            rows.append(dict(target=name, family=fam, family_mi=round(mi, 4),
                             best_existing_mi=round(best_existing, 4), lift=round(lift, 3),
                             verdict=("GENUINE-EDGE" if lift >= 1.20 else "REDUNDANT")))
    total_s = time.perf_counter() - t0

    print(f"{'target':22s} {'family':12s} {'fam_mi':>7s} {'exist':>7s} {'lift':>6s}  verdict")
    for r in rows:
        print(f"{r['target']:22s} {r['family']:12s} {r['family_mi']:7.3f} {r['best_existing_mi']:7.3f} "
              f"{r['lift']:6.2f}  {r['verdict']}")
    print(f"\ntotal {total_s:.2f}s")

    _RESULTS.mkdir(exist_ok=True)
    out = _RESULTS / f"extra_basis_fe_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(dict(generated=datetime.now().isoformat(), degree=_DEG, rows=rows, total_s=total_s), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
