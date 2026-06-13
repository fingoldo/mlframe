"""Bench: integer-lattice operators (gcd / bitwise XOR/AND) vs the best EXISTING mlframe basis on their natural targets.

Question: are gcd / bitwise genuine gaps (like the modular operator), or already covered by the rich existing catalog
(arithmetic ops mul/add/sub/div, modular, poly/Fourier/RBF/sigmoid/spline/wavelet)? A candidate counts as a gap only if it
MEASURABLY beats the best existing basis on its natural target AND is SPECIFIC (does not fire on smooth / noise controls).

For each TARGET we report:
  proto_mi  -- best MI the integer-lattice prototype recovers (the winning op's column),
  best_existing_mi -- best MI over a panel of existing-basis surrogates: raw cols, their products/sums/diffs/ratios
                      (arithmetic registry), modular residues (the modular operator), and a Chebyshev degree-6 fit
                      (the smooth-poly catalog). This is what the current pipeline could at best recover.
  lift = proto_mi / max(best_existing_mi, eps).

GENUINE EDGE  := lift >= 1.20 on the natural target AND proto fires (a hit) there.
SPECIFIC      := proto emits NO hit (or far lower MI) on the smooth-monotone + noise controls.

Run: CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 MLFRAME_DISABLE_GPU=1 PYTHONPATH=src python -m \
     mlframe.feature_selection._benchmarks.bench_integer_lattice_fe
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from mlframe.feature_selection.filters._integer_lattice_fe_proto import (
    INTEGER_LATTICE_OPS,
    apply_integer_lattice,
    scan_integer_lattice_pairs,
)
from mlframe.feature_selection.filters._pairwise_modular_fe import _mi

_RESULTS = Path(__file__).parent / "_results"
_NBINS = 12


def _cheb_fit_mi(a, b, y, degree=6):
    """Best MI a degree-`degree` Chebyshev surrogate of each raw column (the smooth-poly catalog) can recover vs y."""
    from numpy.polynomial.chebyshev import chebvander

    best = 0.0
    for col in (a, b):
        c = col.astype(np.float64)
        z = 2 * (c - c.min()) / (np.ptp(c) + 1e-9) - 1
        V = chebvander(z, degree)
        coef, *_ = np.linalg.lstsq(V, y.astype(np.float64), rcond=None)
        best = max(best, _mi(V @ coef, y, nbins=_NBINS))
    return best


def _best_existing_mi(a, b, y):
    """What the current pipeline could at best recover: raw cols, arithmetic combos, modular residues, Chebyshev fit."""
    af, bf = a.astype(np.float64), b.astype(np.float64)
    cands = [af, bf, af * bf, af + bf, af - bf, af / (np.abs(bf) + 1.0)]
    for m in (2, 3, 4, 5, 6, 7, 8, 10, 12, 16):  # modular operator's coarse ladder, on raw cols + their sum/diff
        for base in (af, bf, af + bf, af - bf, af * bf):
            cands.append(np.mod(np.rint(base).astype(np.int64), m).astype(np.float64))
    best = max(_mi(c, y, nbins=_NBINS) for c in cands)
    return max(best, _cheb_fit_mi(a, b, y))


def _make_targets(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}

    # gcd target: y keyed on the shared factor of two integer columns (grid alignment / common-divisor structure).
    a = rng.integers(1, 60, n)
    b = rng.integers(1, 60, n)
    out["gcd_shared_factor"] = (a, b, (np.gcd(a, b) >= 3).astype(int))

    # XOR-of-low-bits target: zero-marginal bit interaction -- step keyed on parity of shared low bits.
    a2 = rng.integers(0, 256, n)
    b2 = rng.integers(0, 256, n)
    out["bitwise_xor_lowbits"] = (a2, b2, (np.bitwise_xor(a2, b2) & 0x0F).astype(int) % 5)

    # AND-flag co-occurrence: target fires when both share a high bit set (flag-AND interaction).
    a3 = rng.integers(0, 256, n)
    b3 = rng.integers(0, 256, n)
    out["bitwise_and_flag"] = (a3, b3, ((np.bitwise_and(a3, b3) & 0x80) > 0).astype(int))

    # CONTROL 1 -- smooth monotone: y = round(a) + noise; no lattice structure. Proto must NOT fire / not beat existing.
    a4 = rng.integers(0, 100, n)
    b4 = rng.integers(0, 100, n)
    out["control_smooth_monotone"] = (a4, b4, (a4 + 0.3 * b4 > 60).astype(int))

    # CONTROL 2 -- pure noise: y independent of integer cols. Specificity floor.
    a5 = rng.integers(0, 100, n)
    b5 = rng.integers(0, 100, n)
    out["control_noise"] = (a5, b5, rng.integers(0, 4, n))

    return out


def main():
    targets = _make_targets()
    rows = []
    t0 = time.perf_counter()
    for name, (a, b, y) in targets.items():
        y = np.asarray(y).astype(np.int64)
        X = np.column_stack([a, b]).astype(np.float64)
        ts = time.perf_counter()
        hits = scan_integer_lattice_pairs(X, y, ["a", "b"], nbins=_NBINS)
        scan_s = time.perf_counter() - ts
        proto_mi = max((_mi(apply_integer_lattice(a, b, op), y, nbins=_NBINS) for op in INTEGER_LATTICE_OPS))
        best_existing = _best_existing_mi(a, b, y)
        lift = proto_mi / max(best_existing, 1e-6)
        is_control = name.startswith("control")
        verdict = ("SPECIFIC-OK" if not hits else "FALSE-POSITIVE") if is_control else (
            "GENUINE-EDGE" if (lift >= 1.20 and hits) else "REDUNDANT")
        rows.append(dict(
            target=name, proto_mi=round(proto_mi, 4), best_existing_mi=round(best_existing, 4),
            lift=round(lift, 3), n_hits=len(hits), top_hit=(hits[0]["op"] if hits else None),
            scan_s=round(scan_s, 3), verdict=verdict))
    total_s = time.perf_counter() - t0

    print(f"{'target':28s} {'proto':>7s} {'exist':>7s} {'lift':>6s} {'hits':>5s} {'op':>5s}  verdict")
    for r in rows:
        print(f"{r['target']:28s} {r['proto_mi']:7.3f} {r['best_existing_mi']:7.3f} {r['lift']:6.2f} "
              f"{r['n_hits']:5d} {str(r['top_hit']):>5s}  {r['verdict']}")
    print(f"\ntotal {total_s:.2f}s")

    _RESULTS.mkdir(exist_ok=True)
    out = _RESULTS / f"integer_lattice_fe_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(dict(generated=datetime.now().isoformat(), nbins=_NBINS, total_s=total_s, rows=rows), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
