"""Compare 4 orthogonal polynomial bases on synthetic + UCI regimes.

For each (regime, basis) pair: run ``optimise_hermite_pair`` with the
chosen basis and report MI / uplift / wall time.

Hypothesis (verified by the table):
* Hermite wins on z-score-normalisable Gaussian-ish inputs (XOR /
  saddle / polynomial -- inputs are np.random.normal).
* Legendre / Chebyshev win on bounded Uniform inputs
  (uniform_xor regime).
* Laguerre wins on heavy-tailed positive inputs (some California
  Housing / Diabetes columns).

Run::

    python -m mlframe.feature_selection._benchmarks.bench_polynomial_bases
    python -m mlframe.feature_selection._benchmarks.bench_polynomial_bases --regimes xor,circle --bases hermite,legendre
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--regimes", default="all")
    parser.add_argument("--bases", default="hermite,legendre,chebyshev,laguerre")
    parser.add_argument("--max-degree", type=int, default=4)
    args = parser.parse_args()

    from mlframe.feature_selection._benchmarks.bench_hermite_fe import _REGIMES, _ksg_baseline_pair
    from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair

    regimes = list(_REGIMES) if args.regimes == "all" else args.regimes.split(",")
    bases = args.bases.split(",")

    print(f"\n=== Polynomial-bases comparison, n={args.n}, trials={args.n_trials}, max_degree={args.max_degree} ===\n")
    header = f"  {'regime':22s} {'baseline':>10s} " + " ".join(f"{b:>10s}" for b in bases)
    print(header)
    print("  " + "-" * (24 + 11 + 11 * len(bases)))

    for regime in regimes:
        if regime not in _REGIMES:
            print(f"  {regime}: unknown regime, skipping")
            continue
        x1, x2, y = _REGIMES[regime](n=args.n)
        baseline = _ksg_baseline_pair(x1, x2, y)

        cells = []
        times = []
        for basis in bases:
            t0 = time.perf_counter()
            try:
                res = optimise_hermite_pair(
                    x1, x2, y,
                    discrete_target=True,
                    max_degree=args.max_degree,
                    n_trials=args.n_trials,
                    seed=42,
                    basis=basis,
                )
                dt = time.perf_counter() - t0
                if res is None:
                    cells.append("none")
                else:
                    cells.append(f"{res.mi:.3f}")
                times.append(dt)
            except Exception as e:
                cells.append(f"ERR")
                times.append(0.0)
                print(f"  {regime}+{basis}: ERROR {type(e).__name__}: {e}")

        # Highlight winning basis (highest mi, ignoring "none"/"ERR").
        numeric = [(i, float(c)) for i, c in enumerate(cells) if c not in ("none", "ERR")]
        winner_idx = max(numeric, key=lambda kv: kv[1])[0] if numeric else None
        rendered = []
        for i, c in enumerate(cells):
            if i == winner_idx:
                rendered.append(f"{'*'+c:>10s}")
            else:
                rendered.append(f"{c:>10s}")
        print(f"  {regime:22s} {baseline:10.3f} " + " ".join(rendered)
              + f"   ({np.mean(times):.1f}s avg)")

    print("\n  * = winning basis (highest engineered MI)")


if __name__ == "__main__":
    main()
