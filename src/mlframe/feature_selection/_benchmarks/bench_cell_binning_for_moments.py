"""Which binning forms the CELLS for per-cell higher-moment features (scenario 2: aggregate a feature's
std/skew/kurt over quantile cells of a numeric)? The bin count drives reliability: a k-way cross has
~n/nbins^k rows per cell, and a higher moment needs more rows per cell to be non-garbage (mean ~5, std
~10, skew ~30, kurt ~100). So fixed-10 over-bins at small n / high moment order, and plain Freedman-Diaconis
(tuned for 1-D density, not per-cell sample count) can over-bin too. The moment-aware cap
``nbins = floor((n / n_min(order))^(1/k))`` ties resolution to the highest requested moment.

Honest train/test: bin on TRAIN edges, replay on held-out TEST, predict a spread-driven target with a GBM
on the per-cell std feature. Compares fixed-10 / Freedman-Diaconis / moment-aware-cap by OOS R2.

Run:  python -m mlframe.feature_selection._benchmarks.bench_cell_binning_for_moments
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

SEEDS = (0, 1, 2, 3, 4)
# Minimum rows-per-cell for a stable estimate of each moment order (rule-of-thumb).
_N_MIN = {"mean": 5, "std": 12, "skew": 30, "kurt": 100}


def _fd_nbins(x: np.ndarray) -> int:
    """Freedman-Diaconis bin count for a 1-D array (density-optimal, NOT moment-aware)."""
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 1
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return 10
    width = 2.0 * iqr * n ** (-1.0 / 3.0)
    if width <= 0:
        return 10
    return int(np.clip(np.ceil((x.max() - x.min()) / width), 2, 256))


def _moment_aware_nbins(n: int, k: int, highest_moment: str) -> int:
    """Cap per-axis bins so each of nbins^k cells holds >= n_min(highest_moment) rows."""
    n_min = _N_MIN[highest_moment]
    return int(max(2, np.floor((n / n_min) ** (1.0 / k))))


def _edges(x, nbins):
    qs = np.linspace(0, 1, nbins + 1)[1:-1]
    return np.unique(np.quantile(x, qs))


def _codes(x, edges):
    return np.searchsorted(edges, x, side="right")


def _cell_std(codes_tr, aux_tr, n_cells, codes_te):
    df = pd.DataFrame({"c": codes_tr, "v": aux_tr})
    g = df.groupby("c")["v"].std().to_dict()
    gl = float(np.nanstd(aux_tr))
    f_tr = np.array([g.get(int(c), gl) for c in codes_tr])
    f_te = np.array([g.get(int(c), gl) for c in codes_te])
    return np.nan_to_num(f_tr, nan=gl)[:, None], np.nan_to_num(f_te, nan=gl)[:, None]


def run(n, seed, k=2):
    rng = np.random.default_rng(seed)
    xs = [rng.uniform(0, 1, n) for _ in range(k)]
    sigma = 0.5 + 2.0 * sum(np.abs(x - 0.5) for x in xs)
    aux = rng.normal(0, sigma, n)
    T = sigma
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2 :]
    out = {}
    for name, nbins_fn in (
        ("fixed10", lambda: 10),
        ("FD", lambda: int(np.median([_fd_nbins(x[tr]) for x in xs]))),
        # The feature here is per-cell STD, so the cap uses std's n_min (the moment actually computed),
        # not the highest-possible moment. Capping by kurt when only std is needed needlessly over-coarsens.
        ("moment_aware(std)", lambda: _moment_aware_nbins(len(tr), k, "std")),
        ("moment_aware(kurt)", lambda: _moment_aware_nbins(len(tr), k, "kurt")),
    ):
        nb = nbins_fn()
        edges = [_edges(x[tr], nb) for x in xs]
        nbset = [int(e.size + 1) for e in edges]
        def cross(ix):
            c = _codes(xs[0][ix], edges[0])
            mult = 1
            for d in range(1, k):
                mult *= nbset[d - 1]
                c = c + _codes(xs[d][ix], edges[d]) * mult
            return c
        ctr, cte = cross(tr), cross(te)
        f_tr, f_te = _cell_std(ctr, aux[tr], int(np.prod(nbset)), cte)
        m = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0).fit(f_tr, T[tr])
        out[name] = (r2_score(T[te], m.predict(f_te)), nb)
    return out


def main():
    print("\nCell binning for per-cell std feature (scenario 2), 2-way cross, 5 seeds -- OOS R2\n")
    for n in (2000, 8000, 30000):
        rows = [run(n, s) for s in SEEDS]
        line = f"n={n:>6}  "
        for name in ("fixed10", "FD", "moment_aware(std)", "moment_aware(kurt)"):
            r2 = np.mean([r[name][0] for r in rows])
            nb = int(np.median([r[name][1] for r in rows]))
            line += f"{name}: R2={r2:.3f} (nbins~{nb})   "
        print(line)
    print()


if __name__ == "__main__":
    main()
