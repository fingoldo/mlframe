"""Does ``fe_target_mix_candidates=True`` pay for itself? y = 3*x0 + weak share * (x1*x2) + noise, candidates OFF vs ON.

Usage: ``python mix_value.py --share 0.018 --noise 0.6 --seeds 3``. The held-out R2 comes from a ridge on ``transform`` of the selection; ``weak_term`` says whether a selected column is built from both x1 and x2.
Results of 2026-11-23 (n=6000, p=10, 3 seeds each): see ``audits/2026-11-23/PLAN_target_transform_in_mrmr.md``, section PLAN-05.
"""

from __future__ import annotations

import argparse
import re
import time
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")


def make(seed: int, n: int, p: int, share: float, noise: float):
    """Dominant linear term, a weak x1*x2 interaction (no marginal effect) with the given variance share, Gaussian noise."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)])
    dominant = 3.0 * X["x0"].to_numpy()
    weak_raw = X["x1"].to_numpy() * X["x2"].to_numpy()
    weak = weak_raw / weak_raw.std() * np.sqrt(share / (1 - share)) * dominant.std()
    return X, dominant + weak + noise * rng.standard_normal(n)


def sources(name: str) -> set:
    """Indices of the raw columns named inside a feature name."""
    return {int(i) for i in re.findall(r"x(\d+)", name)}


def r2(y, p) -> float:
    """Coefficient of determination."""
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def main() -> None:
    """Run the OFF / ON comparison and print one line per seed."""
    from sklearn.linear_model import Ridge

    from mlframe.feature_selection.filters.mrmr import MRMR

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--share", type=float, default=0.04)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    for seed in range(a.seeds):
        X, y = make(seed, a.n, a.p, a.share, a.noise)
        tr = np.arange(len(y)) < int(len(y) * 0.7)
        te = ~tr
        out = {}
        for cand in (False, True):
            t = time.perf_counter()
            m = MRMR(verbose=0, random_seed=seed, fe_max_steps=1, fe_target_mix_enable=True, fe_target_mix_candidates=cand, fe_target_mix_max_bases=1)
            m.fit(X.iloc[tr], y[tr])
            wall = time.perf_counter() - t
            Xt = np.asarray(m.transform(X.iloc[tr]), dtype=float)
            Xe = np.asarray(m.transform(X.iloc[te]), dtype=float)
            ok = np.isfinite(Xt).all(0) & np.isfinite(Xe).all(0)
            pred = Ridge(1.0).fit(Xt[:, ok], y[tr]).predict(Xe[:, ok])
            names = [str(c) for c in m.get_feature_names_out()]
            out[cand] = (wall, r2(y[te], pred), any({1, 2} <= sources(nm) for nm in names), len(names))
        print(f"seed {seed}: OFF wall {out[False][0]:.1f}s R2 {out[False][1]:.4f} weak_term {out[False][2]} n {out[False][3]} | ON wall {out[True][0]:.1f}s R2 {out[True][1]:.4f} weak_term {out[True][2]} n {out[True][3]}", flush=True)


if __name__ == "__main__":
    main()
