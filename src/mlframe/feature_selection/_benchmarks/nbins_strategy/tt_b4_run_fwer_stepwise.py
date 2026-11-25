"""False positives of the STEPWISE driver on mostly-noise panels (research B4, companion of tt_b4_run_fwer.py).

N0: y pure noise (any committed column is a false positive). N1: y = 4 linear informative columns + noise, blocks of 5 correlated at 0.5 (a committed non-identity transform,
or a committed column that is neither informative nor ... is a false positive; block mates of informative columns are counted separately).
Usage: python tt_b4_run_fwer_stepwise.py --reps 100 --n 5000 --p 100
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings


warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
sys.path.insert(0, __file__.replace("\\", "/").rsplit("/", 1)[0])
from tt_b4_panel import make_null_panel
from tt_b4_reference import recommend_stepwise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--p", type=int, default=100)
    a = ap.parse_args()
    for sc, n_lin in (("N0", 0), ("N1", 4)):
        t0 = time.perf_counter()
        any_commit = any_transform = any_noise_commit = any_mate = 0
        n_commit = n_transform = n_noise = n_mate = n_inf_missed = 0
        for s in range(a.reps):
            X, y, info = make_null_panel(a.n, a.p, s, n_lin=n_lin)
            r = recommend_stepwise(X, y, n_null=50, seed=s)
            ch = r.table["chosen"]
            lin = set(info["lin_cols"]); mates = set(info["mates"])
            tr = {j for j, f in ch.items() if f != "id"}
            noise = {j for j in ch if j not in lin and j not in mates}
            mt = {j for j in ch if j in mates}
            any_commit += bool(ch); any_transform += bool(tr); any_noise_commit += bool(noise); any_mate += bool(mt)
            n_commit += len(ch); n_transform += len(tr); n_noise += len(noise); n_mate += len(mt)
            n_inf_missed += len(lin - set(ch))
        R = a.reps
        print(f"{sc} n={a.n} p={a.p} reps={R} ({time.perf_counter() - t0:.0f} s): datasets with any committed column {any_commit / R:.3f}; any non-identity transform {any_transform / R:.3f}; "
              f"any committed pure-noise column (not informative, not a block mate) {any_noise_commit / R:.3f}; any committed block mate of an informative column {any_mate / R:.3f}; "
              f"mean committed {n_commit / R:.2f}, mean transforms {n_transform / R:.3f}, informative linear columns not committed {n_inf_missed}/{R * n_lin}", flush=True)


if __name__ == "__main__":
    main()
