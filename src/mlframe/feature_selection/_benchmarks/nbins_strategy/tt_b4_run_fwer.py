"""Family-wise error of the feature-transform recommender when most columns are noise (research B4, direction (1)).

Scenarios on a mixed-marginal panel (Gaussian, lognormal, Student-t3, Poisson, 5-level ordinal, exponential, uniform; blocks of 5 correlated at 0.5):
  N0  y is pure noise (global null)
  N1  y = 4 LINEAR informative columns + noise (every true link is linear: any accepted transform is a false positive)
  N1s same, with the 4 informative columns passed as `selected` (what MRMR's support would supply)
Rules compared on the same permutations: column (per-column null), maxT (max over columns), pooled (null pooled over columns, Bonferroni), gated (relevance maxT gate,
then a longer per-column null on the relevant columns only). Each is shown without and with replication on the held-out half.
Reports the share of datasets with at least one accepted transform (family-wise false-positive rate) and the mean number of accepted columns per dataset.
Usage: python tt_b4_run_fwer.py [--reps 100] [--n 5000] [--p 100] [--scenario N0,N1]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
sys.path.insert(0, __file__.replace("\\", "/").rsplit("/", 1)[0])
from tt_b4_panel import make_null_panel
from tt_b4_reference import recommend_feature_transforms

RULES = ("column", "maxT", "pooled", "gated")


def run(n, p, reps, n_lin, studentize=False, n_null=50, seed0=0, noise_rho=0.5, use_selected=False):
    acc_any = {(r, rep): 0 for r in RULES for rep in (0, 1)}
    acc_cnt = {(r, rep): 0 for r in RULES for rep in (0, 1)}
    by_role = {(r, role): 0 for r in RULES for role in ("informative_linear", "block_mate_of_informative", "other_noise")}
    t0 = time.perf_counter()
    for s in range(reps):
        X, y, info = make_null_panel(n, p, seed0 + s, n_lin=n_lin, noise_rho=noise_rho)
        sel = list(info["lin_cols"]) if use_selected else None
        res = recommend_feature_transforms(X, y, selected=sel, n_null=n_null, seed=s, rule="gated", replicate_rules=RULES, studentize=studentize)
        tb = res.table
        lin = set(info["lin_cols"]); mates = set(info["mates"])
        for r in RULES:
            d = tb["disc_by_rule"][r]
            a = tb["accepted_by_rule"][r]
            acc_any[(r, 0)] += bool(d.any()); acc_cnt[(r, 0)] += int(d.sum())
            acc_any[(r, 1)] += bool(a.any()); acc_cnt[(r, 1)] += int(a.sum())
            for j in np.flatnonzero(a):
                role = "informative_linear" if int(j) in lin else ("block_mate_of_informative" if int(j) in mates else "other_noise")
                by_role[(r, role)] += 1
    secs = time.perf_counter() - t0
    return acc_any, acc_cnt, secs, by_role


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--scenario", default="N0,N1")
    ap.add_argument("--studentize", action="store_true")
    ap.add_argument("--noise_rho", type=float, default=0.5)
    a = ap.parse_args()
    for sc in a.scenario.split(","):
        n_lin = 0 if sc == "N0" else 4
        any_, cnt, secs, by_role = run(a.n, a.p, a.reps, n_lin, a.studentize, noise_rho=a.noise_rho, use_selected=(sc == "N1s"))
        print(f"\n== {sc} n={a.n} p={a.p} reps={a.reps} studentize={a.studentize} noise_rho={a.noise_rho} ({secs:.0f} s)", flush=True)
        print("rule     | FWER no-repl | FWER +repl | mean #acc no-repl | mean #acc +repl", flush=True)
        for r in RULES:
            print(f"{r:8s} | {any_[(r, 0)] / a.reps:12.3f} | {any_[(r, 1)] / a.reps:10.3f} | {cnt[(r, 0)] / a.reps:17.3f} | {cnt[(r, 1)] / a.reps:15.3f}", flush=True)
        print("accepted (with replication) by column role, total over reps (informative_linear / block_mate_of_informative / other_noise):", flush=True)
        for r in RULES:
            print(f"  {r:8s} {by_role[(r, 'informative_linear')]:5d} {by_role[(r, 'block_mate_of_informative')]:5d} {by_role[(r, 'other_noise')]:5d}", flush=True)


if __name__ == "__main__":
    main()
