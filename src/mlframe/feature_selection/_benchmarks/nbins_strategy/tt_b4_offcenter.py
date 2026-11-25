"""Candidate-set limit: informative columns whose even link is centred away from 0 (research B4).

The panel is generated as usual (links square / abs / sigmoid act on a zero-centred Gaussian), then the SAME columns are shifted by +3 AFTER y is built, so the truth is
(x - 3)^2, |x - 3|, tanh(2 (x - 3)). Candidate set A = CAND_DEFAULT (median-centred square / abs / tanh). Candidate set B = MRMR's fixed-zero unary table
(filters/_mi_greedy_fe.py UNARY_TRANSFORMS: square = x^2, abs = |x|, tanh(x), log1p|x|, ...). Reports, per set, the stepwise picks on those columns and the judge gain
(test R2 of ridge with the column replaced) of the pick vs the best candidate of the same set.
Usage: python tt_b4_offcenter.py [--seeds 8] [--n 20000]
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
sys.path.insert(0, __file__.replace("\\", "/").rsplit("/", 1)[0])
from tt_b4_models import replacement_table
from tt_b4_panel import make_panel
from tt_b4_reference import CAND_DEFAULT, CAND_MRMR, recommend_stepwise
from tt_b4_run_detect import config_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--p", type=int, default=20)
    a = ap.parse_args()
    tot = {"default": [0, 0, 0.0, 0.0], "mrmr": [0, 0, 0.0, 0.0]}
    for s in range(a.seeds):
        cfg = config_for(a.p, s)
        Xall, yall, info = make_panel(2 * a.n, a.p, s, **cfg)
        shifted = [c for c in info["nl_cols"] if info["link"][c] in ("square", "abs", "sigmoid") and info["marginal"][c] in ("gauss", "gauss2")]
        Xall[:, shifted] += 3.0
        X, y, Xt, yt = Xall[: a.n], yall[: a.n], Xall[a.n:], yall[a.n:]
        for name, cands in (("default", CAND_DEFAULT), ("mrmr", CAND_MRMR)):
            r = recommend_stepwise(X, y, candidates=cands, n_null=50, seed=s)
            base, tab = replacement_table(X, y, Xt, yt, cols=shifted, cands=cands)
            for a_, c in enumerate(shifted):
                best = np.nanmax(tab[a_]) - base
                pick = r.table["chosen"].get(c, None)
                realized = 0.0 if pick in (None, "id") else tab[a_, cands.index(pick)] - base
                tot[name][0] += 1
                tot[name][1] += int(best > 0.01 and realized >= 0.8 * best)
                tot[name][2] += best
                tot[name][3] += max(realized, 0.0)
                print(f"seed {s} col {c} link {info['link'][c]} set {name}: pick {pick}, best candidate {cands[int(np.nanargmax(tab[a_]))]} gain {best:+.3f}, pick gain {realized:+.3f}", flush=True)
    for name, (n_, hit, best, real) in tot.items():
        print(f"SUMMARY {name}: {n_} shifted even/saturating columns, hit {hit}, mean best-candidate gain {best / n_:.3f}, mean realized gain {real / n_:.3f}")


if __name__ == "__main__":
    main()
