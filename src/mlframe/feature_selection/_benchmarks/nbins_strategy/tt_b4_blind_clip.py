"""Fairness check for the blind-add baseline (research B4): ridge and GAM with every candidate added, with and without winsorising each added column to the
training 0.5 / 99.5 percentiles (an unclipped 1/x of a positive column with tiny values explodes on test rows). Ridge also reports the recommended stepwise set for reference.
Usage: python tt_b4_blind_clip.py --p 100 --n 5000 --seeds 4
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
sys.path.insert(0, __file__.replace("\\", "/").rsplit("/", 1)[0])
from tt_b4_models import gam_fit_predict, r2_rmse, ridge_fit_predict
from tt_b4_panel import make_panel
from tt_b4_reference import all_candidate_matrix, recommend_stepwise
from tt_b4_run_detect import config_for
from tt_b4_run_e2e import add_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--gam", action="store_true")
    a = ap.parse_args()
    rows = []
    for s in range(a.seeds):
        Xall, yall, info = make_panel(2 * a.n, a.p, s, **config_for(a.p, s))
        X, y, Xt, yt = Xall[: a.n], yall[: a.n], Xall[a.n:], yall[a.n:]
        Bm, _ = all_candidate_matrix(X, X)
        Bt, _ = all_candidate_matrix(X, Xt)
        lo, hi = np.percentile(Bm, [0.5, 99.5], axis=0)
        Bmc, Btc = np.clip(Bm, lo, hi), np.clip(Bt, lo, hi)
        ch = {j: f for j, f in recommend_stepwise(X, y, n_null=50, seed=s).table["chosen"].items() if f != "id"}
        sets = {"none": (X, Xt), "blind": (np.c_[X, Bm], np.c_[Xt, Bt]), "blind_clipped": (np.c_[X, Bmc], np.c_[Xt, Btc]), "stepwise": add_cols(X, Xt, ch)}
        r = {}
        for k, (A, B) in sets.items():
            r[k] = {"ridge": r2_rmse(yt, ridge_fit_predict(A, y, B))[0]}
            if a.gam and not (k.startswith("blind") and a.p > 20):
                r[k]["gam"] = r2_rmse(yt, gam_fit_predict(A, y, B))[0]
        rows.append(r)
        print(f"seed {s}: " + "; ".join(f"{k} ridge {v['ridge']:.4f}" + (f" gam {v['gam']:.4f}" if "gam" in v else "") for k, v in r.items()), flush=True)
    print(f"MEAN p={a.p} n={a.n}: " + "; ".join(f"{k} ridge {np.mean([r[k]['ridge'] for r in rows]):.4f}" + (f" gam {np.mean([r[k]['gam'] for r in rows]):.4f}" if "gam" in rows[0][k] else "") for k in rows[0]))


if __name__ == "__main__":
    main()
