"""MRMR's own fe_max_steps=1 output on a correlated-feature panel (research B4, direction (d)).

Fits MRMR on a 20000 x 20 panel (4 nonlinear informative columns with known links, 1 linear informative, partners, noise), lists the engineered columns it produced
(provenance), and scores ridge on MRMR's transform() output on a fresh 20000-row test set. Also reports ridge on the raw columns and on raw + the recommender's picks.
Usage: python tt_b4_mrmr_run.py [--seeds 3] [--n 20000] [--p 20]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
sys.path.insert(0, __file__.replace("\\", "/").rsplit("/", 1)[0])
from tt_b4_panel import make_panel
from tt_b4_reference import Recipes, recommend_stepwise


def ridge_r2(Atr, ytr, Ate, yte, lam=1.0):
    mu, sd = Atr.mean(0), Atr.std(0) + 1e-12
    Z = (Atr - mu) / sd
    Zt = (Ate - mu) / sd
    Z1 = np.c_[np.ones(len(Z)), Z]
    Zt1 = np.c_[np.ones(len(Zt)), Zt]
    pen = lam * np.eye(Z1.shape[1]); pen[0, 0] = 0
    w = np.linalg.solve(Z1.T @ Z1 + pen, Z1.T @ ytr)
    pr = Zt1 @ w
    return 1 - ((yte - pr) ** 2).sum() / ((yte - yte.mean()) ** 2).sum(), float(np.sqrt(((yte - pr) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--p", type=int, default=20)
    a = ap.parse_args()
    from mlframe.feature_selection.filters import MRMR
    for seed in range(a.seeds):
        Xall, yall, info = make_panel(2 * a.n, a.p, seed)
        X, y, Xt, yt = Xall[: a.n], yall[: a.n], Xall[a.n:], yall[a.n:]
        names = [f"c{j}" for j in range(a.p)]
        df, dft = pd.DataFrame(X, columns=names), pd.DataFrame(Xt, columns=names)
        truth = {names[c]: info["link"][c] for c in info["inf_cols"]}
        print(f"## seed {seed}: informative columns and links {truth}", flush=True)
        r2raw, rmse_raw = ridge_r2(X, y, Xt, yt)
        print(f"## ridge on all {a.p} raw columns: R2 {r2raw:.4f} RMSE {rmse_raw:.4f}", flush=True)
        t = time.perf_counter()
        m = MRMR(fe_max_steps=1, verbose=0, random_state=seed).fit(df, y)
        fit_s = time.perf_counter() - t
        pv = m.fe_provenance_
        sel = pv[pv["in_output"] == True]  # noqa: E712
        print(f"## MRMR fit {fit_s:.1f} s; in_output rows:\n{sel[['feature_name', 'origin', 'mrmr_gain']].to_string(index=False)}", flush=True)
        eng = [f for f in pv["feature_name"] if f not in names]
        print(f"## all engineered names in provenance ({len(eng)}): {eng[:40]}", flush=True)
        tr, te = m.transform(df), m.transform(dft)
        tr, te = np.asarray(tr, dtype=np.float64), np.asarray(te, dtype=np.float64)
        ok = np.isfinite(tr).all(0) & np.isfinite(te).all(0)
        r2m, rmse_m = ridge_r2(tr[:, ok], y, te[:, ok], yt)
        print(f"## ridge on MRMR transform() output ({tr.shape[1]} columns, {int(ok.sum())} finite): R2 {r2m:.4f} RMSE {rmse_m:.4f}", flush=True)
        out_names = list(m.get_feature_names_out()) if hasattr(m, "get_feature_names_out") else []
        print(f"## MRMR output names: {out_names}", flush=True)
        # the recommender (stepwise) on the same fit rows: raw + the transforms it commits
        res = recommend_stepwise(X, y, n_null=50, seed=seed)
        rec = Recipes(X, res.table["chosen"])
        extra = rec.transform_cols(X); extra_t = rec.transform_cols(Xt)
        A = np.c_[X, np.column_stack([extra[j] for j in extra])] if extra else X
        At = np.c_[Xt, np.column_stack([extra_t[j] for j in extra])] if extra else Xt
        r2r, rmse_r = ridge_r2(A, y, At, yt)
        print(f"## recommender stepwise: chosen {res.table['chosen']} ({res.seconds:.1f} s); ridge on raw + added transforms: R2 {r2r:.4f} RMSE {rmse_r:.4f}", flush=True)
        # MRMR output plus the recommender transforms (credit on top of MRMR's selection)
        names_in_out = [c for c in names if c in set(out_names)]
        print(f"## raw columns kept by MRMR: {names_in_out}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
