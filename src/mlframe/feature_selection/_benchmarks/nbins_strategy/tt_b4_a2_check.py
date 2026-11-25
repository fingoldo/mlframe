"""Does the vectorised reference reproduce the wave-1 (tt_a2) single-feature result? (research B4 sanity check)

Same 9 fixtures (y = g(x) + 0.5 z + 0.7 noise, 20000 rows, 7 seeds) and the same decision rule: correct = (a transform with judge gain > 0.02 R2 exists: accepted and the
pick within 0.02 R2 of the judge best) or (no gain exists: nothing accepted). The recommender sees the 2-column frame [x, z] (z is a second linear feature) and runs in
three modes: marginal (selected=None), conditional (selected=[z]) and with replication on a held-out half. Candidate set = CAND_DEFAULT (9, incl. rank-gauss).
"""
from __future__ import annotations

import collections
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, __file__.replace("\\", "/").rsplit("/", 1)[0])
from tt_a2_feature_transform import KINDS, make
from tt_b4_models import ridge_fit_predict, r2_rmse
from tt_b4_reference import CAND_DEFAULT, ColParams, _col_params_j, design_column, recommend_feature_transforms, rg_grid, valid_mask


def judge(x, z, y, h):
    p = ColParams.fit(x[:h, None]); valid = valid_mask(p, CAND_DEFAULT)
    out = {}
    for k, c in enumerate(CAND_DEFAULT):
        if not valid[0, k]:
            continue
        g = rg_grid(x[:h]) if c == "rankgauss" else None
        a = design_column(x[:h], _col_params_j(p, 0), c, g); b = design_column(x[h:], _col_params_j(p, 0), c, g)
        out[c] = r2_rmse(y[h:], ridge_fit_predict(np.c_[z[:h], a], y[:h], np.c_[z[h:], b]))[0]
    return out


def main(seeds=7, n=20000):
    modes = {"marginal": dict(selected=None), "conditional(z)": dict(selected=[1])}
    print("fixture | judge best (gain over id) | " + " | ".join(f"{m}: correct/{seeds} (picks)" for m in modes))
    tot = {m: 0 for m in modes}; fp = {m: 0 for m in modes}; ctl = {m: 0 for m in modes}
    for kind in KINDS:
        cor = {m: 0 for m in modes}; picks = {m: collections.Counter() for m in modes}; gains = []; bests = collections.Counter()
        for s in range(seeds):
            x, z, y = make(kind, n, s)
            h = n // 2
            j = judge(x, z, y, h)
            best = max(j, key=j.get); gain = j[best] - j["id"]; gains.append(gain); bests[best] += 1
            X = np.c_[x, z]
            for m, kw in modes.items():
                r = recommend_feature_transforms(X, y, n_null=50, seed=s, rule="gated", **kw)
                acc = bool(r.table["accepted"][0]); pk = r.table["best"][0]
                picks[m][pk if acc else "-"] += 1
                ok = (acc and j.get(pk, -9) >= j[best] - 0.02) if gain > 0.02 else (not acc)
                cor[m] += int(ok)
                if kind.startswith("ctl"):
                    ctl[m] += 1; fp[m] += int(acc)
        for m in modes:
            tot[m] += cor[m]
        print(f"{kind:22s} | {bests.most_common(2)} gain {np.mean(gains):.3f} | " + " | ".join(f"{cor[m]} {dict(picks[m])}" for m in modes), flush=True)
    print("TOTAL correct:", {m: f"{tot[m]}/{len(KINDS) * seeds}" for m in modes}, " controls accepted (false positives):", {m: f"{fp[m]}/{ctl[m]}" for m in modes})


if __name__ == "__main__":
    main()
