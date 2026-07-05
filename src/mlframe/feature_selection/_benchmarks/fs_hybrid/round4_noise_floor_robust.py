"""Robustness probe for A4-2 plateau noise-floor stop: does madelon N* stay ~8 across permutation seeds / n_perm,
and is the downstream win a property of the AUTO-LOCATED N (not just 'any small N')? Also sweeps the held-out AUC
across a neighborhood of N so we can see the plateau the rule landed on.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import load_real, downstream
from round4_noise_floor_bench import lgbm_gain_ranking, cv_curve, noise_floor_plateau, noise_floor_first, _mk_model

OUT = "D:/Temp/rfecv_floor_robust.txt"


def log(msg):
    with open(OUT, "a") as fh:
        fh.write(msg + "\n")
    print(msg, flush=True)


def main():
    open(OUT, "w").close()
    X, y, name = load_real()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    p = Xtr.shape[1]
    log(f"madelon {name} shape={X.shape}")

    # ranking stability across LGBM-gain seeds
    rankings = {}
    for seed in (0, 1, 2):
        m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, n_jobs=4,
                               importance_type="gain", random_state=seed, verbose=-1).fit(Xtr, ytr)
        order = np.argsort(m.feature_importances_)[::-1]
        rankings[seed] = [Xtr.columns[i] for i in order]
    top8_overlap = len(set(rankings[0][:8]) & set(rankings[1][:8]) & set(rankings[2][:8]))
    log(f"top-8 ranking overlap across gain-seeds {{0,1,2}}: {top8_overlap}/8")

    ranked = rankings[0]
    n_grid = sorted(set([1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50, 75, 100, 150, 200, 251, p]))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    real_curve = cv_curve(Xtr, ytr, ranked, n_grid, cv, permute=False)

    # N* stability across n_perm and permutation base_seed
    log("\nN* (plateau) vs n_perm / base_seed:")
    for n_perm in (1, 3, 5):
        for bs in (100, 200, 300):
            _, perm_curves = cv_curve(Xtr, ytr, ranked, n_grid, cv, permute=True, n_perm=n_perm, base_seed=bs)
            Np, _, _, _ = noise_floor_plateau(n_grid, real_curve, perm_curves, pct=95.0)
            Nf, _, _, _ = noise_floor_first(n_grid, real_curve, perm_curves, pct=95.0)
            log(f"  n_perm={n_perm} base_seed={bs}: plateau N*={Np}  first N*={Nf}")

    # held-out AUC sweep across a neighborhood of N (is the win specific to the auto N, or flat across small N?)
    log("\nheld-out AUC by N (LGBM-gain top-N), to show the plateau the rule landed on:")
    log(f"  {'N':>4} {'lgbm':>7} {'logit':>7} {'knn':>7} {'mean':>7}")
    for N in (3, 5, 6, 8, 10, 12, 16, 20, 30, 50, 100, 251, p):
        cols = ranked[:N]
        a = downstream(Xtr[cols], Xte[cols], ytr, yte)
        am = round(float(np.nanmean(list(a.values()))), 4)
        log(f"  {N:>4} {a['lgbm']:>7} {a['logit']:>7} {a['knn']:>7} {am:>7}")
    log("\nDONE")


if __name__ == "__main__":
    main()
