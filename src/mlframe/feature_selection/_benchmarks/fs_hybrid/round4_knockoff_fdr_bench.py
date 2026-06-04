"""Round-4 idea A4-1: knockoff-FDR post-hoc cut for RFECV's madelon OVER-selection (251 features -> ~20).

RFECV's CV curve is flat on noise-robust GBM, so its N-rule lands high (251/500 on madelon -> noisy). Knockoffs
attack on a different axis: build a draw-INDEPENDENT Gaussian knockoff X_tilde for the chosen support, fit on
[X, X_tilde], W_j = imp(real_j) - imp(knockoff_j); a pure-noise feature's W is symmetric around 0 -> dropped by
the Barber-Candes FDR threshold. Plumbing (_knockoffs.py) exists and is UNUSED by the N-rule. Run ONCE on the
already-chosen 251-survivor set (p=251, 2p<n so not p>n-degenerate), with a premerge collapse to reduce collinearity.

FALSIFIABLE: does select_features_fdr cut 251 -> ~20-40 WITHOUT losing the all-features AUC (lgbm 0.87)?
KILL: FDR returns [] (no threshold achieves q -> knockoffs degenerate), OR keeps >100, OR AUC drops > 0.01.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import load_real, downstream
import fs_selectors as S
from mlframe.feature_selection.wrappers import knockoff_importance, select_features_fdr


def main():
    X, y, name = load_real()
    print(f"REAL: {name} shape={X.shape}", flush=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    rows = []

    def emit(tag, cols, t0):
        cols = [c for c in cols if c in Xtr.columns]
        if not cols:
            print(f"  {tag:24s} EMPTY (KILL signal)", flush=True); return
        a = downstream(Xtr[cols], Xte[cols], ytr, yte); am = round(float(np.nanmean(list(a.values()))), 4)
        rows.append(dict(variant=tag, n=len(cols), fit_s=round(time.time()-t0, 1), auc_mean=am, **a))
        print(f"  {tag:24s} n={len(cols):3d} {rows[-1]['fit_s']:6.1f}s mean={am} {a}", flush=True)

    t0 = time.time(); emit("all", list(Xtr.columns), t0)

    # 1) RFECV -> its (over-selected) support (cached: the fit is ~220s, deterministic at random_state=0)
    import pickle
    cache = "D:/Temp/rfecv_madelon_support.pkl"
    t0 = time.time()
    if os.path.exists(cache):
        support = pickle.load(open(cache, "rb")); print("  (loaded cached RFECV support)", flush=True)
    else:
        sel = S.RFECVSel("lgbm_perm"); sel.fit(Xtr, ytr)
        support = [c for c in sel.raw_selected_ if c in Xtr.columns]
        pickle.dump(support, open(cache, "wb"))
    emit("rfecv (support)", support, t0)
    print(f"  -> RFECV kept {len(support)} features", flush=True)

    mk = lambda: lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.06, n_jobs=-1, verbose=-1)

    def knockoff_cut(feats, tag):
        if len(feats) < 5:
            return
        t0 = time.time()
        W = knockoff_importance(mk, Xtr[feats], ytr, random_state=0, w_statistic="gain")
        print(f"  -> [{tag}] knockoff W on {len(feats)} feats ({round(time.time()-t0,1)}s); W>0: {sum(1 for v in W.values() if v>0)}", flush=True)
        for q in (0.05, 0.1, 0.2):
            t0 = time.time(); emit(f"{tag}_fdr_q{q}", select_features_fdr(W, q=q), t0)

    # 2) raw knockoff cut on the 251 support (degenerate on madelon's collinear probes -- the flagged risk)
    knockoff_cut(support, "knockoff")

    # 3) WORKAROUND: premerge-collapse correlated clusters first (reduce collinearity so knockoffs are non-singular),
    #    run knockoffs on the cluster reps, then re-expand kept reps to their members.
    from hybrid_selector import corr_clusters
    reps, members = corr_clusters(Xtr[support], thr=0.5)   # aggressive collapse for madelon's probe clusters
    print(f"  -> premerge: {len(support)} support -> {len(reps)} reps (corr_thr=0.5)", flush=True)
    if len(reps) >= 5:
        t0 = time.time()
        W = knockoff_importance(mk, Xtr[reps], ytr, random_state=0, w_statistic="gain")
        print(f"  -> [premerge] knockoff W on {len(reps)} reps ({round(time.time()-t0,1)}s); W>0: {sum(1 for v in W.values() if v>0)}", flush=True)
        for q in (0.05, 0.1, 0.2):
            kept_reps = select_features_fdr(W, q=q)
            expanded = [m for r in kept_reps for m in members.get(r, [r])]
            t0 = time.time(); emit(f"premerge_knockoff_fdr_q{q}", expanded, t0)
    df = pd.DataFrame(rows)
    print("\n=== ranked ===")
    print(df.sort_values("auc_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
