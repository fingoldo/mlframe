"""Round-4 synergy idea A2-2: RESIDUAL-RELAY (boosting logic for FS).

Hypothesis: after the hybrid selects S and a model fits it, the OOF RESIDUAL carries the complementary signal the
selection missed (e.g. hard_synth's weak-sparse block). Features correlated with the residual -- especially ones
NOT in S -- are exactly what a second selection pass should recover. Cheap falsifiable test: screen the DROPPED
features by |corr|/MI with the OOF residual, add the top-k, and see if AUC improves over S alone.

KILL: adding residual-screened dropped features does NOT improve AUC (residual is noise-dominated, or S already
captured the recoverable signal -- the tree member may have closed this gap).
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import load_real, downstream
from hard_synth import make_hard_dataset
from hybrid_selector import HybridSelector


def run_bed(name, X, y, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    rows = []

    def emit(tag, cols, t0):
        cols = [c for c in dict.fromkeys(cols) if c in Xtr.columns]
        a = downstream(Xtr[cols], Xte[cols], ytr, yte); am = round(float(np.nanmean(list(a.values()))), 4)
        rows.append(dict(bed=name, variant=tag, n=len(cols), fit_s=round(time.time()-t0, 1), auc_mean=am, **a))
        print(f"[{name}] {tag:22s} n={len(cols):3d} {rows[-1]['fit_s']:6.1f}s mean={am} {a}", flush=True)
        return set(cols)

    # baseline: tree-member hybrid
    t0 = time.time(); h = HybridSelector(vote=1, use_fe=True, random_state=seed).fit(Xtr, ytr)
    S = [c for c in h.raw_selected_ if c in Xtr.columns]
    Sset = emit("hybrid", S, t0)

    # OOF residual from a model on S
    m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, n_jobs=-1, verbose=-1)
    p_oof = cross_val_predict(m, Xtr[S], ytr, cv=4, method="predict_proba", n_jobs=-1)[:, 1]
    resid = ytr.values.astype(float) - p_oof   # deviance-ish residual

    # screen DROPPED raw features by |corr| with the residual, and (separately) by MI with a binarized residual
    dropped = [c for c in Xtr.columns if c not in Sset]
    if dropped:
        Xd = Xtr[dropped].values
        corr = np.abs(np.nan_to_num([np.corrcoef(Xd[:, j], resid)[0, 1] for j in range(Xd.shape[1])]))
        order_corr = [dropped[j] for j in np.argsort(corr)[::-1]]
        rbin = (resid > np.median(resid)).astype(int)
        mi = mutual_info_classif(Xd, rbin, random_state=seed)
        order_mi = [dropped[j] for j in np.argsort(mi)[::-1]]
        print(f"[{name}] top residual-corr dropped: {order_corr[:6]}", flush=True)
        print(f"[{name}] top residual-MI   dropped: {order_mi[:6]}", flush=True)
        for k in (5, 10):
            t0 = time.time(); emit(f"relay_corr_top{k}", S + order_corr[:k], t0)
            t0 = time.time(); emit(f"relay_mi_top{k}", S + order_mi[:k], t0)
    df = pd.DataFrame(rows)
    base = float(df[df.variant == "hybrid"]["auc_mean"].iloc[0])
    best = df[df.variant != "hybrid"]["auc_mean"].max()
    print(f"[{name}] VERDICT: best relay {best} vs hybrid {base} (d={round(best-base,4):+}) "
          f"{'-> residual-relay HELPS' if best > base + 0.003 else '-> no gain / KILL'}", flush=True)
    return rows


def main():
    allrows = []
    Xh, yh, _ = make_hard_dataset(n_samples=5000, seed=0)
    print(f"=== hard_synth {Xh.shape} ===", flush=True)
    allrows += run_bed("hard_synth", Xh, yh)
    Xr, yr, rname = load_real()
    print(f"\n=== {rname} {Xr.shape} ===", flush=True)
    allrows += run_bed(rname, Xr, yr)
    pd.DataFrame(allrows).to_csv("D:/Temp/round4_residual_relay_rows.csv", index=False)


if __name__ == "__main__":
    main()
