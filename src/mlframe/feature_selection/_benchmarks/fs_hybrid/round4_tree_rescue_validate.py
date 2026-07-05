"""Round-4: validate the madelon tree-importance recovery across SEEDS, vs mrmr_fe AND the production hybrid.

The single-seed isolation bench showed a 0.4s shallow-GBM importance pass recovers madelon's collapse-to-3
(mrmr_fe lgbm 0.69 -> tree_top15 lgbm 0.90 -> tree_top25+cooccur_fe lgbm 0.91, matching the hybrid's 0.915).
Before treating that as real, confirm it is not a single-split fluke and place it against the production hybrid.

Variants on madelon x 3 split-seeds:
  all, mrmr_fe (collapse ref), hybrid (production winner), tree_top20, tree_top20+cooccur_fe.
Report mean +/- std AUC + n. KILL the finding if tree variants do NOT robustly beat mrmr_fe, or are not
competitive with the hybrid, across seeds.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import load_real, downstream
from round4_jmim_bench import mrmr_fe
from round4_tree_seed_bench import shallow_tree_signals, engineer_products
from hybrid_selector import HybridSelector


def main():
    X, y, name = load_real()
    print(f"REAL: {name} shape={X.shape} pos={round(float(y.mean()),3)}", flush=True)
    seeds = [0, 1, 2]
    rows = []
    for sd in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)

        def emit(tag, Ztr, Zte, t0):
            a = downstream(Ztr, Zte, ytr, yte); am = round(float(np.nanmean(list(a.values()))), 4)
            rows.append(dict(seed=sd, variant=tag, n=int(Ztr.shape[1]), fit_s=round(time.time()-t0, 1), auc_mean=am, **a))
            print(f"  sd{sd} {tag:24s} n={int(Ztr.shape[1]):3d} {rows[-1]['fit_s']:6.1f}s mean={am} {a}", flush=True)

        t0 = time.time(); emit("all", Xtr, Xte, t0)
        t0 = time.time(); T, _ = mrmr_fe(Xtr, ytr); emit("mrmr_fe", T.transform(Xtr), T.transform(Xte), t0)
        t0 = time.time(); h = HybridSelector(vote=1, use_fe=True); h.fit(Xtr, ytr); emit("hybrid", h.transform(Xtr), h.transform(Xte), t0)
        t0 = time.time(); ranked, pairs = shallow_tree_signals(Xtr, ytr)
        sel = ranked[:20]; emit("tree_top20", Xtr[sel], Xte[sel], t0)
        t0 = time.time(); Ztr = engineer_products(Xtr[sel], pairs); Zte = engineer_products(Xte[sel], pairs)
        emit("tree_top20+cooccur_fe", Ztr, Zte, t0)

    df = pd.DataFrame(rows)
    print("\n=== mean +/- std over seeds ===")
    agg = df.groupby("variant").agg(auc_mean=("auc_mean", "mean"), auc_std=("auc_mean", "std"), n=("n", "mean"), fit_s=("fit_s", "mean")).round(4)
    agg = agg.sort_values("auc_mean", ascending=False)
    print(agg.to_string())
    mr = float(agg.loc["mrmr_fe", "auc_mean"]); hy = float(agg.loc["hybrid", "auc_mean"])
    tr = float(agg.loc["tree_top20+cooccur_fe", "auc_mean"])
    print(f"\nmadelon recovery: tree_top20+cooccur_fe {tr} vs mrmr_fe {mr} (d={round(tr-mr,4):+}) vs hybrid {hy} (d={round(tr-hy,4):+})")
    print("VERDICT:", "tree-recovery ROBUST (beats mrmr_fe, competitive w/ hybrid)" if tr > mr + 0.02 else "weak / KILL")


if __name__ == "__main__":
    main()
