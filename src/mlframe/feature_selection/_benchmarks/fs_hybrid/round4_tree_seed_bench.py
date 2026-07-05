"""Round-4 convergent ideas A1-3/A3-1 (tree split-co-occurrence) + A1-2/A3-2 (the p>60 synergy gate).

Decisive ISOLATION test (bench-local, NO production edit) of the two distinct uses of a cheap shallow-tree
supervised signal, against the MEASURED madelon collapse-to-3:

  (1) tree-importance SELECTION   -- does picking the shallow GBM's top-K features (a supervised signal that
      sees interaction operands because the tree branches on them) recover madelon where MRMR's marginal-MI
      greedy collapses to 3?  This is the SELECTION fix (seed/rescue MRMR's under-selection).
  (2) tree-co-occurrence FE       -- does engineering raw[i]*raw[j] for the top tree-co-occurrence pairs
      (features appearing together in the same tree = the tree is exploiting their interaction) recover the
      a*b signal?  This is the FE-product fix.  hard_synth (true a*b/a^2 interactions) is the clean bed for it;
      madelon (5-dim XOR cluster) is the stress bed where pairwise products may NOT linearize the signal.

KILL: (1) tree-topK does not beat mrmr_fe collapse on madelon -> the supervised SELECTION signal is no better.
      (2) tree-cooccur products do not recover ia*ib on hard_synth -> the co-occurrence proposer is useless.
The point: learn WHICH mechanism to wire into MRMR before editing the concurrently-owned _mrmr_fe_step.py.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from itertools import combinations
from collections import Counter
from sklearn.model_selection import train_test_split
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import load_real, downstream
from synth import make_dataset
from hard_synth import make_hard_dataset
from round4_jmim_bench import mrmr_fe


def shallow_tree_signals(X, y, n_estimators=80, max_depth=3, top_pairs=12):
    """One cheap depth-limited GBM -> (importance-ranked feature list, top co-occurring raw pairs).

    Co-occurrence proxy: for each tree, take the set of features it splits on; every within-tree pair
    gets a count weighted by the tree's total split gain. Pairs the tree branches on together are the
    interactions it is exploiting -- a supervised operand proposer blind to marginal MI.
    """
    m = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, num_leaves=2**max_depth, learning_rate=0.1, n_jobs=-1, verbose=-1, random_state=0)
    m.fit(X, y)
    cols = list(X.columns)
    imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
    ranked = [c for c in imp.index if imp[c] > 0]
    # co-occurrence from the tree dump
    tdf = m.booster_.trees_to_dataframe()
    tdf = tdf[tdf["split_feature"].notna()]
    pair_w = Counter()
    for tree_id, g in tdf.groupby("tree_index"):
        feats = sorted(set(g["split_feature"].tolist()))
        gain = float(g["split_gain"].sum()) + 1e-9
        for a, b in combinations(feats, 2):
            pair_w[(a, b)] += gain
    top = [p for p, _ in pair_w.most_common(top_pairs)]
    return ranked, top


def engineer_products(X, pairs):
    """Append raw[a]*raw[b] product columns for the given pairs."""
    new = {}
    for i, (a, b) in enumerate(pairs):
        if a in X.columns and b in X.columns:
            new[f"prod_{i}"] = X[a].values * X[b].values
    if not new:
        return X.copy()
    return pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1)


def run_bed(name, X, y, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    rows = []

    def emit(tag, Ztr, Zte, t0):
        a = downstream(Ztr, Zte, ytr, yte); am = round(float(np.nanmean(list(a.values()))), 4)
        rows.append(dict(bed=name, variant=tag, n=int(Ztr.shape[1]), fit_s=round(time.time()-t0, 1), auc_mean=am, **a))
        print(f"[{name}] {tag:26s} n={int(Ztr.shape[1]):3d} {rows[-1]['fit_s']:6.1f}s mean={am} {a}", flush=True)

    # baseline: all + mrmr_fe (the collapse reference)
    t0 = time.time(); emit("all", Xtr, Xte, t0)
    t0 = time.time(); T, n = mrmr_fe(Xtr, ytr); emit("mrmr_fe", T.transform(Xtr), T.transform(Xte), t0)

    # shallow-tree signals (one fit, shared)
    t0 = time.time(); ranked, pairs = shallow_tree_signals(Xtr, ytr); tree_s = round(time.time()-t0, 1)
    print(f"[{name}] shallow-tree: {len(ranked)} nonzero-imp feats, top pairs {pairs[:5]}  ({tree_s}s)", flush=True)

    # (1) tree-importance SELECTION at a few K
    for K in (15, 25, 40):
        sel = ranked[:K]
        t0 = time.time(); emit(f"tree_top{K}", Xtr[sel], Xte[sel], t0)

    # (2) tree-co-occurrence FE products on top of tree_top25
    base = ranked[:25]
    t0 = time.time()
    Ztr = engineer_products(Xtr[base], pairs); Zte = engineer_products(Xte[base], pairs)
    emit("tree_top25+cooccur_fe", Ztr, Zte, t0)
    return rows


def main():
    allrows = []
    Xr, yr, rname = load_real()
    print(f"\n=== REAL: {rname} shape={Xr.shape} ===", flush=True)
    allrows += run_bed(rname, Xr, yr, seed=0)
    Xh, yh, _ = make_hard_dataset(n_samples=5000, seed=0)
    print(f"\n=== hard_synth shape={Xh.shape} ===", flush=True)
    allrows += run_bed("hard_synth", Xh, yh, seed=0)
    Xs, ys, _ = make_dataset(n_samples=5000, seed=0)
    print(f"\n=== synth shape={Xs.shape} ===", flush=True)
    allrows += run_bed("synth", Xs, ys, seed=0)

    df = pd.DataFrame(allrows)
    print("\n=== ALL ===")
    print(df.to_string(index=False))
    print("\n=== verdicts ===")
    for bed in df.bed.unique():
        b = df[df.bed == bed].set_index("variant")
        mr = float(b.loc["mrmr_fe", "auc_mean"]) if "mrmr_fe" in b.index else float("nan")
        best_tree = b.loc[[i for i in b.index if i.startswith("tree_top") and "+" not in i], "auc_mean"].max()
        co = float(b.loc["tree_top25+cooccur_fe", "auc_mean"]) if "tree_top25+cooccur_fe" in b.index else float("nan")
        print(f"  {bed:12s} mrmr_fe={mr}  best_tree_select={round(best_tree,4)} (d={round(best_tree-mr,4):+})  "
              f"+cooccur_fe={co} (d vs tree_top25 {round(co-float(b.loc['tree_top25','auc_mean']),4):+})")


if __name__ == "__main__":
    main()
