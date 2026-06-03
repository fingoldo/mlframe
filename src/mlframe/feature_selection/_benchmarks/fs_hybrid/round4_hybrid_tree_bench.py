"""Round-4: does ADDING the tree-importance member lift the hybrid on madelon WITHOUT regressing synth/hard_synth?

The standalone tree_top20+cooccur_fe beat the production hybrid on madelon (3-seed 0.840 vs 0.805) but LOST on the
FE-saturated synth (0.745 vs mrmr_fe 0.834) -- the two FE styles fail oppositely. The fix is to add the tree signal
as a MEMBER so the composition keeps both. The incremental madelon win came from the co-occurrence PRODUCTS, not the
top-k raw votes (the hybrid's existing members already match tree-selection), so test products-only too.

Variants (all else equal):
  hybrid_notree   = production hybrid (use_tree_member=False) -- the baseline to beat / not regress
  hybrid_tree     = + tree member (top_k=20 votes + FI-gated cooccur products)
  hybrid_treeprod = + tree member, PRODUCTS ONLY (tree_top_k=0) -- isolates the product contribution, no raw-vote dilution
PASS: a tree variant lifts madelon >= +0.02 AND does NOT regress synth/hard_synth by > 0.005. KILL otherwise.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import load_real, downstream
from synth import make_dataset
from hard_synth import make_hard_dataset
from hybrid_selector import HybridSelector


def run_bed(name, X, y, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    rows = []
    variants = {
        "notree":        dict(use_tree_member=False),
        "gate_synergy":  dict(use_tree_member=True, tree_top_k=0, tree_prod_gate="synergy"),
        "gate_relmed":   dict(use_tree_member=True, tree_top_k=0, tree_prod_gate="relevant_median"),
        "gate_rawmed":   dict(use_tree_member=True, tree_top_k=0, tree_prod_gate="raw_median"),
    }
    for tag, kw in variants.items():
        t0 = time.time()
        h = HybridSelector(vote=1, use_fe=True, random_state=seed, **kw); h.fit(Xtr, ytr)
        Ztr, Zte = h.transform(Xtr), h.transform(Xte)
        a = downstream(Ztr, Zte, ytr, yte); am = round(float(np.nanmean(list(a.values()))), 4)
        neng = getattr(h, "n_engineered_", 0)
        rows.append(dict(bed=name, variant=tag, n=int(Ztr.shape[1]), n_eng=int(neng),
                         fit_s=round(time.time()-t0, 1), auc_mean=am, **a))
        print(f"[{name}] {tag:16s} n={int(Ztr.shape[1]):3d} eng={int(neng):2d} {rows[-1]['fit_s']:6.1f}s mean={am} {a}", flush=True)
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
    print("\n=== verdict (vs notree per bed) ===")
    for bed in df.bed.unique():
        b = df[df.bed == bed].set_index("variant")
        base = float(b.loc["notree", "auc_mean"])
        for v in ("gate_synergy", "gate_relmed", "gate_rawmed"):
            d = round(float(b.loc[v, "auc_mean"]) - base, 4)
            print(f"  {bed:12s} {v:14s} {b.loc[v,'auc_mean']} (d={d:+}) vs notree {base}")


if __name__ == "__main__":
    main()
