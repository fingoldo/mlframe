"""Round-4 FINAL confirm: tree member (synergy gate, default config) vs notree, 3 seeds, all 3 beds.

Settles whether the madelon win is robust across seeds AND whether the tiny synth/hard_synth deltas are within
seed noise (so the member can default ON without a real regression). PASS to ship default-ON: madelon mean lift
>= +0.03 across seeds AND synth/hard_synth mean delta >= -0.005 (within noise).
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


def run(name, X, y, seeds=(0, 1, 2)):
    rows = []
    for sd in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
        for tag, kw in (("notree", dict(use_tree_member=False)), ("tree", dict(use_tree_member=True))):
            t0 = time.time()
            h = HybridSelector(vote=1, use_fe=True, random_state=sd, **kw); h.fit(Xtr, ytr)
            a = downstream(h.transform(Xtr), h.transform(Xte), ytr, yte); am = round(float(np.nanmean(list(a.values()))), 4)
            rows.append(dict(bed=name, seed=sd, variant=tag, n=int(h.transform(Xte).shape[1]), auc_mean=am))
            print(f"  [{name}] sd{sd} {tag:7s} n={rows[-1]['n']:3d} {time.time()-t0:6.1f}s mean={am}", flush=True)
    return rows


def main():
    allrows = []
    Xr, yr, rname = load_real(); print(f"=== {rname} {Xr.shape} ===", flush=True)
    allrows += run(rname, Xr, yr)
    Xh, yh, _ = make_hard_dataset(n_samples=5000, seed=0); print(f"=== hard_synth {Xh.shape} ===", flush=True)
    allrows += run("hard_synth", Xh, yh)
    Xs, ys, _ = make_dataset(n_samples=5000, seed=0); print(f"=== synth {Xs.shape} ===", flush=True)
    allrows += run("synth", Xs, ys)

    df = pd.DataFrame(allrows)
    print("\n=== mean over seeds ===")
    piv = df.groupby(["bed", "variant"]).agg(auc=("auc_mean", "mean"), std=("auc_mean", "std"), n=("n", "mean")).round(4)
    print(piv.to_string())
    print("\n=== verdict ===")
    ok = True
    for bed in df.bed.unique():
        t = df[(df.bed == bed) & (df.variant == "tree")]["auc_mean"].mean()
        nt = df[(df.bed == bed) & (df.variant == "notree")]["auc_mean"].mean()
        d = round(t - nt, 4)
        thr = 0.03 if bed == rname else -0.005
        good = (d >= thr) if bed == rname else (d >= thr)
        ok = ok and good
        print(f"  {bed:12s} tree {round(t,4)} vs notree {round(nt,4)}  d={d:+}  ({'OK' if good else 'FAIL'})")
    print("\nSHIP default-ON:", "YES" if ok else "NO -- needs regime gate")


if __name__ == "__main__":
    main()
