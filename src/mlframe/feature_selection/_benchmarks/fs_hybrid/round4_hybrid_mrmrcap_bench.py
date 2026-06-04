"""Round-4: does raising the hybrid's MRMR-member synergy cap lift the HYBRID on hard_synth?

Agent-B finding: raising MRMR's fe_synergy_screen_max_features lifts STANDALONE MRMR on hard_synth (220 cols >
default cap 60, so the default SKIPS the synergy bootstrap) 0.7576 -> 0.8030. Question: given the hybrid already
has the tree member (which adds co-occurrence products), is the MRMR-member synergy bootstrap additive or redundant?
Test hybrid(mrmr_synergy_cap=None=default) vs hybrid(mrmr_synergy_cap=250), 3 seeds, hard_synth + synth + madelon.
PASS: hard_synth >= +0.01 mean AND synth/madelon within +/-0.005.
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
        for tag, cap in (("default", None), ("cap250", 250)):
            t0 = time.time()
            h = HybridSelector(vote=1, use_fe=True, mrmr_synergy_cap=cap, random_state=sd).fit(Xtr, ytr)
            a = downstream(h.transform(Xtr), h.transform(Xte), ytr, yte); am = round(float(np.nanmean(list(a.values()))), 4)
            rows.append(dict(bed=name, seed=sd, variant=tag, n=int(h.transform(Xte).shape[1]), fit_s=round(time.time()-t0,1), auc_mean=am))
            print(f"  [{name}] sd{sd} {tag:8s} n={rows[-1]['n']:3d} {rows[-1]['fit_s']:6.1f}s mean={am}", flush=True)
    return rows


def main():
    allrows = []
    Xh, yh, _ = make_hard_dataset(n_samples=5000, seed=0); print(f"=== hard_synth {Xh.shape} ===", flush=True)
    allrows += run("hard_synth", Xh, yh)
    Xs, ys, _ = make_dataset(n_samples=5000, seed=0); print(f"=== synth {Xs.shape} ===", flush=True)
    allrows += run("synth", Xs, ys)
    Xr, yr, rname = load_real(); print(f"=== {rname} {Xr.shape} ===", flush=True)
    allrows += run(rname, Xr, yr)
    df = pd.DataFrame(allrows)
    print("\n=== mean over seeds ===")
    print(df.groupby(["bed", "variant"]).agg(auc=("auc_mean","mean"), std=("auc_mean","std"), n=("n","mean")).round(4).to_string())
    print("\n=== verdict ===")
    for bed in df.bed.unique():
        d0 = df[(df.bed==bed)&(df.variant=="default")]["auc_mean"].mean()
        d1 = df[(df.bed==bed)&(df.variant=="cap250")]["auc_mean"].mean()
        print(f"  {bed:12s} cap250 {round(d1,4)} vs default {round(d0,4)}  d={round(d1-d0,4):+}")
    df.to_csv("D:/Temp/round4_hybrid_mrmrcap_rows.csv", index=False)


if __name__ == "__main__":
    main()
