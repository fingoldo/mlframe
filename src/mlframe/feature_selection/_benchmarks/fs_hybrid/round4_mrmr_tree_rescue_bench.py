"""Round-4 #1 (biggest unsolved): fix MRMR-STANDALONE's selection-gate collapse with a tree-importance RESCUE.

CONFIRMED root cause (agent B + MDLP diag + FE-cap bench): MRMR's marginal-MI greedy under-selects to <=3 on
interaction-heavy data (madelon: lgbm 0.69 vs all-features 0.87) because the informative operands have ~0 marginal
MI. It is the SELECTION gate, not FE or binning. The tree member fixed the HYBRID; standalone MRMR is still broken.
Agent B showed injecting tree PRODUCTS into MRMR's FE fails (the greedy discards them). This tests a different
mechanism: when MRMR under-selects on a wide frame, UNION its selection with a cheap shallow-GBM importance top-K
(post-selection rescue -- not FE, not via the greedy). Gated to the under-selection regime so it is a no-op where
MRMR already selects well (synth).

Gate: rescue fires iff len(mrmr_raw_selected) < max(5, ceil(0.04*p)) AND p > 60 (wide, under-selected).
Variants: mrmr_fe (baseline) ; mrmr_fe + tree_rescue(K in {15,25}). PASS: madelon >= +0.05 (toward tree_top's 0.84)
AND synth/hard_synth within +/-0.005 (gate must not fire / not hurt where MRMR is fine).
"""
from __future__ import annotations
import os, sys, time, math
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import load_real, downstream
from synth import make_dataset
from hard_synth import make_hard_dataset
from round4_jmim_bench import mrmr_fe
from round4_tree_seed_bench import shallow_tree_signals


def mrmr_raw_selected(X, y):
    """Fit mrmr_fe; return (transformer, raw-selected col list, n_total_out)."""
    from mlframe.feature_selection.filters import MRMR
    import re
    _SAFE = re.compile(r"^[A-Za-z0-9_]+$")
    m = MRMR(verbose=0, fe_max_steps=1, n_jobs=-1, random_seed=0).fit(X, y)
    out = list(m.transform(X.iloc[:5]).columns)
    raw = [c for c in out if _SAFE.match(str(c)) and c in X.columns]
    ren = {c: (c if _SAFE.match(str(c)) else f"eng_{i}") for i, c in enumerate(out)}

    class _T:
        def transform(self, Z):
            df = m.transform(Z).copy(); df.columns = [ren[c] for c in df.columns]; return df
    return _T(), raw, len(out)


def run_bed(name, X, y, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    p = Xtr.shape[1]
    rows = []

    def emit(tag, T, t0, extra=None):
        # ALWAYS the full mrmr_fe transformer output (raw + engineered); rescue APPENDS the tree-top-K raw cols.
        Ztr, Zte = T.transform(Xtr).reset_index(drop=True), T.transform(Xte).reset_index(drop=True)
        if extra:
            ex = [c for c in extra if c in Xtr.columns and c not in Ztr.columns]
            if ex:
                Ztr = pd.concat([Ztr, Xtr[ex].reset_index(drop=True)], axis=1)
                Zte = pd.concat([Zte, Xte[ex].reset_index(drop=True)], axis=1)
        a = downstream(Ztr, Zte, ytr, yte); am = round(float(np.nanmean(list(a.values()))), 4)
        rows.append(dict(bed=name, variant=tag, n=int(Ztr.shape[1]), fit_s=round(time.time()-t0,1), auc_mean=am, **a))
        print(f"[{name}] {tag:20s} n={int(Ztr.shape[1]):3d} {rows[-1]['fit_s']:6.1f}s mean={am} {a}", flush=True)

    t0 = time.time(); T, raw, n_out = mrmr_raw_selected(Xtr, ytr)
    emit("mrmr_fe", T, t0)
    under = len(raw) < max(5, math.ceil(0.04 * p)) and p > 60
    print(f"[{name}] mrmr raw-selected={len(raw)} (of p={p}); under-select regime={under}", flush=True)

    # tree importance (shared, one fit)
    t0 = time.time(); ranked, _pairs = shallow_tree_signals(Xtr, ytr); tsig = round(time.time()-t0,1)
    for K in (15, 25):
        # rescue = mrmr_fe output PLUS tree top-K raw cols, ONLY when under-select detected (else a true no-op = baseline)
        extra = [c for c in ranked[:K] if c in Xtr.columns] if under else None
        t0 = time.time(); emit(f"mrmr_fe+rescue{K}", T, t0, extra=extra)
    return rows


def main():
    allrows = []
    Xr, yr, rname = load_real(); print(f"=== {rname} {Xr.shape} ===", flush=True)
    allrows += run_bed(rname, Xr, yr)
    Xs, ys, _ = make_dataset(n_samples=5000, seed=0); print(f"=== synth {Xs.shape} ===", flush=True)
    allrows += run_bed("synth", Xs, ys)
    Xh, yh, _ = make_hard_dataset(n_samples=5000, seed=0); print(f"=== hard_synth {Xh.shape} ===", flush=True)
    allrows += run_bed("hard_synth", Xh, yh)
    df = pd.DataFrame(allrows)
    print("\n=== ALL ===\n" + df.to_string(index=False))
    print("\n=== verdict (vs mrmr_fe per bed) ===")
    for bed in df.bed.unique():
        b = df[df.bed == bed].set_index("variant")
        base = float(b.loc["mrmr_fe", "auc_mean"])
        for v in ("mrmr_fe+rescue15", "mrmr_fe+rescue25"):
            if v in b.index:
                print(f"  {bed:12s} {v:18s} {b.loc[v,'auc_mean']} (d={round(float(b.loc[v,'auc_mean'])-base,4):+})")
    df.to_csv("D:/Temp/round4_mrmr_rescue_rows.csv", index=False)


if __name__ == "__main__":
    main()
