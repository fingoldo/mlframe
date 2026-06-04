"""Multi-seed confirm of the MRMR tree-importance rescue on madelon (the under-selection regime).

Single-seed showed mrmr_fe+rescue +0.10 on madelon (collapse 0.70 -> 0.80), byte-identical no-op on synth/hard_synth
(gate doesn't fire). madelon has high seed variance, so confirm the rescue win across 3 split-seeds before integrating.
"""
from __future__ import annotations
import os, sys, time, math
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import load_real, downstream
from round4_tree_seed_bench import shallow_tree_signals
from round4_mrmr_tree_rescue_bench import mrmr_raw_selected


def main():
    X, y, name = load_real()
    print(f"REAL: {name} {X.shape}", flush=True)
    rows = []
    for sd in (0, 1, 2):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
        p = Xtr.shape[1]
        T, raw, _ = mrmr_raw_selected(Xtr, ytr)
        under = len(raw) < max(5, math.ceil(0.04 * p)) and p > 60
        ranked, _ = shallow_tree_signals(Xtr, ytr)

        def score(extra):
            Ztr, Zte = T.transform(Xtr).reset_index(drop=True), T.transform(Xte).reset_index(drop=True)
            if extra:
                ex = [c for c in extra if c in Xtr.columns and c not in Ztr.columns]
                Ztr = pd.concat([Ztr, Xtr[ex].reset_index(drop=True)], axis=1)
                Zte = pd.concat([Zte, Xte[ex].reset_index(drop=True)], axis=1)
            a = downstream(Ztr, Zte, ytr, yte)
            return round(float(np.nanmean(list(a.values()))), 4), int(Ztr.shape[1])

        base, nb = score(None)
        resc, nr = score([c for c in ranked[:20] if c in Xtr.columns] if under else None)
        rows.append(dict(seed=sd, raw=len(raw), under=under, mrmr_fe=base, n_base=nb, rescue=resc, n_resc=nr, d=round(resc-base, 4)))
        print(f"  sd{sd} raw={len(raw)} under={under} mrmr_fe={base}(n{nb}) +rescue20={resc}(n{nr}) d={resc-base:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    print("\n=== mean over seeds ===")
    print(f"  mrmr_fe={df.mrmr_fe.mean():.4f}  +rescue20={df.rescue.mean():.4f}  d={df.d.mean():+.4f} (std {df.d.std():.4f})")
    print("VERDICT:", "ROBUST rescue win" if df.d.mean() > 0.04 else "weak / KILL")


if __name__ == "__main__":
    main()
