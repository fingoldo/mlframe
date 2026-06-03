"""Round-4 idea #1 (the ONLY genuinely-unmeasured round-3 idea): JMIM redundancy aggregator.

Round-3 marked JMIM "njit-gated, untested". That was WRONG: the thread-local set by
MRMR(redundancy_aggregator='jmim') (mrmr.py:2570) is read at the PYTHON level in
evaluate_candidate (evaluation.py:631) and PASSED as the njit arg use_jmim into evaluate_gain
(evaluation.py:336/411) -- the njit kernel receives the bool, it does not read the thread-local.
So JMIM is fully wired and benchable with ZERO code change.

HYPOTHESIS: MRMR's CMIM conditional-min stop (current default) discards synergy -- on madelon it
collapsed to 3 features (lgbm 0.69 vs all-features 0.87). JMIM scores a candidate by its best JOINT
relevance I({X,Z};Y) with an already-selected feature, so an interaction operand whose marginal ~0
but whose PAIR with a selected feature carries signal gets a high score.

FALSIFIABLE TEST: on madelon, does redundancy_aggregator='jmim' select >3 features AND lift AUC over
the 0.69 baseline -- WITHOUT readmitting the whole noise pool (JMIM's known multi-collinear failure)?
Also guarded on synth (must not regress the FE-saturated easy bed) + hard_synth (split-signal).
KILL: still <=3 on madelon, OR readmits noise (AUC drops like the rescue), OR regresses synth.
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
import re

_SAFE = re.compile(r"^[A-Za-z0-9_]+$")


def mrmr_fe(X, y, fe_strict=False, jmim=False):
    """Fit MRMR+FE; return (selected_df_transformer, n_selected, raw_selected_count)."""
    from mlframe.feature_selection.filters import MRMR
    kw = {}
    if fe_strict:
        kw.update(fe_synergy_min_prevalence=1.5, fe_min_engineered_mi_prevalence=0.97)
    if jmim:
        kw.update(redundancy_aggregator="jmim")
    m = MRMR(verbose=0, fe_max_steps=1, n_jobs=-1, random_seed=0, **kw)
    m.fit(X, y)
    out = list(m.transform(X.iloc[:5]).columns)
    ren = {c: (c if _SAFE.match(str(c)) else f"eng_{i}") for i, c in enumerate(out)}

    class _T:
        def transform(self, Z):
            df = m.transform(Z).copy(); df.columns = [ren[c] for c in df.columns]; return df
    return _T(), len(out)


def run_bed(name, X, y, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    rows = []
    variants = {
        "all": None,
        "mrmr_fe_default": dict(fe_strict=False, jmim=False),
        "mrmr_fe_strict": dict(fe_strict=True, jmim=False),
        "mrmr_fe_jmim": dict(fe_strict=False, jmim=True),
        "mrmr_fe_strict_jmim": dict(fe_strict=True, jmim=True),
    }
    for nm, kw in variants.items():
        t0 = time.time()
        if kw is None:
            Ztr, Zte, n = Xtr, Xte, Xtr.shape[1]
        else:
            T, n = mrmr_fe(Xtr, ytr, **kw)
            Ztr, Zte = T.transform(Xtr), T.transform(Xte)
        a = downstream(Ztr, Zte, ytr, yte); am = round(float(np.nanmean(list(a.values()))), 4)
        rows.append(dict(bed=name, variant=nm, n=int(n), fit_s=round(time.time()-t0, 1), auc_mean=am, **a))
        print(f"[{name}] {nm:22s} n={int(n):3d} {rows[-1]['fit_s']:6.1f}s mean={am} {a}", flush=True)
    return rows


def main():
    allrows = []
    # 1) REAL madelon -- the headline collapse-to-3 failure mode
    Xr, yr, rname = load_real()
    print(f"\n=== REAL: {rname} shape={Xr.shape} pos={round(float(yr.mean()),3)} ===", flush=True)
    allrows += run_bed(rname, Xr, yr, seed=0)
    # 2) synth (FE-saturated easy bed) -- guard against regression
    Xs, ys, _ = make_dataset(n_samples=5000, seed=0)
    print(f"\n=== synth shape={Xs.shape} ===", flush=True)
    allrows += run_bed("synth", Xs, ys, seed=0)
    # 3) hard_synth (split-signal)
    Xh, yh, _ = make_hard_dataset(n_samples=5000, seed=0)
    print(f"\n=== hard_synth shape={Xh.shape} ===", flush=True)
    allrows += run_bed("hard_synth", Xh, yh, seed=0)

    df = pd.DataFrame(allrows)
    print("\n=== ALL ===")
    print(df.to_string(index=False))
    print("\n=== verdict (madelon) ===")
    md = df[df.bed == rname].set_index("variant")
    base = md.loc["mrmr_fe_default"]
    for v in ("mrmr_fe_jmim", "mrmr_fe_strict_jmim"):
        if v in md.index:
            r = md.loc[v]
            d = round(float(r.auc_mean - base.auc_mean), 4)
            print(f"  {v}: n {int(base.n)}->{int(r.n)}  auc {base.auc_mean}->{r.auc_mean} (delta {d:+})  "
                  f"{'RECOVERS (>3 + AUC up)' if r.n > base.n and d > 0 else 'no recovery / KILL'}")


if __name__ == "__main__":
    main()
