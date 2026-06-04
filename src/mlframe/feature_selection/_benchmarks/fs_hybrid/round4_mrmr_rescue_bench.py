"""Round-4: can MRMR-STANDALONE be rescued on madelon WITHOUT the hybrid, via ctor kwargs / wrapping?

MEASURED context: MRMR-standalone collapses to ~3 features on madelon (downstream lgbm ~0.69 vs
all-features ~0.87) because (a) its synergy bootstrap is SKIPPED on wide frames (madelon p=500 >
cap fe_synergy_screen_max_features=60) so it never engineers the interaction products, and (b) the
interaction operands have ~0 marginal MI so the greedy never selects them.

Three rescue routes, all WITHOUT editing the (concurrently-owned) MRMR source -- only ctor kwargs /
external wrapping:

  1. BASELINE    -- mrmr_fe (default kwargs) on each bed. Confirm the collapse on madelon.
  2. RAISE CAP   -- MRMR(fe_synergy_screen_max_features=N) so the synergy bootstrap RUNS on madelon's
                    500 cols. Tested at N in {150, 250, 600}. Watch O(p^2) cost (124750 pairs at 500;
                    note _MRMR_BATCH_PRECOMPUTE_MAX_K=200 -> pool>200 falls to the slow legacy path).
  3. TREE-SEED   -- shallow_tree_signals -> top co-occurrence pairs -> engineer raw[a]*raw[b], CONCAT
                    to X as extra columns, THEN run plain MRMR on the augmented frame. Pre-injecting
                    the tree-co-occurrence products gives the greedy operands with REAL marginal MI.

Measured on madelon (load_real) + synth (FE-saturated guard) + hard_synth (split-signal). Reports
n_selected + per-model AUC + fit time for every variant/bed.

VERDICT goal: which (if any) lifts madelon >3 feats toward ~0.84 WITHOUT regressing synth, and is
raising fe_synergy_screen_max_features a safe default-raise or does O(p^2)/synth forbid it?
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import re
from itertools import combinations
from collections import Counter
from sklearn.model_selection import train_test_split
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import load_real, downstream
from synth import make_dataset
from hard_synth import make_hard_dataset

_SAFE = re.compile(r"^[A-Za-z0-9_]+$")
PROGRESS = r"D:\Temp\mrmr_fix_progress.txt"
N_JOBS = 4  # cap -- machine under concurrent load


def ckpt(msg: str):
    with open(PROGRESS, "a") as f:
        f.write(msg.rstrip() + "\n")
    print("CKPT " + msg, flush=True)


def build_mrmr(**extra):
    """Build a fitted-on-demand MRMR+FE wrapper. extra -> MRMR ctor kwargs."""
    from mlframe.feature_selection.filters import MRMR

    base_kw = dict(verbose=0, fe_max_steps=1, n_jobs=N_JOBS, random_seed=0)
    base_kw.update(extra)  # extra (incl. fe_max_steps) overrides defaults

    class _Sel:
        def fit(self, X, y):
            self.m_ = MRMR(**base_kw)
            self.m_.fit(X, y)
            out = list(self.m_.transform(X.iloc[:5]).columns)
            self.ren_ = {c: (c if _SAFE.match(str(c)) else f"eng_{i}") for i, c in enumerate(out)}
            # count how many SELECTED outputs are raw vs engineered
            self.n_raw_ = sum(1 for c in out if _SAFE.match(str(c)) and c in X.columns)
            self.n_eng_ = len(out) - self.n_raw_
            return self

        def transform(self, X):
            df = self.m_.transform(X).copy()
            df.columns = [self.ren_[c] for c in df.columns]
            return df

    return _Sel()


def shallow_tree_signals(X, y, n_estimators=80, max_depth=3, top_pairs=12):
    """One cheap depth-limited GBM -> (importance-ranked feature list, top co-occurring raw pairs)."""
    m = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, num_leaves=2 ** max_depth,
                           learning_rate=0.1, n_jobs=N_JOBS, verbose=-1, random_state=0)
    m.fit(X, y)
    cols = list(X.columns)
    imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
    ranked = [c for c in imp.index if imp[c] > 0]
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
    """Return a NEW df with raw[a]*raw[b] product columns appended (named tprod_*)."""
    new = {}
    for i, (a, b) in enumerate(pairs):
        if a in X.columns and b in X.columns:
            new[f"tprod_{i}"] = (X[a].values * X[b].values)
    if not new:
        return X.copy()
    return pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1)


def emit(rows, name, tag, Ztr, Zte, ytr, yte, t0, extra=""):
    a = downstream(Ztr, Zte, ytr, yte)
    am = round(float(np.nanmean(list(a.values()))), 4)
    n = int(Ztr.shape[1])
    rows.append(dict(bed=name, variant=tag, n=n, fit_s=round(time.time() - t0, 1), auc_mean=am, **a))
    print(f"[{name}] {tag:30s} n={n:4d} {rows[-1]['fit_s']:7.1f}s mean={am} {a} {extra}", flush=True)


def run_bed(name, X, y, cap_grid, seed=0):
    ckpt(f"bed={name} shape={X.shape} start")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    p = Xtr.shape[1]
    rows = []

    # 0) all features
    t0 = time.time(); emit(rows, name, "all", Xtr, Xte, ytr, yte, t0)

    # 1) baseline mrmr_fe (default cap=60)
    t0 = time.time()
    sel = build_mrmr().fit(Xtr, ytr)
    emit(rows, name, "mrmr_fe_default", sel.transform(Xtr), sel.transform(Xte), ytr, yte, t0,
         extra=f"(raw={sel.n_raw_} eng={sel.n_eng_})")
    ckpt(f"bed={name} mrmr_fe_default done n_raw={sel.n_raw_} n_eng={sel.n_eng_}")

    # 2) raise the synergy cap so the bootstrap runs on wide frames
    for cap in cap_grid:
        if cap <= p:
            # cap below p -> bootstrap STILL skipped; only run caps that actually enable it (>p) plus
            # exactly one informative "still-skipped" sanity if desired. We only run caps >= p here.
            continue
        t0 = time.time()
        try:
            sel = build_mrmr(fe_synergy_screen_max_features=cap).fit(Xtr, ytr)
            emit(rows, name, f"mrmr_cap{cap}", sel.transform(Xtr), sel.transform(Xte), ytr, yte, t0,
                 extra=f"(raw={sel.n_raw_} eng={sel.n_eng_})")
            ckpt(f"bed={name} mrmr_cap{cap} done n_raw={sel.n_raw_} n_eng={sel.n_eng_} t={round(time.time()-t0,1)}s")
        except Exception as e:
            print(f"[{name}] mrmr_cap{cap} FAILED: {type(e).__name__}: {e}", flush=True)
            ckpt(f"bed={name} mrmr_cap{cap} FAILED {type(e).__name__}")

    # 3) tree-seed FE: engineer top tree co-occurrence products, concat, then plain MRMR
    t0 = time.time()
    ranked, pairs = shallow_tree_signals(Xtr, ytr)
    tree_s = round(time.time() - t0, 1)
    print(f"[{name}] shallow-tree: {len(ranked)} nonzero-imp feats; top pairs {pairs[:5]} ({tree_s}s)", flush=True)
    Xtr_aug = engineer_products(Xtr, pairs)
    Xte_aug = engineer_products(Xte, pairs)
    n_added = Xtr_aug.shape[1] - Xtr.shape[1]

    # 3a) plain MRMR (FE off) on the tree-augmented frame -- pure SELECTION over raw + tree-products
    t0 = time.time()
    sel = build_mrmr(fe_max_steps=0).fit(Xtr_aug, ytr)
    emit(rows, name, "treeseed+mrmr_nofe", sel.transform(Xtr_aug), sel.transform(Xte_aug), ytr, yte, t0,
         extra=f"(+{n_added} tprods, raw={sel.n_raw_} eng={sel.n_eng_})")
    ckpt(f"bed={name} treeseed+mrmr_nofe done n_raw={sel.n_raw_}")

    # 3b) MRMR-FE (default) on the tree-augmented frame -- selection + native FE on top of tree-products
    t0 = time.time()
    sel = build_mrmr().fit(Xtr_aug, ytr)
    emit(rows, name, "treeseed+mrmr_fe", sel.transform(Xtr_aug), sel.transform(Xte_aug), ytr, yte, t0,
         extra=f"(+{n_added} tprods, raw={sel.n_raw_} eng={sel.n_eng_})")
    ckpt(f"bed={name} treeseed+mrmr_fe done n_raw={sel.n_raw_}")

    # 3c) reference: lgbm on [tree_top25 + tree-products] (the standalone non-MRMR baseline ~0.84)
    base = ranked[:25]
    t0 = time.time()
    Ztr = engineer_products(Xtr[base], pairs); Zte = engineer_products(Xte[base], pairs)
    emit(rows, name, "tree_top25+cooccur_ref", Ztr, Zte, ytr, yte, t0)

    return rows


def main():
    allrows = []

    # 1) REAL madelon -- the headline collapse-to-3 failure mode. p=500.
    Xr, yr, rname = load_real()
    print(f"\n=== REAL: {rname} shape={Xr.shape} pos={round(float(yr.mean()),3)} ===", flush=True)
    allrows += run_bed(rname, Xr, yr, cap_grid=(150, 250, 600), seed=0)

    # 2) synth (FE-saturated easy bed) -- guard against regression. p~67.
    Xs, ys, _ = make_dataset(n_samples=5000, seed=0)
    print(f"\n=== synth shape={Xs.shape} ===", flush=True)
    allrows += run_bed("synth", Xs, ys, cap_grid=(150, 600), seed=0)

    # 3) hard_synth (split-signal). p~222.
    Xh, yh, _ = make_hard_dataset(n_samples=5000, seed=0)
    print(f"\n=== hard_synth shape={Xh.shape} ===", flush=True)
    allrows += run_bed("hard_synth", Xh, yh, cap_grid=(250, 600), seed=0)

    df = pd.DataFrame(allrows)
    print("\n=== ALL ===")
    print(df.to_string(index=False))

    print("\n=== verdicts ===")
    for bed in df.bed.unique():
        b = df[df.bed == bed].set_index("variant")
        base_auc = float(b.loc["mrmr_fe_default", "auc_mean"]) if "mrmr_fe_default" in b.index else float("nan")
        base_n = int(b.loc["mrmr_fe_default", "n"]) if "mrmr_fe_default" in b.index else -1
        allf = float(b.loc["all", "auc_mean"]) if "all" in b.index else float("nan")
        print(f"\n  [{bed}] all={allf}  mrmr_fe_default n={base_n} auc={base_auc}")
        for v in b.index:
            if v in ("all", "mrmr_fe_default"):
                continue
            r = b.loc[v]
            d = round(float(r.auc_mean) - base_auc, 4)
            flag = "RECOVERS" if (int(r.n) > base_n and d > 0.02) else ("up" if d > 0 else "")
            print(f"     {v:28s} n={int(r.n):4d} auc={r.auc_mean} (d vs base {d:+.4f}) {flag}")

    out_csv = r"D:\Temp\mrmr_fix_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
