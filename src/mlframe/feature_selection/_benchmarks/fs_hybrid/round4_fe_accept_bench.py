"""Round-4 FE-acceptance ideas A3-3 (holdout_greedy_FE) + A3-6 (fe_residual_target).

Setup (EXTERNAL, no production edit): per bed, one cheap shallow GBM proposes co-occurrence PAIRS
(features the tree branches on together). We engineer raw[a]*raw[b] product columns from a GENEROUS pool of
those pairs, then ask: which ACCEPTANCE RULE for keeping products beats keeping all of them?

Base feature set is ALWAYS tree_top25 raw. The variable is WHICH engineered products to append:
  (1) keep_all          -- append every candidate product (the round4_tree_seed reference behaviour).
  (2) holdout_greedy    -- forward-select products one at a time by HELD-OUT incremental AUC gain on an
                           inner honest split; stop when best remaining gain < floor (A3-3).
  (3) resid_mi_topk     -- score each candidate product by MI with the RESIDUAL (y - OOF p) of a model on
                           the raw selected set; keep top-k (A3-6: interaction signal lives in the residual).
  (4) rawy_mi_topk      -- score each candidate product by MI with raw y; keep top-k (the naive control).

Downstream AUC = mean over {lgbm, logit, knn} of [tree_top25 raw + kept products].

VERDICT: does holdout_greedy or resid_mi_topk beat keep_all by >= +0.005 on a bed? Report all numbers.
Plain negative is fine.

Memory-frugal (heavy concurrent load): n_jobs=4, n_estimators<=200, synth/hard_synth primary, madelon final
stress only (artifact cached). stdout -> file. ASCII-only prints (cp1251).
"""
from __future__ import annotations
import os, sys, time, json
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from itertools import combinations
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import load_real, downstream
from synth import make_dataset
from hard_synth import make_hard_dataset

NJ = 4                  # n_jobs cap under concurrent load
PROGRESS = "D:/Temp/fe_accept_progress.txt"


def checkpoint(msg):
    with open(PROGRESS, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(f"  [ckpt] {msg}", flush=True)


def shallow_tree_signals(X, y, n_estimators=120, max_depth=3, top_pairs=30, seed=0):
    """One cheap depth-limited GBM -> (importance-ranked feature list, top co-occurring raw pairs).

    GENEROUS top_pairs (30) so acceptance rules have a real candidate pool to prune (the whole point: if
    keep-all of a generous pool dilutes, a good acceptance rule should recover by pruning).
    """
    m = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, num_leaves=2 ** max_depth,
                           learning_rate=0.1, n_jobs=NJ, verbose=-1, random_state=seed)
    m.fit(X, y)
    cols = list(X.columns)
    imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
    ranked = [c for c in imp.index if imp[c] > 0]
    tdf = m.booster_.trees_to_dataframe()
    tdf = tdf[tdf["split_feature"].notna()]
    pair_w = Counter()
    for _, g in tdf.groupby("tree_index"):
        feats = sorted(set(g["split_feature"].tolist()))
        gain = float(g["split_gain"].sum()) + 1e-9
        for a, b in combinations(feats, 2):
            pair_w[(a, b)] += gain
    top = [p for p, _ in pair_w.most_common(top_pairs)]
    return ranked, top


def build_products(X, pairs):
    """Return dict name -> product array for the given (a,b) pairs present in X."""
    prods = {}
    for i, (a, b) in enumerate(pairs):
        if a in X.columns and b in X.columns:
            prods[f"prod_{i}"] = (X[a].values * X[b].values).astype(np.float64)
    return prods


def _auc_cv(Xmat, y, n_estimators=150, seed=0):
    """Honest OOF AUC of a small lgbm on a numeric matrix (used inside the greedy inner loop)."""
    m = lgb.LGBMClassifier(n_estimators=n_estimators, num_leaves=31, learning_rate=0.05,
                           n_jobs=NJ, verbose=-1, random_state=seed)
    p = cross_val_predict(m, Xmat, y, cv=3, method="predict_proba", n_jobs=1)[:, 1]
    return roc_auc_score(y, p)


def holdout_greedy_accept(base_df, prod_dict, y, floor=0.0015, seed=0):
    """Forward-select products by HELD-OUT incremental AUC gain on an inner honest split.

    Inner split of the TRAIN data only -> fit on inner-train, score gain on inner-val. Add the best product
    each round; stop when the best remaining held-out gain < floor. Returns the accepted product names.
    base_df / prod_dict are positionally aligned (RangeIndex 0..n-1), so we split on row POSITIONS.
    """
    if not prod_dict:
        return []
    n = len(base_df)
    tr_pos, va_pos = train_test_split(
        np.arange(n), test_size=0.4, random_state=seed, stratify=y.values)
    base_mat = base_df.values.astype(np.float64)
    base_tr, base_va = base_mat[tr_pos], base_mat[va_pos]
    yi_tr, yi_va = y.values[tr_pos], y.values[va_pos]
    Ptr = {k: v[tr_pos] for k, v in prod_dict.items()}
    Pva = {k: v[va_pos] for k, v in prod_dict.items()}

    def fit_auc(extra_tr, extra_va):
        Mtr = np.hstack([base_tr] + extra_tr) if extra_tr else base_tr
        Mva = np.hstack([base_va] + extra_va) if extra_va else base_va
        m = lgb.LGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.05,
                               n_jobs=NJ, verbose=-1, random_state=seed)
        m.fit(Mtr, yi_tr)
        return roc_auc_score(yi_va, m.predict_proba(Mva)[:, 1])

    accepted, acc_tr, acc_va = [], [], []
    cur = fit_auc([], [])
    remaining = list(prod_dict.keys())
    while remaining:
        best_name, best_auc = None, cur
        for name in remaining:
            a = fit_auc(acc_tr + [Ptr[name].reshape(-1, 1)], acc_va + [Pva[name].reshape(-1, 1)])
            if a > best_auc:
                best_auc, best_name = a, name
        if best_name is None or (best_auc - cur) < floor:
            break
        accepted.append(best_name)
        acc_tr.append(Ptr[best_name].reshape(-1, 1)); acc_va.append(Pva[best_name].reshape(-1, 1))
        cur = best_auc
        remaining.remove(best_name)
    return accepted


def resid_mi_rank(base_df, prod_dict, y, seed=0):
    """Rank products by MI with the OOF residual of a model on the raw base set (A3-6)."""
    if not prod_dict:
        return []
    m = lgb.LGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.05,
                           n_jobs=NJ, verbose=-1, random_state=seed)
    p_oof = cross_val_predict(m, base_df.values, y, cv=4, method="predict_proba", n_jobs=1)[:, 1]
    resid = y.values.astype(float) - p_oof
    rbin = (resid > np.median(resid)).astype(int)
    names = list(prod_dict.keys())
    Pmat = np.column_stack([prod_dict[k] for k in names])
    mi = mutual_info_classif(Pmat, rbin, random_state=seed)
    return [names[j] for j in np.argsort(mi)[::-1]]


def rawy_mi_rank(prod_dict, y, seed=0):
    """Rank products by MI with raw y (the naive control)."""
    if not prod_dict:
        return []
    names = list(prod_dict.keys())
    Pmat = np.column_stack([prod_dict[k] for k in names])
    mi = mutual_info_classif(Pmat, y.values, random_state=seed)
    return [names[j] for j in np.argsort(mi)[::-1]]


def run_bed(name, X, y, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    # IMPORTANT: reset to clean RangeIndex so position<->index mapping is unambiguous in the greedy loop
    Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
    ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)
    rows = []

    def emit(tag, base_cols, prod_names, prod_tr, prod_te, t0):
        Ztr = Xtr[base_cols].copy(); Zte = Xte[base_cols].copy()
        for pn in prod_names:
            Ztr[pn] = prod_tr[pn]; Zte[pn] = prod_te[pn]
        a = downstream(Ztr, Zte, ytr, yte); am = round(float(np.nanmean(list(a.values()))), 4)
        rows.append(dict(bed=name, variant=tag, n_base=len(base_cols), n_prod=len(prod_names),
                         n=int(Ztr.shape[1]), fit_s=round(time.time() - t0, 1), auc_mean=am, **a))
        print(f"[{name}] {tag:22s} base={len(base_cols):2d} prod={len(prod_names):2d} "
              f"n={int(Ztr.shape[1]):3d} {rows[-1]['fit_s']:6.1f}s mean={am} {a}", flush=True)
        return am

    # shallow-tree signals (one fit, shared across all variants)
    t0 = time.time(); ranked, pairs = shallow_tree_signals(Xtr, ytr, seed=seed)
    print(f"[{name}] shallow-tree: {len(ranked)} nonzero-imp feats, {len(pairs)} candidate pairs, "
          f"top {pairs[:4]}  ({round(time.time()-t0,1)}s)", flush=True)
    base = ranked[:25] if len(ranked) >= 1 else list(Xtr.columns)[:25]

    # build all candidate products once (train + test)
    prod_tr = build_products(Xtr, pairs)
    prod_te = build_products(Xte, pairs)
    cand = list(prod_tr.keys())
    print(f"[{name}] {len(cand)} candidate products engineered from pairs", flush=True)

    # base-only reference (no products) so we can see the FE lift at all
    t0 = time.time(); emit("base_only", base, [], prod_tr, prod_te, t0)

    # (1) keep_all
    t0 = time.time(); auc_keepall = emit("keep_all", base, cand, prod_tr, prod_te, t0)
    checkpoint(f"{name} seed{seed} keep_all done")

    # base_df for ranking/greedy (raw selected set as a frame with RangeIndex)
    base_df = Xtr[base].reset_index(drop=True)

    # (2) holdout_greedy
    t0 = time.time()
    acc = holdout_greedy_accept(base_df, prod_tr, ytr, floor=0.0015, seed=seed)
    print(f"[{name}] holdout_greedy accepted {len(acc)}/{len(cand)}: {acc[:8]}", flush=True)
    emit("holdout_greedy", base, acc, prod_tr, prod_te, t0)
    checkpoint(f"{name} seed{seed} holdout_greedy done (kept {len(acc)})")

    # (3) resid_mi_topk
    t0 = time.time()
    resid_order = resid_mi_rank(base_df, prod_tr, ytr, seed=seed)
    print(f"[{name}] resid-MI order top6: {resid_order[:6]}", flush=True)
    for k in (5, 10):
        kk = min(k, len(resid_order))
        emit(f"resid_mi_top{k}", base, resid_order[:kk], prod_tr, prod_te, t0); t0 = time.time()
    checkpoint(f"{name} seed{seed} resid_mi done")

    # (4) rawy_mi_topk
    t0 = time.time()
    rawy_order = rawy_mi_rank(prod_tr, ytr, seed=seed)
    print(f"[{name}] rawy-MI order top6: {rawy_order[:6]}", flush=True)
    for k in (5, 10):
        kk = min(k, len(rawy_order))
        emit(f"rawy_mi_top{k}", base, rawy_order[:kk], prod_tr, prod_te, t0); t0 = time.time()
    checkpoint(f"{name} seed{seed} rawy_mi done")

    return rows


def verdict(df):
    lines = ["", "=== VERDICTS (delta vs keep_all; >=+0.005 = acceptance rule WINS) ==="]
    for bed in df.bed.unique():
        b = df[df.bed == bed]
        ka = b[b.variant == "keep_all"]["auc_mean"].mean()
        bo = b[b.variant == "base_only"]["auc_mean"].mean()
        lines.append(f"  {bed:14s} base_only={round(bo,4)} keep_all={round(ka,4)} (FE lift {round(ka-bo,4):+})")
        for v in ["holdout_greedy", "resid_mi_top5", "resid_mi_top10", "rawy_mi_top5", "rawy_mi_top10"]:
            sub = b[b.variant == v]
            if len(sub):
                m = sub["auc_mean"].mean()
                win = " <== WIN" if (m - ka) >= 0.005 else ""
                lines.append(f"      {v:18s} {round(m,4)} (d vs keep_all {round(m-ka,4):+}){win}")
    return "\n".join(lines)


def main():
    open(PROGRESS, "w").close()
    checkpoint("START fe_accept bench")
    seeds = (0, 1)
    allrows = []

    # PRIMARY light beds first
    for seed in seeds:
        Xs, ys, _ = make_dataset(n_samples=5000, seed=seed)
        print(f"\n=== synth seed={seed} shape={Xs.shape} ===", flush=True)
        allrows += run_bed("synth", Xs, ys, seed=seed)
        checkpoint(f"synth seed{seed} bed DONE")
    for seed in seeds:
        Xh, yh, _ = make_hard_dataset(n_samples=5000, seed=seed)
        print(f"\n=== hard_synth seed={seed} shape={Xh.shape} ===", flush=True)
        allrows += run_bed("hard_synth", Xh, yh, seed=seed)
        checkpoint(f"hard_synth seed{seed} bed DONE")

    df = pd.DataFrame(allrows)
    df.to_csv("D:/Temp/fe_accept_light_rows.csv", index=False)
    print("\n=== ALL (light beds) ===")
    print(df.to_string(index=False))
    print(verdict(df))

    # FINAL stress check on madelon (real). Cache the loaded dataset artifact.
    do_madelon = os.environ.get("FE_ACCEPT_SKIP_MADELON", "0") != "1"
    if do_madelon:
        try:
            checkpoint("loading madelon (cached if available)")
            Xr, yr, rname = load_real()
            print(f"\n=== REAL stress: {rname} shape={Xr.shape} ===", flush=True)
            mrows = run_bed(rname, Xr, yr, seed=0)
            checkpoint(f"{rname} seed0 bed DONE")
            allrows += mrows
            df = pd.DataFrame(allrows)
            df.to_csv("D:/Temp/fe_accept_all_rows.csv", index=False)
            print("\n=== madelon rows ===")
            print(pd.DataFrame(mrows).to_string(index=False))
            print(verdict(df))
        except Exception as e:
            print(f"  (madelon stress skipped: {type(e).__name__}: {e})", flush=True)

    # write results markdown
    out = ["# FE-acceptance bench (A3-3 holdout_greedy + A3-6 resid_mi) results", ""]
    out.append("Base feature set = tree_top25 raw. Variable = which engineered co-occurrence products to keep.")
    out.append("Downstream AUC = mean over {lgbm, logit, knn}. 2 seeds on light beds; madelon seed0 stress.")
    out.append("")
    out.append("```")
    out.append(pd.DataFrame(allrows).to_string(index=False))
    out.append("```")
    out.append(verdict(pd.DataFrame(allrows)))
    with open("D:/Temp/fe_accept_results.md", "w") as f:
        f.write("\n".join(out))
    checkpoint("DONE all beds; results written")


if __name__ == "__main__":
    main()
