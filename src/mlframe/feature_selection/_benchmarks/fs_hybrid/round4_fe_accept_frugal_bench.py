"""Round-4 FE-acceptance ideas A3-3 (holdout_greedy_FE) + A3-6 (fe_residual_target) -- MEMORY-FRUGAL.

Same question as round4_fe_accept_bench.py, rebuilt to survive HEAVY concurrent load (a sibling rich-tree
madelon bench was eating RAM -> the original died with a LightGBM model_to_string MemoryError inside the
greedy inner loop, which refit a full LightGBM O(candidates^2) times).

Setup (EXTERNAL, no production edit): per bed, one cheap shallow GBM proposes co-occurrence PAIRS (features the
tree branches on together). We engineer raw[a]*raw[b] product columns from a GENEROUS pool of those pairs, then
ask: which ACCEPTANCE RULE for keeping products beats keeping all of them?

Base feature set is ALWAYS tree_top25 raw. The variable is WHICH engineered products to append:
  (1) keep_all  -- append every candidate product (the round4_tree_seed reference behaviour).
  (2) holdout_greedy  -- forward-select products one at a time by HELD-OUT incremental gain on an inner honest
                           split; stop when best remaining gain < floor (A3-3). Inner gain scored with a CHEAP
                           standardized logistic regression (NOT a full LightGBM) -> no big-model allocation,
                           O(cand^2) cheap fits. The product-pruning decision is what we test; the downstream
                           AUC of the kept set is still judged by the full {lgbm,logit,knn} panel.
  (3) resid_mi_topk  -- score each candidate product by MI with the RESIDUAL (y - OOF p) of a cheap model on
                           the raw selected set; keep top-k (A3-6: interaction signal lives in the residual).
  (4) rawy_mi_topk  -- score each candidate product by MI with raw y; keep top-k (the naive control).

Downstream AUC = mean over {lgbm, logit, knn} of [tree_top25 raw + kept products].

VERDICT: does holdout_greedy or resid_mi_topk beat keep_all by >= +0.005 on a bed (cleaner set at >= AUC, or
higher AUC)? Report all numbers. Plain negative is fine.

Memory-frugal: n_jobs=2, n_estimators<=200, free boosters, gc between beds, single-thread MI; synth/hard_synth
primary, madelon final stress only (cached + env-skippable). stdout -> file. ASCII-only prints (cp1251).
On OOM the outer driver sleeps 90s and retries once.
"""
from __future__ import annotations
import os, sys, time, gc
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from itertools import combinations
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import load_real, downstream
from synth import make_dataset
from hard_synth import make_hard_dataset

NJ = 2  # n_jobs cap under heavy concurrent load
PROGRESS = "D:/Temp/fe_accept_progress.txt"
GREEDY_FLOOR = 0.0015  # held-out AUC gain floor to accept another product
TOPK = (5, 10)


def checkpoint(msg):
    with open(PROGRESS, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(f"  [ckpt] {msg}", flush=True)


def shallow_tree_signals(X, y, n_estimators=120, max_depth=3, top_pairs=30, seed=0):
    """One cheap depth-limited GBM -> (importance-ranked feature list, top co-occurring raw pairs).

    GENEROUS top_pairs (30) so acceptance rules have a real candidate pool to prune (the whole point: if
    keep-all of a generous pool dilutes, a good acceptance rule should recover by pruning).
    """
    m = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, num_leaves=2**max_depth, learning_rate=0.1, n_jobs=NJ, verbose=-1, random_state=seed)
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
    del m; gc.collect()
    return ranked, top


def build_products(X, pairs):
    """Return dict name -> product array for the given (a,b) pairs present in X."""
    prods = {}
    for i, (a, b) in enumerate(pairs):
        if a in X.columns and b in X.columns:
            prods[f"prod_{i}"] = (X[a].values * X[b].values).astype(np.float64)
    return prods


def holdout_greedy_accept(base_mat, prod_dict, y, floor=GREEDY_FLOOR, seed=0):
    """Forward-select products by HELD-OUT incremental AUC gain on an inner honest split.

    The inner gain is scored with a CHEAP standardized logistic regression (not a full LightGBM): we are
    measuring whether each candidate product adds OUT-OF-SAMPLE signal beyond the already-accepted set, and a
    linear probe on standardized columns is a low-memory, fast, monotone proxy for that incremental gain. This
    sidesteps the LightGBM model_to_string allocation that OOM'd under concurrent load. The FINAL downstream AUC
    of the chosen product set is still judged by the full {lgbm,logit,knn} panel in emit(), so the comparison
    against keep_all is apples-to-apples on the same scoring panel.

    base_mat: (n, p) float64 raw selected matrix (RangeIndex aligned). prod_dict: name -> (n,) product array.
    Returns the accepted product names in acceptance order.
    """
    if not prod_dict:
        return []
    n = base_mat.shape[0]
    tr_pos, va_pos = train_test_split(np.arange(n), test_size=0.4, random_state=seed, stratify=y)
    yi_tr, yi_va = y[tr_pos], y[va_pos]
    base_tr, base_va = base_mat[tr_pos], base_mat[va_pos]
    Ptr = {k: v[tr_pos].reshape(-1, 1) for k, v in prod_dict.items()}
    Pva = {k: v[va_pos].reshape(-1, 1) for k, v in prod_dict.items()}

    def probe_auc(extra_tr, extra_va):
        Mtr = np.hstack([base_tr] + extra_tr) if extra_tr else base_tr
        Mva = np.hstack([base_va] + extra_va) if extra_va else base_va
        sc = StandardScaler().fit(Mtr)
        clf = LogisticRegression(max_iter=500, C=1.0, n_jobs=1)
        clf.fit(sc.transform(Mtr), yi_tr)
        return roc_auc_score(yi_va, clf.predict_proba(sc.transform(Mva))[:, 1])

    accepted, acc_tr, acc_va = [], [], []
    cur = probe_auc([], [])
    remaining = list(prod_dict.keys())
    while remaining:
        best_name, best_auc = None, cur
        for name in remaining:
            a = probe_auc(acc_tr + [Ptr[name]], acc_va + [Pva[name]])
            if a > best_auc:
                best_auc, best_name = a, name
        if best_name is None or (best_auc - cur) < floor:
            break
        accepted.append(best_name)
        acc_tr.append(Ptr[best_name]); acc_va.append(Pva[best_name])
        cur = best_auc
        remaining.remove(best_name)
    return accepted


def resid_mi_rank(base_mat, prod_dict, y, seed=0):
    """Rank products by MI with the OOF residual of a cheap model on the raw base set (A3-6)."""
    if not prod_dict:
        return []
    m = lgb.LGBMClassifier(n_estimators=120, num_leaves=31, learning_rate=0.05, n_jobs=NJ, verbose=-1, random_state=seed)
    p_oof = cross_val_predict(m, base_mat, y, cv=4, method="predict_proba", n_jobs=1)[:, 1]
    resid = y.astype(float) - p_oof
    rbin = (resid > np.median(resid)).astype(int)
    names = list(prod_dict.keys())
    Pmat = np.column_stack([prod_dict[k] for k in names])
    mi = mutual_info_classif(Pmat, rbin, random_state=seed)
    del m, Pmat; gc.collect()
    return [names[j] for j in np.argsort(mi)[::-1]]


def rawy_mi_rank(prod_dict, y, seed=0):
    """Rank products by MI with raw y (the naive control)."""
    if not prod_dict:
        return []
    names = list(prod_dict.keys())
    Pmat = np.column_stack([prod_dict[k] for k in names])
    mi = mutual_info_classif(Pmat, y, random_state=seed)
    del Pmat; gc.collect()
    return [names[j] for j in np.argsort(mi)[::-1]]


def run_bed(name, X, y, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
    ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)
    yv = ytr.values
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
        del Ztr, Zte; gc.collect()
        return am

    t0 = time.time(); ranked, pairs = shallow_tree_signals(Xtr, ytr, seed=seed)
    print(f"[{name}] shallow-tree: {len(ranked)} nonzero-imp feats, {len(pairs)} candidate pairs, "
          f"top {pairs[:4]}  ({round(time.time()-t0,1)}s)", flush=True)
    base = ranked[:25] if len(ranked) >= 1 else list(Xtr.columns)[:25]

    prod_tr = build_products(Xtr, pairs)
    prod_te = build_products(Xte, pairs)
    cand = list(prod_tr.keys())
    print(f"[{name}] {len(cand)} candidate products engineered from pairs", flush=True)

    t0 = time.time(); emit("base_only", base, [], prod_tr, prod_te, t0)
    t0 = time.time(); auc_keepall = emit("keep_all", base, cand, prod_tr, prod_te, t0)
    checkpoint(f"{name} seed{seed} keep_all done")

    base_mat = Xtr[base].values.astype(np.float64)

    # (2) holdout_greedy (cheap linear probe inner loop)
    t0 = time.time()
    acc = holdout_greedy_accept(base_mat, prod_tr, yv, floor=GREEDY_FLOOR, seed=seed)
    print(f"[{name}] holdout_greedy accepted {len(acc)}/{len(cand)}: {acc[:8]}", flush=True)
    emit("holdout_greedy", base, acc, prod_tr, prod_te, t0)
    checkpoint(f"{name} seed{seed} holdout_greedy done (kept {len(acc)})")

    # (3) resid_mi_topk
    resid_order = resid_mi_rank(base_mat, prod_tr, yv, seed=seed)
    print(f"[{name}] resid-MI order top6: {resid_order[:6]}", flush=True)
    for k in TOPK:
        kk = min(k, len(resid_order))
        t0 = time.time(); emit(f"resid_mi_top{k}", base, resid_order[:kk], prod_tr, prod_te, t0)
    checkpoint(f"{name} seed{seed} resid_mi done")

    # (4) rawy_mi_topk
    rawy_order = rawy_mi_rank(prod_tr, yv, seed=seed)
    print(f"[{name}] rawy-MI order top6: {rawy_order[:6]}", flush=True)
    for k in TOPK:
        kk = min(k, len(rawy_order))
        t0 = time.time(); emit(f"rawy_mi_top{k}", base, rawy_order[:kk], prod_tr, prod_te, t0)
    checkpoint(f"{name} seed{seed} rawy_mi done")

    del prod_tr, prod_te, base_mat, Xtr, Xte; gc.collect()
    return rows


def verdict(df):
    lines = ["", "=== VERDICTS (delta vs keep_all; >=+0.005 = acceptance rule WINS on AUC) ==="]
    for bed in df.bed.unique():
        b = df[df.bed == bed]
        ka = b[b.variant == "keep_all"]["auc_mean"].mean()
        ka_np = b[b.variant == "keep_all"]["n_prod"].mean()
        bo = b[b.variant == "base_only"]["auc_mean"].mean()
        lines.append(f"  {bed:14s} base_only={round(bo,4)} keep_all={round(ka,4)} (prod={ka_np:.0f}, " f"FE lift {round(ka-bo,4):+})")
        for v in ["holdout_greedy", "resid_mi_top5", "resid_mi_top10", "rawy_mi_top5", "rawy_mi_top10"]:
            sub = b[b.variant == v]
            if len(sub):
                m = sub["auc_mean"].mean(); npr = sub["n_prod"].mean()
                d = m - ka
                tag = ""
                if d >= 0.005:
                    tag = " <== WIN (higher AUC)"
                elif d >= -0.002 and npr < ka_np:
                    tag = f" <== CLEANER (>=AUC at {npr:.0f} vs {ka_np:.0f} prod)"
                lines.append(f"      {v:18s} {round(m,4)} prod={npr:4.1f} (d vs keep_all {round(d,4):+}){tag}")
    return "\n".join(lines)


def run_light(seeds):
    allrows = []
    for seed in seeds:
        Xs, ys, _ = make_dataset(n_samples=5000, seed=seed)
        print(f"\n=== synth seed={seed} shape={Xs.shape} ===", flush=True)
        allrows += run_bed("synth", Xs, ys, seed=seed)
        checkpoint(f"synth seed{seed} bed DONE")
        del Xs, ys; gc.collect()
    for seed in seeds:
        Xh, yh, _ = make_hard_dataset(n_samples=5000, seed=seed)
        print(f"\n=== hard_synth seed={seed} shape={Xh.shape} ===", flush=True)
        allrows += run_bed("hard_synth", Xh, yh, seed=seed)
        checkpoint(f"hard_synth seed{seed} bed DONE")
        del Xh, yh; gc.collect()
    return allrows


def main():
    with open(PROGRESS, "w"):
        pass
    checkpoint("START fe_accept FRUGAL bench")
    seeds = (0, 1)

    allrows = run_light(seeds)

    df = pd.DataFrame(allrows)
    df.to_csv("D:/Temp/fe_accept_light_rows.csv", index=False)
    print("\n=== ALL (light beds) ===")
    print(df.to_string(index=False))
    print(verdict(df))

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
        except (MemoryError, Exception) as e:
            print(f"  (madelon stress skipped: {type(e).__name__}: {e})", flush=True)

    final_df = pd.DataFrame(allrows)
    print("\n=== FINAL ALL ===")
    print(final_df.to_string(index=False))
    print(verdict(final_df))

    out = ["# FE-acceptance bench (A3-3 holdout_greedy + A3-6 resid_mi) results -- frugal rerun", ""]
    out.append("Base feature set = tree_top25 raw. Variable = which engineered co-occurrence products to keep.")
    out.append("Downstream AUC = mean over {lgbm, logit, knn}. 2 seeds on light beds; madelon seed0 stress.")
    out.append("holdout_greedy inner gain scored by a cheap held-out logistic probe (LightGBM refit OOM'd under")
    out.append("concurrent load); downstream AUC of the kept set still judged by the full lgbm/logit/knn panel.")
    out.append("")
    out.append("```")
    out.append(final_df.to_string(index=False))
    out.append("```")
    out.append(verdict(final_df))
    with open("D:/Temp/fe_accept_results.md", "w") as f:
        f.write("\n".join(out))
    checkpoint("DONE all beds; results written")


if __name__ == "__main__":
    main()
