"""Round-4 idea A5-6 union_backward: replace the hybrid's cluster VOTE with one honest RFECV backward
elimination over the UNION of all member picks.

Mechanism (distinct from every rejected combine refinement):
  - The shipped hybrid combine is a cluster-aware VOTE over members. Adding RFECV as a 4th MEMBER was rejected
    because it re-discovers from the full noisy pool and over-selects (madelon: 251 features).
  - A5-6 instead runs RFECV over JUST the UNION of member_selections_ (raw + engineered cols already present in
    _Xaug_) -- a small, clean candidate set (~20-40 cols). RFECV PRUNES this clean set with the PROVEN one_se_min
    rule + permutation importance + cv=3, rather than re-discovering from scratch. The union is the recall ceiling
    (every member's pick is on the table); RFECV's backward elimination is a precision pass over it.
  - This is NOT the rejected disagreement-referee (a forward-AUC gate on contested clusters); it is a single
    honest CV backward elimination over the whole union, the regime where RFECV's one_se_min is known to behave.

ONE hybrid fit per (bed, seed). The RFECV-over-union is the only extra cost. We compare:
  baseline_hybrid : the shipped tree-member cluster vote (h.raw_selected_).
  union_all       : the raw union of member_selections_ (the candidate set RFECV prunes; recall ceiling).
  union_backward  : RFECV(one_se_min, permutation, cv=3) survivors over X_aug[union]  <-- the A5-6 idea.

VERDICT: union_backward PASSES if it beats baseline_hybrid by >= +0.005 on a bed without regressing the others
> 0.005; otherwise it is subsumed by (or worse than) the vote.

Memory-frugal under heavy concurrent load: all LightGBM / RFECV / permutation n_jobs capped at N_JOBS=2, modest
n_estimators, RFECV max_refits capped low (the union is small so few refits are needed). madelon cached to disk.
"""
from __future__ import annotations
import os, sys, time, pickle  # nosec B403 - module used safely in this file, see call sites below (no untrusted input reaches it)
os.environ.setdefault("TQDM_DISABLE", "1")
# cap thread oversubscription from the many native-threaded estimators running concurrently with sibling agents
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import load_real, downstream
from synth import make_dataset
from hard_synth import make_hard_dataset
from hybrid_selector import HybridSelector

N_JOBS = 2
PROGRESS = "D:/Temp/union_bw_progress.txt"
MADELON_CACHE = "D:/Temp/union_bw_madelon_cache.pkl"


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(PROGRESS, "a", encoding="ascii", errors="replace") as fh:
            fh.write(line + "\n")
    except Exception:  # nosec B110 - best-effort path
        pass


def union_of_members(member_sel: dict, cols) -> list:
    """The union of every member's picks, restricted to columns physically present in the augmented frame.
    This is the candidate set RFECV prunes -- the recall ceiling (raw survivors + engineered eng_/tprod cols)."""
    colset = set(cols)
    seen, out = set(), []
    for _m, sel in member_sel.items():
        for c in sel:
            if c in colset and c not in seen:
                seen.add(c); out.append(c)
    return out


def rfecv_backward_union(Xtr_aug, ytr, union, seed):
    """One honest RFECV backward elimination over X_aug[union]: one_se_min rule (proven parsimonious), permutation
    importance (debiases impurity, the proven QUALITY driver), cv=3. Capped for the concurrent-load machine:
    LightGBM n_estimators modest, RFECV max_refits low (union is small so the curve is short), permutation repeats=3.
    Returns the survivor column list."""
    from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig
    import lightgbm as lgb
    union = [c for c in union if c in Xtr_aug.columns]
    if len(union) < 3:
        return list(union)
    est = lgb.LGBMClassifier(n_estimators=120, num_leaves=31, learning_rate=0.06, n_jobs=N_JOBS, verbose=-1)
    # one_se_min = the PROVEN parsimonious rule (beats one_se_max/argmax/plateau here); permutation = proven driver.
    fi = FIConfig(importance_getter="permutation", n_features_selection_rule="one_se_min", n_repeats=3)
    # max_refits low: the union is small (~20-40 cols), so a short MBH budget reaches the elbow; max_runtime caps
    # the worst case on the concurrent-load box. n_jobs=1 (the LightGBM estimator is already native-multithreaded).
    sc = SearchConfig(max_refits=12, max_runtime_mins=4)
    r = RFECV(estimator=est, cv=3, scoring=None, verbose=0, fi_config=fi, search_config=sc, n_jobs=1, random_state=seed)
    r.fit(Xtr_aug[union], ytr)
    surv = [c for c in r.get_feature_names_out() if c in Xtr_aug.columns]
    return surv or list(union)


def run_bed(name, X, y, seed):
    log(f"{name} seed={seed}: split + ONE hybrid fit (n={len(X)}, p={X.shape[1]})")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    t0 = time.time()
    h = HybridSelector(vote=1, use_fe=True, random_state=seed).fit(Xtr, ytr)
    fit_s = round(time.time() - t0, 1)
    Ztr, Zte = h._augment(Xtr), h._augment(Xte)  # replay FE once for ALL downstream evals
    cols = list(Ztr.columns)
    ms = h.member_selections_
    union = union_of_members(ms, cols)
    log(f"{name} seed={seed}: hybrid fit {fit_s}s; members={ {k: len(v) for k, v in ms.items()} }; " f"|union|={len(union)} |baseline|={len(h.raw_selected_)}")

    rows = []

    def evalsel(tag, selected):
        sel = [c for c in dict.fromkeys(selected) if c in Ztr.columns and c in Zte.columns]
        if not sel:
            sel = list(h.raw_selected_)
        a = downstream(Ztr[sel], Zte[sel], ytr, yte)
        am = round(float(np.nanmean(list(a.values()))), 4)
        rows.append(dict(bed=name, seed=seed, variant=tag, n=len(sel), auc_mean=am, **a))
        log(f"{name} seed={seed}  {tag:16s} n={len(sel):3d} mean={am} {a}")
        return am

    evalsel("baseline_hybrid", list(h.raw_selected_))
    evalsel("union_all", union)
    t1 = time.time()
    surv = rfecv_backward_union(Ztr, ytr, union, seed)
    rfe_s = round(time.time() - t1, 1)
    evalsel("union_backward", surv)
    log(f"{name} seed={seed}: union_backward RFECV {rfe_s}s -> {len(surv)} survivors")
    return rows


def get_madelon():
    """load_real() (madelon) with disk cache so the concurrent madelon bench / network is hit at most once."""
    if os.path.exists(MADELON_CACHE):
        try:
            with open(MADELON_CACHE, "rb") as fh:
                X, y, nm = pickle.load(fh)  # nosec B301 - dev-only benchmark cache; local file this script itself wrote
            log(f"madelon loaded from cache {MADELON_CACHE} shape={X.shape}")
            return X, y, nm
        except Exception as e:
            log(f"madelon cache read failed ({type(e).__name__}); reloading")
    X, y, nm = load_real()
    try:
        with open(MADELON_CACHE, "wb") as fh:
            pickle.dump((X, y, nm), fh, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"madelon cached to {MADELON_CACHE} shape={X.shape}")
    except Exception as e:
        log(f"madelon cache write failed ({type(e).__name__})")
    return X, y, nm


def main():
    seeds = [0, 1]
    allrows = []

    # PRIMARY beds first (preferred over madelon per the concurrent-load constraint).
    for seed in seeds:
        Xh, yh, _ = make_hard_dataset(n_samples=5000, seed=seed)
        allrows += run_bed("hard_synth", Xh, yh, seed)
    for seed in seeds:
        Xs, ys, _ = make_dataset(n_samples=5000, seed=seed)
        allrows += run_bed("synth", Xs, ys, seed)

    # madelon stress check (cached). Failures here must NOT lose the primary-bed results.
    try:
        Xr, yr, rname = get_madelon()
        for seed in seeds:
            allrows += run_bed(rname, Xr, yr, seed)
    except Exception as e:
        log(f"madelon bed FAILED ({type(e).__name__}: {e}); primary beds stand")

    df = pd.DataFrame(allrows)
    df.to_csv("D:/Temp/union_bw_rows.csv", index=False)
    log("=== ALL ROWS ===\n" + df.to_string(index=False))

    # per-bed seed-averaged verdict vs baseline_hybrid
    log("=== VERDICT (seed-averaged auc_mean per bed, vs baseline_hybrid) ===")
    summary = []
    for bed in df.bed.unique():
        b = df[df.bed == bed]
        piv = b.groupby("variant")["auc_mean"].mean()
        npiv = b.groupby("variant")["n"].mean()
        base = float(piv.get("baseline_hybrid", float("nan")))
        for v in ("baseline_hybrid", "union_all", "union_backward"):
            if v not in piv.index:
                continue
            d = round(float(piv[v]) - base, 4)
            row = dict(bed=bed, variant=v, auc_mean=round(float(piv[v]), 4), n=round(float(npiv[v]), 1), delta_vs_baseline=d)
            summary.append(row)
            log(f"  {bed:12s} {v:16s} auc={row['auc_mean']:.4f} n={row['n']:.1f} d={d:+.4f}")
    sdf = pd.DataFrame(summary)
    sdf.to_csv("D:/Temp/union_bw_summary.csv", index=False)

    # machine verdict line
    log("=== A5-6 union_backward PASS/FAIL ===")
    beds = list(df.bed.unique())
    ub = {bed: None for bed in beds}
    for bed in beds:
        b = df[df.bed == bed].groupby("variant")["auc_mean"].mean()
        ub[bed] = round(float(b.get("union_backward", float("nan"))) - float(b.get("baseline_hybrid", float("nan"))), 4)
    wins = [bed for bed, d in ub.items() if d is not None and d >= 0.005]
    regress = [bed for bed, d in ub.items() if d is not None and d <= -0.005]
    if wins and not regress:
        log(f"PASS: union_backward beats baseline by >=+0.005 on {wins} with no >0.005 regression elsewhere. deltas={ub}")
    elif regress:
        log(f"FAIL: union_backward REGRESSES on {regress} (>0.005). deltas={ub}")
    else:
        log(f"SUBSUMED: union_backward within +/-0.005 of the baseline vote on every bed (no win). deltas={ub}")
    log("DONE")


if __name__ == "__main__":
    main()
