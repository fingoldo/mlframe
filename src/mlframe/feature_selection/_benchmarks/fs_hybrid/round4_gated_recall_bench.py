"""Round-4 synergy idea: PERMUTATION-NULL-GATED residual recall-add (the gated successor to residual-relay).

MEASURED CONTEXT (round4_residual_relay_bench / round4_synergy_combine_bench): an UNGATED residual recall-add
(screen DROPPED raw features by MI with the hybrid's OOF residual, add the top-k) RECOVERS real weak features on
hard_synth (+0.006..0.017) but HURTS madelon (-0.008..-0.011): the tree-member hybrid already captures madelon's
signal, so its OOF residual is essentially NOISE there, and the residual-screened additions only dilute the set.

THE FIX (this bench): a PERMUTATION NULL gate that fires the recall-add ONLY when the residual genuinely carries
signal a dropped feature can explain.
  1. Fit production HybridSelector(vote=1, use_fe=True) -> selected set S; AUC(S) via downstream().
  2. OOF residual on S (cross_val_predict, r = y - p_oof).
  3. For DROPPED raw features compute MI(feature; binarized residual). Build a PERMUTATION NULL: shuffle the
     residual K times, recompute the MAX dropped-feature residual-MI per shuffle -> null distribution of the max.
     ADMIT a dropped feature iff its real residual-MI exceeds the 95th percentile of the null-max distribution.
     -> on madelon (residual=noise) ~nothing clears the null; on hard_synth the real weak/FE features clear it.
  4. S' = S + admitted. Compare AUC(S') vs AUC(S).

VERDICT TARGET: CLEAN GATED WIN iff hard_synth mean delta >= +0.01 AND madelon/synth means within +/-0.005.

Memory-frugal: every lgbm/MI/RF this bench controls is capped n_jobs=4 + modest n_estimators. The hybrid itself
uses n_jobs=-1 internally (unavoidable). Fitted hybrid outputs are CACHED to disk per (bed, seed) so multi-seed
reruns skip the expensive (esp. madelon ~50-90s) fits.
"""
from __future__ import annotations
import os, sys, time, pickle, hashlib  # nosec B403 - pickle used only for trusted same-process/dev-local round-trips, see call sites in this file
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import load_real
from hard_synth import make_hard_dataset
from synth import make_dataset
from hybrid_selector import HybridSelector

NJOBS = 4  # cap for everything THIS bench controls (concurrent-load friendly)
CACHE_DIR = "D:/Temp/gated_recall_cache"
PROGRESS = "D:/Temp/gated_recall_progress.txt"
K_PERM = 20  # permutation-null shuffles
NULL_PCTILE = 95  # admit a dropped feature iff real residual-MI > this pct of the null-max distribution
SEEDS = (0, 1, 2)


def checkpoint(msg: str):
    with open(PROGRESS, "a", encoding="ascii", errors="replace") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def downstream(Xtr, Xte, ytr, yte):
    """3-model honest held-out AUC (lgbm/logit/knn). Same shape as round3_realdata_bench.downstream but n_jobs-capped."""
    o = {}
    o["lgbm"] = roc_auc_score(yte, lgb.LGBMClassifier(n_estimators=300, verbose=-1, n_jobs=NJOBS).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    o["logit"] = roc_auc_score(yte, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    o["knn"] = roc_auc_score(yte, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    return {k: round(float(v), 4) for k, v in o.items()}


def _cache_key(bed, seed, shape):
    h = hashlib.md5(f"{bed}|{seed}|{shape}".encode(), usedforsecurity=False).hexdigest()[:10]
    return os.path.join(CACHE_DIR, f"hybrid_{bed}_s{seed}_{h}.pkl")


def fit_hybrid_cached(bed, X, y, seed):
    """Fit HybridSelector(vote=1, use_fe=True) on a 60/40 train split and CACHE everything the gate needs:
    the train/test split indices, the selected set S, and the augmented train/test frames. The hybrid fit (n_jobs=-1
    internally) is the only expensive step; caching makes multi-seed reruns cheap (madelon ~50-90s -> instant)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(bed, seed, X.shape)
    if os.path.exists(key):
        with open(key, "rb") as f:
            return pickle.load(f)  # nosec B301 - dev-only benchmark cache; local file this script itself wrote
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    t0 = time.time()
    h = HybridSelector(vote=1, use_fe=True, random_state=seed).fit(Xtr, ytr)
    fit_s = round(time.time() - t0, 1)
    S = [c for c in h.raw_selected_ if c in (set(Xtr.columns) | set(h._augment(Xtr.iloc[:2]).columns))]
    Ztr, Zte = h._augment(Xtr), h._augment(Xte)
    payload = dict(
        Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte,
        Ztr=Ztr, Zte=Zte, S=list(S), fit_s=fit_s,
        raw_cols=list(Xtr.columns),                 # original raw columns (dropped pool drawn from here)
    )
    with open(key, "wb") as f:
        pickle.dump(payload, f)
    return payload


def gated_recall(bed, X, y, seed):
    """Run the full gated-recall pipeline for one (bed, seed). Returns a dict of rows + the gate diagnostics."""
    p = fit_hybrid_cached(bed, X, y, seed)
    Xtr, Xte, ytr, yte = p["Xtr"], p["Xte"], p["ytr"], p["yte"]
    Ztr, Zte, S = p["Ztr"], p["Zte"], p["S"]
    raw_cols = p["raw_cols"]

    # S evaluated on the augmented frame (so engineered eng_/tprod_ survivors are present) -- the honest baseline.
    Scols = [c for c in dict.fromkeys(S) if c in Ztr.columns and c in Zte.columns]
    base_auc = downstream(Ztr[Scols], Zte[Scols], ytr, yte)
    base_mean = round(float(np.nanmean(list(base_auc.values()))), 4)

    # OOF residual on S (deviance-ish r = y - p_oof). lgbm n_jobs-capped.
    m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, n_jobs=NJOBS, verbose=-1)
    p_oof = cross_val_predict(m, Ztr[Scols], ytr, cv=4, method="predict_proba", n_jobs=NJOBS)[:, 1]
    resid = ytr.values.astype(float) - p_oof
    rbin = (resid > np.median(resid)).astype(int)

    # DROPPED raw features (original raw cols not selected). Residual-MI of each.
    Sset = set(Scols)
    dropped = [c for c in raw_cols if c not in Sset]
    diag = dict(bed=bed, seed=seed, n_S=len(Scols), n_dropped=len(dropped), fit_s=p["fit_s"])
    if not dropped:
        diag.update(n_admitted=0, null_thresh=None, top_real_mi=None, admitted=[])
        rows = [dict(bed=bed, seed=seed, variant="hybrid_S", n=len(Scols), auc_mean=base_mean, **base_auc),
                dict(bed=bed, seed=seed, variant="gated_recall", n=len(Scols), auc_mean=base_mean, **base_auc)]
        return rows, diag

    Xd = Ztr[dropped].values if all(c in Ztr.columns for c in dropped) else Xtr[dropped].values
    # real residual-MI (capped n_jobs via single-threaded mutual_info_classif; it ignores n_jobs but is cheap)
    real_mi = mutual_info_classif(Xd, rbin, random_state=seed)
    real_mi = np.nan_to_num(real_mi)

    # PERMUTATION NULL: shuffle the residual K times, recompute the MAX dropped-feature residual-MI per shuffle.
    rng = np.random.default_rng(seed)
    null_max = np.empty(K_PERM)
    for k in range(K_PERM):
        perm = rng.permutation(len(rbin))
        rbin_s = rbin[perm]
        mi_s = np.nan_to_num(mutual_info_classif(Xd, rbin_s, random_state=int(rng.integers(1 << 30))))
        null_max[k] = float(mi_s.max()) if mi_s.size else 0.0
    null_thresh = float(np.percentile(null_max, NULL_PCTILE))

    # ADMIT dropped features whose real residual-MI exceeds the null-max threshold.
    admit_idx = [j for j in range(len(dropped)) if real_mi[j] > null_thresh]
    # rank admitted by real_mi desc (so if we cap, strongest first); no cap here (the gate IS the selector)
    admit_idx = sorted(admit_idx, key=lambda j: real_mi[j], reverse=True)
    admitted = [dropped[j] for j in admit_idx]

    order = np.argsort(real_mi)[::-1]
    top_real = [(dropped[j], round(float(real_mi[j]), 5)) for j in order[:8]]
    diag.update(
        n_admitted=len(admitted),
        null_thresh=round(null_thresh, 5),
        null_max_mean=round(float(null_max.mean()), 5),
        null_max_max=round(float(null_max.max()), 5),
        top_real_mi=top_real,
        admitted=admitted[:25],
    )

    Sprime = Scols + [c for c in admitted if c in Ztr.columns and c in Zte.columns]
    Sprime = [c for c in dict.fromkeys(Sprime) if c in Ztr.columns and c in Zte.columns]
    if admitted:
        gr_auc = downstream(Ztr[Sprime], Zte[Sprime], ytr, yte)
        gr_mean = round(float(np.nanmean(list(gr_auc.values()))), 4)
    else:
        gr_auc, gr_mean = dict(base_auc), base_mean  # gate fired nothing -> identical to S (the intended madelon case)

    rows = [
        dict(bed=bed, seed=seed, variant="hybrid_S", n=len(Scols), auc_mean=base_mean, **base_auc),
        dict(bed=bed, seed=seed, variant="gated_recall", n=len(Sprime), auc_mean=gr_mean, **gr_auc),
    ]
    return rows, diag


def main():
    checkpoint("=== gated_recall bench start ===")
    beds = {}
    # build datasets once per seed where the generator is seeded; madelon (load_real) is seed-invariant -> load once.
    Xr, yr, rname = load_real()
    checkpoint(f"loaded real bed {rname} {Xr.shape}")
    print(f"REAL bed: {rname} {Xr.shape} pos_rate={round(float(yr.mean()),3)}", flush=True)

    all_rows, all_diag = [], []
    for seed in SEEDS:
        Xh, yh, _ = make_hard_dataset(n_samples=5000, seed=seed)
        Xs, ys, _ = make_dataset(n_samples=5000, seed=seed)
        bed_specs = [("hard_synth", Xh, yh), (rname, Xr, yr), ("synth", Xs, ys)]
        for bed, X, y in bed_specs:
            checkpoint(f"seed={seed} bed={bed} fitting/gating")
            t0 = time.time()
            rows, diag = gated_recall(bed, X, y, seed)
            all_rows += rows; all_diag.append(diag)
            base = [r for r in rows if r["variant"] == "hybrid_S"][0]["auc_mean"]
            gr = [r for r in rows if r["variant"] == "gated_recall"][0]["auc_mean"]
            print(f"[seed={seed}] {bed:12s} S_mean={base} gated_mean={gr} d={round(gr-base,4):+} "
                  f"admitted={diag['n_admitted']}/{diag['n_dropped']} null_thr={diag.get('null_thresh')} "
                  f"({round(time.time()-t0,1)}s)", flush=True)
            print(f"            top_real_mi={diag.get('top_real_mi')}", flush=True)
            if diag.get("admitted"):
                print(f"            admitted={diag['admitted']}", flush=True)
            checkpoint(f"seed={seed} bed={bed} done d={round(gr-base,4):+} admitted={diag['n_admitted']}")

    df = pd.DataFrame(all_rows)
    df.to_csv("D:/Temp/gated_recall_rows.csv", index=False)
    pd.DataFrame(all_diag).to_csv("D:/Temp/gated_recall_diag.csv", index=False)

    # per-bed per-seed deltas + means
    print("\n=== PER-SEED (auc_mean = mean of lgbm/logit/knn) ===", flush=True)
    summary = []
    for bed in df.bed.unique():
        for seed in SEEDS:
            b = df[(df.bed == bed) & (df.seed == seed)]
            if b.empty:
                continue
            s_auc = float(b[b.variant == "hybrid_S"]["auc_mean"].iloc[0])
            g_auc = float(b[b.variant == "gated_recall"]["auc_mean"].iloc[0])
            summary.append(dict(bed=bed, seed=seed, hybrid_S=s_auc, gated_recall=g_auc, delta=round(g_auc - s_auc, 4)))
    sdf = pd.DataFrame(summary)
    print(sdf.to_string(index=False), flush=True)

    print("\n=== MEAN OVER SEEDS ===", flush=True)
    means = sdf.groupby("bed").agg(hybrid_S=("hybrid_S", "mean"), gated_recall=("gated_recall", "mean"), delta=("delta", "mean")).round(4)
    print(means.to_string(), flush=True)

    # VERDICT
    print("\n=== VERDICT ===", flush=True)
    hs_d = float(means.loc["hard_synth", "delta"]) if "hard_synth" in means.index else float("nan")
    other_ok = True
    for bed in means.index:
        if bed == "hard_synth":
            continue
        d = float(means.loc[bed, "delta"])
        if d < -0.005:
            other_ok = False
    win = (hs_d >= 0.01) and other_ok
    print(f"hard_synth mean delta = {round(hs_d,4):+} (target >= +0.01)", flush=True)
    for bed in means.index:
        if bed != "hard_synth":
            print(f"{bed} mean delta = {round(float(means.loc[bed,'delta']),4):+} (target within +/-0.005)", flush=True)
    print(f"VERDICT: {'CLEAN GATED WIN -> SHIP' if win else 'NOT A CLEAN WIN -> KILL (or needs more work)'}", flush=True)
    checkpoint(f"=== done; verdict win={win} hard_synth_d={round(hs_d,4)} ===")
    return sdf, means


if __name__ == "__main__":
    main()
