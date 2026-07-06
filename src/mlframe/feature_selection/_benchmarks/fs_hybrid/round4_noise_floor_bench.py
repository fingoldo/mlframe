"""Round-4 idea A4-2: permuted-y NOISE-FLOOR stop for RFECV's madelon OVER-selection (251 -> ~20-40).

RFECV's CV curve is flat on a noise-robust GBM, so its N-rule (one_se_min/max) lands high (251/500 on madelon).
Knockoff-FDR (A4-1) FAILED to cut this (Gaussian knockoffs degenerate on madelon's correlated probes -> empty FDR).

A4-2 is a DIFFERENT mechanism: detect when the plateau is just NOISE TOLERANCE. Compare the REAL per-N CV curve to a
PERMUTED-y reference curve (same top-N feature sets, y shuffled). Stop at the smallest N where the REAL curve's gain
over its own start exceeds the shuffled-y noise envelope (95th pct of the permuted-curve gain). Past that N the real
curve's rise is within noise = those features are indistinguishable from noise.

FALSIFIABLE: does the noise-floor stop cut madelon 251 -> ~20-40 WITHOUT losing the all-features AUC (lgbm 0.87)?
KILL: returns near-251 (no cut), OR drops AUC > 0.01 vs all-features, OR over-cuts a real signal curve on synth.

This is a STANDALONE bench / post-hoc cut on the CV curve. Does NOT edit production files.

MEASURED VERDICT (2026-06-04) -> SHIP (as a post-hoc CV-curve cut). Full numbers in D:/Temp/rfecv_floor_results.md.
  - The LITERAL task spec ('smallest N where real gain over real[0] clears the 95th-pct permuted gain', here
    noise_floor_first) is a SIGNAL-ONSET detector, NOT a stop rule: it fires at N=2 on BOTH madelon (AUC 0.68) and
    synth (0.63) -> over-cuts catastrophically. Kept only as a contrast. DO NOT use it.
  - The CORRECTED rule (noise_floor_plateau) -- smallest N past which the REMAINING climb is within the permuted
    noise envelope -- cuts madelon 251 -> N*=8..16 (modal 12 over n_perm/seed); lgbm 0.9135 (N=8) / 0.940 (N=12)
    vs all-features 0.872 and vs RFECV-251 0.868. knn 0.61 -> 0.91 (noise probes uncurse the distance metric).
  - synth guard: plateau N*=20 keeps 7/8 causal (base_recall 0.875), AUC 0.7535 vs all 0.7603 (-0.007, in noise) ->
    does NOT over-cut a real signal curve.
  - Needs >=3 permutations (n_perm=1 is noisy: N* in {12,12,25}); 3-5 perms -> N* in {8,12,16}. ~110s for the 3
    permuted CV curves on a LGBM-gain ranking; NO 220s RFECV refit needed (the cut runs on top-N-by-FI).
  - NOTE: the current RFECVSel('lgbm_perm') config (SearchConfig max_runtime_mins=3) TIMES OUT on madelon (2600x500
    with permutation FI) and returns all-500 with a 2-point cv_results_ curve -- so this bench uses the cached 251
    support (D:/Temp/rfecv_madelon_support.pkl) as the over-selection baseline and a clean LGBM-gain ranking for the
    per-N cut ordering. fit_rfecv_full/global_ranking below are retained but unused (the degenerate RFECV path).
"""
from __future__ import annotations
import os, sys, time, pickle
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import load_real, downstream
import fs_selectors as S

PROGRESS = "D:/Temp/rfecv_floor_progress.txt"
FULL_CACHE = "D:/Temp/rfecv_madelon_full.pkl"  # full curve + per-N feature lists (NEW; old cache was support-only)


def checkpoint(msg: str):
    with open(PROGRESS, "a") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    print(f"  ## {msg}", flush=True)


# ----------------------------------------------------------------------------- cached RFECV fit (full curve)
def fit_rfecv_full(Xtr, ytr):
    """Fit RFECVSel('lgbm_perm') once; cache support + cv_results_ + per-N selected_features_ + global ranking.

    The per-N selected_features_ lists are RFECV's elimination-rank survivors at each evaluated N; we reconstruct a
    GLOBAL feature ordering (most-important first) by the smallest N at which each feature first appears as a survivor.
    """
    if os.path.exists(FULL_CACHE):
        checkpoint(f"loading cached full RFECV from {FULL_CACHE}")
        return pickle.load(open(FULL_CACHE, "rb"))  # nosec B301 - dev-only benchmark cache; local file this script itself wrote
    checkpoint("no full cache -> fitting RFECVSel('lgbm_perm') on madelon (~220s)")
    t0 = time.time()
    sel = S.RFECVSel("lgbm_perm")
    sel.fit(Xtr, ytr)
    r = sel.r_  # the inner RFECV
    cvres = {k: list(v) for k, v in r.cv_results_.items()}
    # selected_features_ maps N -> ordered feature list (the survivors RFECV scored at that N).
    sel_by_n = {int(k): [str(c) for c in v] for k, v in r.selected_features_.items()}
    support = [c for c in sel.raw_selected_ if c in Xtr.columns]
    out = dict(
        support=support,
        cv_results=cvres,
        selected_features=sel_by_n,
        feature_names_in=[str(c) for c in r.feature_names_in_],
        fit_s=round(time.time() - t0, 1),
    )
    pickle.dump(out, open(FULL_CACHE, "wb"))
    checkpoint(f"fitted + cached full RFECV in {out['fit_s']}s; |support|={len(support)}, "
               f"|cv_results N|={len(cvres.get('nfeatures', []))}, |selected_features N|={len(sel_by_n)}")
    return out


def global_ranking(sel_by_n, feature_names_in, support):
    """Reconstruct a global most-important-first ordering of features from the per-N survivor lists.

    A feature that survives down to a SMALL N is more important. We order each feature by (smallest N at which it
    appears, then its position in that smallest-N list). Features never appearing in any survivor list are appended
    in feature_names_in order (they were eliminated earliest). Returns a full ordered list over all input features.
    """
    first_n = {}
    pos_in_first = {}
    for n in sorted(sel_by_n):
        for pos, f in enumerate(sel_by_n[n]):
            if f not in first_n:
                first_n[f] = n
                pos_in_first[f] = pos
    ranked = sorted(first_n, key=lambda f: (first_n[f], pos_in_first[f]))
    # append any never-seen features (earliest-eliminated) in original order
    seen = set(ranked)
    for f in feature_names_in:
        if f not in seen:
            ranked.append(f)
            seen.add(f)
    return ranked


# ----------------------------------------------------------------------------- per-N CV curves (real + permuted)
def _mk_model():
    # modest, memory-frugal (concurrent load): cap n_jobs=4, modest n_estimators
    return lgb.LGBMClassifier(n_estimators=120, num_leaves=31, learning_rate=0.06, n_jobs=4, verbose=-1)


def cv_curve(X, y, ranked, n_grid, cv, permute=False, n_perm=1, base_seed=100):
    """Per-N CV ROC-AUC on the top-N features by `ranked`. If permute, average over n_perm shuffles of y.

    Returns, for permute=False: array of real cv-mean per N.
    For permute=True: (mean_curve, all_perm_curves [n_perm x len(n_grid)]).
    """
    curves = []
    n_iter = n_perm if permute else 1
    for p in range(n_iter):
        if permute:
            rng = np.random.default_rng(base_seed + p)
            yy = pd.Series(rng.permutation(y.values), index=y.index)
        else:
            yy = y
        row = []
        for n in n_grid:
            cols = ranked[:n]
            sc = cross_val_score(_mk_model(), X[cols], yy, cv=cv, scoring="roc_auc", n_jobs=1)
            row.append(float(np.mean(sc)))
        curves.append(row)
    arr = np.asarray(curves)
    if permute:
        return arr.mean(axis=0), arr
    return arr[0]


def noise_floor_first(n_grid, real_curve, perm_curves, pct=95.0):
    """LITERAL spec: smallest N where real gain over real[0] exceeds 95th-pct of permuted-curve gain over its own start.

    NOTE (measured 2026-06-04): this FIRST-clears rule is a SIGNAL-ONSET detector, not a STOP rule. On any dataset with
    signal it fires at the first N>=1 that beats noise (N~2), catastrophically over-cutting (synth N*=2 AUC 0.60 vs all
    0.76). Kept only as a contrast; the production-relevant rule is noise_floor_plateau below.
    Returns (N*, idx*, real_gain array, perm_envelope array).
    """
    real_gain = real_curve - real_curve[0]
    perm_gain = perm_curves - perm_curves[:, [0]]  # each permutation's gain over its own N_min
    envelope = np.percentile(perm_gain, pct, axis=0)
    star_idx = next((i for i in range(len(n_grid)) if real_gain[i] > envelope[i]), len(n_grid) - 1)
    return n_grid[star_idx], star_idx, real_gain, envelope


def noise_floor_plateau(n_grid, real_curve, perm_curves, pct=95.0):
    """STOP rule: smallest N past which the real curve's REMAINING climb is within the shuffled-y noise envelope.

    Mechanism: the over-selection plateau is just noise tolerance when going from N to a larger N' adds no more than a
    shuffled-y curve would over the SAME N->N' span. For each grid index i we ask: does ANY larger j add real gain
    (real[j]-real[i]) exceeding the 95th-pct of the permuted incremental gain (perm[:,j]-perm[:,i]) over that span?
    N* = the smallest N for which NO larger j clears -> i.e. everything beyond N* is indistinguishable from noise.

    This is the parsimonious analogue of RFECV 'plateau' but referenced to a PERMUTED-y noise floor instead of a fixed
    1-SE band, so on a noise-robust GBM (flat real plateau) it stops where real signal genuinely flattens vs noise.
    Returns (N*, idx*, remaining_gain array, remaining_envelope array) where remaining_*[i] = max over j>i.
    """
    G = len(n_grid)
    remaining_gain = np.full(G, -np.inf)
    remaining_env = np.zeros(G)
    star_idx = G - 1
    found = False
    for i in range(G):
        best_excess = -np.inf
        best_rg = -np.inf
        best_env = 0.0
        for j in range(i + 1, G):
            rg = real_curve[j] - real_curve[i]
            inc = perm_curves[:, j] - perm_curves[:, i]
            env = float(np.percentile(inc, pct))
            if (rg - env) > best_excess:
                best_excess = rg - env
                best_rg = rg
                best_env = env
        remaining_gain[i] = best_rg if i < G - 1 else 0.0
        remaining_env[i] = best_env
        # no future N adds signal beyond noise -> plateau reached at this i
        if i < G - 1 and best_excess <= 0 and not found:
            star_idx = i
            found = True
    return n_grid[star_idx], star_idx, remaining_gain, remaining_env


# ----------------------------------------------------------------------------- ranking builders
def lgbm_gain_ranking(Xtr, ytr):
    """Clean, fast global most-important-first ordering by LightGBM split-gain importance on the full feature set.

    Used as the per-N cut ordering when RFECV's own per-N survivor lists are degenerate (RFECV timed out at all-500 on
    madelon -> selected_features_ had 1 entry). A 'top-N by FI' cut is exactly what the noise-floor needs and this is a
    standard, valid FI ranking.
    """
    m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, n_jobs=4, importance_type="gain", verbose=-1).fit(Xtr, ytr)
    order = np.argsort(m.feature_importances_)[::-1]
    return [Xtr.columns[i] for i in order]


def _curves(Xtr, ytr, ranked, n_grid, n_perm):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    t0 = time.time()
    real_curve = cv_curve(Xtr, ytr, ranked, n_grid, cv, permute=False)
    t_real = round(time.time() - t0, 1)
    t0 = time.time()
    perm_mean, perm_curves = cv_curve(Xtr, ytr, ranked, n_grid, cv, permute=True, n_perm=n_perm)
    t_perm = round(time.time() - t0, 1)
    return real_curve, perm_mean, perm_curves, t_real, t_perm


def _curve_table(n_grid, real_curve, perm_mean, first_env, plateau_gain, plateau_env):
    return pd.DataFrame(dict(
        N=n_grid,
        real_cv=np.round(real_curve, 4),
        real_gain=np.round(real_curve - real_curve[0], 4),
        perm_mean=np.round(perm_mean, 4),
        first_noise95=np.round(first_env, 4),
        remain_gain=np.round(plateau_gain, 4),
        remain_noise95=np.round(plateau_env, 4),
    ))


# ----------------------------------------------------------------------------- madelon bench
def run_madelon(n_perm=3):
    X, y, name = load_real()
    checkpoint(f"madelon loaded shape={X.shape}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    p = Xtr.shape[1]

    # RFECV baseline = the documented over-selection (251). The pre-existing cache from the knockoff bench holds it;
    # the CURRENT RFECVSel config (max_runtime_mins=3) times out on madelon and returns all-500, so we DON'T use that
    # degenerate curve. We use a clean LGBM-gain ranking for the per-N cut ordering.
    rfecv_support = pickle.load(open("D:/Temp/rfecv_madelon_support.pkl", "rb"))  # nosec B301 - dev-only benchmark cache; local file this script itself wrote
    rfecv_support = [c for c in rfecv_support if c in Xtr.columns]
    checkpoint(f"loaded RFECV 251-support cache: |support|={len(rfecv_support)}")

    ranked = lgbm_gain_ranking(Xtr, ytr)
    checkpoint(f"LGBM-gain ranking built; |ranked|={len(ranked)}")

    n_grid = sorted(set([1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50, 75, 100, 150, 200, len(rfecv_support), p]))
    n_grid = [n for n in n_grid if 1 <= n <= p]
    checkpoint(f"madelon N-grid={n_grid}")

    real_curve, perm_mean, perm_curves, t_real, t_perm = _curves(Xtr, ytr, ranked, n_grid, n_perm)
    checkpoint(f"madelon curves done real={t_real}s perm={t_perm}s ({n_perm} perms)")

    N_first, _, _, first_env = noise_floor_first(n_grid, real_curve, perm_curves, pct=95.0)
    N_plat, _, plat_gain, plat_env = noise_floor_plateau(n_grid, real_curve, perm_curves, pct=95.0)
    checkpoint(f"madelon first-clears N*={N_first}, plateau N*={N_plat}")

    rows = []

    def emit(tag, cols):
        cols = [c for c in cols if c in Xtr.columns]
        if not cols:
            print(f"  {tag:24s} EMPTY", flush=True); return None
        a = downstream(Xtr[cols], Xte[cols], ytr, yte)
        am = round(float(np.nanmean(list(a.values()))), 4)
        rows.append(dict(variant=tag, n=len(cols), auc_mean=am, **a))
        print(f"  {tag:24s} n={len(cols):3d} mean={am} {a}", flush=True)
        return am

    emit("all", list(Xtr.columns))
    emit("rfecv_251", rfecv_support)
    emit(f"floor_FIRST_N{N_first}", ranked[:N_first])
    emit(f"floor_PLATEAU_N{N_plat}", ranked[:N_plat])

    curve_table = _curve_table(n_grid, real_curve, perm_mean, first_env, plat_gain, plat_env)
    return dict(name=name, N_first=N_first, N_plateau=N_plat, n_grid=n_grid, support=len(rfecv_support), p=p, rows=rows, curve=curve_table)


# ----------------------------------------------------------------------------- synth sanity (must NOT over-cut)
def run_synth(n_samples=5000, n_perm=3):
    from synth import make_dataset
    X, y, truth = make_dataset(n_samples=n_samples, seed=0)
    base = truth["base"]; relevant = set(truth["relevant"])
    checkpoint(f"synth loaded shape={X.shape}; n_base={len(base)}, n_relevant={len(relevant)}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    p = Xtr.shape[1]

    ranked = lgbm_gain_ranking(Xtr, ytr)
    n_grid = sorted(set([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 40, p]))
    n_grid = [n for n in n_grid if 1 <= n <= p]
    checkpoint(f"synth N-grid={n_grid}")

    real_curve, perm_mean, perm_curves, t_real, t_perm = _curves(Xtr, ytr, ranked, n_grid, n_perm)
    N_first, _, _, first_env = noise_floor_first(n_grid, real_curve, perm_curves, pct=95.0)
    N_plat, _, plat_gain, plat_env = noise_floor_plateau(n_grid, real_curve, perm_curves, pct=95.0)
    checkpoint(f"synth first-clears N*={N_first}, plateau N*={N_plat}")

    def recalls(N):
        picked = set(ranked[:N])
        return (round(len(picked & set(base)) / len(base), 3), round(len(picked & relevant) / len(relevant), 3))

    br_f, rr_f = recalls(N_first)
    br_p, rr_p = recalls(N_plat)

    rows = []

    def emit(tag, cols):
        cols = [c for c in cols if c in Xtr.columns]
        a = downstream(Xtr[cols], Xte[cols], ytr, yte)
        am = round(float(np.nanmean(list(a.values()))), 4)
        rows.append(dict(variant=tag, n=len(cols), auc_mean=am, **a))
        print(f"  {tag:24s} n={len(cols):3d} mean={am} {a}", flush=True)
        return am

    emit("all", list(Xtr.columns))
    emit(f"floor_FIRST_N{N_first}", ranked[:N_first])
    emit(f"floor_PLATEAU_N{N_plat}", ranked[:N_plat])
    curve_table = _curve_table(n_grid, real_curve, perm_mean, first_env, plat_gain, plat_env)
    return dict(N_first=N_first, N_plateau=N_plat, n_base=len(base),
                first_base_recall=br_f, first_rel_recall=rr_f,
                plateau_base_recall=br_p, plateau_rel_recall=rr_p,
                rows=rows, curve=curve_table)


def main():
    print("=" * 78, flush=True)
    print("A4-2 permuted-y NOISE-FLOOR stop bench", flush=True)
    print("=" * 78, flush=True)
    md = run_madelon(n_perm=3)
    sy = run_synth(n_samples=5000, n_perm=3)

    # ---- write results md ----
    lines = []
    lines.append("# A4-2 permuted-y noise-floor stop -- results\n")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("\nTwo stop rules tested on a LGBM-gain feature ranking:\n")
    lines.append("- FIRST-clears (literal task spec): smallest N where real gain over real[0] clears the 95th-pct permuted gain.\n")
    lines.append("- PLATEAU (corrected stop rule): smallest N past which the REMAINING climb is within the permuted noise envelope.\n")
    lines.append("\n## madelon (load_real, test_size=0.4 seed 0)\n")
    lines.append(f"- RFECV over-selection baseline = {md['support']} features (of p={md['p']})\n")
    lines.append(f"- FIRST-clears N* = {md['N_first']} ; PLATEAU N* = {md['N_plateau']}\n\n")
    lines.append("### per-N curve (real vs permuted-y noise envelope)\n\n")
    lines.append("```\n" + md["curve"].to_string(index=False) + "\n```\n\n")
    lines.append("### downstream AUC (madelon held-out)\n\n")
    lines.append("```\n" + pd.DataFrame(md["rows"]).sort_values("auc_mean", ascending=False).to_string(index=False) + "\n```\n\n")
    lines.append("## synth sanity (n=5000) -- must NOT over-cut a real signal curve\n")
    lines.append(f"- FIRST-clears N*={sy['N_first']} (base_recall={sy['first_base_recall']}, rel_recall={sy['first_rel_recall']})\n")
    lines.append(f"- PLATEAU N*={sy['N_plateau']} (base_recall={sy['plateau_base_recall']}, rel_recall={sy['plateau_rel_recall']}); n_base={sy['n_base']}\n\n")
    lines.append("```\n" + sy["curve"].to_string(index=False) + "\n```\n\n")
    lines.append("```\n" + pd.DataFrame(sy["rows"]).sort_values("auc_mean", ascending=False).to_string(index=False) + "\n```\n")

    with open("D:/Temp/rfecv_floor_results.md", "w") as fh:
        fh.writelines(lines)

    # ---- verdict ----
    md_rows = {r["variant"]: r for r in md["rows"]}
    all_lgbm = md_rows["all"]["lgbm"]; all_mean = md_rows["all"]["auc_mean"]
    first = md_rows.get(f"floor_FIRST_N{md['N_first']}", {})
    plat = md_rows.get(f"floor_PLATEAU_N{md['N_plateau']}", {})
    print("\n" + "=" * 78, flush=True)
    print("VERDICT INPUTS", flush=True)
    print(f"  madelon: RFECV over-selection={md['support']}, p={md['p']}", flush=True)
    print(f"  all-features        lgbm={all_lgbm} mean={all_mean}", flush=True)
    print(f"  FIRST-clears  N*={md['N_first']:3d} lgbm={first.get('lgbm')} mean={first.get('auc_mean')}", flush=True)
    print(f"  PLATEAU       N*={md['N_plateau']:3d} lgbm={plat.get('lgbm')} mean={plat.get('auc_mean')}", flush=True)
    print(f"  synth FIRST N*={sy['N_first']} base_recall={sy['first_base_recall']} | "
          f"PLATEAU N*={sy['N_plateau']} base_recall={sy['plateau_base_recall']}", flush=True)
    print("=" * 78, flush=True)
    checkpoint(f"DONE madelon plateau N*={md['N_plateau']} lgbm={plat.get('lgbm')} vs all {all_lgbm}; "
               f"synth plateau N*={sy['N_plateau']} base_recall={sy['plateau_base_recall']}")


if __name__ == "__main__":
    main()
