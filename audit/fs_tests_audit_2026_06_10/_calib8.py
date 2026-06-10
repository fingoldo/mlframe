"""Verify the ROBUST, honest claims across 6 seeds:
 (1) redundancy-avoidance: MRMR top-K covers strictly more latent clusters than
     SelectKBest top-K (the structural mRMR property, robust to LogReg's redundancy
     tolerance);
 (2) MI baseline is provably redundancy-trapped: its top-4 picks are dominated by
     ONE cluster's copies (skb_clu0 high), and its quality frontier PLATEAUS for
     small K then only improves at larger K;
 (3) downstream AUC: MRMR full vs SKB@same-K is a within-epsilon TIE (honest), and
     MRMR top-4 >= random-K floor (sanity)."""
import sys, warnings, numpy as np, pandas as pd
sys.path.insert(0, "tests")
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def make_fix(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    zc = [rng.standard_normal(n) for _ in range(3)]
    cols = {}
    copies = (6, 3, 3); tight = (0.05, 0.25, 0.25)
    for c in range(3):
        for k in range(copies[c]):
            cols[f"clu{c}_m{k}"] = zc[c] + tight[c] * rng.standard_normal(n)
    i0 = rng.standard_normal(n); i1 = rng.standard_normal(n)
    cols["indep0"] = i0; cols["indep1"] = i1
    for j in range(20):
        cols[f"noise{j}"] = rng.standard_normal(n)
    score = 2.2*zc[0] + 1.3*zc[1] + 1.3*zc[2] + 1.1*i0 + 1.1*i1 + 0.3*rng.standard_normal(n)
    y = (score > np.median(score)).astype(np.int64)
    return pd.DataFrame(cols), pd.Series(y)


def auc_cols(X, y, cols, cv=5):
    cols = list(cols)
    if not cols:
        return float("nan")
    return float(cross_val_score(LogisticRegression(max_iter=400), X[cols], y, cv=cv, scoring="roc_auc").mean())


def clusters_in(cols):
    return {c.split("_")[0] for c in cols if c.startswith("clu")}


def mrmr_order(X, y, seed):
    from mlframe.feature_selection.filters.mrmr import MRMR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(random_seed=seed, verbose=0, cv=3, run_additional_rfecv_minutes=False,
                   use_simple_mode=False, dcd_enable=False, min_relevance_gain=0.0,
                   min_relevance_gain_relative_to_first=0.0, max_consec_unconfirmed=60,
                   min_features_fallback=4, full_npermutations=5).fit(X, y)
    names = list(X.columns)
    return [names[i] for i in np.asarray(sel.support_, dtype=int)]


KS = [1, 2, 3, 4, 5, 6, 8]
cov_win = skb_trap = plateau_jump = tie = randbeat = 0
print("=== robust claims (seeds 0-5) ===")
for seed in range(6):
    X, y = make_fix(n=2000, seed=seed)
    order = mrmr_order(X, y, seed)
    Km = len(order)
    mi = mutual_info_classif(X.values, y.values, random_state=seed)
    i_ord = [X.columns[i] for i in np.argsort(mi)[::-1]]
    # (1) cluster coverage of top-4
    K4 = min(4, Km) if Km else 0
    m_cov = len(clusters_in(order[:4]))
    i_cov = len(clusters_in(i_ord[:4]))
    # (2) skb trap: how many of MI top-4 are cluster0
    skb4 = i_ord[:4]
    skb_clu0 = sum(1 for c in skb4 if c.startswith("clu0_"))
    # MI frontier plateau then jump
    auc_i = {k: auc_cols(X, y, i_ord[:k]) for k in KS}
    plateau = auc_i[6] - auc_i[1]   # small if trapped
    jump = auc_i[8] - auc_i[6]      # positive when MI escapes
    # (3) tie + rand
    a_m = auc_cols(X, y, order) if Km else float("nan")
    skb_same = [X.columns[i] for i in np.flatnonzero(
        SelectKBest(mutual_info_classif, k=max(Km, 1)).fit(X.values, y.values).get_support())] if Km else []
    a_s = auc_cols(X, y, skb_same) if Km else float("nan")
    rng = np.random.default_rng(7000 + seed)
    a_rand = float(np.mean([auc_cols(X, y, list(rng.choice(X.columns, 4, replace=False))) for _ in range(5)]))
    a_m4 = auc_cols(X, y, order[:4]) if Km else float("nan")

    if m_cov > i_cov: cov_win += 1
    if skb_clu0 >= 3: skb_trap += 1
    if jump >= plateau + 0.02: plateau_jump += 1   # the escape jump dwarfs the in-trap gain
    if not np.isnan(a_m) and a_m >= a_s - 0.05: tie += 1
    if not np.isnan(a_m4) and a_m4 >= a_rand + 0.03: randbeat += 1
    print(f" s={seed} Km={Km} m_cov4={m_cov} i_cov4={i_cov} skb_clu0={skb_clu0}/4 "
          f"plateau(i6-i1)={plateau:+.4f} jump(i8-i6)={jump:+.4f} "
          f"a_m={a_m if not np.isnan(a_m) else float('nan'):.4f} a_skb@K={a_s if not np.isnan(a_s) else float('nan'):.4f} "
          f"a_m4={a_m4 if not np.isnan(a_m4) else float('nan'):.4f} a_rand={a_rand:.4f}")
print(f"COV_WIN(m_cov>i_cov)={cov_win}/6  SKB_TRAP(clu0>=3)={skb_trap}/6  "
      f"PLATEAU_JUMP={plateau_jump}/6  TIE(a_m>=a_skb-0.05)={tie}/6  RANDBEAT(a_m4>=rand+0.03)={randbeat}/6")
