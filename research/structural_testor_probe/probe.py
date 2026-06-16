"""Go/no-go probe for the Structural-Testor idea. PRINTS tables -- not a pytest.

Run:  D:/ProgramData/anaconda3/python.exe research/structural_testor_probe/probe.py

Order (per plan):
  (A) ReliefF kill-test  -- XOR + niche feature_37 + Spearman(rare_pair, ReliefF).
                            If rho > 0.8 -> rare_pair is a Relief reskin -> stop A.
  (B) separation_profile redundancy for mRMR  -- LEAD bet, real data (Madelon + control).
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

import scorers as S

RNG_SEED = 12345
MAX_PAIRS = 20_000


# --------------------------------------------------------------------------- #
# shared helpers                                                              #
# --------------------------------------------------------------------------- #
def _kinds_stds(X):
    kinds = ["num"] * X.shape[1]
    stds = X.std(axis=0)
    return kinds, stds


def make_D(X, y, seed=RNG_SEED, max_pairs=MAX_PAIRS):
    rng = np.random.default_rng(seed)
    ia, ib = S.sample_cross_class_pairs(y, max_pairs, rng)
    kinds, stds = _kinds_stds(X)
    return S.build_difference_matrix(X, ia, ib, kinds, stds)


def relief_scores(X, y, n_neighbors=50):
    from skrebate import ReliefF
    r = ReliefF(n_neighbors=n_neighbors, n_jobs=1)
    r.fit(X.astype(np.float64), y.astype(np.int64))
    return np.asarray(r.feature_importances_, dtype=np.float64)


def mi_relevance(X, y):
    return mutual_info_classif(X, y, random_state=RNG_SEED)


# --------------------------------------------------------------------------- #
# synthetics for the kill-test                                               #
# --------------------------------------------------------------------------- #
def make_xor(n=2000, p=20, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (np.sign(X[:, 0] * X[:, 1]) > 0).astype(np.int64)
    return X, y, [0, 1]


def make_niche(n=4000, p=40, niche_idx=37, frac=0.05, seed=RNG_SEED):
    """95% of rows: class from a weak global signal. 5% 'hard' rows: class
    fully determined by feature `niche_idx`, which is pure noise elsewhere."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    hard = rng.random(n) < frac
    # on hard rows, the niche feature's sign IS the label (and signal_0 is uninformative there)
    y[hard] = (X[hard, niche_idx] > 0).astype(np.int64)
    return X, y, niche_idx, hard


# --------------------------------------------------------------------------- #
# (A) ReliefF kill-test                                                       #
# --------------------------------------------------------------------------- #
def kill_test():
    print("=" * 78)
    print("(A) ReliefF KILL-TEST  -- is rare_pair_coverage just (distance-weighted) ReliefF?")
    print("=" * 78)
    verdict = {}

    # --- XOR ---
    X, y, sig = make_xor()
    D = make_D(X, y)
    rp_soft, rp_tau = S.rare_pair_coverage(D)
    cov = S.coverage(D)
    rel = relief_scores(X, y)
    rho = spearmanr(rp_soft, rel).correlation
    rank_rp = {f: int(np.argsort(-rp_soft).tolist().index(f)) for f in sig}
    rank_rel = {f: int(np.argsort(-rel).tolist().index(f)) for f in sig}
    rank_cov = {f: int(np.argsort(-cov).tolist().index(f)) for f in sig}
    print(f"\nXOR (signal={sig}, p={X.shape[1]}):")
    print(f"  rank of signal feats -- rare_pair_soft: {rank_rp}")
    print(f"                          ReliefF        : {rank_rel}")
    print(f"                          coverage       : {rank_cov}")
    print(f"  Spearman(rare_pair_soft, ReliefF) = {rho:+.3f}")
    verdict["xor_rho"] = rho

    # --- niche ---
    X, y, ni, hard = make_niche()
    D = make_D(X, y)
    rp_soft, rp_tau = S.rare_pair_coverage(D)
    cov = S.coverage(D)
    rel = relief_scores(X, y)
    mi = mi_relevance(X, y)
    rho_n = spearmanr(rp_soft, rel).correlation
    def rk(s, f):
        return int(np.argsort(-s).tolist().index(f))
    print(f"\nNiche (feature_{ni} decides {hard.mean():.0%} hard rows; signal_0 decides rest):")
    print(f"  rank of feature_{ni} -- rare_pair_soft:{rk(rp_soft,ni):3d}  rare_pair_tau:{rk(rp_tau,ni):3d}"
          f"  ReliefF:{rk(rel,ni):3d}  coverage:{rk(cov,ni):3d}  MI:{rk(mi,ni):3d}")
    print(f"  Spearman(rare_pair_soft, ReliefF) = {rho_n:+.3f}")
    verdict["niche_rho"] = rho_n
    verdict["niche_rank_rp"] = rk(rp_soft, ni)
    verdict["niche_rank_relief"] = rk(rel, ni)
    verdict["niche_rank_mi"] = rk(mi, ni)

    print("\n--- A verdict ---")
    max_rho = max(abs(verdict["xor_rho"]), abs(verdict["niche_rho"]))
    if max_rho > 0.8:
        print(f"  Spearman vs ReliefF up to {max_rho:+.3f} (>0.8) -> REDISCOVERY of ReliefF. STOP A.")
    else:
        print(f"  Spearman vs ReliefF stays {max_rho:+.3f} (<=0.8) -> genuinely diverges; A may continue.")
    return verdict


# --------------------------------------------------------------------------- #
# (B) separation_profile redundancy for mRMR                                  #
# --------------------------------------------------------------------------- #
def _abs_corr_matrix(X):
    c = np.corrcoef(X, rowvar=False)
    return np.abs(np.nan_to_num(c))


def _su_matrix(X, y, n_bins=8):
    """Symmetric-uncertainty feature-feature matrix on discretized X (baseline redundancy)."""
    Xd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal",
                          strategy="quantile", subsample=None).fit_transform(X)
    F = X.shape[1]

    def entropy(col):
        _, c = np.unique(col, return_counts=True)
        p = c / c.sum()
        return -(p * np.log(p + 1e-12)).sum()

    H = np.array([entropy(Xd[:, f]) for f in range(F)])
    SU = np.eye(F)
    # pairwise MI via sklearn on discrete features
    for f in range(F):
        mi_row = mutual_info_classif(Xd, Xd[:, f], discrete_features=True, random_state=RNG_SEED)
        denom = H[f] + H
        denom = np.where(denom > 0, denom, 1.0)
        SU[f] = 2.0 * mi_row / denom
    np.fill_diagonal(SU, 1.0)
    return np.clip(SU, 0, 1)


def _eval_selection(X, y, idx, model_kind="gb", cv=5):
    if len(idx) == 0:
        return float("nan")
    Xs = X[:, idx]
    if model_kind == "gb":
        m = GradientBoostingClassifier(random_state=RNG_SEED)
    else:
        m = LogisticRegression(max_iter=1000)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RNG_SEED)
    return float(np.mean(cross_val_score(m, Xs, y, cv=skf, scoring="roc_auc")))


def _diversity(X, idx):
    """Mean abs input-correlation among selected feats (lower = more diverse)."""
    if len(idx) < 2:
        return 0.0
    c = np.abs(np.corrcoef(X[:, idx], rowvar=False))
    iu = np.triu_indices(len(idx), 1)
    return float(np.mean(c[iu]))


def _stability(X, y, sim_fn, relevance_fn, k, n_folds=5):
    """Jaccard overlap of selected sets across folds (re-fit sim+relevance per fold)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RNG_SEED)
    sels = []
    for tr, _ in skf.split(X, y):
        rel = relevance_fn(X[tr], y[tr])
        sim = sim_fn(X[tr], y[tr])
        sels.append(set(S.mrmr_select(rel, sim, k)))
    js = []
    for i in range(len(sels)):
        for j in range(i + 1, len(sels)):
            u = len(sels[i] | sels[j])
            js.append(len(sels[i] & sels[j]) / u if u else 0.0)
    return float(np.mean(js))


def make_decorrelated_redundant(n=3000, p_noise=12, seed=RNG_SEED):
    """THE home-turf case for separation_profile. Two independent signals z1,z2;
    y = (z1+z2>0). f0=z1 has a redundant TWIN f1 = monotone(z1) with LOW Pearson
    corr to f0 (heavy-tail map) but an IDENTICAL separation profile. f2=z2 is the
    second signal. At small k, corr-mRMR sees f0,f1 as non-redundant (low corr) and
    can waste a slot on the twin, missing z2; separation_profile penalises the twin
    (same separated pairs) and should keep f2 -> higher downstream AUC."""
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    y = (z1 + z2 > 0).astype(np.int64)
    f0 = z1 + 0.05 * rng.normal(size=n)
    f1 = np.sign(z1) * np.expm1(2.0 * np.abs(z1))   # monotone in z1, same separation, low Pearson
    f2 = z2 + 0.05 * rng.normal(size=n)
    noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([f0, f1, f2, noise])
    return X, y, [0, 1, 2]


def run_B_dataset(name, X, y, k=20):
    print(f"\n{'-'*78}\n(B) dataset: {name}  (n={X.shape[0]}, F={X.shape[1]}, k={k})\n{'-'*78}")
    X = X.astype(np.float64)
    rel = mi_relevance(X, y)

    if "decorrelated" in name:
        sep = S.separation_profile_similarity(make_D(X, y))
        c01 = abs(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
        print(f"  mechanism: twin f0/f1 -> |corr(x)|={c01:.3f} (looks non-redundant)"
              f"  sep_profile_sim={sep[0,1]:.3f} (should expose redundancy)")

    sims = {
        "corr":         lambda Xt, yt: _abs_corr_matrix(Xt),
        "SU":           lambda Xt, yt: _su_matrix(Xt, yt),
        "sep_profile":  lambda Xt, yt: np.clip(S.separation_profile_similarity(make_D(Xt, yt)), 0, 1),
    }

    show_sel = X.shape[1] <= 20
    print(f"  {'variant':<12} {'GB_AUC':>8} {'LR_AUC':>8} {'diversity':>10} {'stability':>10}  overlap_vs_corr"
          + ("  selected" if show_sel else ""))
    sel_by = {}
    for vname, sim_fn in sims.items():
        sim = sim_fn(X, y)
        idx = S.mrmr_select(rel, sim, k)
        sel_by[vname] = set(idx)
        gb = _eval_selection(X, y, idx, "gb")
        lr = _eval_selection(X, y, idx, "lr")
        div = _diversity(X, idx)
        stab = _stability(X, y, sim_fn, mi_relevance, k)
        ov = (len(sel_by[vname] & sel_by["corr"]) / len(sel_by[vname] | sel_by["corr"])
              if "corr" in sel_by else 1.0)
        tail = f"  {sorted(idx)}" if show_sel else ""
        print(f"  {vname:<12} {gb:8.4f} {lr:8.4f} {div:10.4f} {stab:10.4f}  {ov:.2f}{tail}")


# --------------------------------------------------------------------------- #
def _load_repo_synth():
    """Reuse redundancy-structured synthetics from tests/_biz_val_synth.py (home turf
    for a redundancy metric). Returns list of (name, X, y, k)."""
    import importlib.util, os
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "..", "tests", "feature_selection", "_biz_val_synth.py"))
    spec = importlib.util.spec_from_file_location("_biz_val_synth", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = []
    X, y, _ = m.make_correlated_redundant(n=2500, n_corr=4, p_noise=6, corr=0.95)
    out.append(("synth:correlated_redundant", X, y.astype(np.int64), 3))
    if hasattr(m, "make_latent_reflections"):
        X, y, *_ = m.make_latent_reflections(n=2500)
        out.append(("synth:latent_reflections", X, np.asarray(y).astype(np.int64), 4))
    if hasattr(m, "make_two_latent_groups"):
        X, y, *_ = m.make_two_latent_groups(n=2500)
        out.append(("synth:two_latent_groups", X, np.asarray(y).astype(np.int64), 4))
    return out


def load_B_datasets():
    from sklearn.datasets import fetch_openml, load_breast_cancer
    out = []
    # *** home-turf discriminating synthetic: low-corr redundant twin, small k ***
    Xd, yd, _ = make_decorrelated_redundant()
    out.append(("synth:decorrelated_redundant*", Xd, yd, 2))
    # redundancy-structured repo synthetics
    try:
        out.extend(_load_repo_synth())
    except Exception as e:
        print(f"  [skip repo synth: {e}]")
    # interaction-heavy real, designed for FS
    try:
        d = fetch_openml("madelon", version=1, as_frame=False, parser="liac-arff")
        y = (np.asarray(d.target) == np.unique(d.target)[1]).astype(np.int64)
        out.append(("madelon", np.asarray(d.data, dtype=np.float64), y, 20))
    except Exception as e:
        print(f"  [skip madelon: {e}]")
    # non-interaction controls
    bc = load_breast_cancer()
    out.append(("breast_cancer(control)", bc.data, bc.target.astype(np.int64), 20))
    try:
        d = fetch_openml("ionosphere", version=1, as_frame=False, parser="liac-arff")
        y = (np.asarray(d.target) == np.unique(d.target)[1]).astype(np.int64)
        out.append(("ionosphere(control)", np.asarray(d.data, dtype=np.float64), y, 20))
    except Exception as e:
        print(f"  [skip ionosphere: {e}]")
    return out


def main():
    kill_test()
    print("\n" + "=" * 78)
    print("(B) separation_profile as a NEW mRMR redundancy metric (LEAD bet)")
    print("=" * 78)
    for name, X, y, k in load_B_datasets():
        run_B_dataset(name, X, y, k=k)


if __name__ == "__main__":
    main()
