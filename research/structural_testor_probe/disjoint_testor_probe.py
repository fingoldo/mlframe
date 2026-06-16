"""Disjoint-testor structure probe -- the Zhuravlev-spirit question we did NOT test before.

NOT feature ranking. Question: can a testor-style search find MULTIPLE near-disjoint
minimal-sufficient feature systems (alternative 'explanations' of y), and does an ensemble
over them beat (a) one model on the full X and (b) Random Subspaces of matched sizes?

Success = (1) recovery: the disjoint systems map onto the planted alternative groups;
          (2) value: disjoint-testor ensemble OOS-AUC / diversity > Random-Subspaces ensemble.
Failure (ensemble no better than random subspaces) closes this branch too, honestly.
"""
from __future__ import annotations
import warnings, time
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

SEED = 12345


# --------------------------------------------------------------------------- #
def make_two_solution(n=3000, pA=20, pB=20, p_noise=20, noise=2.0, seed=SEED):
    """Two near-independent feature groups A,B that EACH independently predict y
    (both are noisy reflections of one latent s; cross-corr kept low by high noise),
    plus pure-noise columns. Planted groups: A=0..pA-1, B=pA..pA+pB-1, noise=rest."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    y = (s > 0).astype(np.int64)
    A = s[:, None] * rng.uniform(0.5, 1.5, pA)[None, :] + noise * rng.normal(size=(n, pA))
    B = s[:, None] * rng.uniform(0.5, 1.5, pB)[None, :] + noise * rng.normal(size=(n, pB))
    N = rng.normal(size=(n, p_noise))
    X = np.hstack([A, B, N]).astype(np.float64)
    groups = {"A": list(range(pA)), "B": list(range(pA, pA + pB)), "noise": list(range(pA + pB, pA + pB + p_noise))}
    return X, y, groups


def _cv_auc(X, y, cols):
    if not cols:
        return 0.5
    skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
    return float(np.mean(cross_val_score(LogisticRegression(max_iter=500), X[:, cols], y, cv=skf, scoring="roc_auc")))


def forward_select(X, y, candidate, target_auc=0.90, max_size=25, min_gain=0.003):
    """Greedy forward selection within `candidate` feature indices until CV-AUC plateaus
    or hits target. Returns the selected subset (a 'minimal sufficient' system)."""
    chosen, pool, last = [], list(candidate), 0.5
    while pool and len(chosen) < max_size:
        best_f, best_auc = None, last
        for f in pool:
            auc = _cv_auc(X, y, chosen + [f])
            if auc > best_auc:
                best_auc, best_f = auc, f
        if best_f is None or best_auc - last < min_gain:
            break
        chosen.append(best_f); pool.remove(best_f); last = best_auc
        if last >= target_auc:
            break
    return chosen, last


def find_disjoint_testors(X, y, max_systems=4, target_auc=0.90, min_auc=0.75):
    """Repeatedly extract a minimal-sufficient system, then REMOVE its features and
    search the remainder -> near-disjoint alternative solutions."""
    remaining = list(range(X.shape[1]))
    systems = []
    for _ in range(max_systems):
        if len(remaining) < 2:
            break
        sub, auc = forward_select(X, y, remaining, target_auc=target_auc)
        if not sub or auc < min_auc:
            break
        systems.append((sub, auc))
        remaining = [f for f in remaining if f not in sub]
    return systems


# --------------------------------------------------------------------------- #
def _ensemble_auc(Xtr, ytr, Xte, yte, subsets, model="gb"):
    """Average member probabilities over the given feature subsets; return (AUC, member_disagreement)."""
    probs = []
    for cols in subsets:
        if not cols:
            continue
        m = (GradientBoostingClassifier(random_state=SEED) if model == "gb"
             else LogisticRegression(max_iter=800))
        m.fit(Xtr[:, cols], ytr)
        probs.append(m.predict_proba(Xte[:, cols])[:, 1])
    P = np.array(probs)
    auc = roc_auc_score(yte, P.mean(0))
    # diversity: mean pairwise (1 - corr) of member probability vectors
    div = 0.0
    if len(P) > 1:
        c = np.corrcoef(P)
        iu = np.triu_indices(len(P), 1)
        div = float(np.mean(1 - c[iu]))
    return float(auc), div


def _compose(sub, groups):
    inv = {}
    for g, idxs in groups.items():
        for i in idxs:
            inv[i] = g
    from collections import Counter
    c = Counter(inv.get(i, "?") for i in sub)
    return dict(c)


def run(name, X, y, groups=None):
    print(f"\n{'='*82}\nDATASET {name}  shape={X.shape}\n{'='*82}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=SEED, stratify=y)
    rng = np.random.default_rng(SEED)

    systems = find_disjoint_testors(Xtr, ytr)
    sizes = [len(s) for s, _ in systems]
    print(f"disjoint testors found: {len(systems)}  sizes={sizes}")
    for i, (s, auc) in enumerate(systems):
        comp = f" composition={_compose(s, groups)}" if groups else ""
        print(f"  system {i}: size={len(s)} train_cv_auc={auc:.3f}{comp}")
    if not systems:
        print("  (no sufficient systems found)"); return

    subs = [s for s, _ in systems]
    for model in ("gb", "logit"):
        e_auc, e_div = _ensemble_auc(Xtr, ytr, Xte, yte, subs, model)
        # baseline 1: single model on full X
        full = (GradientBoostingClassifier(random_state=SEED) if model == "gb"
                else LogisticRegression(max_iter=800)).fit(Xtr, ytr)
        f_auc = roc_auc_score(yte, full.predict_proba(Xte)[:, 1])
        # baseline 2: Random Subspaces of matched sizes (avg over 3 draws)
        rs_aucs, rs_divs = [], []
        for d in range(3):
            rsubs = [list(rng.choice(X.shape[1], size=sz, replace=False)) for sz in sizes]
            a, dv = _ensemble_auc(Xtr, ytr, Xte, yte, rsubs, model)
            rs_aucs.append(a); rs_divs.append(dv)
        print(f"  [{model}] disjoint_ens AUC={e_auc:.4f} div={e_div:.3f} | "
              f"full_X AUC={f_auc:.4f} | rand_subspaces AUC={np.mean(rs_aucs):.4f} div={np.mean(rs_divs):.3f}"
              f"  -> dELTA vs RS={e_auc-np.mean(rs_aucs):+.4f}")


def main():
    t0 = time.time()
    # planted two-solution synthetic (the home-turf structure case)
    X, y, g = make_two_solution()
    run("synth:two_solutions", X, y, g)
    # honesty check: a single-solution synthetic (no alternative system) -- disjoint search
    # should NOT manufacture a strong second system, and ensemble should not beat full X.
    from sklearn.datasets import load_breast_cancer
    bc = load_breast_cancer()
    run("breast_cancer(real)", bc.data.astype(np.float64), bc.target.astype(np.int64))
    print(f"\n[total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
