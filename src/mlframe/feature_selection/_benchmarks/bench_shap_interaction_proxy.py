"""Bench: interaction-aware subset proxy (``proxy_mode="interaction"``) vs additive (default).

Decides whether ``proxy_mode="interaction"`` should be the default for ShapProxiedFS's subset proxy.
Fits the selector machinery DIRECTLY on synthetic frames (no full train_mlframe_models_suite -- that
native-segfaults under contention via the cupy probe): build the OOF main-effect SHAP ``phi`` + the
TreeSHAP interaction tensor ``Phi`` from one xgboost model, run the additive search AND the interaction
proxy, then HONEST-holdout score each selected subset by refitting a fresh model on a disjoint test
split.

HONEST metric = held-out AUC (clf) of a model refit on the selected subset (the model never saw test).

Beds (5 x 3 seeds): interaction-heavy (XOR pairs, multiplicative) AND additive-only (no regression
where interactions are absent). FLIP the default only on a MAJORITY win (esp. interaction beds) that
REPLICATES across seeds AND does not regress additive beds.

Run:
  CUDA_VISIBLE_DEVICES="" MLFRAME_NO_CUDA_AUTOCONFIG=1 MLFRAME_KEEP_BROKEN_CUPY=1 \
      python -m mlframe.feature_selection._benchmarks.bench_shap_interaction_proxy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import compute_shap_matrix
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import compute_interaction_tensor
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interaction_proxy import interaction_proxy_top_n
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n


def _xgb():
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=120, max_depth=4, learning_rate=0.15, subsample=0.9,
        n_jobs=1, tree_method="hist", verbosity=0, random_state=0)


def make_bed(kind: str, n: int, p_noise: int, seed: int):
    """Return (X df, y, informative_idx_set). Beds keep n<=2000, p<=~60 for a tight wave."""
    rng = np.random.default_rng(seed)
    cols = {}
    inf = []

    def add(name, vec):
        cols[name] = vec
        return name

    if kind == "xor2":  # two independent XOR pairs (pure interaction; ~0 marginal)
        for k in range(2):
            a = rng.normal(size=n); b = rng.normal(size=n)
            inf += [add(f"xa{k}", a), add(f"xb{k}", b)]
        logit = 2.5 * (np.sign(cols["xa0"] * cols["xb0"]) + np.sign(cols["xa1"] * cols["xb1"]))
    elif kind == "xor_distract":  # ONE XOR pair + many WEAK additive distractors with higher marginal
        a = rng.normal(size=n); b = rng.normal(size=n)
        inf += [add("xa", a), add("xb", b)]
        logit = 3.5 * np.sign(cols["xa"] * cols["xb"])  # operands have ~0 marginal
        for k in range(6):  # weak additive distractors: nonzero marginal, individually informative-ish
            v = rng.normal(size=n)
            cols[f"dz{k}"] = v
            logit = logit + 0.35 * v  # NOT counted in inf (they are real but weak; goal is XOR recovery)
    elif kind == "mult":  # multiplicative interaction
        a = rng.normal(size=n); b = rng.normal(size=n)
        inf += [add("ma", a), add("mb", b)]
        logit = 3.0 * (cols["ma"] * cols["mb"])
    elif kind == "mixed":  # one additive strong feature + one XOR pair
        s = rng.normal(size=n)
        a = rng.normal(size=n); b = rng.normal(size=n)
        inf += [add("add0", s), add("xa", a), add("xb", b)]
        logit = 1.8 * s + 2.2 * np.sign(cols["xa"] * cols["xb"])
    elif kind == "additive4":  # purely additive (regression guard: interactions absent)
        logit = np.zeros(n)
        for k in range(4):
            v = rng.normal(size=n)
            inf.append(add(f"ad{k}", v))
            logit = logit + (1.5 - 0.2 * k) * v
    elif kind == "additive_redundant":  # additive + correlated redundant copies (over-score guard)
        logit = np.zeros(n)
        for k in range(3):
            v = rng.normal(size=n)
            inf.append(add(f"ad{k}", v))
            logit = logit + 1.4 * v
            cols[f"red{k}"] = v + 0.1 * rng.normal(size=n)  # redundant, NOT counted informative
    else:
        raise ValueError(kind)

    for j in range(p_noise):
        cols[f"nz{j}"] = rng.normal(size=n)
    X = pd.DataFrame(cols)
    p = 1.0 / (1.0 + np.exp(-(logit - logit.mean())))
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y, set(inf), list(X.columns)


def honest_auc(X, y, sel_names, seed):
    if not sel_names:
        return 0.5
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    m = _xgb(); m.set_params(random_state=seed + 1)
    m.fit(Xtr[sel_names], ytr)
    if len(np.unique(yte)) < 2:
        return 0.5
    return float(roc_auc_score(yte, m.predict_proba(Xte[sel_names])[:, 1]))


def pick_from_candidates(cands, col_names, n_keep=None):
    if not cands:
        return []
    loss, idx = cands[0]
    return [col_names[i] for i in idx]


def run():
    beds = ["xor_distract", "xor2", "mult", "mixed", "additive4", "additive_redundant"]
    seeds = [0, 1, 2]
    n, p_noise = 1500, 30
    # Tight cardinality cap so the proxy must RANK-PRUNE (the regime where the additive proxy's
    # blindness to non-additive pairs actually changes the pick); larger caps let additive recover both
    # operands anyway (verified: no delta at max_card=6).
    cap = 2
    rows = []
    for bed in beds:
        for seed in seeds:
            X, y, inf, names = make_bed(bed, n, p_noise, seed)
            # SHAP + interaction tensor from the selector's own machinery (search split only).
            Xs, _Xh, ys, _yh = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
            Xs = Xs.reset_index(drop=True); ys = np.asarray(ys)
            phi, base, y_phi = compute_shap_matrix(
                _xgb(), Xs, ys, classification=True, out_of_fold=True, n_splits=3, n_models=1,
                rng=np.random.default_rng(seed), n_jobs=1)
            Phi, ibase = compute_interaction_tensor(
                _xgb(), Xs, ys, classification=True, rng=np.random.default_rng(seed))
            # ADDITIVE proxy (current default): brute force over the (small) width.
            P = phi.shape[1]
            add_c = brute_force_top_n(phi, base, y_phi, classification=True, metric="brier",
                                      min_card=1, max_card=min(cap, P), top_n=30, parallel=False)
            # INTERACTION proxy: re-score additive candidates + sweep gated pairs.
            int_c = interaction_proxy_top_n(
                phi, Phi, base, y_phi, classification=True, metric="brier",
                min_card=1, max_card=min(cap, P), top_n=30, interaction_top_k=30,
                candidate_subsets=[c for _l, c in add_c])
            add_sel = pick_from_candidates(add_c, list(Xs.columns))
            int_sel = pick_from_candidates(int_c, list(Xs.columns))
            a_auc = honest_auc(X, y, add_sel, seed)
            i_auc = honest_auc(X, y, int_sel, seed)
            a_rec = len(set(add_sel) & inf) / max(1, len(inf))
            i_rec = len(set(int_sel) & inf) / max(1, len(inf))
            rows.append((bed, seed, round(a_auc, 4), round(i_auc, 4), round(i_auc - a_auc, 4),
                         round(a_rec, 2), round(i_rec, 2), len(add_sel), len(int_sel)))
            print(f"{bed:20s} s{seed}  add_auc={a_auc:.4f} int_auc={i_auc:.4f} "
                  f"delta={i_auc - a_auc:+.4f}  rec a/i={a_rec:.2f}/{i_rec:.2f}  "
                  f"|S| a/i={len(add_sel)}/{len(int_sel)}")
    print("\n=== per-bed mean delta (interaction - additive) ===")
    for bed in beds:
        ds = [r[4] for r in rows if r[0] == bed]
        wins = sum(1 for d in ds if d > 1e-4)
        print(f"{bed:20s} mean_delta={np.mean(ds):+.4f}  seed_wins={wins}/{len(ds)}  deltas={ds}")
    return rows


if __name__ == "__main__":
    run()
