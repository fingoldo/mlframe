"""A4-4 su_seeded_interactions (ShapProxiedFS) -- cheap falsifiable test.

PRODUCTION FACT (read-only from shap_proxied_fs.py:1500 + _shap_proxy_interactions.py):
  ShapProxiedFS's interaction_aware objective ADDS candidate subsets ranked by the SHAP-interaction
  coalition value ``base + sum_{i,k in S} Phi_ik`` -- but it is GATED to ``phi.shape[1] <=
  max_interaction_features`` (default 16) because the interaction tensor Phi is (n, P, P) -- O(P^2)
  memory + compute. Round-3 verdict: raising the gate engages a prohibitively slow O(P^2) TreeSHAP
  tensor on wide data, so interaction_aware is a no-op default.

A4-4 PROPOSAL: a CHEAP pairwise-SU screen picks the top-K candidate interaction PAIRS first, then run
  the interaction objective on ONLY those K pairs (a sparse tensor over K pairs, not the dense PxP).
  This decouples the interaction recall win from the O(P^2) wall.

THE WHOLE IDEA HINGES ON ONE FALSIFIABLE PREMISE:
  Does a cheap pairwise-SU(a, b ; y) screen actually RANK the true interaction operand pairs near the
  top, on data where each operand has ~0 MARGINAL signal? If the screen cannot find {ia, ib} cheaply,
  there is nothing to seed and the idea is dead -- with NO need to build the sparse-tensor machinery.

So this bench measures the screen's discriminative power directly + the downstream recall lift of
  seeding the product columns the screen proposes. Three beds:
   (1) designed pure-interaction bed y=sign(a*b): operands have EXACTLY 0 marginal MI -- the hardest
       case + the exact regime interaction_aware exists for.
   (2) hard_synth: ia*ib operands buried in 200 noise + linear/weak signal (the real split-signal bed).
   (3) synth: inf_4*inf_5 operands among redundant clusters + noise.

SU(pair;y) is computed on quantile-binned columns with the standard discrete MI estimator (the same
  quantity MRMR's symmetric_uncertainty computes; reimplemented on numpy bins here because the prod
  njit kernel needs prebinned ``factors_data`` plumbing not worth standing up for a screen prototype).

PASS premise (1): the true operand pair is in the SU-screen top-K (K=20) on the pure-interaction bed.
PASS end-to-end: ShapSel + SU-top-K product cols beats plain ShapSel recall by >= +1 informative
  WITHOUT the O(P^2) cost (screen is O(P) univariate SU + O(K) pair SU).
"""
from __future__ import annotations
import os, sys, time, itertools
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import make_dataset
from hard_synth import make_hard_dataset
import fs_selectors as S

CK = "D:/Temp/queue_ideas_progress.txt"
def ck(msg):
    with open(CK, "a") as f:
        f.write(msg + "\n")


# ----------------------------------------------------------- discrete MI / SU on quantile bins
def _qbin(col, nbins=10):
    col = np.asarray(col, dtype=float)
    try:
        edges = np.unique(np.quantile(col, np.linspace(0, 1, nbins + 1)))
        if len(edges) <= 2:
            return np.zeros(len(col), dtype=np.int64)
        return np.clip(np.digitize(col, edges[1:-1]), 0, len(edges) - 2).astype(np.int64)
    except Exception as exc:
        ck(f"_qbin: quantile binning failed, returning zeros: {exc!r}")
        return np.zeros(len(col), dtype=np.int64)

def _entropy(labels):
    _, cnt = np.unique(labels, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log(p)).sum())

def _mi(a, b):
    # joint label via cantor-ish pairing on small-card ints
    j = a.astype(np.int64) * (b.max() + 1) + b
    return _entropy(a) + _entropy(b) - _entropy(j)

def _su(a, b):
    ha, hb = _entropy(a), _entropy(b)
    if ha + hb <= 1e-12:
        return 0.0
    return 2.0 * _mi(a, b) / (ha + hb)


def pairwise_su_screen(X, y, top_k=20, nbins=8, max_cols=120):
    """A4-4 SCREEN: rank feature pairs by how much their JOINT bin carries about y BEYOND the marginals.

    For each pair (a,b): synergy_su = SU(joint_bin(a,b); y) - max(SU(a;y), SU(b;y)).
    A pure interaction (operands ~0 marginal SU, high joint SU) scores high; two strong-marginal but
    non-interacting cols score ~0 (joint adds nothing past the better marginal). O(P) marginal bins +
    O(P^2) cheap discrete-SU on bins (no model, no TreeSHAP). Cap cols to keep the pair scan bounded.
    """
    cols = list(X.columns)
    yb = _qbin(np.asarray(y), nbins=min(nbins, max(2, len(np.unique(y)))))
    if len(np.unique(y)) <= 20:
        yb = np.asarray(y).astype(np.int64)
    bins = {c: _qbin(X[c].values, nbins=nbins) for c in cols}
    marg = {c: _su(bins[c], yb) for c in cols}
    # restrict pair scan to the most promising marginals + a noise sample (keeps O(P^2) bounded)
    ranked = sorted(cols, key=lambda c: -marg[c])
    pool = ranked[:max_cols]
    pairs = []
    for a, b in itertools.combinations(pool, 2):
        ja = bins[a].astype(np.int64) * (bins[b].max() + 1) + bins[b]
        # relabel joint to dense ints
        _, ja = np.unique(ja, return_inverse=True)
        joint_su = _su(ja.astype(np.int64), yb)
        syn = joint_su - max(marg[a], marg[b])
        pairs.append((syn, joint_su, a, b))
    pairs.sort(key=lambda t: -t[0])
    return pairs[:top_k], marg


def make_pure_interaction(n=5000, p_noise=60, seed=0):
    """Designed bed: y = sign(a*b) -- a,b have EXACTLY 0 marginal MI, all signal is the interaction."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n); b = rng.standard_normal(n)
    c = rng.standard_normal(n)  # a weak linear control so something has marginal signal
    logit = 2.2 * np.sign(a * b) + 0.7 * c
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    cols = {"op_a": a, "op_b": b, "lin_c": c}
    for i in range(p_noise):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    order = list(X.columns); rng.shuffle(order)
    truth = {"operands": ["op_a", "op_b"], "informative": ["op_a", "op_b", "lin_c"]}
    return X[order], pd.Series(y, name="target"), truth


def downstream_auc(Xtr, Xte, ytr, yte):
    m = lgb.LGBMClassifier(n_estimators=300, verbose=-1, n_jobs=4).fit(Xtr, ytr)
    return round(float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1])), 4)


def run_bed(name, X, y, truth):
    operands = truth["operands"]
    print(f"\n=== {name} {X.shape} ===", flush=True); ck(f"A4-4 {name} start {X.shape}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)

    # --- the A4-4 screen (cheap; no TreeSHAP) -------------------------------------------------
    t0 = time.time()
    top_pairs, marg = pairwise_su_screen(Xtr, ytr, top_k=20)
    scr_s = round(time.time() - t0, 2)
    op_set = set(operands)
    rank_of_true = None
    for r, (syn, jsu, a, b) in enumerate(top_pairs):
        if {a, b} == op_set:
            rank_of_true = r
            break
    print(f"  SU-screen {scr_s}s | marginal SU(op_a)={marg.get(operands[0],float('nan')):.4f} "
          f"SU(op_b)={marg.get(operands[1],float('nan')):.4f}", flush=True)
    print("  top-6 synergy pairs (syn, joint_su, a, b):", flush=True)
    for syn, jsu, a, b in top_pairs[:6]:
        star = "  <-- TRUE PAIR" if {a, b} == op_set else ""
        print(f"    syn={syn:+.4f} joint_su={jsu:.4f}  {a} x {b}{star}", flush=True)
    print(f"  >> true operand pair rank in SU-top-20: " f"{'NOT FOUND' if rank_of_true is None else '#'+str(rank_of_true)}", flush=True)

    # --- end-to-end: plain ShapSel vs ShapSel + SU-top-K product columns ----------------------
    def recall_of(sel_cols):
        s = set(c for c in sel_cols)
        # an operand counts as recovered if the raw col OR a product col containing it is selected
        rec = 0
        for op in operands:
            if op in s or any(op in str(c) and "__x__" in str(c) for c in sel_cols):
                rec += 1
        return rec

    rows = []
    # plain      : additive default selector, no products.
    # su_seeded  : PROTOTYPE proof -- top-8 SU pairs pre-engineered as product cols, plain selector.
    # su_inclass : PRODUCTION in-class path -- selector runs its OWN SNR-gated SU synergy screen
    #              (su_seeded_interactions=True) on the RAW frame; no externally-seeded columns. This
    #              is the path shipped in ShapProxiedFS; it must reproduce the prototype's win on the
    #              clear-SNR beds and NO-OP (== plain) on hard_synth where the SNR gate clears nothing.
    for tag, add_products, inclass in (
        ("plain", False, False),
        ("su_seeded", True, False),
        ("su_inclass", False, True),
    ):
        Xtr2, Xte2 = Xtr.copy(), Xte.copy()
        added = []
        if add_products:
            for syn, jsu, a, b in top_pairs[:8]:  # seed top-8 SU pairs as product cols
                nm = f"{a}__x__{b}"
                Xtr2[nm] = Xtr[a].values * Xtr[b].values
                Xte2[nm] = Xte[a].values * Xte[b].values
                added.append(nm)
        t1 = time.time()
        sel = S.ShapSel(su_seeded_interactions=inclass); sel.fit(Xtr2, ytr)
        fit_s = round(time.time() - t1, 1)
        chosen = list(sel.all_selected_)
        rec = recall_of(chosen)
        # downstream AUC on the selected set
        keep_tr = [c for c in chosen if c in Xtr2.columns]
        keep_te = [c for c in keep_tr if c in Xte2.columns]
        if not keep_te:
            keep_te = list(Xtr2.columns[:1])
        auc = downstream_auc(Xtr2[keep_te], Xte2[keep_te], ytr, yte)
        n_prod_sel = sum(1 for c in chosen if "__x__" in str(c))
        rows.append(dict(bed=name, variant=tag, n_sel=len(chosen), op_recall=rec,
                         n_added_prod=len(added), n_prod_selected=n_prod_sel, auc=auc, fit_s=fit_s))
        print(f"  [{tag:10s}] n_sel={len(chosen):3d} op_recall={rec}/{len(operands)} "
              f"prod_added={len(added)} prod_selected={n_prod_sel} auc={auc} ({fit_s}s)", flush=True)
        ck(f"A4-4 {name} {tag} recall={rec}/{len(operands)} auc={auc}")
    return rows, rank_of_true


def main():
    allrows = []; ranks = {}
    # bed 1: designed pure interaction (the hardest + the exact regime)
    Xp, yp, tp = make_pure_interaction(n=5000, p_noise=60, seed=0)
    r, rk = run_bed("pure_interaction", Xp, yp, tp); allrows += r; ranks["pure_interaction"] = rk
    # bed 2: hard_synth (ia*ib operands)
    Xh, yh, th = make_hard_dataset(n_samples=5000, seed=0)
    thr = {"operands": ["ia", "ib"], "informative": th["base"]}
    r, rk = run_bed("hard_synth", Xh, yh, thr); allrows += r; ranks["hard_synth"] = rk
    # bed 3: synth (inf_4*inf_5)
    Xs, ys, ts = make_dataset(n_samples=5000, seed=0)
    tsr = {"operands": ["inf_4", "inf_5"], "informative": ts["base"]}
    r, rk = run_bed("synth", Xs, ys, tsr); allrows += r; ranks["synth"] = rk

    df = pd.DataFrame(allrows)
    print("\n=== ALL ===\n" + df.to_string(index=False), flush=True)
    print("\n=== A4-4 VERDICT (proto = externally-seeded prototype; inclass = production in-class path) ===", flush=True)
    for bed in df.bed.unique():
        b = df[df.bed == bed].set_index("variant")
        d_rec = int(b.loc["su_seeded", "op_recall"]) - int(b.loc["plain", "op_recall"])
        d_auc = round(float(b.loc["su_seeded", "auc"]) - float(b.loc["plain", "auc"]), 4)
        d_rec_ic = int(b.loc["su_inclass", "op_recall"]) - int(b.loc["plain", "op_recall"])
        d_auc_ic = round(float(b.loc["su_inclass", "auc"]) - float(b.loc["plain", "auc"]), 4)
        rk = ranks[bed]
        print(f"  {bed:18s} screen_rank_of_true_pair="
              f"{'MISS' if rk is None else '#'+str(rk):5s}  "
              f"proto[d_recall={d_rec:+d} d_auc={d_auc:+}]  "
              f"inclass[d_recall={d_rec_ic:+d} d_auc={d_auc_ic:+}]", flush=True)
    df.to_csv("D:/Temp/round4_su_seeded_rows.csv", index=False)
    ck("A4-4 DONE")


if __name__ == "__main__":
    main()
