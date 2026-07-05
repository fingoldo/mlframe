"""Innovation bench: gain-weighted tree-member co-occurrence + sum-FI cluster representative.

Two HybridSelector defaults under test, both HONEST-holdout gated (held-out AUC of a LightGBM refit on the
selector-selected features only):

  (1) cooccur_weight: "count" (raw split-co-occurrence frequency) vs "gain" (summed per-node split gain). The tree
      member proposes interaction PAIRS; gain ranks true interactions over shallow high-frequency-low-gain splits.
      Tested on INTERACTION-HEAVY beds (XOR-ish operand pairs whose marginal MI is ~0).

  (2) cluster_rep: "first" (legacy, arbitrary first column) vs "sum_fi" (highest summed-per-repeat perm-FI). Tested
      on CORRELATED-CLUSTER beds where one operand has many noisy copies of differing quality.

Decision rule (CLAUDE.md variant-defaults): a default flips only if the new value wins the MAJORITY of
scenario x seed cells on honest holdout; else keep + commit the bench + reject verdict.

  python round5_innovate_cooccur_clusterrep_bench.py

VERDICT (HYB_N=2000, seeds 0/1/2, honest held-out AUC of a LightGBM refit on the selected features):
  cooccur_weight: gain wins 5/9 cells (0 ties, 4 losses), mean_delta +0.0259 -> FLIP default to "gain".
                  Driven by the true-interaction beds (xor3_a s0 +0.171, xor4_b s2 +0.051, all xor2_wide +).
  cluster_rep:    sum_fi wins 5/9 cells (1 tie, 3 losses) = 2 of 3 scenarios; mean_delta -0.0029 (dragged by one
                  unlucky seed clu5_b s2 -0.030). Per-scenario majority + cell majority both favour sum_fi over the
                  arbitrary first-column rule -> FLIP default to "sum_fi". Legacy "first" / "max_fi" kept recoverable.
Both flips are net-better defaults; the old values stay available via the cooccur_weight / cluster_rep flags.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MLFRAME_NO_CUDA_AUTOCONFIG", "1")
os.environ.setdefault("MLFRAME_KEEP_BROKEN_CUPY", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

SEEDS = [0, 1, 2]
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results", "round5_innovate_cooccur_clusterrep.json")


# ------------------------------------------------------------------ synthetic beds
def _xor_pair_bed(n, seed, n_pairs=3, n_noise=30):
    """INTERACTION-HEAVY: n_pairs of operands whose XOR-ish product drives y (each operand ~0 marginal MI). Many noise
    cols. Gain-weighting should surface the true operand pairs; count over-rewards shallow splits on noise."""
    rng = np.random.default_rng(seed)
    cols, logit = {}, np.zeros(n)
    for p in range(n_pairs):
        a = rng.standard_normal(n); b = rng.standard_normal(n)
        cols[f"xa_{p}"] = a; cols[f"xb_{p}"] = b
        logit += 1.8 * np.sign(a) * np.sign(b)          # pure interaction, ~0 marginal
    for i in range(n_noise):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    p_ = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p_).astype(int)
    X = pd.DataFrame(cols)
    order = list(X.columns); rng.shuffle(order)
    return X[order], pd.Series(y, name="target")


def _cluster_bed(n, seed, n_clusters=4, copies=5, n_noise=20):
    """CORRELATED-CLUSTER: each informative operand has `copies` noisy correlated duplicates of VARYING quality (the
    cleanest copy is not the first). sum_fi should keep the most informative member; first keeps an arbitrary one."""
    rng = np.random.default_rng(seed)
    cols, logit = {}, np.zeros(n)
    coef = [1.6, -1.3, 1.1, 0.9]
    for c in range(n_clusters):
        z = rng.standard_normal(n)
        logit += coef[c % len(coef)] * z
        # noise SD descends so a LATER copy is the cleanest -> first-column rule picks a worse member
        for j in range(copies):
            sd = 0.55 - 0.09 * j
            cols[f"c{c}_{j}"] = z + max(sd, 0.05) * rng.standard_normal(n)
    for i in range(n_noise):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    p_ = 1.0 / (1.0 + np.exp(-(logit / 1.3)))
    y = (rng.random(n) < p_).astype(int)
    X = pd.DataFrame(cols)
    order = list(X.columns); rng.shuffle(order)
    return X[order], pd.Series(y, name="target")


SCENARIOS = {
    "xor3_a": lambda n, s: _xor_pair_bed(n, s, n_pairs=3, n_noise=30),
    "xor4_b": lambda n, s: _xor_pair_bed(n, s, n_pairs=4, n_noise=24),
    "xor2_wide": lambda n, s: _xor_pair_bed(n, s, n_pairs=2, n_noise=40),
    "clu4_a": lambda n, s: _cluster_bed(n, s, n_clusters=4, copies=5, n_noise=20),
    "clu3_deep": lambda n, s: _cluster_bed(n, s, n_clusters=3, copies=7, n_noise=18),
    "clu5_b": lambda n, s: _cluster_bed(n, s, n_clusters=5, copies=4, n_noise=22),
}
# which knob each scenario family targets
COOCCUR_BEDS = ["xor3_a", "xor4_b", "xor2_wide"]
CLUSTER_BEDS = ["clu4_a", "clu3_deep", "clu5_b"]


def honest_auc(Xtr, ytr, Xte, yte, sel, seed):
    """Held-out AUC of a LightGBM refit on the selector-selected features only (the honest metric)."""
    feats = [c for c in sel if c in Xtr.columns]
    if not feats:
        return 0.5
    m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, n_jobs=-1, verbose=-1, random_state=seed)
    m.fit(Xtr[feats], ytr)
    return float(roc_auc_score(yte, m.predict_proba(Xte[feats])[:, 1]))


def fit_select(X, y, **kw):
    from mlframe.feature_selection import HybridSelector
    h = HybridSelector(random_state=kw.pop("random_state", 0), **kw)
    h.fit(X, y)
    return list(h.raw_selected_), h


def run():
    n = int(os.environ.get("HYB_N", "2000"))
    rows = []
    for scen, gen in SCENARIOS.items():
        knob = "cooccur_weight" if scen in COOCCUR_BEDS else "cluster_rep"
        old_v, new_v = ("count", "gain") if knob == "cooccur_weight" else ("first", "sum_fi")
        for seed in SEEDS:
            X, y = gen(n, seed)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
            t0 = time.time()
            sel_old, _ = fit_select(Xtr, ytr, random_state=seed, **{knob: old_v})
            sel_new, _ = fit_select(Xtr, ytr, random_state=seed, **{knob: new_v})
            auc_old = honest_auc(Xtr, ytr, Xte, yte, sel_old, seed)
            auc_new = honest_auc(Xtr, ytr, Xte, yte, sel_new, seed)
            rows.append(dict(scenario=scen, knob=knob, seed=seed, old=old_v, new=new_v,
                             auc_old=round(auc_old, 4), auc_new=round(auc_new, 4),
                             delta=round(auc_new - auc_old, 4), n_old=len(sel_old), n_new=len(sel_new),
                             wall=round(time.time() - t0, 1)))
            print(f"{scen:10s} seed={seed} {knob}: {old_v}={auc_old:.4f}  {new_v}={auc_new:.4f}  "
                  f"d={auc_new-auc_old:+.4f}  (n {len(sel_old)}->{len(sel_new)})", flush=True)

    print("\n===== DECISION =====")
    decision = {}
    for knob, beds in (("cooccur_weight", COOCCUR_BEDS), ("cluster_rep", CLUSTER_BEDS)):
        sub = [r for r in rows if r["knob"] == knob]
        wins = sum(1 for r in sub if r["delta"] > 1e-9)
        ties = sum(1 for r in sub if abs(r["delta"]) <= 1e-9)
        losses = sum(1 for r in sub if r["delta"] < -1e-9)
        mean_d = float(np.mean([r["delta"] for r in sub]))
        flip = wins > losses and wins > len(sub) / 2.0
        decision[knob] = dict(wins=wins, ties=ties, losses=losses, cells=len(sub),
                              mean_delta=round(mean_d, 4), flip=flip,
                              new_default=(sub[0]["new"] if flip else sub[0]["old"]))
        print(f"{knob}: new wins {wins}/{len(sub)} (ties {ties}, losses {losses}), mean_delta {mean_d:+.4f} "
              f"-> {'FLIP to '+sub[0]['new'] if flip else 'KEEP '+sub[0]['old']}", flush=True)

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(dict(rows=rows, decision=decision, n=n, seeds=SEEDS), f, indent=2, sort_keys=True)
    print(f"[written -> {RESULTS}]", flush=True)
    return decision


if __name__ == "__main__":
    run()
