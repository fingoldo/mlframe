"""A2-7 tentative_to_cmi -- route Boruta's tentative set to MRMR conditional-MI, admit if it adds
conditional information beyond the accepted core. Cheap falsifiable test.

PRODUCTION FACT: BorutaShap partitions features into accepted / rejected / TENTATIVE (the shadow gate
  could not decide). On hard_synth the WEAK-SPARSE block (8 features, individually small coef ~0.40
  buried in 200 noise) is exactly the block a shallow shadow gate misses and a marginal-MI greedy MRMR
  drops -- it is the documented "block MRMR-FE alone loses." Many of those weak features land in Boruta's
  TENTATIVE bucket rather than rejected.

A2-7 PROPOSAL: for each tentative feature f, compute MRMR conditional-MI  I(f ; y | accepted_core).
  Admit f if it clears a permutation null (it carries information about y BEYOND what the accepted core
  already explains). The accepted core conditions out redundant copies; weak-but-complementary features
  survive, pure noise does not.

FALSIFIABLE QUESTION: does CMI-gating Boruta's tentative set on hard_synth RECOVER weak-sparse features
  (raising weak-block recall) at acceptable noise admission, and lift downstream AUC over plain Boruta?

METHOD: CMI computed on quantile-binned columns with the standard discrete plug-in estimator
  I(X;Y|Z) = H(X,Z)+H(Y,Z)-H(Z)-H(X,Y,Z) (the same quantity mrmr.conditional_mi computes; reimplemented
  on numpy bins because the njit kernel needs prebinned factors_data plumbing not worth standing up for a
  prototype gate). Z = accepted core (joined-bin), permutation null from shuffling f's bins B times.
  Baselines: plain boruta accepted; boruta accepted + CMI-admitted tentative; (control) boruta
  accepted + ALL tentative (no gate) to show the gate is doing work, not just "add everything".

PASS: CMI-gate raises weak-block recall by >= +2 over plain boruta WITHOUT admitting > the all-tentative
  control's noise, AND lifts downstream LGBM AUC on hard_synth.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hard_synth import make_hard_dataset
from synth import make_dataset
import fs_selectors as S

CK = "D:/Temp/queue_ideas_progress.txt"
def ck(m):
    with open(CK, "a") as f:
        f.write(m + "\n")


def _qbin(col, nbins=8):
    col = np.asarray(col, dtype=float)
    edges = np.unique(np.quantile(col, np.linspace(0, 1, nbins + 1)))
    if len(edges) <= 2:
        return np.zeros(len(col), dtype=np.int64)
    return np.clip(np.digitize(col, edges[1:-1]), 0, len(edges) - 2).astype(np.int64)

def _entropy(labels):
    _, cnt = np.unique(labels, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log(p)).sum())

def _joint(*arrs):
    out = arrs[0].astype(np.int64).copy()
    for a in arrs[1:]:
        out = out * (int(a.max()) + 1) + a.astype(np.int64)
        _, out = np.unique(out, return_inverse=True)
    return out.astype(np.int64)

def cmi(xb, yb, zb):
    """I(X;Y|Z) = H(X,Z)+H(Y,Z)-H(Z)-H(X,Y,Z) on discrete bins."""
    if zb is None or (hasattr(zb, "size") and zb.size == 0):
        # unconditional MI
        return _entropy(xb) + _entropy(yb) - _entropy(_joint(xb, yb))
    Hxz = _entropy(_joint(xb, zb)); Hyz = _entropy(_joint(yb, zb))
    Hz = _entropy(zb); Hxyz = _entropy(_joint(xb, yb, zb))
    return Hxz + Hyz - Hz - Hxyz


def cmi_gate(X, y, accepted, tentative, nbins=8, n_perm=40, alpha=0.1, seed=0):
    """Return tentative features whose CMI(f;y|accepted_core) exceeds the permutation-null (1-alpha) quantile."""
    rng = np.random.default_rng(seed)
    yb = np.asarray(y).astype(np.int64) if len(np.unique(y)) <= 20 else _qbin(np.asarray(y), nbins)
    bins = {c: _qbin(X[c].values, nbins) for c in (list(accepted) + list(tentative)) if c in X.columns}
    core = [c for c in accepted if c in bins]
    # cap conditioning set to the strongest few accepted (full joint of many cols explodes cardinality)
    if len(core) > 6:
        core = sorted(core, key=lambda c: -cmi(bins[c], yb, None))[:6]
    zb = _joint(*[bins[c] for c in core]) if core else None
    admitted = []
    diag = []
    for f in tentative:
        if f not in bins:
            continue
        xb = bins[f]
        obs = cmi(xb, yb, zb)
        null = np.array([cmi(xb[rng.permutation(len(xb))], yb, zb) for _ in range(n_perm)])
        thr = float(np.quantile(null, 1 - alpha))
        if obs > thr:
            admitted.append(f)
        diag.append((f, round(obs, 5), round(thr, 5)))
    return admitted, diag


def downstream(Xtr, Xte, ytr, yte):
    keep = [c for c in Xtr.columns if c in Xte.columns]
    if not keep:
        keep = list(Xtr.columns[:1])
    m = lgb.LGBMClassifier(n_estimators=300, verbose=-1, n_jobs=4).fit(Xtr[keep], ytr)
    return round(float(roc_auc_score(yte, m.predict_proba(Xte[keep])[:, 1])), 4)


def block_recall(sel, block):
    s = set(sel); return sum(1 for f in block if f in s)


def run_bed(name, X, y, truth, weak_key, seed=0):
    print(f"\n=== {name} {X.shape} ===", flush=True); ck(f"A2-7 {name} start")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    weak = truth.get(weak_key, [])
    relevant = set(truth["relevant"]); noise = set(truth["noise"])

    t0 = time.time()
    bs = S.BorutaSel(); bs.fit(Xtr, ytr)
    b = bs.b_
    accepted = [c for c in getattr(b, "accepted", []) if c in Xtr.columns]
    tentative = [c for c in getattr(b, "tentative", []) if c in Xtr.columns]
    fit_s = round(time.time() - t0, 1)
    print(f"  boruta {fit_s}s: accepted={len(accepted)} tentative={len(tentative)} "
          f"(weak in accepted={block_recall(accepted, weak)}/{len(weak)}, "
          f"weak in tentative={block_recall(tentative, weak)}/{len(weak)})", flush=True)

    t1 = time.time()
    admitted, diag = cmi_gate(Xtr, ytr, accepted, tentative, seed=seed)
    gate_s = round(time.time() - t1, 2)
    weak_admitted = [f for f in admitted if f in weak]
    noise_admitted = [f for f in admitted if f in noise]
    print(f"  CMI-gate {gate_s}s: admitted {len(admitted)} of {len(tentative)} tentative "
          f"(weak={len(weak_admitted)}, noise={len(noise_admitted)}, other={len(admitted)-len(weak_admitted)-len(noise_admitted)})", flush=True)

    variants = {
        "boruta_accepted": accepted,
        "cmi_admit": accepted + admitted,
        "all_tentative(ctrl)": accepted + tentative,
    }
    rows = []
    for tag, sel in variants.items():
        sel = [c for c in dict.fromkeys(sel) if c in Xtr.columns]
        if not sel:
            sel = list(Xtr.columns[:1])
        auc = downstream(Xtr[sel], Xte[sel], ytr, yte)
        rec_rel = sum(1 for c in sel if c in relevant)
        rec_weak = block_recall(sel, weak)
        n_noise = sum(1 for c in sel if c in noise)
        rows.append(dict(bed=name, variant=tag, n=len(sel), weak_recall=rec_weak, rel_recall=rec_rel, n_noise=n_noise, auc=auc))
        print(f"  [{tag:20s}] n={len(sel):3d} weak={rec_weak}/{len(weak)} rel={rec_rel} " f"noise={n_noise} auc={auc}", flush=True)
        ck(f"A2-7 {name} {tag} weak={rec_weak}/{len(weak)} noise={n_noise} auc={auc}")
    return rows


def main():
    rows = []
    Xh, yh, th = make_hard_dataset(n_samples=5000, seed=0)
    rows += run_bed("hard_synth", Xh, yh, th, "weak_sparse")
    df = pd.DataFrame(rows)
    print("\n=== ALL ===\n" + df.to_string(index=False), flush=True)
    print("\n=== A2-7 VERDICT ===", flush=True)
    for bed in df.bed.unique():
        b = df[df.bed == bed].set_index("variant")
        base = b.loc["boruta_accepted"]; gate = b.loc["cmi_admit"]; ctrl = b.loc["all_tentative(ctrl)"]
        d_weak = int(gate.weak_recall) - int(base.weak_recall)
        d_auc = round(float(gate.auc) - float(base.auc), 4)
        d_noise = int(gate.n_noise) - int(base.n_noise)
        print(f"  {bed:12s} cmi_admit vs boruta: d_weak_recall={d_weak:+d} d_noise={d_noise:+d} d_auc={d_auc:+}", flush=True)
        print(f"           (gate admitted noise={int(gate.n_noise)} vs ALL-tentative ctrl noise={int(ctrl.n_noise)}; "
              f"ctrl auc={ctrl.auc} vs gate auc={gate.auc})", flush=True)
    df.to_csv("D:/Temp/round4_tentative_cmi_rows.csv", index=False)
    ck("A2-7 DONE")


if __name__ == "__main__":
    main()
