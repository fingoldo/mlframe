"""A2-4 prior_protected_rfecv -- seed RFECV must_include with a high-confidence core (Boruta hits /
MRMR gain) so RFECV only TRIMS the tail. Cheap falsifiable test.

PRODUCTION FACT: RFECV (wrappers/_rfecv.py) supports ``must_include`` (wrappers/_rfecv_must_include.py):
  pinned features bypass elimination; the optimiser searches ONLY the complement; final support_ =
  must_include UNION optimiser pick. Plain RFECV-as-a-hybrid-member was REJECTED (H3-4): on real data
  it OVER-selects (madelon kept 251) or its CV one_se_min over-prunes the clean union (A5-6 union_backward).

A2-4 DISTINCTION vs the rejected plain RFECV: do NOT let RFECV re-discover the core from scratch (where
  it over/under-selects). PIN a high-confidence core (Boruta-accepted UNION MRMR-top-gain) via
  must_include, so RFECV only decides the TAIL. The question is whether protecting the core changes
  RFECV's output enough to beat plain RFECV downstream.

FALSIFIABLE QUESTION: does prior-protected RFECV beat PLAIN RFECV on honest held-out AUC (hard_synth +
  synth), and does it land a more sensible support (recover the core RFECV would otherwise trim, without
  re-admitting the noise RFECV would otherwise keep)?

Beds: hard_synth (split signal; RFECV's known regime) + synth. Core = Boruta-accepted UNION MRMR raw
  selected (both already in fs_selectors). Variants:
   plain_rfecv         : S.RFECVSel('lgbm') -- the rejected baseline.
   protected_rfecv     : same RFECV but must_include = (boruta_accepted | mrmr_selected) high-conf core.
   core_only           : the pinned core alone (no RFECV) -- shows whether RFECV's tail adds anything.
PASS: protected_rfecv beats plain_rfecv AUC by >= +0.005 on a bed without regressing the other > 0.005.
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


def downstream(Xtr, Xte, ytr, yte, sel):
    sel = [c for c in dict.fromkeys(sel) if c in Xtr.columns and c in Xte.columns]
    if not sel:
        sel = list(Xtr.columns[:1])
    m = lgb.LGBMClassifier(n_estimators=300, verbose=-1, n_jobs=4).fit(Xtr[sel], ytr)
    return round(float(roc_auc_score(yte, m.predict_proba(Xte[sel])[:, 1])), 4), len(sel)


def high_conf_core(Xtr, ytr):
    """Boruta-accepted UNION MRMR raw selected -- the high-confidence core both selectors agree carries signal."""
    core = set()
    try:
        bs = S.BorutaSel(); bs.fit(Xtr, ytr)
        core |= set(c for c in bs.b_.accepted if c in Xtr.columns)
    except Exception as e:
        print(f"  (boruta core skip: {type(e).__name__})", flush=True)
    try:
        mr = S.MRMRSel(fe=False); mr.fit(Xtr, ytr)
        core |= set(c for c in mr.raw_selected_ if c in Xtr.columns)
    except Exception as e:
        print(f"  (mrmr core skip: {type(e).__name__})", flush=True)
    return [c for c in Xtr.columns if c in core]


def protected_rfecv(Xtr, ytr, core):
    from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig
    fi = FIConfig(importance_getter="auto", n_features_selection_rule="one_se_min")
    sc = SearchConfig(max_refits=18, max_runtime_mins=3)
    est = lgb.LGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.06, n_jobs=4, verbose=-1)
    r = RFECV(estimator=est, cv=3, scoring=None, verbose=0, fi_config=fi, search_config=sc,
              random_state=0, must_include=list(core))
    r.fit(Xtr, ytr)
    return [c for c in r.get_feature_names_out() if c in Xtr.columns]


def run_bed(name, X, y, truth, seed=0):
    print(f"\n=== {name} {X.shape} ===", flush=True); ck(f"A2-4 {name} start")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    relevant = set(truth["relevant"]); noise = set(truth["noise"])

    core = high_conf_core(Xtr, ytr)
    print(f"  high-conf core: n={len(core)} (rel={sum(1 for c in core if c in relevant)} "
          f"noise={sum(1 for c in core if c in noise)})", flush=True)

    rows = []
    # plain RFECV (rejected baseline)
    t0 = time.time()
    pr = S.RFECVSel("lgbm"); pr.fit(Xtr, ytr)
    plain_sel = pr.raw_selected_
    t_plain = round(time.time() - t0, 1)
    # protected RFECV
    t1 = time.time()
    prot_sel = protected_rfecv(Xtr, ytr, core)
    t_prot = round(time.time() - t1, 1)

    for tag, sel, ts in (("plain_rfecv", plain_sel, t_plain),
                         ("protected_rfecv", prot_sel, t_prot),
                         ("core_only", core, 0.0)):
        auc, n = downstream(Xtr, Xte, ytr, yte, sel)
        rec = sum(1 for c in sel if c in relevant); nn = sum(1 for c in sel if c in noise)
        rows.append(dict(bed=name, variant=tag, n=n, rel_recall=rec, n_noise=nn, auc=auc, fit_s=ts))
        print(f"  [{tag:18s}] n={n:3d} rel={rec} noise={nn} auc={auc} ({ts}s)", flush=True)
        ck(f"A2-4 {name} {tag} n={n} rel={rec} noise={nn} auc={auc}")
    return rows


def main():
    rows = []
    Xh, yh, th = make_hard_dataset(n_samples=5000, seed=0)
    rows += run_bed("hard_synth", Xh, yh, th)
    Xs, ys, ts = make_dataset(n_samples=5000, seed=0)
    rows += run_bed("synth", Xs, ys, ts)
    df = pd.DataFrame(rows)
    print("\n=== ALL ===\n" + df.to_string(index=False), flush=True)
    print("\n=== A2-4 VERDICT (protected_rfecv vs plain_rfecv) ===", flush=True)
    for bed in df.bed.unique():
        b = df[df.bed == bed].set_index("variant")
        d = round(float(b.loc["protected_rfecv", "auc"]) - float(b.loc["plain_rfecv", "auc"]), 4)
        d_core = round(float(b.loc["protected_rfecv", "auc"]) - float(b.loc["core_only", "auc"]), 4)
        print(f"  {bed:12s} d_auc_vs_plain={d:+}  (plain n={int(b.loc['plain_rfecv','n'])} "
              f"protected n={int(b.loc['protected_rfecv','n'])} core n={int(b.loc['core_only','n'])}; "
              f"protected-vs-core_only d_auc={d_core:+})", flush=True)
    df.to_csv("D:/Temp/round4_prior_protected_rfecv_rows.csv", index=False)
    ck("A2-4 DONE")


if __name__ == "__main__":
    main()
