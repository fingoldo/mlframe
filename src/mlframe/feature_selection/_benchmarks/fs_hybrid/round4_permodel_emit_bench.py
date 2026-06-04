"""Round-4 A5-4 per_model_emit: does emitting a DIFFERENT feature set per downstream FAMILY beat the
shared one-size-fits-all set ON SPLIT-SIGNAL data?

The production HybridSelector emits ONE `raw_selected_` set S consumed by every downstream model. Round-3
validated "one-size-fits-all" only on the FE-saturated easy bed (everything at ceiling). On SPLIT-SIGNAL
regimes (hard_synth, madelon) it is UNTESTED. Hypothesis: linear (logit) and tree (lgbm) downstreams want
DIFFERENT sets -- logit benefits from the engineered/tprod interaction COLUMNS + a harder de-collinearized
set (1 rep/cluster); lgbm tolerates redundancy and wants the RAW operands that built those products.

Method (NO re-fit -- all sets derive from ONE fitted HybridSelector state per (bed, seed)):
  baseline  = shared S: every downstream scored on h.raw_selected_  (exactly the production behavior).
  per-family: each family gets ITS OWN set, derived from the stashed fi_ / members_ / cluster_of_ /
              member_selections_ / _Xaug_ columns; each model scored on its set; the per-family MEAN is the
              average of (logit on its set, lgbm on its set, knn on its set).

Per-family rules (2-3 each, all measured):
  LINEAR  L1 eng_decollin : S, force-add every engineered/tprod col that cleared the FI prescreen, then keep
                            exactly 1 (highest-FI) representative per corr-cluster (hard de-collinearize).
          L2 eng_priority : like S but for each kept engineered-bearing cluster prefer the ENGINEERED member,
                            and force-add high-FI (top-quartile) engineered cols even if only 1 member voted.
  TREE    T1 raw_operands : S plus the RAW operands (a,b) behind every included tprod / interaction column,
                            no de-dup pressure (keep redundant copies tree tolerates).
          T2 fi_rep       : the FI-rep set == S (control: lgbm on the shared set).
  KNN     K1 clean_decollin : 1 highest-FI rep per voted cluster, engineered included, low-dim (== a clean S).

Beds: hard_synth (n=5000), madelon (load_real), synth (n=5000). Seeds 0,1,2.
VERDICT: per-family beats shared by >= +0.005 on hard_synth or madelon WITHOUT hurting synth (> -0.005)?
A clean NEGATIVE (one-size-fits-all holds even on split-signal) is a valuable, reportable result.
"""
from __future__ import annotations
import os, sys, time, gc
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import load_real
from synth import make_dataset
from hard_synth import make_hard_dataset
from hybrid_selector import HybridSelector

PROG = "D:/Temp/permodel_progress.txt"


def _ckpt(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
    with open(PROG, "a", encoding="ascii", errors="replace") as f:
        f.write(line)
    print(line.rstrip(), flush=True)


# memory-frugal downstream models (capped n_jobs / n_estimators); one model per family.
def _lgbm_auc(Xtr, Xte, ytr, yte):
    m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, n_jobs=4, verbose=-1, random_state=0)
    m.fit(Xtr, ytr)
    return round(float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1])), 4)


def _logit_auc(Xtr, Xte, ytr, yte):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(Xtr, ytr)
    return round(float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1])), 4)


def _knn_auc(Xtr, Xte, ytr, yte):
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, n_jobs=4))
    m.fit(Xtr, ytr)
    return round(float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1])), 4)


class PerModelHybrid(HybridSelector):
    """Hybrid that derives per-family feature sets from the fitted state (no re-fit). All helpers are pure
    functions of fi_ / members_ / cluster_of_ / member_selections_ / _eng_rename / _tree_prod_* and the
    augmented column list."""

    # -------- introspection over the fitted state
    def _engineered_cols(self):
        """All engineered column names present in the augmented frame: MRMR eng_N + admitted tree tprod_N."""
        eng = set(self._eng_rename.values()) if getattr(self, "_eng_rename", None) else set()
        eng |= set(getattr(self, "_tree_prod_names_", []) or [])
        cols = set(self._Xaug_.columns)
        return {c for c in eng if c in cols}

    def _voted_clusters(self):
        """rep -> set of members that voted, over the production member_selections_ (>=1 vote => kept by vote=1)."""
        from collections import defaultdict
        cv = defaultdict(set)
        for m, sel in self.member_selections_.items():
            for f in sel:
                r = self.cluster_of_.get(f)
                if r is not None:
                    cv[r].add(m)
        return cv

    def _top_fi_rep(self, rep, cols):
        ms = [m for m in self.members_[rep] if m in cols]
        return max(ms, key=lambda f: self.fi_.get(f, 0.0)) if ms else None

    def _vote_count(self):
        """rep -> number of distinct members that voted for the cluster."""
        return {r: len(v) for r, v in self._voted_clusters().items()}

    def _consensus_floor_fi(self, vc):
        """Median rep-FI over consensus (>=2 vote) clusters -- the credibility floor for single-vote clusters."""
        cols = list(self._Xaug_.columns)
        cons = [self.fi_.get(self._top_fi_rep(r, cols), 0.0) for r, n in vc.items() if n >= 2]
        return float(np.median(cons)) if cons else float("-inf")

    # The fitted state on split-signal beds (verified on hard_synth): the ONLY corr-cluster with >1 member is the
    # redundant block; S already collapses it to one rep AND already carries every engineered col. So the genuine
    # per-family divergence is NOT "force engineered / harder de-collinearize" (S already does both) but:
    #   * LINEAR / KNN are hurt by the single-vote NOISE leaks S admits under vote=1 -> they want a CLEANER set
    #     (consensus-only, or FI-floored single-vote) while still keeping the engineered interaction columns.
    #   * TREE tolerates noise + redundancy -> it can take S verbatim, or even the EXPANDED redundant copies + the
    #     raw operands behind the engineered/product columns (operands the tree branches on natively).

    # -------- LINEAR family (wants the engineered cols but a cleaner, lower-noise set)
    def set_linear_consensus_eng(self):
        """Keep only CONSENSUS clusters (>=2 member votes), 1 highest-FI rep each, PLUS every engineered col (the
        explicit interaction columns linear models need). Drops the single-vote noise leaks that cap logit."""
        cols = list(self._Xaug_.columns)
        eng = self._engineered_cols()
        vc = self._vote_count()
        out = [self._top_fi_rep(r, cols) for r, n in vc.items() if n >= 2]
        out = [c for c in out if c is not None]
        out += [c for c in eng if c in cols]                       # always keep engineered interaction cols
        if len(out) < 2:                                           # never collapse to nothing
            out = list(self.raw_selected_)
        return [c for c in dict.fromkeys(out) if c in cols]

    def set_linear_fi_floored(self):
        """S, but drop single-vote clusters whose rep-FI is below the median consensus-rep FI (softer noise cut that
        keeps high-FI single-vote REAL features while shedding the low-FI noise leaks). Engineered always kept."""
        cols = list(self._Xaug_.columns)
        eng = self._engineered_cols()
        vc = self._vote_count()
        floor = self._consensus_floor_fi(vc)
        out = []
        for r, n in vc.items():
            rep = self._top_fi_rep(r, cols)
            if rep is None:
                continue
            if n >= 2 or rep in eng or self.fi_.get(rep, 0.0) >= floor:
                out.append(rep)
        out += [c for c in eng if c in cols]
        if len(out) < 2:
            out = list(self.raw_selected_)
        return [c for c in dict.fromkeys(out) if c in cols]

    # -------- TREE family (tolerates noise + redundancy; wants raw operands)
    def set_tree_expand_operands(self):
        """S, but EXPAND each kept cluster to ALL its members (redundant copies the tree tolerates) and ADD the raw
        operands (a,b) behind every included tprod/engineered column the tree would branch on natively."""
        cols = list(self._Xaug_.columns)
        out = list(self.raw_selected_)
        sel = set(out)
        # expand redundant clusters back to all members
        for c in list(self.raw_selected_):
            r = self.cluster_of_.get(c)
            if r is not None:
                for m in self.members_[r]:
                    if m in cols and m not in sel:
                        out.append(m); sel.add(m)
        # add raw operands behind any included tprod column
        pairs = getattr(self, "_tree_prod_pairs_", []) or []
        names = getattr(self, "_tree_prod_names_", []) or []
        for nm, (a, b) in zip(names, pairs):
            if nm in sel:
                for op in (a, b):
                    if op in cols and op not in sel:
                        out.append(op); sel.add(op)
        # add raw cols any tree-member voted for but that got collapsed away
        for c in self.member_selections_.get("tree", []):
            if c in cols and c not in sel:
                out.append(c); sel.add(c)
        return [c for c in dict.fromkeys(out) if c in cols]

    def set_tree_shared_S(self):
        """Control: the shared production selection S (lgbm on exactly what every model gets today)."""
        return list(self.raw_selected_)

    # -------- KNN family (most noise/redundancy sensitive; wants the tightest clean de-collinearized set)
    def set_knn_consensus_clean(self):
        """Tightest clean set: consensus clusters only (>=2 votes), 1 highest-FI rep each, engineered included."""
        cols = list(self._Xaug_.columns)
        eng = self._engineered_cols()
        vc = self._vote_count()
        out = [self._top_fi_rep(r, cols) for r, n in vc.items() if n >= 2]
        out = [c for c in out if c is not None]
        out += [c for c in eng if c in cols]
        if len(out) < 2:
            out = list(self.raw_selected_)
        return [c for c in dict.fromkeys(out) if c in cols]


def run_bed(name, X, y, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    t0 = time.time()
    h = PerModelHybrid(vote=1, use_fe=True, random_state=seed).fit(Xtr, ytr)
    fit_s = round(time.time() - t0, 1)
    cols = list(h._Xaug_.columns)
    Ztr, Zte = h._augment(Xtr), h._augment(Xte)

    def slice_ok(sel):
        s = [c for c in dict.fromkeys(sel) if c in Ztr.columns and c in Zte.columns]
        return s if s else list(h.raw_selected_)

    S = slice_ok(list(h.raw_selected_))
    L1 = slice_ok(h.set_linear_consensus_eng())   # consensus + engineered (drop single-vote noise)
    L2 = slice_ok(h.set_linear_fi_floored())      # FI-floored single-vote cut + engineered
    T1 = slice_ok(h.set_tree_expand_operands())   # expand redundant copies + raw operands
    T2 = slice_ok(h.set_tree_shared_S())          # control == S
    K1 = slice_ok(h.set_knn_consensus_clean())    # tightest clean consensus set

    def auc(model_fn, sel):
        return model_fn(Ztr[sel], Zte[sel], ytr, yte)

    # baseline: every family on the shared set S
    base_lgbm = auc(_lgbm_auc, S)
    base_logit = auc(_logit_auc, S)
    base_knn = auc(_knn_auc, S)
    base_mean = round(float(np.mean([base_lgbm, base_logit, base_knn])), 4)

    # per-family: each model on its own set. Picking the best per family from its candidate sets is NOT allowed
    # (that would be test-set leakage). We FIX one rule per family a priori but report all rule AUCs so the
    # reader sees the spread; the per-family MEAN uses the designated default rule per family:
    #   linear -> L1 (consensus_eng), tree -> T1 (expand_operands), knn -> K1 (consensus_clean).
    pf_logit_L1 = auc(_logit_auc, L1)
    pf_logit_L2 = auc(_logit_auc, L2)
    pf_lgbm_T1 = auc(_lgbm_auc, T1)
    pf_lgbm_T2 = auc(_lgbm_auc, T2)
    pf_knn_K1 = auc(_knn_auc, K1)

    # designated per-family default mean (a priori choice, no leakage)
    pf_mean = round(float(np.mean([pf_lgbm_T1, pf_logit_L1, pf_knn_K1])), 4)
    # also an ORACLE per-family mean (best rule per family) to show the UPPER BOUND of the approach
    oracle_logit = max(pf_logit_L1, pf_logit_L2)
    oracle_lgbm = max(pf_lgbm_T1, pf_lgbm_T2)
    oracle_mean = round(float(np.mean([oracle_lgbm, oracle_logit, pf_knn_K1])), 4)

    row = dict(
        bed=name, seed=seed, fit_s=fit_s,
        n_S=len(S), n_L1=len(L1), n_L2=len(L2), n_T1=len(T1), n_T2=len(T2), n_K1=len(K1),
        base_lgbm=base_lgbm, base_logit=base_logit, base_knn=base_knn, base_mean=base_mean,
        pf_lgbm_T1=pf_lgbm_T1, pf_lgbm_T2=pf_lgbm_T2,
        pf_logit_L1=pf_logit_L1, pf_logit_L2=pf_logit_L2,
        pf_knn_K1=pf_knn_K1,
        pf_mean=pf_mean, oracle_mean=oracle_mean,
        d_pf=round(pf_mean - base_mean, 4), d_oracle=round(oracle_mean - base_mean, 4),
    )
    _ckpt(f"{name} seed={seed} fit={fit_s}s | base_mean={base_mean} pf_mean={pf_mean} (d={row['d_pf']:+}) "
          f"oracle={oracle_mean} (d={row['d_oracle']:+}) | "
          f"lgbm base={base_lgbm} T1={pf_lgbm_T1} | logit base={base_logit} L1={pf_logit_L1} L2={pf_logit_L2} | "
          f"knn base={base_knn} K1={pf_knn_K1} | n_S={len(S)} n_L1={len(L1)} n_T1={len(T1)} n_K1={len(K1)}")
    del h, Ztr, Zte, Xtr, Xte; gc.collect()
    return row


def run_with_retry(name, X, y, seed):
    try:
        return run_bed(name, X, y, seed)
    except Exception as e:
        _ckpt(f"RETRY {name} seed={seed} after error: {type(e).__name__}: {e}")
        time.sleep(60)
        return run_bed(name, X, y, seed)


def main():
    open(PROG, "w").close()
    _ckpt("START per_model_emit bench")
    seeds = [0, 1, 2]
    allrows = []

    # build datasets once (madelon loaded once -> resample-free; seeds vary the train/test split)
    _ckpt("loading madelon ...")
    Xr, yr, rname = load_real()
    _ckpt(f"madelon loaded: {Xr.shape} name={rname}")

    beds = []
    for seed in seeds:
        Xh, yh, _ = make_hard_dataset(n_samples=5000, seed=seed)
        beds.append(("hard_synth", Xh, yh, seed))
    for seed in seeds:
        beds.append((rname, Xr, yr, seed))
    for seed in seeds:
        Xs, ys, _ = make_dataset(n_samples=5000, seed=seed)
        beds.append(("synth", Xs, ys, seed))

    for name, X, y, seed in beds:
        allrows.append(run_with_retry(name, X, y, seed))

    df = pd.DataFrame(allrows)
    df.to_csv("D:/Temp/permodel_rows.csv", index=False)

    # aggregate per bed
    print("\n=== ALL ROWS ===", flush=True)
    print(df.to_string(index=False), flush=True)

    print("\n=== PER-BED MEANS (across seeds) ===", flush=True)
    agg = df.groupby("bed").agg(
        base_mean=("base_mean", "mean"), pf_mean=("pf_mean", "mean"), oracle_mean=("oracle_mean", "mean"),
        base_lgbm=("base_lgbm", "mean"), pf_lgbm_T1=("pf_lgbm_T1", "mean"),
        base_logit=("base_logit", "mean"), pf_logit_L1=("pf_logit_L1", "mean"), pf_logit_L2=("pf_logit_L2", "mean"),
        base_knn=("base_knn", "mean"), pf_knn_K1=("pf_knn_K1", "mean"),
    ).round(4)
    agg["d_pf"] = (agg["pf_mean"] - agg["base_mean"]).round(4)
    agg["d_oracle"] = (agg["oracle_mean"] - agg["base_mean"]).round(4)
    print(agg.to_string(), flush=True)

    print("\n=== VERDICT ===", flush=True)
    for bed in ["hard_synth", rname, "synth"]:
        if bed not in agg.index:
            continue
        d = float(agg.loc[bed, "d_pf"])
        do = float(agg.loc[bed, "d_oracle"])
        print(f"  {bed:14s} per-family-default d={d:+.4f}  oracle d={do:+.4f}", flush=True)
    hard_d = float(agg.loc["hard_synth", "d_pf"]) if "hard_synth" in agg.index else 0.0
    real_d = float(agg.loc[rname, "d_pf"]) if rname in agg.index else 0.0
    synth_d = float(agg.loc["synth", "d_pf"]) if "synth" in agg.index else 0.0
    win = (hard_d >= 0.005 or real_d >= 0.005) and synth_d >= -0.005
    print(f"\n  PASS criterion (>= +0.005 on hard_synth or {rname} AND synth not hurt > 0.005): "
          f"{'WIN' if win else 'NEGATIVE (one-size-fits-all holds)'}", flush=True)
    _ckpt("DONE")


if __name__ == "__main__":
    main()
