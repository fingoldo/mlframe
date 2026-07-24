"""Round-4 cross-selector SYNERGY combine-refinements, measured against the tree-member hybrid.

Three ideas, all sharing ONE member fit (the combine rule is cheap -> fit once per bed, apply each rule):

  A2-3/A5-1 confidence_prior : per-FEATURE confidence from the members the hybrid currently DISCARDS
      (Boruta accepted=1.0 / tentative=0.5 tier + distinct-member vote count + normalized shared FI). A single-
      member cluster is kept only if its max-confidence clears the consensus-cluster confidence floor; consensus
      (>=2 members) always kept. A multi-signal generalization of fi_guard (which used FI-median alone and cut recall).
  A2-1 disagreement_referee  : consensus clusters kept, zero-vote dropped, CONTESTED (1-member) clusters routed
      through a held-out forward-AUC referee (admit the rep only if it lifts held-out AUC over the consensus core).
  A5-3 tentative_rescue      : binary vote=1 baseline PLUS readmit Boruta-tentative features that a DIFFERENT
      member (mrmr/shap/tree) also selected OR whose shared FI is top-quartile (disagreement-as-signal, recall-add).

Baseline = the shipped tree-member hybrid (binary cluster vote=1). PASS for any variant: beats baseline by
>= +0.005 on a bed without regressing the others > 0.005. Reported with ALL numbers (no top-N filtering).
"""
from __future__ import annotations
import logging, os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import load_real, downstream
from synth import make_dataset
from hard_synth import make_hard_dataset
from hybrid_selector import HybridSelector


class SynergyHybrid(HybridSelector):
    """Tree-member hybrid that ALSO captures the per-member confidence the base class discards (Boruta
    accepted/tentative tiers), and exposes the three combine-refinement rules as methods over the fitted state."""

    def _run_boruta_premerge(self, X, y, relevant):
        # replicate the base logic but stash the boruta object's accepted/tentative (expanded to cluster members)
        from mlframe.feature_selection.boruta_shap import BorutaShap
        from sklearn.ensemble import RandomForestClassifier
        rep_members = {}
        for r, ms in self.members_.items():
            keep = [m for m in ms if m in relevant]
            if keep:
                rep = max(keep, key=lambda f: self.fi_.get(f, 0.0))
                rep_members[rep] = keep
        reps = list(rep_members.keys())
        if len(reps) < 2:
            reps = list(relevant); rep_members = {c: [c] for c in reps}
        driver = str(self.boruta_driver).lower()
        tot = "test" if driver == "permutation" else "train"
        b = BorutaShap(model=RandomForestClassifier(n_estimators=80, n_jobs=-1, random_state=self.random_state),
                       importance_measure=self.boruta_driver, permutation_n_repeats=2, classification=True,
                       n_trials=50, percentile=95, train_or_test=tot, verbose=False, random_state=self.random_state)
        b.fit(X[reps], y)
        def _expand(names):
            out = []
            for r in names:
                out.extend(rep_members.get(r, [r]))
            return [c for c in dict.fromkeys(out) if c in X.columns]
        accepted = [c for c in getattr(b, "accepted", []) if c in reps]
        tentative = [c for c in list(getattr(b, "tentative", [])) if c in reps]
        self._boruta_accepted_ = set(_expand(accepted))
        self._boruta_tentative_ = set(_expand(tentative))
        return _expand(accepted)

    # -------------------------------------------------- shared helpers over the fitted state
    def _conf(self, member_sel):
        """Per-feature continuous confidence from the members + Boruta tiers + normalized shared FI."""
        fi = self.fi_; fmax = max(fi.values()) if fi else 1.0
        sel_sets = {m: set(s) for m, s in member_sel.items()}
        acc = getattr(self, "_boruta_accepted_", set()); ten = getattr(self, "_boruta_tentative_", set())
        conf = {}
        for f in self.cluster_of_:
            votes = sum(1 for m, s in sel_sets.items() if f in s)
            tier = 1.0 if f in acc else (0.5 if f in ten else 0.0)
            conf[f] = votes + tier + (fi.get(f, 0.0) / fmax if fmax > 0 else 0.0)
        return conf

    def _cluster_votes(self, member_sel):
        cv = defaultdict(set)
        for m, sel in member_sel.items():
            for f in sel:
                r = self.cluster_of_.get(f)
                if r is not None:
                    cv[r].add(m)
        return cv

    def _emit(self, chosen_reps, cols):
        out = []
        for r in chosen_reps:
            ms = [m for m in self.members_[r] if m in cols]
            if ms:
                out.append(max(ms, key=lambda f: self.fi_.get(f, 0.0)))
        return out

    # -------------------------------------------------- the three rules (return selected feature list)
    def combine_confidence(self, member_sel, cols):
        cv = self._cluster_votes(member_sel); conf = self._conf(member_sel)
        score = {r: max((conf.get(m, 0.0) for m in self.members_[r]), default=0.0) for r in cv}
        consensus = [r for r, v in cv.items() if len(v) >= 2]
        floor = float(np.percentile([score[r] for r in consensus], 40)) if consensus else -np.inf
        chosen = [r for r, v in cv.items() if len(v) >= 2 or score[r] >= floor]
        if not chosen:
            chosen = list(cv.keys())
        return self._emit(chosen, cols)

    def combine_referee(self, member_sel, cols):
        cv = self._cluster_votes(member_sel)
        consensus = [r for r, v in cv.items() if len(v) >= 2]
        contested = [r for r, v in cv.items() if len(v) == 1]
        core = self._emit(consensus, cols)
        # held-out forward-AUC referee on the stashed augmented frame
        Xa, yv = self._Xaug_, self._y_
        from sklearn.model_selection import train_test_split as _tts
        try:
            Xtr, Xva, ytr, yva = _tts(Xa, yv, test_size=0.3, random_state=self.random_state, stratify=yv)
        except Exception as exc:
            logger.debug("combine_referee: held-out split failed, keeping all voted: %s", exc)
            return self._emit(consensus + contested, cols)  # fallback: keep all voted
        def auc_of(feats):
            feats = [c for c in feats if c in Xa.columns]
            if not feats:
                return 0.5
            m = lgb.LGBMClassifier(n_estimators=120, num_leaves=15, learning_rate=0.08, n_jobs=-1, verbose=-1)
            m.fit(Xtr[feats], ytr)
            return roc_auc_score(yva, m.predict_proba(Xva[feats])[:, 1])
        kept = list(core); base_auc = auc_of(core) if core else 0.5
        # order contested reps by shared FI, greedily admit those that lift held-out AUC
        contested_reps = sorted(contested, key=lambda r: max((self.fi_.get(m, 0.0) for m in self.members_[r]), default=0.0), reverse=True)
        for r in contested_reps:
            rep = self._emit([r], cols)
            if not rep:
                continue
            trial = kept + rep
            a = auc_of(trial)
            if a >= base_auc + 1e-4:
                kept = trial; base_auc = a
        return kept

    def combine_tentative_rescue(self, member_sel, cols):
        # binary vote=1 baseline
        base = set(self._combine(member_sel, cols))
        fi = self.fi_
        fi_vals = sorted(fi.values())
        q75 = fi_vals[int(0.75 * len(fi_vals))] if fi_vals else np.inf
        other = set().union(*[set(member_sel.get(m, [])) for m in ("mrmr", "shap", "tree")]) if member_sel else set()
        rescued = list(base)
        for f in getattr(self, "_boruta_tentative_", set()):
            r = self.cluster_of_.get(f)
            if r is None:
                continue
            rep = self._emit([r], cols)
            if rep and rep[0] not in base and (f in other or fi.get(f, 0.0) >= q75):
                rescued.extend(rep)
        return [c for c in dict.fromkeys(rescued) if c in cols]


def run_bed(name, X, y, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    t0 = time.time()
    h = SynergyHybrid(vote=1, use_fe=True, random_state=seed).fit(Xtr, ytr)
    fit_s = round(time.time() - t0, 1)
    cols = list(h._Xaug_.columns)
    Ztr_full, Zte_full = h._augment(Xtr), h._augment(Xte)
    ms = h.member_selections_
    rows = []

    def evalsel(tag, selected):
        sel = [c for c in dict.fromkeys(selected) if c in Ztr_full.columns and c in Zte_full.columns]
        if not sel:
            sel = list(h.raw_selected_)
        a = downstream(Ztr_full[sel], Zte_full[sel], ytr, yte); am = round(float(np.nanmean(list(a.values()))), 4)
        rows.append(dict(bed=name, variant=tag, n=len(sel), auc_mean=am, **a))
        print(f"[{name}] {tag:18s} n={len(sel):3d} mean={am} {a}", flush=True)

    evalsel("baseline_vote1", list(h.raw_selected_))
    evalsel("confidence", h.combine_confidence(ms, cols))
    evalsel("referee", h.combine_referee(ms, cols))
    evalsel("tentative_rescue", h.combine_tentative_rescue(ms, cols))
    print(f"[{name}] (1 fit {fit_s}s; boruta acc={len(getattr(h,'_boruta_accepted_',[]))} ten={len(getattr(h,'_boruta_tentative_',[]))})", flush=True)
    return rows


def main():
    allrows = []
    Xh, yh, _ = make_hard_dataset(n_samples=5000, seed=0); print(f"=== hard_synth {Xh.shape} ===", flush=True)
    allrows += run_bed("hard_synth", Xh, yh)
    Xr, yr, rname = load_real(); print(f"\n=== {rname} {Xr.shape} ===", flush=True)
    allrows += run_bed(rname, Xr, yr)
    Xs, ys, _ = make_dataset(n_samples=5000, seed=0); print(f"\n=== synth {Xs.shape} ===", flush=True)
    allrows += run_bed("synth", Xs, ys)

    df = pd.DataFrame(allrows)
    print("\n=== ALL ===\n" + df.to_string(index=False))
    print("\n=== verdict (vs baseline_vote1 per bed) ===")
    for bed in df.bed.unique():
        b = df[df.bed == bed].set_index("variant")
        base = float(b.loc["baseline_vote1", "auc_mean"])
        for v in ("confidence", "referee", "tentative_rescue"):
            d = round(float(b.loc[v, "auc_mean"]) - base, 4)
            print(f"  {bed:12s} {v:18s} {b.loc[v,'auc_mean']} (d={d:+})")
    df.to_csv("D:/Temp/round4_synergy_rows.csv", index=False)


if __name__ == "__main__":
    main()
