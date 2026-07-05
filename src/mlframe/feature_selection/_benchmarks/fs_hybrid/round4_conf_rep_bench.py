"""A5-5 conf_rep -- choose the hybrid cluster REPRESENTATIVE by cross-member confidence rank-aggregation
instead of permutation-FI alone. Cheap, low-ceiling (tracker: 'cheap, low ceiling; ship-if-positive').

PRODUCTION FACT (hybrid_selector.py:474): for each KEPT redundant cluster, the hybrid emits exactly ONE
  representative = ``max(ms, key=lambda f: self.fi_.get(f, 0.0))`` -- the highest shared permutation-FI
  member. The set of KEPT clusters (the vote decision) is UNCHANGED by A5-5; only WHICH member of a
  multi-candidate cluster is emitted changes.

A5-5 PROPOSAL: pick the rep by cross-member confidence rank-aggregation -- a member ranks higher if MORE
  distinct selectors (mrmr/shap/boruta/tree) picked it AND its perm-FI rank is high (Borda over the two
  signals), instead of perm-FI alone. Rationale: on a redundant cluster the highest-FI copy may be a
  lucky-noise copy; the copy MORE selectors independently picked is a more reproducible representative.

FALSIFIABLE QUESTION: does the rank-agg rep differ from the FI rep on any kept multi-member cluster, and
  if so does it change downstream AUC? Low ceiling because (a) within a redundant cluster all copies carry
  near-identical signal so the downstream barely cares, and (b) FI and cross-member agreement are highly
  correlated. PASS: AUC delta >= +0.003 on a bed without regressing others; else low-ceiling NEGATIVE.

Beds: synth + hard_synth (both have engineered redundant clusters: red_0_*, red_4_*, red_6_*). Subclasses
  the production HybridSelector, fits ONCE, then re-emits with the two rep rules over the SAME kept
  clusters + same augmented frame (the combine rule is cheap).
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round3_realdata_bench import downstream
from synth import make_dataset
from hard_synth import make_hard_dataset
from hybrid_selector import HybridSelector

CK = "D:/Temp/queue_ideas_progress.txt"
def ck(m):
    with open(CK, "a") as f:
        f.write(m + "\n")


class ConfRepHybrid(HybridSelector):
    """Re-emit kept clusters with FI-argmax rep (prod) vs cross-member confidence rank-agg rep (A5-5)."""

    def _kept_clusters(self, member_sel, cols):
        """Reproduce prod's vote decision -> list of kept cluster reps (the cluster KEYS), unchanged by A5-5."""
        cluster_votes = defaultdict(set)
        for m, sel in member_sel.items():
            for f in sel:
                r = self.cluster_of_.get(f)
                if r is not None:
                    cluster_votes[r].add(m)
        support = {r: len(v) for r, v in cluster_votes.items()}
        chosen = [r for r, s in support.items() if s >= max(self.vote, 2) or s >= self.vote]
        if not chosen:
            chosen = list(cluster_votes.keys())
        return chosen, cluster_votes

    def emit_fi(self, member_sel, cols):
        chosen, _ = self._kept_clusters(member_sel, cols)
        out = []
        for r in chosen:
            ms = [m for m in self.members_[r] if m in cols]
            if ms:
                out.append(max(ms, key=lambda f: self.fi_.get(f, 0.0)))
        return out

    def emit_confrep(self, member_sel, cols):
        """A5-5: rep = Borda rank-agg of (cross-member vote count, perm-FI) over the cluster's members."""
        chosen, _ = self._kept_clusters(member_sel, cols)
        sel_sets = {m: set(s) for m, s in member_sel.items()}
        out = []; n_diff = 0
        for r in chosen:
            ms = [m for m in self.members_[r] if m in cols]
            if not ms:
                continue
            # signal 1: how many distinct selectors picked this member
            votes = {f: sum(1 for s in sel_sets.values() if f in s) for f in ms}
            # signal 2: perm-FI
            fi = {f: self.fi_.get(f, 0.0) for f in ms}
            # Borda: rank each member by votes (desc) and by fi (desc); sum ranks; lower=better
            vr = {f: i for i, f in enumerate(sorted(ms, key=lambda x: -votes[x]))}
            fr = {f: i for i, f in enumerate(sorted(ms, key=lambda x: -fi[x]))}
            rep = min(ms, key=lambda f: vr[f] + fr[f])
            fi_rep = max(ms, key=lambda f: fi[f])
            if rep != fi_rep:
                n_diff += 1
            out.append(rep)
        self._last_n_diff_ = n_diff
        return out


def run_bed(name, X, y, truth, seed=0):
    print(f"\n=== {name} {X.shape} ===", flush=True); ck(f"A5-5 {name} start")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    relevant = set(truth["relevant"])
    t0 = time.time()
    h = ConfRepHybrid(vote=1, use_fe=True, random_state=seed).fit(Xtr, ytr)
    fit_s = round(time.time() - t0, 1)
    cols = list(h._Xaug_.columns)
    Ztr, Zte = h._augment(Xtr), h._augment(Xte)
    ms = h.member_selections_

    # cluster structure diagnostic
    multi = sum(1 for r, mem in h.members_.items() if len([m for m in mem if m in cols]) > 1)
    print(f"  fit {fit_s}s | clusters={len(h.members_)} (multi-member={multi})", flush=True)

    rows = []
    for tag, emit in (("fi_rep(prod)", h.emit_fi), ("conf_rep(A5-5)", h.emit_confrep)):
        sel = [c for c in dict.fromkeys(emit(ms, cols)) if c in Ztr.columns and c in Zte.columns]
        if not sel:
            sel = list(h.raw_selected_)
        a = downstream(Ztr[sel], Zte[sel], ytr, yte)
        am = round(float(np.nanmean(list(a.values()))), 4)
        rec = sum(1 for c in sel if c in relevant)
        ndiff = getattr(h, "_last_n_diff_", 0) if "A5-5" in tag else 0
        rows.append(dict(bed=name, variant=tag, n=len(sel), rel_recall=rec, reps_changed=ndiff, auc_mean=am, **a))
        print(f"  [{tag:16s}] n={len(sel):3d} rel={rec} reps_changed={ndiff} mean={am} {a}", flush=True)
        ck(f"A5-5 {name} {tag} n={len(sel)} reps_changed={ndiff} mean={am}")
    return rows


def main():
    rows = []
    Xs, ys, ts = make_dataset(n_samples=5000, seed=0)
    rows += run_bed("synth", Xs, ys, ts)
    Xh, yh, th = make_hard_dataset(n_samples=5000, seed=0)
    rows += run_bed("hard_synth", Xh, yh, th)
    df = pd.DataFrame(rows)
    print("\n=== ALL ===\n" + df.to_string(index=False), flush=True)
    print("\n=== A5-5 VERDICT (conf_rep vs fi_rep) ===", flush=True)
    for bed in df.bed.unique():
        b = df[df.bed == bed].set_index("variant")
        d = round(float(b.loc["conf_rep(A5-5)", "auc_mean"]) - float(b.loc["fi_rep(prod)", "auc_mean"]), 4)
        nd = int(b.loc["conf_rep(A5-5)", "reps_changed"])
        print(f"  {bed:12s} reps_changed_by_A5-5={nd}  d_auc_mean={d:+}", flush=True)
    df.to_csv("D:/Temp/round4_conf_rep_rows.csv", index=False)
    ck("A5-5 DONE")


if __name__ == "__main__":
    main()
