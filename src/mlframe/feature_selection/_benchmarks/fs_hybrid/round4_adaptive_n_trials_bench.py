"""A4-5 adaptive_n_trials (BorutaShap) -- cheap falsifiable speed-lever test.

PRODUCTION FACT (read-only from boruta_shap/_fit_explain.py):
  BorutaShap ALREADY early-terminates: each trial accumulates per-feature ``hits`` and runs the
  binomial H0 test (n=trial, p=0.5); the loop BREAKS once every feature is decided (zero tentative)
  -- ``n_trials_run_`` records the actual count. So "stop trials by binomial convergence" is LARGELY
  ALREADY SHIPPED (the tracker QUEUE note flagged exactly this: "early-stop may already harvest it").

A4-5's ONLY remaining marginal value vs the shipped stop: the existing stop requires ALL features
  decided (tentative == 0). If a STABLE TENTATIVE TAIL exists -- features whose binomial p-value sits
  permanently between the accept and reject thresholds and will NEVER resolve within n_trials -- the
  prod loop runs to the full n_trials cap for nothing. A4-5 would detect "the accepted/rejected SET has
  not changed for W trials AND every still-tentative feature is provably unresolvable soon" and stop.

FALSIFIABLE QUESTIONS (the whole idea lives or dies on these):
  Q1. Does prod's all-decided stop already fire WELL BEFORE n_trials on these beds? If it always stops
      early on its own, A4-5 adds ~nothing (no residual budget to reclaim).
  Q2. When prod runs to the cap, is there a residual TENTATIVE TAIL (decision never completes)? Only
      then does A4-5's "stop on accepted-set stability" reclaim wall.
  Q3. For the runs WITH a tail: does an A4-5 "accepted-set unchanged for W=10 trials" stop cut trials
      at a DECISION-EQUIVALENT accepted set (Jaccard ~ 1.0)?

METHOD (cheap, fit-once): run prod BorutaShap once at a HIGH n_trials cap with verbose history; the
  class stores ``history_hits`` (cumulative hits per trial). Replay the exact prod decision rule
  (binomial H0 + Bonferroni, test_features) offline per trial to reconstruct the accepted/tentative
  trajectory, find prod's actual all-decided stop trial, and the A4-5 stop trial (accepted set stable
  for W). Compare accepted-set Jaccard(A4-5 stop vs full-cap) and trial savings. No extra model fits.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import make_dataset
from hard_synth import make_hard_dataset

try:
    from scipy.stats import binomtest as _binomtest
    def _binom_p(x, n, p, alt):
        return _binomtest(int(x), n=int(n), p=p, alternative=alt).pvalue
except ImportError:
    from scipy.stats import binom_test as _bt
    def _binom_p(x, n, p, alt):
        return _bt(int(x), n=int(n), p=p, alternative=alt)

CK = "D:/Temp/queue_ideas_progress.txt"
def ck(m):
    with open(CK, "a") as f:
        f.write(m + "\n")


def replay_decisions(history_hits, n_cols, pvalue=0.05):
    """Replay prod test_features per trial from cumulative hit history.

    history_hits[t] = cumulative hits after trial t (row 0 is the zeros seed -> trials are rows 1..T).
    Returns per-trial (accepted_set, rejected_set, tentative_count) lists using the EXACT prod rule:
    binomial H0 (greater -> accept, less -> reject), Bonferroni-correct over n_cols, threshold < pvalue,
    accumulated across trials (prod appends to accepted_columns/rejected_columns lists -> a feature stays
    accepted/rejected once it ever crosses).
    """
    acc_cum, rej_cum = set(), set()
    traj = []
    T = history_hits.shape[0]
    for t in range(1, T):  # row t corresponds to iteration=t (1-based), matching test_features(iteration=trial+1)
        hits = history_hits[t]
        it = t
        acc_p = np.array([_binom_p(h, it, 0.5, "greater") for h in hits])
        rej_p = np.array([_binom_p(h, it, 0.5, "less") for h in hits])
        # Bonferroni: prod multiplies by n_tests (multipletests 'bonferroni' caps at 1.0)
        acc_pc = np.minimum(acc_p * n_cols, 1.0)
        rej_pc = np.minimum(rej_p * n_cols, 1.0)
        newly_acc = {i for i in range(len(hits)) if acc_pc[i] < pvalue}
        newly_rej = {i for i in range(len(hits)) if rej_pc[i] < pvalue}
        acc_cum |= newly_acc
        rej_cum |= newly_rej
        decided = acc_cum | rej_cum
        tent = n_cols - len(decided - (acc_cum & rej_cum)) if False else n_cols - len(acc_cum | rej_cum)
        traj.append((set(acc_cum), set(rej_cum - acc_cum), tent))
    return traj


def a45_stop_trial(traj, W=10):
    """A4-5 stop: first trial after which the ACCEPTED set has been unchanged for W consecutive trials."""
    stable = 0
    for t in range(1, len(traj)):
        if traj[t][0] == traj[t - 1][0]:
            stable += 1
        else:
            stable = 0
        if stable >= W:
            return t  # 0-based index into traj; trial number = t+1
    return len(traj) - 1


def a45_stop_safe(traj, hist_hits, n_cols, W=20, margin=0.15, pvalue=0.05):
    """SAFER A4-5 stop: require (a) accepted set unchanged for W trials AND (b) NO still-tentative feature
    is within ``margin`` of either binomial threshold (i.e. no feature is about to cross). The naive W=10
    rule fired on hard_synth's transient plateau while weak features were still slowly crossing; gating on
    'no tentative feature near a decision boundary' prevents stopping mid-convergence."""
    stable = 0
    for t in range(1, len(traj)):
        acc, rej, _ = traj[t]
        if acc == traj[t - 1][0]:
            stable += 1
        else:
            stable = 0
        if stable < W:
            continue
        it = t
        hits = hist_hits[t]
        decided = acc | rej
        near = False
        for i in range(n_cols):
            if i in decided:
                continue
            ap = min(_binom_p(hits[i], it, 0.5, "greater") * n_cols, 1.0)
            rp = min(_binom_p(hits[i], it, 0.5, "less") * n_cols, 1.0)
            # a tentative feature within margin of crossing either threshold -> not safe to stop
            if ap < pvalue * (1 + margin) or rp < pvalue * (1 + margin):
                near = True; break
        if not near:
            return t
    return len(traj) - 1


def prod_alldecided_stop(traj, n_cols):
    """Trial at which prod's shipped early-stop fires (tentative == 0)."""
    for t in range(len(traj)):
        if traj[t][2] == 0:
            return t
    return None  # never fully decided -> prod runs to cap


def run_bed(name, X, y, n_trials_cap=120, seed=0):
    from mlframe.feature_selection.boruta_shap import BorutaShap
    print(f"\n=== {name} {X.shape} (n_trials_cap={n_trials_cap}) ===", flush=True)
    ck(f"A4-5 {name} start {X.shape}")
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    t0 = time.time()
    b = BorutaShap(
        model=RandomForestClassifier(n_estimators=80, n_jobs=4, random_state=seed),
        importance_measure="gini", classification=True, n_trials=n_trials_cap, percentile=95,
        pvalue=0.05, verbose=False, random_state=seed,
    )
    b.fit(Xtr, ytr)
    wall = time.time() - t0
    n_run = int(getattr(b, "n_trials_run_", n_trials_cap))
    n_cols = int(b.ncols)
    hh = np.asarray(b.history_hits)  # (n_run+1, n_cols)
    traj = replay_decisions(hh, n_cols)
    prod_stop = prod_alldecided_stop(traj, n_cols)   # 0-based traj idx
    a45_stop = a45_stop_trial(traj, W=10)
    a45_safe = a45_stop_safe(traj, hh, n_cols, W=20, margin=0.15)
    full_accepted = traj[-1][0]
    a45_accepted = traj[a45_stop][0]
    a45_safe_accepted = traj[a45_safe][0]
    prod_accepted_at_run = traj[min(n_run - 1, len(traj) - 1)][0]

    def jac(a, b_):
        a, b_ = set(a), set(b_)
        return 1.0 if not a and not b_ else round(len(a & b_) / max(1, len(a | b_)), 3)

    prod_stop_trial = (prod_stop + 1) if prod_stop is not None else None
    a45_stop_trial_n = a45_stop + 1
    per_trial_s = wall / max(1, n_run)
    a45_wall = per_trial_s * a45_stop_trial_n
    prod_actual_wall = wall  # prod ran n_run trials (its own stop already applied)

    tail = traj[-1][2]  # residual tentative at cap
    j_a45 = jac(a45_accepted, full_accepted)
    a45_safe_trial_n = a45_safe + 1
    j_safe = jac(a45_safe_accepted, full_accepted)
    safe_wall = per_trial_s * a45_safe_trial_n
    print(f"  prod n_trials_run_={n_run}/{n_trials_cap}  wall={wall:.1f}s  ({per_trial_s:.2f}s/trial)", flush=True)
    print(f"  residual tentative at cap = {tail}  (0 => prod's all-decided stop fired)", flush=True)
    print(f"  prod all-decided stop trial = {prod_stop_trial}  | A4-5 naive(W=10) stop = {a45_stop_trial_n} "
          f"| A4-5 SAFE(W=20+margin) stop = {a45_safe_trial_n}", flush=True)
    print(f"  accepted@cap={sorted(full_accepted)[:8]}... (n={len(full_accepted)})", flush=True)
    print(f"  naive A4-5 Jaccard={j_a45} (save {max(0.0, wall-a45_wall):.1f}s) | "
          f"SAFE A4-5 Jaccard={j_safe} (save {max(0.0, wall-safe_wall):.1f}s)", flush=True)
    ck(f"A4-5 {name} n_run={n_run} tail={tail} naive_stop={a45_stop_trial_n} jac={j_a45} safe_stop={a45_safe_trial_n} safe_jac={j_safe}")
    return dict(bed=name, n_cols=n_cols, n_trials_run=n_run, wall_s=round(wall, 1),
                per_trial_s=round(per_trial_s, 2), residual_tentative=tail,
                prod_alldecided_stop=prod_stop_trial, a45_naive_stop=a45_stop_trial_n,
                a45_naive_jaccard=j_a45, a45_safe_stop=a45_safe_trial_n, a45_safe_jaccard=j_safe,
                safe_est_save_s=round(max(0.0, wall - safe_wall), 1), n_accepted=len(full_accepted))


def main():
    rows = []
    Xs, ys, _ = make_dataset(n_samples=5000, seed=0)
    rows.append(run_bed("synth", Xs, ys, n_trials_cap=120))
    Xh, yh, _ = make_hard_dataset(n_samples=5000, seed=0)
    rows.append(run_bed("hard_synth", Xh, yh, n_trials_cap=120))
    df = pd.DataFrame(rows)
    print("\n=== ALL ===\n" + df.to_string(index=False), flush=True)
    print("\n=== A4-5 VERDICT ===", flush=True)
    for _, r in df.iterrows():
        print(f"  {r.bed:12s} n_run={r.n_trials_run} tail={r.residual_tentative} prod_alldecided={r.prod_alldecided_stop} | "
              f"naive(W=10) stop={r.a45_naive_stop} jac={r.a45_naive_jaccard} | "
              f"SAFE stop={r.a45_safe_stop} jac={r.a45_safe_jaccard} save~{r.safe_est_save_s}s", flush=True)
    df.to_csv("D:/Temp/round4_adaptive_n_trials_rows.csv", index=False)
    ck("A4-5 DONE")


if __name__ == "__main__":
    main()
