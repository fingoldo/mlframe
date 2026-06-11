"""biz_value measurement harness for the RFECV ``n_features_selection_rule='auto'``
flat-curve / pure-noise false-positive-control fix.

THE DEFECT: ``auto`` resolves to ``one_se_max`` which, on a statistically FLAT
CV-vs-N curve (pure noise, where no feature count significantly beats the
smallest), takes the LARGEST N in the 1-SE band -> lands at ~all features. That
is a false-positive-control gap: a pure-noise input selects ~all columns.

THE FIX (gated, default-on): in the ``auto`` branch, detect a flat curve --
the best smoothed mean is within ~1 SE of the score at the SMALLEST evaluated N
(i.e. no larger feature set is SE-significantly better than the minimum) -- and
in that case resolve to ``one_se_min`` (parsimonious minimum). On a genuinely
improving curve the best mean exceeds (min-N score + 1 SE), so ``auto`` stays
``one_se_max`` and signal-recall is UNAFFECTED.

This harness measures the three gating scenarios across multiple seeds and
reports the MAJORITY vote, comparing pre-fix vs post-fix. Run it directly:

    PYTHONPATH=src python src/mlframe/feature_selection/wrappers/_benchmarks/bench_auto_rule_noise_fp.py

Decision rule (CLEAN WIN required to ship the prod change):
  1. PURE NOISE  : selected <= p/3  (pre-fix ~p).
  2. STRONG      : recovery >= 4/5 informative AND drops most noise; NO regression vs pre-fix.
  3. WEAK        : weak-informative recovery MUST NOT drop materially vs pre-fix.

MEASURED VERDICT (RandomForest, n=400, p=15, seeds=[0,1,2], MAJORITY=median): *** NO CLEAN WIN -- TRADEOFF, REVERTED ***
On the three harness fixtures below the rule-resolution-layer reject LOOKED clean:
  | Fixture | PRE-FIX (auto=one_se_max)              | ATTEMPTED FIX (auto + dummy-reject gate)  |
  |---------|----------------------------------------|-------------------------------------------|
  | NOISE   | selected/seed=[15,15,15] MED=15 (ALL)  | selected/seed=[0,0,0]    MED=0   REJECTED |
  | STRONG  | rec [4,4,5] MED=4/5, noise_kept MED=0  | rec [4,4,5] MED=4/5, noise_kept MED=0     |
  | WEAK    | rec [3,3,3] MED=3/3, noise_kept MED=2  | rec [3,3,3] MED=3/3, noise_kept MED=2     |
-- pure noise flipped 15 -> 0 with STRONG/WEAK (detectable-signal) recall BIT-IDENTICAL.

BUT two INDEPENDENT real-signal fixtures showed the gate rejecting RECOVERABLE signal -- a SIGNAL-RECALL REGRESSION:
  (R1) 6-informative multi-estimator MIN-AGGREGATION (knockoffs K3 fixture, n=400, p=12, 6 informative):
       pre-fix one_se_max selects 12; attempted fix selects 0. The info6 subset scores +0.16 (well above the
       -0.15 dummy) under min-aggregation, but the FULL 12-feature set sits at/below the dummy, and RFECV's search
       early-exits at {0, 12} BEFORE ever evaluating info6 -- so the gate (seeing only {0, 12}) rejects.
  (R2) recency SAMPLE-WEIGHTED 2-feature (A predictive under the weighting): pre-fix selects 2; attempted fix selects 0.
       The 2-feature (A+B) weighted score is -1.70 (far below the -0.15 dummy) because B hurts under the weighting,
       but A-only would win -- again never explored before the {0, full} early-exit.

ROOT CAUSE: RFECV's "all-features can't beat the no-features dummy" early-exit (_rfecv_fit_outer_loop.py ~351) stops
the search at {N=0, N=full} on BOTH pure noise AND noise-diluted-but-recoverable signal. The two are therefore
INDISTINGUISHABLE at the select_optimal_nfeatures_ rule-resolution layer -- any reject there sacrifices recoverable
signal. Tightening the predicate to "best is SE-significantly WORSE than the dummy" did NOT separate them (R2's
-1.70 << dummy is MORE extreme than pure noise's -0.70, yet R2 has real signal). The only safe noise rejection
requires the outer-loop search to actually EXPLORE smaller subsets before concluding (an outer-loop change, out of
scope for a rule-resolution fix). Prod change REVERTED; see the bench-attempt-rejected note in
wrappers/_rfecv_stability_select.py (select_optimal_nfeatures_, 'auto' branch).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _pure_noise(n=400, p=15, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = rng.integers(0, 2, size=n).astype(np.int64)
    return X, y, []


def _strong_signal(n=400, p_signal=5, p_noise=10, seed=0):
    rng = np.random.default_rng(seed)
    X_sig = rng.normal(size=(n, p_signal))
    X_noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([X_sig, X_noise])
    # Strong linear signal: 5 informative features dominate.
    y = (X_sig.sum(axis=1) + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    informative = list(range(p_signal))
    return X, y, informative


def _weak_signal(n=400, p_signal=3, p_noise=12, seed=0):
    rng = np.random.default_rng(seed)
    X_sig = rng.normal(size=(n, p_signal))
    X_noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([X_sig, X_noise])
    # GENUINELY-detectable low SNR: the 3 informative features carry a real but
    # weak signal (coef 1.2, label-noise 1.2) that the estimator CAN extract --
    # both the signal-only and all-features subsets beat the no-features dummy by
    # >1 SE, so the CV curve is NOT flat. This is the critical no-regression
    # guard: an over-aggressive parsimony rule that rejected this (flat-curve
    # mis-classification) would sacrifice recoverable weak signal. Contrast with
    # _pure_noise where the all-features subset CANNOT beat the dummy.
    logit = 1.2 * X_sig.sum(axis=1) + 1.2 * rng.normal(size=n)
    y = (logit > 0).astype(np.int64)
    informative = list(range(p_signal))
    return X, y, informative


def _support_indices(sel):
    s = sel.support_
    if s.dtype == bool:
        return [int(i) for i in np.flatnonzero(s)]
    return [int(i) for i in s]


def _fit_rule(X, y, seed, rule="auto"):
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    sel = RFECV(
        estimator=RandomForestClassifier(random_state=seed, n_estimators=40),
        cv=3, max_refits=10, verbose=0, random_state=seed,
        max_noimproving_iters=4,
        n_features_selection_rule=rule,
    )
    sel.fit(df, y)
    return set(_support_indices(sel))


def _fit_auto(X, y, seed):
    return _fit_rule(X, y, seed, rule="auto")


def _run_scenario(make, seeds, label, rule="auto"):
    sel_counts = []
    recoveries = []
    noise_kept = []
    p = None
    n_inform = None
    for seed in seeds:
        X, y, informative = make(seed=seed)
        p = X.shape[1]
        n_inform = len(informative)
        selected = _fit_rule(X, y, seed, rule=rule)
        sel_counts.append(len(selected))
        if informative:
            inf_set = set(informative)
            recoveries.append(len(selected & inf_set))
            noise_idx = set(range(p)) - inf_set
            noise_kept.append(len(selected & noise_idx))
    # MAJORITY = median across seeds (robust to one outlier seed).
    med_sel = int(np.median(sel_counts))
    out = {"label": label, "p": p, "n_inform": n_inform,
           "sel_counts": sel_counts, "median_selected": med_sel}
    if recoveries:
        out["recoveries"] = recoveries
        out["median_recovery"] = int(np.median(recoveries))
        out["noise_kept"] = noise_kept
        out["median_noise_kept"] = int(np.median(noise_kept))
    return out


def _report(tag, rule):
    seeds = [0, 1, 2]
    noise = _run_scenario(_pure_noise, seeds, "PURE_NOISE", rule=rule)
    strong = _run_scenario(_strong_signal, seeds, "STRONG", rule=rule)
    weak = _run_scenario(_weak_signal, seeds, "WEAK", rule=rule)
    print(f"--- {tag} (rule={rule!r}) ---")
    print(f"[1] PURE NOISE   p={noise['p']}  selected/seed={noise['sel_counts']}  "
          f"MEDIAN={noise['median_selected']}  (target <= {noise['p']//3})")
    print(f"[2] STRONG       recovery/seed={strong['recoveries']} MEDIAN={strong['median_recovery']}/{strong['n_inform']}  "
          f"noise_kept/seed={strong['noise_kept']} MEDIAN={strong['median_noise_kept']}")
    print(f"[3] WEAK         recovery/seed={weak['recoveries']} MEDIAN={weak['median_recovery']}/{weak['n_inform']}  "
          f"noise_kept/seed={weak['noise_kept']} MEDIAN={weak['median_noise_kept']}")
    return noise, strong, weak


def main():
    print("=== RFECV auto-rule noise-FP biz_value harness (seeds=[0,1,2], majority=median) ===\n")
    # 'one_se_max' is what pre-fix 'auto' resolved to uniformly; running it gives the pre-fix baseline.
    _report("PRE-FIX baseline", "one_se_max")
    print()
    return _report("POST-FIX", "auto")


if __name__ == "__main__":
    main()
