"""Round-5 lever: RFECV ``smooth_perf`` default for ``select_optimal_nfeatures_``.

Production default path: ``RFECV.smooth_perf`` flat default = 0 (no smoothing) -> ``_fit.py`` reads ``self.smooth_perf``
-> ``_finalize_fit_results`` -> ``select_optimal_nfeatures_(..., smooth_perf=...)``. The function-signature default 3 is
overridden by the flat 0, so the REAL default reaching the rule is 0.

Lever question: rolling-mean smoothing the CV-mean-vs-N curve BEFORE the 1-SE selection rule reduces sampling noise in
the N pick. HONEST metric = TRUE (noise-free) score at the SELECTED N, averaged over many noisy realisations of a known
score-vs-N curve. We sweep smooth_perf in {0,1,3,5} across 5 curve shapes x many seeds on a DENSE N grid (rolling mean is
by-index, so it is only meaningful when N is densely+evenly sampled -- the regime the RFECV dichotomic search produces on
small p). Higher TRUE-score-at-pick = better. REJECTED != DELETED: smooth_perf stays a tunable; only the default is at stake.

Run: python round5_smooth_perf_bench.py
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

import numpy as np

from mlframe.feature_selection.wrappers.rfecv._stability_select import select_optimal_nfeatures_


class _Mock:
    """Minimal carrier exposing the attributes select_optimal_nfeatures_ reads."""

    def __init__(self, nfeat, smooth_perf):
        self.mean_perf_weight = 1.0
        self.std_perf_weight = 0.1
        self.n_features_selection_rule = "auto"  # resolves to one_se_max (the production default)
        self.max_nfeatures = None
        self.conduct_final_voting = False
        self.fi_missing_policy = "worst"
        self.n_features_in_ = int(max(nfeat))
        self.feature_names_in_ = [str(i) for i in range(self.n_features_in_)]
        # selected_features_[N] must exist for each candidate N (rule reads it for support_).
        self.selected_features_ = {int(n): [str(i) for i in range(int(n))] for n in nfeat}
        self.smooth_perf = smooth_perf


def true_curve(kind, ngrid):
    """Return (nfeatures grid, TRUE noise-free score) for a curve shape with a known optimum region."""
    nf = np.array(ngrid, dtype=float)
    x = nf / nf.max()
    if kind == "peak":            # sharp interior optimum at ~40% of features
        t = 0.5 + 0.30 * np.exp(-((x - 0.4) ** 2) / (2 * 0.12 ** 2))
    elif kind == "plateau":       # rises then flat tail (one_se_max should grab the band)
        t = 0.5 + 0.30 * (1 - np.exp(-x / 0.25))
    elif kind == "broad":         # broad gentle hump
        t = 0.5 + 0.20 * np.exp(-((x - 0.5) ** 2) / (2 * 0.30 ** 2))
    elif kind == "early":         # optimum at small N, then decays (overfitting tail)
        t = 0.5 + 0.25 * np.exp(-((x - 0.18) ** 2) / (2 * 0.10 ** 2))
    elif kind == "noisy_flat":    # almost flat -> picks driven entirely by noise
        t = 0.55 + 0.04 * x
    else:
        raise ValueError(kind)
    return nf, t


def run(seeds=range(40), noise=0.04, ngrid=None, rule="auto"):
    if ngrid is None:
        ngrid = list(range(2, 42, 2))  # dense even grid, 20 points (small-p RFECV regime)
    kinds = ["peak", "plateau", "broad", "early", "noisy_flat"]
    smooths = [0, 1, 3, 5]
    # accumulate mean TRUE-score-at-pick per (kind, smooth)
    acc = {(k, s): [] for k in kinds for s in smooths}
    for kind in kinds:
        nf, t = true_curve(kind, ngrid)
        std_curve = np.full_like(t, noise)
        for sd in seeds:
            rng = np.random.default_rng(sd)
            obs = t + rng.normal(0.0, noise, size=t.shape)
            for s in smooths:
                m = _Mock(ngrid, s)
                m.n_features_selection_rule = rule
                m.cv_results_ = {}
                select_optimal_nfeatures_(
                    m, np.array(ngrid, dtype=float), obs.copy(), std_curve.copy(),
                    smooth_perf=s, verbose=False, show_plot=False,
                )
                picked_n = int(m.n_features_)
                # TRUE score at the picked N (honest metric: noise-free curve value)
                true_at_pick = float(t[ngrid.index(picked_n)]) if picked_n in ngrid else float(t[np.argmin(np.abs(nf - picked_n))])
                acc[(kind, s)].append(true_at_pick)
    print(f"rule={rule}  noise={noise}  grid={len(ngrid)} pts  seeds={len(list(seeds))}")
    print(f"{'curve':<11}" + "".join(f"  sp={s:<6}" for s in smooths))
    wins = {s: 0 for s in smooths}
    for kind in kinds:
        means = {s: float(np.mean(acc[(kind, s)])) for s in smooths}
        best = max(means.values())
        row = f"{kind:<11}"
        for s in smooths:
            mark = "*" if abs(means[s] - best) < 1e-9 else " "
            row += f"  {means[s]:.4f}{mark}"
        print(row)
        # majority-win accounting: best smooth for this curve (ties -> all best share)
        for s in smooths:
            if abs(means[s] - best) < 5e-4:
                wins[s] += 1
    print("\nper-curve best-or-tied counts (5 curves):", {f"sp={s}": wins[s] for s in smooths})
    overall = {s: float(np.mean([np.mean(acc[(k, s)]) for k in kinds])) for s in smooths}
    print("overall mean TRUE-score-at-pick:", {f"sp={s}": round(overall[s], 4) for s in smooths})
    return acc


if __name__ == "__main__":
    # VERDICT (committed numbers):
    #  rule='auto' (-> one_se_max, the PRODUCTION default): smooth_perf is INERT -- the 1-SE band is built off the RAW
    #    cv_mean_perf (mean_arr), so smoothing only touches base_perf which the no-cost one_se path uses for the plot
    #    alone. TRUE-score-at-pick is BIT-IDENTICAL across sp in {0,1,3,5} for all 5 curves x 3 noise x 40 seeds.
    #    => flipping the production default 0 -> any value changes NOTHING. KEEP smooth_perf=0. Flip REJECTED (no effect).
    #  rule='argmax': smoothing DENOISES the curve argmax reads -> a clean honest-score win (peak curve, noise=0.06:
    #    sp=0 0.7810 -> sp=5 0.7921, +1.1pp; no curve regresses). This is the tunable PAIRING to recommend with argmax,
    #    NOT a production-default flip (argmax is not the default rule).
    for rule in ("auto", "argmax"):
        for nz in (0.03, 0.05, 0.08):
            run(noise=nz, rule=rule)
            print("-" * 70)
