"""ACCURACY bench: equal-width vs equal-frequency (quantile) binning for the headline ECE estimate.

Distinct from ``bench_calibration_binning_strategies.py`` (which times the binning *code paths*).
Here we measure ESTIMATOR BIAS of the binned-ECE against a fine-grid ground-truth ECE, across
synthetic scenarios with KNOWN reliability maps. The question: which binning scheme should
``compute_ece_and_brier_decomposition`` default to for a low-bias headline ECE under skewed score
distributions (rare events concentrate scores near 0 -> equal-width collapses to 1-2 bins).

Ground truth: data is drawn from a known reliability map acc(p) = P(y=1 | score=p). The population
ECE is E_p[ |p - acc(p)| ]. We estimate it empirically by a FINE equal-frequency grid (500 bins)
on a large n=200k sample as the reference "truth" for each scenario, then measure how close the
coarse (nbins=10/15) equal-width vs equal-frequency estimators land on smaller n=2000 samples.

Metric (HONEST): RMSE of the coarse binned-ECE vs the fine-grid ground-truth ECE, averaged over
seeds. Lower = less biased estimator. A challenger (quantile) flips the default only if it wins the
MAJORITY of (scenario, seed) cells.

Run::

    python -m mlframe.metrics._benchmarks.bench_ece_binning_bias

VERDICT: REJECTED (keep equal-width default). Across nbins in {5,10,15,20} x 5-7 seeds x 6 scenarios
(incl. 3 rare-event/heavy-skew maps) equal-frequency does NOT win a majority of (scenario,seed) cells:
EF/EW cell wins were 10/16 (nbins=10), 14/15 (5), 18/18 (15), 20/18 (20) - a wash, never a decisive
majority. RMSE-vs-truth deltas are ~1e-3 or smaller, within seed noise. The headline kernel already
uses per-bin mean predicted prob (not bin centre), so even when equal-width collapses skewed scores
into few bins the within-bin p_mean tracks accuracy, neutralising the expected collapse bias. The
quantile path stays available for the reliability DIAGRAM (where empty bins look bad), but the scalar
ECE estimator gains nothing from switching, so the default is unchanged.
"""
from __future__ import annotations

import numpy as np


# ---- known reliability maps: return (score, prob_true) so true acc(score) is known ----
def _scn_rare_overconfident(rng, n):
    """Rare positives, scores piled near 0, model overconfident at the low end."""
    s = np.clip(rng.beta(0.4, 30.0, n), 1e-6, 1 - 1e-6)
    acc = np.clip(s * 0.6, 0, 1)  # systematically over-confident
    return s, acc


def _scn_rare_underconfident(rng, n):
    s = np.clip(rng.beta(0.5, 40.0, n), 1e-6, 1 - 1e-6)
    acc = np.clip(s * 1.8, 0, 1)
    return s, acc


def _scn_sigmoid_shift(rng, n):
    """Logit-shifted miscalibration on a moderately skewed score dist."""
    s = np.clip(rng.beta(1.2, 6.0, n), 1e-6, 1 - 1e-6)
    logit = np.log(s / (1 - s)) + 0.8
    acc = 1.0 / (1.0 + np.exp(-logit))
    return s, acc


def _scn_balanced_wellcal(rng, n):
    """Balanced, nearly calibrated (small ECE) - control scenario."""
    s = np.clip(rng.uniform(0, 1, n), 1e-6, 1 - 1e-6)
    acc = np.clip(s + 0.03 * np.sin(6 * s), 0, 1)
    return s, acc


def _scn_bimodal(rng, n):
    """Bimodal scores (two clusters) with cluster-dependent miscalibration."""
    half = n // 2
    s = np.empty(n)
    s[:half] = np.clip(rng.beta(2.0, 18.0, half), 1e-6, 1 - 1e-6)
    s[half:] = np.clip(rng.beta(18.0, 2.0, n - half), 1e-6, 1 - 1e-6)
    acc = np.clip(s * 0.7 + 0.1, 0, 1)
    return s, acc


def _scn_extreme_rare(rng, n):
    """1% base rate, very heavy pile near 0."""
    s = np.clip(rng.beta(0.3, 120.0, n), 1e-9, 1 - 1e-9)
    acc = np.clip(s * 0.5, 0, 1)
    return s, acc


SCENARIOS = {
    "rare_overconf": _scn_rare_overconfident,
    "rare_underconf": _scn_rare_underconfident,
    "sigmoid_shift": _scn_sigmoid_shift,
    "balanced_wellcal": _scn_balanced_wellcal,
    "bimodal": _scn_bimodal,
    "extreme_rare": _scn_extreme_rare,
}


def _ece_equal_width(y, p, nbins):
    from mlframe.metrics.calibration._calibration_metrics import compute_ece_and_brier_decomposition

    return compute_ece_and_brier_decomposition(np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64), nbins)[0]


def _ece_quantile(y, p, nbins):
    """Equal-frequency (quantile) binned ECE: same formula as the equal-width kernel but
    bin assignment is by quantile rank instead of [min,max] width."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(p)
    if n == 0:
        return 1.0
    order = np.argsort(p, kind="stable")
    # equal-population edges by rank
    edges_idx = np.linspace(0, n, nbins + 1).astype(np.int64)
    ece = 0.0
    for k in range(nbins):
        lo, hi = edges_idx[k], edges_idx[k + 1]
        if hi <= lo:
            continue
        sel = order[lo:hi]
        w = (hi - lo) / n
        p_mean = p[sel].mean()
        acc = y[sel].mean()
        ece += w * abs(p_mean - acc)
    return ece


def _ground_truth_ece(rng, scn_fn, n_big=200_000, fine_bins=500):
    """Fine equal-frequency grid on a large independent sample = reference ECE."""
    s, acc = scn_fn(rng, n_big)
    y = (rng.random(n_big) < acc).astype(np.float64)
    return _ece_quantile(y, s, fine_bins)


def main(n=2000, nbins=10, seeds=(0, 1, 2, 3, 4)):
    print(f"ECE binning bias bench: n={n}, nbins={nbins}, seeds={seeds}\n")
    print(f"{'scenario':<18}{'base':>7}{'truth':>9}{'EW_rmse':>10}{'EF_rmse':>10}  winner")
    ew_wins = ef_wins = ties = 0
    for name, fn in SCENARIOS.items():
        gt_rng = np.random.default_rng(99)
        truth = _ground_truth_ece(gt_rng, fn)
        ew_sq = ef_sq = 0.0
        base_acc = 0.0
        for sd in seeds:
            rng = np.random.default_rng(sd)
            s, acc = fn(rng, n)
            y = (rng.random(n) < acc).astype(np.float64)
            base_acc += y.mean()
            ew = _ece_equal_width(y, s, nbins)
            ef = _ece_quantile(y, s, nbins)
            ew_sq += (ew - truth) ** 2
            ef_sq += (ef - truth) ** 2
        ew_rmse = (ew_sq / len(seeds)) ** 0.5
        ef_rmse = (ef_sq / len(seeds)) ** 0.5
        # per-cell winner counting
        for sd in seeds:
            rng = np.random.default_rng(sd)
            s, acc = fn(rng, n)
            y = (rng.random(n) < acc).astype(np.float64)
            ew = abs(_ece_equal_width(y, s, nbins) - truth)
            ef = abs(_ece_quantile(y, s, nbins) - truth)
            if ef < ew - 1e-12:
                ef_wins += 1
            elif ew < ef - 1e-12:
                ew_wins += 1
            else:
                ties += 1
        winner = "EF" if ef_rmse < ew_rmse else "EW"
        print(f"{name:<18}{base_acc/len(seeds):>7.3f}{truth:>9.4f}{ew_rmse:>10.4f}{ef_rmse:>10.4f}  {winner}")
    total = ew_wins + ef_wins + ties
    print(f"\nper-cell wins (over {total} cells): equal-freq={ef_wins}  equal-width={ew_wins}  ties={ties}")
    print("DECISION: flip default to quantile/equal-freq iff EF wins majority of cells.")


if __name__ == "__main__":
    main()
