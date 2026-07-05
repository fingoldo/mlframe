"""ACCURACY bench: plug-in binned ECE vs debiased binned ECE (per-bin sampling-noise correction).

The standard binned ECE ``sum_b (n_b/N) * |conf_b - acc_b|`` is a POSITIVELY BIASED estimator of the
population ECE ``E_p[|p - acc(p)|]``: within each bin ``acc_b`` is a noisy estimate of the bin's true
positive rate, and ``E[|conf_b - acc_b|] >= |conf_b - E[acc_b]|`` by Jensen -- the absolute value of a
noisy quantity is inflated by ~``sqrt(Var(acc_b))``. The bias is worst exactly where it matters: a
PERFECTLY CALIBRATED model has true ECE 0, yet the plug-in estimator returns a strictly positive value
that grows with nbins and shrinks with bin count n_b. This overstates miscalibration in the headline
report and (Kumar et al. NeurIPS 2019 "Verified Uncertainty Calibration"; Roelofs et al. AISTATS 2022
"Mitigating bias in calibration error estimation") biases method comparison toward whichever method
happens to be over-binned.

Debiased estimator (the squared-ECE / Brier-reliability debiasing applied to the L1 ECE via the
per-bin variance of the Bernoulli accuracy estimate): in bin ``b`` with ``n_b`` samples and observed
accuracy ``acc_b``, the sampling variance of ``acc_b`` is ``acc_b*(1-acc_b)/(n_b-1)`` (unbiased).
The plug-in gap ``g_b = |conf_b - acc_b|`` has ``E[g_b^2] = (conf_b-true_acc_b)^2 + Var(acc_b)``, so a
debiased per-bin squared gap is ``max(g_b^2 - Var(acc_b), 0)``; the debiased ECE takes
``sum_b (n_b/N) * sqrt(max(g_b^2 - Var(acc_b), 0))``. Bins with n_b<2 (no variance estimate) keep the
raw gap. This removes the noise floor so a calibrated model scores ~0 instead of a spurious positive.

HONEST metric: bias = (estimated ECE) - (true population ECE). For the calibrated scenarios true ECE=0,
so |estimate| IS the bias. For miscalibrated scenarios we use a fine-grid ground-truth ECE on a large
independent sample. Lower |bias| wins. Multi-seed (>=7), multi-scenario (calibrated + miscalibrated).

Run::

    python -m mlframe.metrics._benchmarks.bench_ece_debiased
"""
from __future__ import annotations

import numpy as np


def _scn_calibrated_uniform(rng, n):
    """Perfectly calibrated, uniform scores. True ECE = 0."""
    s = np.clip(rng.uniform(0.0, 1.0, n), 1e-6, 1 - 1e-6)
    acc = s.copy()  # acc(p) == p -> perfectly calibrated
    return s, acc, 0.0


def _scn_calibrated_beta(rng, n):
    """Perfectly calibrated, skewed (rare-event) scores. True ECE = 0."""
    s = np.clip(rng.beta(0.6, 8.0, n), 1e-6, 1 - 1e-6)
    acc = s.copy()
    return s, acc, 0.0


def _scn_calibrated_bimodal(rng, n):
    """Perfectly calibrated, bimodal scores. True ECE = 0."""
    half = n // 2
    s = np.empty(n)
    s[:half] = np.clip(rng.beta(2.0, 12.0, half), 1e-6, 1 - 1e-6)
    s[half:] = np.clip(rng.beta(12.0, 2.0, n - half), 1e-6, 1 - 1e-6)
    return s, s.copy(), 0.0


def _scn_miscal_sigmoid(rng, n):
    """Logit-shifted miscalibration; nonzero true ECE (fine-grid reference)."""
    s = np.clip(rng.beta(1.2, 6.0, n), 1e-6, 1 - 1e-6)
    logit = np.log(s / (1 - s)) + 0.8
    acc = 1.0 / (1.0 + np.exp(-logit))
    return s, acc, None


def _scn_miscal_overconf(rng, n):
    """Overconfident model; nonzero true ECE."""
    s = np.clip(rng.uniform(0, 1, n), 1e-6, 1 - 1e-6)
    acc = np.clip(0.5 + 0.5 * (s - 0.5) * 0.4, 0, 1)  # squashed toward 0.5
    return s, acc, None


SCENARIOS = {
    "calib_uniform": _scn_calibrated_uniform,
    "calib_beta_rare": _scn_calibrated_beta,
    "calib_bimodal": _scn_calibrated_bimodal,
    "miscal_sigmoid": _scn_miscal_sigmoid,
    "miscal_overconf": _scn_miscal_overconf,
}


def _ece_plugin(y, p, nbins):
    from mlframe.metrics.calibration._calibration_metrics import compute_ece_and_brier_decomposition

    return compute_ece_and_brier_decomposition(np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64), nbins)[0]


def _ece_debiased(y, p, nbins):
    from mlframe.metrics.calibration._calibration_metrics import compute_ece_debiased
    return compute_ece_debiased(np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64), nbins)


def _truth_ece(scn_fn, n_big=400_000, fine=400):
    """Fine equal-frequency reference ECE on a large independent sample."""
    rng = np.random.default_rng(123456)
    s, acc, known = scn_fn(rng, n_big)
    if known is not None:
        return known
    y = (rng.random(n_big) < acc).astype(np.float64)
    order = np.argsort(s, kind="stable")
    edges = np.linspace(0, n_big, fine + 1).astype(np.int64)
    e = 0.0
    for k in range(fine):
        lo, hi = edges[k], edges[k + 1]
        if hi <= lo:
            continue
        sel = order[lo:hi]
        e += (hi - lo) / n_big * abs(s[sel].mean() - y[sel].mean())
    return e


def main(n=2000, nbins=15, seeds=tuple(range(7))):
    print(f"ECE debiasing bench: n={n}, nbins={nbins}, seeds={seeds}\n")
    print(f"{'scenario':<18}{'truth':>8}{'plugin_bias':>13}{'debias_bias':>13}  winner")
    pl_wins = db_wins = ties = 0
    for name, fn in SCENARIOS.items():
        truth = _truth_ece(fn)
        for sd in seeds:
            rng = np.random.default_rng(sd)
            s, acc, _ = fn(rng, n)
            y = (rng.random(n) < acc).astype(np.float64)
            pl = abs(_ece_plugin(y, s, nbins) - truth)
            db = abs(_ece_debiased(y, s, nbins) - truth)
            if db < pl - 1e-12:
                db_wins += 1
            elif pl < db - 1e-12:
                pl_wins += 1
            else:
                ties += 1
        # report mean bias on the scenario
        pl_b = db_b = 0.0
        for sd in seeds:
            rng = np.random.default_rng(sd)
            s, acc, _ = fn(rng, n)
            y = (rng.random(n) < acc).astype(np.float64)
            pl_b += _ece_plugin(y, s, nbins) - truth
            db_b += _ece_debiased(y, s, nbins) - truth
        pl_b /= len(seeds)
        db_b /= len(seeds)
        winner = "DEBIAS" if abs(db_b) < abs(pl_b) else "PLUGIN"
        print(f"{name:<18}{truth:>8.4f}{pl_b:>13.5f}{db_b:>13.5f}  {winner}")
    total = pl_wins + db_wins + ties
    print(f"\nper-cell |bias| wins (over {total} cells): debiased={db_wins}  plugin={pl_wins}  ties={ties}")
    print("DECISION: flip headline ECE to debiased iff debiased wins majority of cells.")


if __name__ == "__main__":
    main()
