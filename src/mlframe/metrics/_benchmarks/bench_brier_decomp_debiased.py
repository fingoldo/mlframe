"""ACCURACY bench: plug-in Brier reliability/resolution vs debiased decomposition (per-bin sampling-noise correction).

The binned Murphy/Brier decomposition ``BinnedBrier = REL - RES + UNC`` with
``REL = sum_b w_b (conf_b - acc_b)^2`` and ``RES = sum_b w_b (acc_b - base_rate)^2`` shares the SAME positive
per-bin Bernoulli-noise bias that inflates the plug-in ECE (qual-1). Within each bin ``acc_b`` is a noisy estimate
of the bin's true positive rate, so ``E[(conf_b - acc_b)^2] = (conf_b - true_acc_b)^2 + Var(acc_b)`` -- the
reliability term is overstated by ``sum_b w_b Var(acc_b)``. A PERFECTLY CALIBRATED model has true REL 0 yet the
plug-in REL is strictly positive and grows with nbins. RES carries an equal-magnitude inflation of the bin-accuracy
spread.

Debiased estimator (Broecker 2009, "Reliability, sufficiency, and the decomposition of proper scores", QJRMS):
subtract the unbiased within-bin variance ``Var(acc_b) = acc_b*(1-acc_b)/(n_b-1)`` from BOTH the REL term (clamp >=0)
and the RES term, so ``REL_db - RES_db == REL_plugin - RES_plugin`` and the Murphy identity / BinnedBrier / UNC are
preserved EXACTLY (only the split shifts). Bins with n_b<2 keep their raw squared term.

HONEST metric: bias = (estimated REL) - (true population REL). For the calibrated scenarios true REL=0, so the
estimated REL IS the bias. Lower |REL bias| wins. Multi-seed (>=7), multi-scenario. We also assert the Murphy
identity (REL_db - RES_db + UNC == BinnedBrier) holds to fp precision for the debiased terms.

Run::

    python -m mlframe.metrics._benchmarks.bench_brier_decomp_debiased
"""
from __future__ import annotations

import numpy as np


def _scn_calibrated_uniform(rng, n):
    s = np.clip(rng.uniform(0.0, 1.0, n), 1e-6, 1 - 1e-6)
    return s, s.copy(), 0.0


def _scn_calibrated_beta(rng, n):
    s = np.clip(rng.beta(0.6, 8.0, n), 1e-6, 1 - 1e-6)
    return s, s.copy(), 0.0


def _scn_calibrated_bimodal(rng, n):
    half = n // 2
    s = np.empty(n)
    s[:half] = np.clip(rng.beta(2.0, 12.0, half), 1e-6, 1 - 1e-6)
    s[half:] = np.clip(rng.beta(12.0, 2.0, n - half), 1e-6, 1 - 1e-6)
    return s, s.copy(), 0.0


def _scn_miscal_sigmoid(rng, n):
    """Logit-shifted miscalibration; nonzero true REL (fine-grid reference)."""
    s = np.clip(rng.beta(1.2, 6.0, n), 1e-6, 1 - 1e-6)
    logit = np.log(s / (1 - s)) + 0.8
    acc = 1.0 / (1.0 + np.exp(-logit))
    return s, acc, None


def _scn_miscal_overconf(rng, n):
    s = np.clip(rng.uniform(0, 1, n), 1e-6, 1 - 1e-6)
    acc = np.clip(0.5 + 0.5 * (s - 0.5) * 0.4, 0, 1)
    return s, acc, None


SCENARIOS = {
    "calib_uniform": _scn_calibrated_uniform,
    "calib_beta_rare": _scn_calibrated_beta,
    "calib_bimodal": _scn_calibrated_bimodal,
    "miscal_sigmoid": _scn_miscal_sigmoid,
    "miscal_overconf": _scn_miscal_overconf,
}


def _rel_plugin(y, p, nbins):
    from mlframe.metrics.calibration._calibration_metrics import compute_ece_and_brier_decomposition

    return compute_ece_and_brier_decomposition(np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64), nbins)[1]


def _decomp_debiased(y, p, nbins):
    from mlframe.metrics.calibration._calibration_metrics import compute_brier_decomposition_debiased

    return compute_brier_decomposition_debiased(np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64), nbins)


def _truth_rel(scn_fn, nbins, n_big=400_000, fine=400):
    """Fine equal-frequency reference REL on a large independent sample (true REL=0 for calibrated)."""
    rng = np.random.default_rng(123456)
    s, acc, known = scn_fn(rng, n_big)
    if known is not None:
        return known
    y = (rng.random(n_big) < acc).astype(np.float64)
    order = np.argsort(s, kind="stable")
    edges = np.linspace(0, n_big, fine + 1).astype(np.int64)
    rel = 0.0
    for k in range(fine):
        lo, hi = edges[k], edges[k + 1]
        if hi <= lo:
            continue
        sel = order[lo:hi]
        rel += (hi - lo) / n_big * (s[sel].mean() - y[sel].mean()) ** 2
    return rel


def main(n=2000, nbins=15, seeds=tuple(range(8))):
    print(f"Brier-decomposition debiasing bench: n={n}, nbins={nbins}, seeds={seeds}\n")
    print(f"{'scenario':<18}{'truth_REL':>10}{'plugin_bias':>13}{'debias_bias':>13}  winner")
    pl_wins = db_wins = ties = 0
    max_identity_err = 0.0
    for name, fn in SCENARIOS.items():
        truth = _truth_rel(fn, nbins)
        pl_b = db_b = 0.0
        for sd in seeds:
            rng = np.random.default_rng(sd)
            s, acc, _ = fn(rng, n)
            y = (rng.random(n) < acc).astype(np.float64)
            rel_pl = _rel_plugin(y, s, nbins)
            rel_db, res_db, unc, binned = _decomp_debiased(y, s, nbins)
            max_identity_err = max(max_identity_err, abs((rel_db - res_db + unc) - binned))
            pl = abs(rel_pl - truth)
            db = abs(rel_db - truth)
            if db < pl - 1e-12:
                db_wins += 1
            elif pl < db - 1e-12:
                pl_wins += 1
            else:
                ties += 1
            pl_b += rel_pl - truth
            db_b += rel_db - truth
        pl_b /= len(seeds)
        db_b /= len(seeds)
        winner = "DEBIAS" if abs(db_b) < abs(pl_b) else "PLUGIN"
        print(f"{name:<18}{truth:>10.5f}{pl_b:>13.5f}{db_b:>13.5f}  {winner}")
    total = pl_wins + db_wins + ties
    print(f"\nper-cell |REL bias| wins (over {total} cells): debiased={db_wins}  plugin={pl_wins}  ties={ties}")
    print(f"max Murphy-identity error (debiased REL-RES+UNC vs BinnedBrier): {max_identity_err:.2e}")
    print("DECISION: flip headline Brier REL/RES to debiased iff debiased wins majority of cells.")


if __name__ == "__main__":
    main()
