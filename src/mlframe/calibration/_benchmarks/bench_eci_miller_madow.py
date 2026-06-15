"""Bench: plug-in Shannon entropy in entropy_calibration_index (ECI) is negatively biased, so ECI = log(bins) - H_plugin is POSITIVELY biased.

Ground truth: a perfectly-calibrated model produces a UNIFORM PIT distribution. The true bin probabilities are all 1/bins, so the true entropy is log(bins) and the true ECI is exactly 0. At finite n, the plug-in entropy estimator H_hat = -sum p_i log p_i underestimates entropy (well-known negative bias ~ -(K-1)/(2N)), which inflates ECI = log(bins) - H_hat above its true value of 0 -- spuriously reporting miscalibration where there is none.

Miller-Madow correction: H_mm = H_hat + (K_obs - 1)/(2N), where K_obs is the number of non-empty bins. This cancels the leading 1/N bias term, pulling ECI back toward the true 0 on calibrated data.

This bench runs >=5 seeds x 2 scenarios (bins=10, bins=20) and reports per-cell |ECI - 0| for plug-in vs Miller-Madow. RESOLVED only on a majority-of-cells reduction. Run:
    python -m mlframe.calibration._benchmarks.bench_eci_miller_madow
"""
from __future__ import annotations

import numpy as np
from scipy.stats import entropy


def _eci_plugin(pit_values, bins=10):
    counts, _ = np.histogram(pit_values, bins=bins, range=(0, 1), density=False)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return float(np.log(bins) - entropy(probs))


def _eci_mm(pit_values, bins=10):
    counts, _ = np.histogram(pit_values, bins=bins, range=(0, 1), density=False)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    h_plugin = entropy(probs)
    k_obs = int(np.count_nonzero(counts))
    h_mm = h_plugin + (k_obs - 1) / (2.0 * total)
    return float(np.log(bins) - h_mm)


def main():
    n = 500  # small-n: where finite-sample entropy bias bites
    seeds = list(range(7))
    scenarios = {"bins=10": 10, "bins=20": 20}

    print(f"Ground truth: perfectly-calibrated => uniform PIT => true ECI = 0. n={n}\n")
    plugin_wins = 0
    mm_wins = 0
    total_cells = 0
    agg_plugin = []
    agg_mm = []
    for sc_name, bins in scenarios.items():
        print(f"--- scenario {sc_name} ---")
        for seed in seeds:
            rng = np.random.default_rng(seed)
            pit = rng.uniform(0.0, 1.0, size=n)  # perfectly-calibrated => uniform PIT
            e_plugin = abs(_eci_plugin(pit, bins) - 0.0)
            e_mm = abs(_eci_mm(pit, bins) - 0.0)
            agg_plugin.append(e_plugin)
            agg_mm.append(e_mm)
            total_cells += 1
            winner = "MM" if e_mm < e_plugin else "plugin"
            if e_mm < e_plugin:
                mm_wins += 1
            else:
                plugin_wins += 1
            print(f"  seed={seed}  |ECI_plugin|={e_plugin:.5f}  |ECI_mm|={e_mm:.5f}  winner={winner}")

    print(f"\nMM wins {mm_wins}/{total_cells} cells; plugin wins {plugin_wins}/{total_cells}")
    print(f"mean |ECI_plugin| = {np.mean(agg_plugin):.5f}   mean |ECI_mm| = {np.mean(agg_mm):.5f}")
    print(f"VERDICT: {'RESOLVED (MM majority win, closer to true 0)' if mm_wins > total_cells / 2 else 'REJECT'}")


if __name__ == "__main__":
    main()
