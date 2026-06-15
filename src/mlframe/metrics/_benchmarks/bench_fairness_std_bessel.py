"""Bench: unbiased (ddof=1, Bessel) vs biased (ddof=0, population) sample std for the
fairness subgroup-dispersion number (``compute_fairness_metrics`` ``metric_std`` column
and the ``robust_mlperf_metric`` penalty term).

The fairness `metric_std` summarises the SPREAD of per-subgroup model performance across a
small number K of subgroups (typically 2-10). It estimates the population dispersion sigma of
subgroup performance from a sample of K observed subgroup metrics. numpy's default ddof=0
(population std) is the maximum-likelihood but DOWNWARD-biased estimator of sigma at small K;
the universal sample-std convention (pandas, R, statistics texts) is ddof=1 (Bessel), which is
unbiased for the VARIANCE and substantially less biased for the std at small K.

Ground truth: we draw K subgroup performance values from N(mu, sigma_true^2) with a KNOWN
sigma_true, then score each estimator by |estimate - sigma_true|. The estimand is the
dispersion of subgroup performance, so sigma_true is the correct target by construction.

Run:
    python -m mlframe.metrics._benchmarks.bench_fairness_std_bessel
"""

from __future__ import annotations

import numpy as np


def _sample_std(x: np.ndarray, ddof: int) -> float:
    return float(np.std(x, ddof=ddof))


def run() -> dict:
    # Scenarios: number of subgroups K (the driver of Bessel bias) x true dispersion sigma_true.
    # Small K is the realistic fairness regime (gender=2, age-decile=3-10, region handful).
    scenarios = [
        ("K2_sigma0.05", 2, 0.05),
        ("K3_sigma0.05", 3, 0.05),
        ("K4_sigma0.10", 4, 0.10),
        ("K6_sigma0.08", 6, 0.08),
        ("K10_sigma0.05", 10, 0.05),
    ]
    seeds = list(range(8))
    mu = 0.80  # e.g. AUC ~ 0.80 baseline; irrelevant to the std estimand.

    # Monte-Carlo over many redraws per (scenario, seed) cell so the |bias| estimate is itself
    # stable: each cell reports the MEAN over n_mc redraws of |estimate - sigma_true|.
    n_mc = 4000

    results = {}
    signed = {}  # (scenario,seed) -> (signed_std_bias_ddof0, signed_std_bias_ddof1, signed_var_bias_ddof0, signed_var_bias_ddof1)
    win_ddof1 = win_ddof0 = ties = 0
    for sc_name, K, sigma_true in scenarios:
        var_true = sigma_true * sigma_true
        for seed in seeds:
            rng = np.random.default_rng(seed * 100003 + K)
            err0 = err1 = 0.0
            sstd0 = sstd1 = svar0 = svar1 = 0.0
            for _ in range(n_mc):
                perfs = rng.normal(mu, sigma_true, size=K)
                s0 = _sample_std(perfs, 0)
                s1 = _sample_std(perfs, 1)
                err0 += abs(s0 - sigma_true)
                err1 += abs(s1 - sigma_true)
                sstd0 += s0 - sigma_true
                sstd1 += s1 - sigma_true
                svar0 += float(np.var(perfs, ddof=0)) - var_true
                svar1 += float(np.var(perfs, ddof=1)) - var_true
            err0 /= n_mc
            err1 /= n_mc
            results[(sc_name, seed)] = (err0, err1)
            signed[(sc_name, seed)] = (sstd0 / n_mc, sstd1 / n_mc, svar0 / n_mc, svar1 / n_mc)
            if err1 < err0:
                win_ddof1 += 1
            elif err0 < err1:
                win_ddof0 += 1
            else:
                ties += 1

    print("=" * 96)
    print("Fairness subgroup-dispersion std: |estimate - sigma_true|  (ddof=0 biased vs ddof=1 Bessel)")
    print("=" * 96)
    print(f"{'scenario':<16}{'seed':<6}{'ddof=0 |bias|':<16}{'ddof=1 |bias|':<16}{'winner':<10}")
    for (sc_name, seed), (e0, e1) in results.items():
        w = "ddof=1" if e1 < e0 else ("ddof=0" if e0 < e1 else "tie")
        print(f"{sc_name:<16}{seed:<6}{e0:<16.5f}{e1:<16.5f}{w:<10}")

    print("-" * 96)
    # Per-scenario aggregate mean |bias|.
    for sc_name, K, sigma_true in scenarios:
        e0 = np.mean([results[(sc_name, s)][0] for s in seeds])
        e1 = np.mean([results[(sc_name, s)][1] for s in seeds])
        print(f"{sc_name:<16} K={K:<3} sigma_true={sigma_true:<6} mean|bias| ddof0={e0:.5f}  ddof1={e1:.5f}  ratio={e0 / e1:.3f}x")

    print("-" * 96)
    print("SIGNED bias (negative = systematic UNDER-estimation of dispersion -- the fairness harm):")
    sstd_win1 = svar_win1 = 0
    for sc_name, K, sigma_true in scenarios:
        b_std0 = np.mean([signed[(sc_name, s)][0] for s in seeds])
        b_std1 = np.mean([signed[(sc_name, s)][1] for s in seeds])
        b_var0 = np.mean([signed[(sc_name, s)][2] for s in seeds])
        b_var1 = np.mean([signed[(sc_name, s)][3] for s in seeds])
        if abs(b_std1) < abs(b_std0):
            sstd_win1 += 1
        if abs(b_var1) < abs(b_var0):
            svar_win1 += 1
        print(f"{sc_name:<16} K={K:<3} std signed-bias ddof0={b_std0:+.5f} ddof1={b_std1:+.5f}  |  var signed-bias ddof0={b_var0:+.6f} ddof1={b_var1:+.6f}")
    print(f"std signed-|bias| ddof1-better scenarios: {sstd_win1}/{len(scenarios)}   var signed-|bias| ddof1-better: {svar_win1}/{len(scenarios)}")

    print("-" * 96)
    total = win_ddof1 + win_ddof0 + ties
    print(f"per-cell |estimate-sigma| wins: ddof=1 {win_ddof1}/{total}  ddof=0 {win_ddof0}/{total}  ties {ties}")
    return {"results": results, "signed": signed, "win_ddof1": win_ddof1, "win_ddof0": win_ddof0, "ties": ties}


if __name__ == "__main__":
    run()
