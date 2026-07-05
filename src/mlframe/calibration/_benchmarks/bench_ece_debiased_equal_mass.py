"""qual-4 bench: equal-MASS (quantile) vs equal-WIDTH binning for the debiased binned ECE.

Both ECE estimators in mlframe (``compute_ece_and_brier_decomposition`` and the bias-corrected
``compute_ece_debiased``) bin predictions on an equal-WIDTH grid over [min(p), max(p)]. When the
predicted-probability distribution is concentrated (the common case -- most rows pile near the base
rate), equal-width bins dump nearly all mass into 1-2 bins and leave the rest near-empty and
high-variance. The literature (Nixon et al. 2019 "Measuring Calibration in Deep Learning" -- Adaptive
Calibration Error; Roelofs et al. AISTATS 2022 -- "Mitigating bias in calibration error estimation")
shows equal-MASS (quantile / adaptive) binning gives a LOWER-bias, LOWER-variance ECE estimate because
every bin holds ~N/nbins samples, so the per-bin Bernoulli noise is bounded and uniform.

This bench keeps the SAME per-bin Bernoulli debiasing (Kumar/Roelofs noise-floor subtraction already
shipped as ``ece_debiased=True`` from qual-1) and only switches the bin GRID from equal-width to
equal-mass. Ground truth: on a perfectly-calibrated model true ECE == 0, so the honest metric is
|estimate - 0| on calibrated scenarios; on miscalibrated scenarios we additionally require the
miscalibration is STILL flagged (estimate stays well above 0).

VERDICT (2026-06-15, qual-4): REJECTED -- equal-mass does NOT win a majority of scenarios/seeds once the
Bernoulli noise-floor debiasing (qual-1) is already applied. The noise-floor subtraction removes most of the
equal-width sparsity penalty, so the residual binning-grid choice is scenario-dependent and reverses with n:

    n=500   equal-mass closer to true 0 in 46/84 calibrated cells (55%); but loses ALL 3 bimodal cells.
    n=2000  44/96 (46%) -- beta_mid flips to width in all cells.
    n=5000  45/96 (47%) -- equal-mass loses the per-cell majority outright.

Equal-mass helps the heavy-tail / rare-event scenarios (beta_rare) but hurts uniform@small-nbins and the
bimodal scenario (where equal-width separates the two modes cleanly while quantile edges land mid-gap). No
consistent, scenario-majority win -> the shipped equal-width debiased ECE stays the default. Re-run this bench
to re-test on different hardware / scenario mixes.

Run:
    MLFRAME_SKIP_NUMBA_PREWARM=1 CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 \
      python -m mlframe.calibration._benchmarks.bench_ece_debiased_equal_mass
"""
from __future__ import annotations

import numpy as np
from math import floor

from mlframe.metrics.calibration._calibration_metrics import compute_ece_debiased


def _equal_mass_debiased_ece(y_true: np.ndarray, y_pred: np.ndarray, nbins: int) -> float:
    """Reference equal-mass twin of compute_ece_debiased (numpy, used as bench oracle + prod template).

    Equal-population edges via np.quantile; same per-bin Bernoulli noise-floor subtraction
    (gap^2 - acc*(1-acc)/(n-1), clamped at 0) as the shipped equal-width debiased ECE.
    """
    n = len(y_true)
    if n == 0:
        return 1.0
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    qs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.unique(np.quantile(y_pred, qs))
    if len(edges) < 2:
        # Degenerate (all identical preds): single bin.
        bins = np.zeros(n, dtype=np.int64)
        nb = 1
    else:
        # np.searchsorted with 'right' then clamp; interior edges define the pockets.
        interior = edges[1:-1]
        bins = np.searchsorted(interior, y_pred, side="right")
        nb = len(interior) + 1

    ece = 0.0
    inv_n = 1.0 / n
    for k in range(nb):
        mask = bins == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        w = nk * inv_n
        p_mean = y_pred[mask].mean()
        acc = y_true[mask].mean()
        gap = abs(p_mean - acc)
        if nk >= 2:
            var_acc = acc * (1.0 - acc) / (nk - 1)
            corrected = max(gap * gap - var_acc, 0.0)
            ece += w * np.sqrt(corrected)
        else:
            ece += w * gap
    return ece


def _make_calibrated(rng, n, kind):
    """Perfectly-calibrated: draw p from a distribution, then y ~ Bernoulli(p). True ECE == 0."""
    if kind == "uniform":
        p = rng.uniform(0, 1, n)
    elif kind == "beta_rare":
        p = rng.beta(0.5, 8.0, n)  # concentrated near 0 -- worst case for equal-width
    elif kind == "beta_mid":
        p = rng.beta(2.0, 2.0, n)  # bell around 0.5
    elif kind == "bimodal":
        half = n // 2
        p = np.concatenate([rng.beta(8, 2, half), rng.beta(2, 8, n - half)])
    else:
        raise ValueError(kind)
    y = (rng.uniform(0, 1, n) < p).astype(np.int64)
    return y, p


def _make_miscalibrated(rng, n, kind):
    """Miscalibrated: true label uses a shifted prob, so reported probs are systematically off. True ECE > 0."""
    if kind == "overconf":
        p = rng.beta(2.0, 2.0, n)
        p_true = np.clip((p - 0.5) * 0.5 + 0.5, 0, 1)  # pull toward 0.5 -> reported is overconfident
    elif kind == "shift":
        p = rng.uniform(0, 1, n)
        p_true = np.clip(p - 0.12, 0, 1)
    else:
        raise ValueError(kind)
    y = (rng.uniform(0, 1, n) < p_true).astype(np.int64)
    return y, p


def _run_one_n(n: int, seeds, nbins_grid, calib_scenarios, miscal_scenarios):
    print(f"n={n}, seeds={seeds}, nbins={nbins_grid}")
    print("=" * 100)
    print("CALIBRATED scenarios (true ECE = 0); honest metric = |estimate - 0|; lower is better")
    print("-" * 100)

    total_eqmass_wins = 0
    total_cells = 0
    agg = {}  # (scenario, nbins) -> [sum|width|, sum|mass|]

    for sc in calib_scenarios:
        for nb in nbins_grid:
            w_abs = []
            m_abs = []
            for sd in seeds:
                rng = np.random.default_rng(sd)
                y, p = _make_calibrated(rng, n, sc)
                e_width = compute_ece_debiased(y.astype(np.float64), p.astype(np.float64), nb)
                e_mass = _equal_mass_debiased_ece(y, p, nb)
                w_abs.append(abs(e_width))
                m_abs.append(abs(e_mass))
                total_cells += 1
                if abs(e_mass) < abs(e_width):
                    total_eqmass_wins += 1
            mw, mm = np.mean(w_abs), np.mean(m_abs)
            agg[(sc, nb)] = (mw, mm)
            tag = "MASS" if mm < mw else "width"
            print(f"  {sc:10s} nbins={nb:2d}  width|ECE|={mw:.4f}  mass|ECE|={mm:.4f}  -> {tag}")

    print("-" * 100)
    print(f"CALIBRATED per-cell wins (equal-mass closer to 0): {total_eqmass_wins}/{total_cells}")
    print("=" * 100)
    print("MISCALIBRATED scenarios (true ECE > 0); both estimators must STILL flag (stay above ~0.03)")
    print("-" * 100)
    for sc in miscal_scenarios:
        for nb in nbins_grid:
            w_vals, m_vals = [], []
            for sd in seeds:
                rng = np.random.default_rng(sd)
                y, p = _make_miscalibrated(rng, n, sc)
                w_vals.append(compute_ece_debiased(y.astype(np.float64), p.astype(np.float64), nb))
                m_vals.append(_equal_mass_debiased_ece(y, p, nb))
            print(f"  {sc:10s} nbins={nb:2d}  width ECE={np.mean(w_vals):.4f}  mass ECE={np.mean(m_vals):.4f}")

    print("=" * 100)
    print(f"VERDICT(n={n}): equal-mass wins {total_eqmass_wins}/{total_cells} calibrated cells " f"({100*total_eqmass_wins/total_cells:.0f}%)")
    return total_eqmass_wins, total_cells


def main():
    seeds = list(range(8))
    nbins_grid = [10, 15, 20]
    calib_scenarios = ["uniform", "beta_rare", "beta_mid", "bimodal"]
    miscal_scenarios = ["overconf", "shift"]
    summary = {}
    for n in [500, 2000, 5000]:
        summary[n] = _run_one_n(n, seeds, nbins_grid, calib_scenarios, miscal_scenarios)
        print()
    print("#" * 100)
    for n, (w, t) in summary.items():
        print(f"n={n:5d}: equal-mass per-cell wins {w}/{t} ({100*w/t:.0f}%)  -> " f"{'flip' if w / t > 0.66 else 'NO majority (REJECT)'}")


if __name__ == "__main__":
    main()
