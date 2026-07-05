"""Isolated bench: ECE n_bins default (DEFAULT_ECE_NBINS=15) bias+variance vs ground truth.

Honest metric. Each scenario fixes a TRUE calibration map g(p)=P(y=1|score=p) on a
known score distribution, so the population calibration error
  ECE_true = E_p |g(p) - p|   (L1, the same functional the binned estimator targets)
is computable to high resolution by integrating the analytic map over the score density.

For a finite sample of size n the equal-width binned estimator
  ECE_hat(n_bins) = sum_b (|B_b|/n) |mean_y(B_b) - mean_p(B_b)|
is biased: too few bins under-resolve the miscalibration shape (bias down when g-p
changes sign within a bin / averages out curvature), too many bins inflate ECE upward
because per-bin |mean_y - mean_p| picks up sampling noise that does not cancel under |.|.

We measure, per (scenario, n, n_bins), over many seeds:
  bias    = mean(ECE_hat) - ECE_true
  std     = std(ECE_hat)
  rmse    = sqrt(mean((ECE_hat - ECE_true)^2))     <- the honest combined metric

Winner per (scenario, n) = n_bins minimising RMSE. We flip DEFAULT_ECE_NBINS only if a
single alternative wins the MAJORITY of (scenario, n) cells, robustly across seeds.

Run:  PYTHONPATH=src python -m mlframe.calibration._benchmarks.bench_ece_nbins
"""

from __future__ import annotations

import numpy as np

from mlframe.calibration.policy import _ece_score

# score distribution: scores ~ Beta(a,b) clipped to (eps,1-eps); calibration map g(p).
EPS = 1e-6
N_GRID = 2_000_00  # high-res grid for ECE_true integration


def _ece_true(g, score_a, score_b, rng_grid):
    """Population E_p|g(p)-p| under score~Beta(a,b), by high-res MC on a fixed grid."""
    s = rng_grid.beta(score_a, score_b, size=N_GRID)
    s = np.clip(s, EPS, 1 - EPS)
    return float(np.mean(np.abs(g(s) - s)))


def _sample(g, score_a, score_b, n, rng):
    s = rng.beta(score_a, score_b, size=n)
    s = np.clip(s, EPS, 1 - EPS)
    y = (rng.random(n) < g(s)).astype(np.float64)
    return y, s


# ---- scenarios: (name, score_a, score_b, g) ----
def _scenarios():
    sc = []
    # S1 perfectly calibrated (ECE_true ~ 0): underdamped est should stay ~0; bins matter for noise floor
    sc.append(("perfect_cal", 2.0, 2.0, lambda p: p))
    # S2 mild overconfidence (logit shrink toward 0.5)
    def g_over(p):
        z = np.log(p / (1 - p)) * 0.6
        return 1.0 / (1.0 + np.exp(-z))
    sc.append(("overconfident", 2.0, 2.0, g_over))
    # S3 underconfidence (logit expand)
    def g_under(p):
        z = np.log(p / (1 - p)) * 1.6
        return 1.0 / (1.0 + np.exp(-z))
    sc.append(("underconfident", 2.0, 2.0, g_under))
    # S4 sigmoidal S-shape miscalibration, skewed score dist
    def g_sshape(p):
        z = np.log(p / (1 - p))
        return 1.0 / (1.0 + np.exp(-(z * 0.7 + 0.4 * np.sin(2.5 * z))))
    sc.append(("s_shape_skew", 1.3, 3.0, g_sshape))
    # S5 constant bias (systematic +0.08, clipped)
    sc.append(("shift_up", 3.0, 3.0, lambda p: np.clip(p + 0.08, 0, 1)))
    # S6 piecewise: well-cal low, overconfident high -> sign change within range
    def g_pw(p):
        return np.where(p < 0.5, p, np.clip(0.5 + (p - 0.5) * 0.5, 0, 1))
    sc.append(("piecewise_high", 2.0, 2.0, g_pw))
    return sc


def run(seeds=12, ns=(1000, 5000, 20000), nbins_grid=(5, 10, 15, 20, 30, 50), verbose=True):
    rng_grid = np.random.default_rng(999)
    scenarios = _scenarios()
    # warm kernel
    _ece_score(np.array([0.0, 1.0]), np.array([0.3, 0.7]), 10)

    cell_winner = {}  # (scenario,n) -> nbins with min rmse
    rows = []
    for name, sa, sb, g in scenarios:
        ece_true = _ece_true(g, sa, sb, rng_grid)
        for n in ns:
            est = {nb: [] for nb in nbins_grid}
            for seed in range(seeds):
                rng = np.random.default_rng(10_000 * seed + n + hash(name) % 1000)
                y, s = _sample(g, sa, sb, n, rng)
                y = np.ascontiguousarray(y)
                s = np.ascontiguousarray(s)
                for nb in nbins_grid:
                    est[nb].append(_ece_score(y, s, nb))
            best_nb, best_rmse = None, np.inf
            for nb in nbins_grid:
                arr = np.array(est[nb])
                bias = arr.mean() - ece_true
                std = arr.std()
                rmse = np.sqrt(np.mean((arr - ece_true) ** 2))
                rows.append((name, n, nb, ece_true, bias, std, rmse))
                if rmse < best_rmse:
                    best_rmse, best_nb = rmse, nb
            cell_winner[(name, n)] = best_nb

    if verbose:
        print(f"ECE n_bins bench | seeds={seeds} | ground-truth grid={N_GRID}")
        print(f"{'scenario':16s} {'n':>6s} {'nbins':>5s} {'ECEtrue':>8s} {'bias':>9s} {'std':>8s} {'rmse':>8s}")
        for name, n, nb, et, bias, std, rmse in rows:
            mark = " *" if cell_winner[(name, n)] == nb else ""
            print(f"{name:16s} {n:6d} {nb:5d} {et:8.4f} {bias:+9.4f} {std:8.4f} {rmse:8.4f}{mark}")
        print("\nPer-cell RMSE winners:")
        from collections import Counter
        c = Counter(cell_winner.values())
        for nb, cnt in sorted(c.items()):
            print(f"  nbins={nb:3d}: wins {cnt}/{len(cell_winner)} cells")
        # default vs best-fixed
        # which single nbins minimises mean RMSE rank across all cells?
        per_nb_rmse = {nb: [] for nb in nbins_grid}
        # rebuild rmse lookup
        rl = {}
        for name, n, nb, et, bias, std, rmse in rows:
            rl[(name, n, nb)] = rmse
        for name, n in cell_winner:
            for nb in nbins_grid:
                per_nb_rmse[nb].append(rl[(name, n, nb)])
        print("\nMean RMSE across all cells per fixed nbins (lower=better):")
        for nb in nbins_grid:
            print(f"  nbins={nb:3d}: mean_rmse={np.mean(per_nb_rmse[nb]):.4f}")
        # head-to-head: candidate vs current default 15
        print("\nHead-to-head cell RMSE win-rate vs default nbins=15:")
        for nb in nbins_grid:
            if nb == 15:
                continue
            wins = sum(1 for (name, n) in cell_winner if rl[(name, n, nb)] < rl[(name, n, 15)])
            print(f"  nbins={nb:3d}: beats 15 in {wins}/{len(cell_winner)} cells")
    return cell_winner, rows


if __name__ == "__main__":
    run()
