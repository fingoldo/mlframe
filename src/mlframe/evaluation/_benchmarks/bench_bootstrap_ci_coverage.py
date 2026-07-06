"""Empirical coverage bench: percentile vs BCa bootstrap CI on skewed sampling distributions.

The percentile bootstrap CI (Efron) silently UNDER-COVERS when the metric's sampling distribution is
skewed and/or bounded -- the canonical cases being ROC-AUC near 1.0 (left-skewed, capped at 1.0) and
Pearson correlation r (skewed, bounded in [-1, 1]). BCa (bias-corrected and accelerated, Efron 1987)
corrects for both median bias (z0) and skew (acceleration via jackknife), recovering close-to-nominal
coverage. This bench MEASURES that gap empirically.

Method (no analytic shortcuts): for each (scenario, master-seed) cell we run MANY Monte-Carlo trials.
Each trial draws a FRESH sample from a population with a KNOWN true metric value, computes a nominal-95%
bootstrap CI by both methods, and records whether the CI contains the true value. Empirical coverage is
``hits / trials``; the better method is the one whose coverage is closest to 0.95 (we report the absolute
miscoverage gap ``|coverage - 0.95|``).

Scenarios (both have a known closed-form / high-precision true value), chosen in the SMALL-n / high-value
regime where the percentile interval's documented skew deficiency actually bites (at large n / mid-value the
sampling distribution is near-symmetric and both methods are ~nominal -- so a fair test of the BCa advantage
must probe the skewed regime, not pad the count with near-symmetric cells):
  - auc_097: binary problem with population AUC ~0.97 at n=150 (strongly left-skewed, capped at 1.0).
  - corr_090: bivariate-normal Pearson r with rho=0.90 at n=40 (strongly skewed, Fisher-z curvature).

Run:
    python -m mlframe.evaluation._benchmarks.bench_bootstrap_ci_coverage

Verdict (recorded in tests/perf/results/_quality_loop_log.md): BCa is closer to nominal on the MAJORITY
of (scenario x seed) cells -> flip the default ``method`` to "bca"; else honest REJECT.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from mlframe.evaluation.bootstrap import bootstrap_metric

_AUC_SEP = 2.66  # two unit-variance Gaussians separated by sep -> true AUC = Phi(sep/sqrt2) ~ 0.97


def _auc_population_value() -> float:
    """True AUC for two unit-variance Gaussians separated by ``sep``: Phi(sep / sqrt(2))."""
    return float(stats.norm.cdf(_AUC_SEP / np.sqrt(2.0)))


def _draw_auc_sample(rng: np.random.Generator, n: int, sep: float = _AUC_SEP) -> tuple[np.ndarray, np.ndarray]:
    y = rng.integers(0, 2, size=n)
    score = rng.normal(loc=sep * y, scale=1.0)
    # Guard against a degenerate single-class draw (rare at n=400 but possible).
    if y.min() == y.max():
        y[0] = 0
        y[1] = 1
    return y, score


def _fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    # Mann-Whitney U via rankdata; matches roc_auc_score, no sklearn dep in the hot loop.
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ties.
    ranks = stats.rankdata(s, method="average")
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _draw_corr_sample(rng: np.random.Generator, n: int, rho: float = 0.90) -> tuple[np.ndarray, np.ndarray]:
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    y = rho * x + np.sqrt(1.0 - rho * rho) * z
    return x, y


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def run_scenario(name: str, draw, metric_fn, true_value: float, seed: int, n_sample: int, n_trials: int = 600, n_bootstrap: int = 800) -> dict:
    rng = np.random.default_rng(seed)
    hits = {"percentile": 0, "bca": 0}
    widths = {"percentile": 0.0, "bca": 0.0}
    valid = 0
    for _ in range(n_trials):
        a, b = draw(rng, n_sample)
        # One resample loop per method but SAME per-trial draw, so the comparison is paired.
        try:
            r_pct = bootstrap_metric(
                a, b, metric_fn=metric_fn, n_bootstrap=n_bootstrap, alpha=0.05, random_state=int(rng.integers(0, 2**31)), method="percentile"
            )
            r_bca = bootstrap_metric(a, b, metric_fn=metric_fn, n_bootstrap=n_bootstrap, alpha=0.05, random_state=int(rng.integers(0, 2**31)), method="bca")
        except Exception:  # nosec B112 - best-effort path
            continue
        valid += 1
        if r_pct["lo"] <= true_value <= r_pct["hi"]:
            hits["percentile"] += 1
        if r_bca["lo"] <= true_value <= r_bca["hi"]:
            hits["bca"] += 1
        widths["percentile"] += r_pct["hi"] - r_pct["lo"]
        widths["bca"] += r_bca["hi"] - r_bca["lo"]
    cov = {m: hits[m] / valid for m in hits}
    w = {m: widths[m] / valid for m in widths}
    gap = {m: abs(cov[m] - 0.95) for m in cov}
    winner = "bca" if gap["bca"] < gap["percentile"] else "percentile"
    return {"scenario": name, "seed": seed, "true": true_value, "valid": valid, "coverage": cov, "miscov_gap": gap, "mean_width": w, "winner": winner}


def main() -> None:
    true_auc = _auc_population_value()
    true_corr = 0.90
    scenarios = [
        ("auc_097", _draw_auc_sample, _fast_auc, true_auc, 150),
        ("corr_090", _draw_corr_sample, _pearson, true_corr, 40),
    ]
    seeds = [11, 23, 37, 51, 67]
    cells = []
    print(f"true AUC = {true_auc:.4f}, true corr = {true_corr:.4f}\n")
    for name, draw, mfn, tv, n_sample in scenarios:
        for seed in seeds:
            res = run_scenario(name, draw, mfn, tv, seed, n_sample)
            cells.append(res)
            print(f"[{name} seed={seed:3d}] cov pct={res['coverage']['percentile']:.3f} "
                  f"bca={res['coverage']['bca']:.3f} | gap pct={res['miscov_gap']['percentile']:.3f} "
                  f"bca={res['miscov_gap']['bca']:.3f} | width pct={res['mean_width']['percentile']:.4f} "
                  f"bca={res['mean_width']['bca']:.4f} -> winner={res['winner']}")

    bca_wins = sum(1 for c in cells if c["winner"] == "bca")
    print(f"\nBCa closer-to-nominal in {bca_wins}/{len(cells)} cells.")
    by_scen: dict[str, list] = {}
    for c in cells:
        by_scen.setdefault(c["scenario"], []).append(c)
    for scen, cs in by_scen.items():
        pct = np.mean([c["coverage"]["percentile"] for c in cs])
        bca = np.mean([c["coverage"]["bca"] for c in cs])
        print(f"  {scen}: mean coverage pct={pct:.3f} bca={bca:.3f} (nominal 0.95)")


if __name__ == "__main__":
    main()
