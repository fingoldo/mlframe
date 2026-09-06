"""Hierarchical posterior over an arm's advantage, by exact quadrature rather than by sampling.

The frequentist paired ``t`` answers "is this arm's advantage over ``all-features`` distinguishable from
zero in this scenario". It cannot answer the two questions the benchmark actually exists to settle:

* pooled across scenarios, how large is the advantage, and is it small enough to be practically nothing;
* does the advantage depend on the task at all, or is one number a fair summary of every bed.

Both are properties of a random-effects model, ``delta_s ~ N(mu, se_s^2 + tau^2)``, where ``mu`` is the
pooled advantage and ``tau`` the between-scenario heterogeneity. The second question is ``tau``, and it is
the more interesting one: a method with a large ``mu`` and a larger ``tau`` is not a better method, it is a
method whose value is decided by which bed you happen to have.

No sampler is needed. Under a flat prior on ``mu`` the conditional posterior is conjugate and integrates in
closed form at each ``tau``, leaving a one-dimensional integral over ``tau`` that a 200-point grid resolves
to far better precision than any Monte Carlo run of comparable cost, with zero simulation error and zero
convergence diagnostics to get wrong. The posterior for ``mu`` is then an exact finite mixture of normals.

The prior on ``tau`` is half-Cauchy, never an inverse-gamma with small parameters. That choice is load
bearing rather than conventional: ``InvGamma(eps, eps)`` places unbounded mass near zero, which fabricates a
confident pooled effect out of genuinely heterogeneous scenarios by asserting they cannot disagree.
``prior_sensitivity`` re-runs the whole fit under the three priors the pre-registration names, so the
sensitivity is reported rather than assumed away.

The scale matters as much as the model. Run this on **normalized skill**, not on a raw Brier difference:
skill is ``(Brier_baserate - Brier_method) / (Brier_baserate - Brier_Bayes)``, so a fixed ROPE means the
same thing at a 50% base rate and at a 1% one, where the raw Brier scale differs by a factor of twenty-five
and a hardcoded band silently changes its meaning between beds.

``rope_curve`` exists so the ROPE width is not a thing to argue about in the abstract: it is the posterior
CDF of ``|mu|``, free to compute, and each reader applies their own threshold to the same curve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ._leaderboard import NULL_ARM
from ._paired_stats import average_over_cv_seed, paired_differences, paired_t_test

logger = logging.getLogger(__name__)

__all__ = [
    "ScenarioEffect",
    "HierarchicalPosterior",
    "extract_skill_rows",
    "scenario_effects",
    "fit_hierarchical",
    "rope_curve",
    "prior_sensitivity",
    "PRIOR_KINDS",
    "DEFAULT_ROPE",
    "hierarchical_report",
]

# One percent of achievable skill. Pre-registered on the normalized-skill scale precisely so it does not
# have to move between beds, models or prevalences.
DEFAULT_ROPE = 0.01
PRIOR_KINDS: Tuple[str, ...] = ("half_cauchy", "half_normal", "half_cauchy_wide")
_TAU_GRID_POINTS = 200
_MU_GRID_POINTS = 1201


@dataclass(frozen=True)
class ScenarioEffect:
    """One scenario's estimated advantage of an arm over the null, with its own standard error."""

    scenario: str
    m: int
    delta: float
    se: float


@dataclass(frozen=True)
class HierarchicalPosterior:
    """Posterior summaries for the pooled advantage ``mu`` and the between-scenario spread ``tau``."""

    n_scenarios: int
    mu_mean: float
    mu_sd: float
    mu_hdi: Tuple[float, float]
    p_positive: float
    p_in_rope: float
    rope: float
    tau_mean: float
    tau_median: float
    tau_hdi: Tuple[float, float]
    prior: str


def extract_skill_rows(records: Iterable[Dict[str, Any]], model: str, k_label: str) -> List[Dict[str, Any]]:
    """Flatten cell records into rows carrying normalized skill for one (model, K) combination.

    Skill rather than a raw metric, because the pooled effect and the ROPE are only comparable across beds
    on a scale where 1.0 is the achievable ceiling and 0.0 is the base rate.
    """
    out: List[Dict[str, Any]] = []
    for rec in records:
        if rec.get("status") != "ok":
            continue
        block = (rec.get("scores") or {}).get(k_label)
        if not isinstance(block, dict):
            continue
        skill = (block.get("skill") or {}).get(model)
        if skill is None or not np.isfinite(float(skill)):
            continue
        out.append(
            {
                "arm": rec["arm"],
                "scenario": rec["scenario"],
                "dataset_seed": int(rec["dataset_seed"]),
                "cv_seed": int(rec.get("cv_seed", 0)),
                "value": float(skill),
            }
        )
    return out


def scenario_effects(rows: Sequence[Dict[str, Any]], arm: str, null_arm: str = NULL_ARM, min_seeds: int = 3) -> List[ScenarioEffect]:
    """Return one ``ScenarioEffect`` per scenario where ``arm`` and the null are paired on enough seeds.

    ``rows`` must already be collapsed over ``cv_seed``; the paired-stats layer asserts that, and a frame
    carrying more than one row per (arm, scenario, dataset_seed) would understate every standard error here
    by the square root of the duplication.
    """
    out: List[ScenarioEffect] = []
    for scenario in sorted({str(r["scenario"]) for r in rows}):
        deltas = paired_differences(rows, arm=arm, null_arm=null_arm, scenario=scenario)
        if len(deltas) < min_seeds:
            continue
        stat = paired_t_test(deltas)
        if stat.se is None or not np.isfinite(stat.se) or stat.se < 0.0:
            logger.info("scenario %s excluded from the hierarchical fit: no usable within-scenario error", scenario)
            continue
        out.append(ScenarioEffect(scenario=scenario, m=stat.m, delta=float(stat.mean_delta), se=float(stat.se)))
    return out


def _integrate(values: np.ndarray, grid: np.ndarray) -> float:
    """Trapezoidal integral, written out because the name for it moved between numpy 1.x and 2.x.

    ``np.trapz`` is gone in numpy 2 and ``np.trapezoid`` does not exist in the 1.23 floor this project still
    supports, so neither name can be called directly here.
    """
    if grid.size < 2:
        return 0.0
    widths = np.diff(grid)
    return float((0.5 * widths * (values[:-1] + values[1:])).sum())


def _tau_prior_density(tau: np.ndarray, kind: str, scale: float) -> np.ndarray:
    """Return the unnormalized prior density on ``tau`` for one of the pre-registered prior families."""
    if kind == "half_cauchy":
        return 1.0 / (1.0 + (tau / scale) ** 2)
    if kind == "half_cauchy_wide":
        wide = 3.0 * scale
        return 1.0 / (1.0 + (tau / wide) ** 2)
    if kind == "half_normal":
        return np.exp(-0.5 * (tau / scale) ** 2)
    raise ValueError(f"unknown tau prior {kind!r}; expected one of {PRIOR_KINDS}")


def _tau_scale(effects: Sequence[ScenarioEffect]) -> float:
    """Return the prior scale for ``tau``: the median within-scenario standard error, floored away from zero.

    Tying the scale to the observed measurement error keeps the prior weakly informative on the scale the
    data actually live on, instead of importing a number whose meaning changes with the metric.
    """
    ses = np.asarray([e.se for e in effects], dtype=np.float64)
    scale = float(np.median(ses))
    return scale if scale > 0.0 else 1e-6


def _hdi_from_grid(grid: np.ndarray, density: np.ndarray, mass: float = 0.95) -> Tuple[float, float]:
    """Return the narrowest interval on ``grid`` holding ``mass`` of the (unnormalized) density."""
    weights = np.clip(density, 0.0, None)
    total = weights.sum()
    if total <= 0.0:
        return (float("nan"), float("nan"))
    cdf = np.cumsum(weights) / total
    lo_target, hi_target = (1.0 - mass) / 2.0, 1.0 - (1.0 - mass) / 2.0
    lo = float(np.interp(lo_target, cdf, grid))
    hi = float(np.interp(hi_target, cdf, grid))
    return (lo, hi)


def _mu_mixture(effects: Sequence[ScenarioEffect], prior: str, tau_grid_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(tau_grid, tau_weights, mu_mean_given_tau, mu_var_given_tau)`` for the conjugate mixture.

    At a fixed ``tau`` the posterior for ``mu`` under a flat prior is normal with precision
    ``sum 1/(se_s^2 + tau^2)``. Integrating ``mu`` out leaves the marginal posterior for ``tau``, evaluated
    here in logs so a scenario with a small standard error cannot overflow the product.
    """
    deltas = np.asarray([e.delta for e in effects], dtype=np.float64)
    variances = np.asarray([e.se for e in effects], dtype=np.float64) ** 2

    scale = _tau_scale(effects)
    # Upper limit generous enough that the posterior has decayed: the observed spread of the effects
    # themselves bounds any tau the data support, and the prior handles the rest.
    tau_max = max(6.0 * scale, 3.0 * float(np.std(deltas)) if deltas.size > 1 else 0.0, 1e-6)
    # A scenario whose seeds all moved by exactly the same amount has zero within-scenario error, which is
    # real information rather than a defect -- but at tau = 0 it would carry infinite weight, so the grid
    # starts one step above zero when that happens. The excluded sliver holds no mass: at tau = 0 a set of
    # effects that disagree at all has likelihood zero, and one that agrees exactly is already summarized by
    # the first grid point.
    tau_lo = (tau_max / tau_grid_points) if float(variances.min()) <= 0.0 else 0.0
    tau_grid = np.linspace(tau_lo, tau_max, tau_grid_points)

    total_var = variances[None, :] + (tau_grid**2)[:, None]
    precision = 1.0 / total_var
    precision_sum = precision.sum(axis=1)
    mu_hat = (precision * deltas[None, :]).sum(axis=1) / precision_sum
    mu_var = 1.0 / precision_sum

    log_lik = -0.5 * np.log(total_var).sum(axis=1) - 0.5 * (precision * (deltas[None, :] - mu_hat[:, None]) ** 2).sum(axis=1)
    log_post = log_lik + 0.5 * np.log(mu_var) + np.log(np.clip(_tau_prior_density(tau_grid, prior, scale), 1e-300, None))
    weights = np.exp(log_post - log_post.max())
    weights /= weights.sum()
    return tau_grid, weights, mu_hat, mu_var


def _mu_grid_density(tau_weights: np.ndarray, mu_mean: np.ndarray, mu_var: np.ndarray, points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the normal mixture for ``mu`` on a grid wide enough to hold its tails."""
    sd = np.sqrt(mu_var)
    lo = float((mu_mean - 6.0 * sd).min())
    hi = float((mu_mean + 6.0 * sd).max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = -1.0, 1.0
    grid = np.linspace(lo, hi, points)
    z = (grid[None, :] - mu_mean[:, None]) / sd[:, None]
    density = (tau_weights[:, None] * np.exp(-0.5 * z**2) / (sd[:, None] * np.sqrt(2.0 * np.pi))).sum(axis=0)
    return grid, density


def fit_hierarchical(
    effects: Sequence[ScenarioEffect],
    rope: float = DEFAULT_ROPE,
    prior: str = "half_cauchy",
    tau_grid_points: int = _TAU_GRID_POINTS,
) -> Optional[HierarchicalPosterior]:
    """Fit the random-effects model by quadrature and summarize it, or return ``None`` below two scenarios.

    Two scenarios is the floor because ``tau`` is not identified from one: with a single effect the model
    has nothing to disagree with, and any reported heterogeneity would be the prior read back.
    """
    if len(effects) < 2:
        return None

    tau_grid, tau_weights, mu_mean, mu_var = _mu_mixture(effects, prior=prior, tau_grid_points=tau_grid_points)
    grid, density = _mu_grid_density(tau_weights, mu_mean, mu_var, _MU_GRID_POINTS)

    norm = _integrate(density, grid)
    pdf = density / norm if norm > 0 else density
    mean = _integrate(grid * pdf, grid)
    second = _integrate((grid**2) * pdf, grid)
    sd = float(np.sqrt(max(second - mean**2, 0.0)))

    positive = grid > 0.0
    p_positive = _integrate(pdf[positive], grid[positive]) if positive.any() else 0.0
    inside = np.abs(grid) < rope
    p_rope = _integrate(pdf[inside], grid[inside]) if inside.any() else 0.0

    tau_cdf_pts = _hdi_from_grid(tau_grid, tau_weights)
    tau_mean = float((tau_grid * tau_weights).sum())
    tau_cum = np.cumsum(tau_weights)
    tau_median = float(np.interp(0.5, tau_cum, tau_grid))

    return HierarchicalPosterior(
        n_scenarios=len(effects),
        mu_mean=mean,
        mu_sd=sd,
        mu_hdi=_hdi_from_grid(grid, pdf),
        p_positive=p_positive,
        p_in_rope=p_rope,
        rope=float(rope),
        tau_mean=tau_mean,
        tau_median=tau_median,
        tau_hdi=tau_cdf_pts,
        prior=prior,
    )


def rope_curve(
    effects: Sequence[ScenarioEffect],
    radii: Sequence[float],
    prior: str = "half_cauchy",
    tau_grid_points: int = _TAU_GRID_POINTS,
) -> List[Tuple[float, float]]:
    """Return ``[(r, P(|mu| < r))]``: the posterior CDF of the absolute pooled effect.

    This is the answer to "how wide should the ROPE be" that does not require agreeing on a number. The
    curve costs nothing beyond the fit that produced it, and a reader who rejects the pre-registered radius
    reads their own off the same posterior.
    """
    if len(effects) < 2:
        return [(float(r), float("nan")) for r in radii]
    tau_grid, tau_weights, mu_mean, mu_var = _mu_mixture(effects, prior=prior, tau_grid_points=tau_grid_points)
    grid, density = _mu_grid_density(tau_weights, mu_mean, mu_var, _MU_GRID_POINTS)
    norm = _integrate(density, grid)
    pdf = density / norm if norm > 0 else density

    out: List[Tuple[float, float]] = []
    for radius in radii:
        inside = np.abs(grid) < float(radius)
        mass = _integrate(pdf[inside], grid[inside]) if inside.any() else 0.0
        out.append((float(radius), mass))
    return out


def prior_sensitivity(effects: Sequence[ScenarioEffect], rope: float = DEFAULT_ROPE) -> Dict[str, Optional[HierarchicalPosterior]]:
    """Refit under every pre-registered ``tau`` prior, so the dependence on that choice is visible."""
    return {kind: fit_hierarchical(effects, rope=rope, prior=kind) for kind in PRIOR_KINDS}


def hierarchical_report(
    records: Sequence[Dict[str, Any]],
    model: str,
    k_label: str,
    arms: Sequence[str] = (),
    rope: float = DEFAULT_ROPE,
    radii: Sequence[float] = (0.002, 0.005, 0.01, 0.02, 0.05, 0.10),
) -> str:
    """Return a markdown block: pooled effect, heterogeneity, ROPE mass and the ROPE curve, per arm."""
    rows = average_over_cv_seed(extract_skill_rows(records, model=model, k_label=k_label))
    present = sorted({str(r["arm"]) for r in rows if str(r["arm"]) != NULL_ARM})
    wanted = [a for a in (arms or present) if a in present]

    lines = [
        f"## Pooled advantage over `{NULL_ARM}` -- {model} @ {k_label}",
        "",
        "Normalized skill, so a fixed ROPE means the same thing at every base rate. `mu` is the pooled",
        f"advantage, `tau` the between-scenario spread. ROPE = {rope:g} of achievable skill.",
        "",
        "| arm | scenarios | mu | sd | 95% HDI | P(mu>0) | P(in ROPE) | tau (median) | tau 95% HDI |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    fits: Dict[str, HierarchicalPosterior] = {}
    for arm in wanted:
        effects = scenario_effects(rows, arm=arm)
        fit = fit_hierarchical(effects, rope=rope)
        if fit is None:
            lines.append(f"| `{arm}` | {len(effects)} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        fits[arm] = fit
        lines.append(
            f"| `{arm}` | {fit.n_scenarios} | {fit.mu_mean:+.4f} | {fit.mu_sd:.4f} | "
            f"[{fit.mu_hdi[0]:+.4f}, {fit.mu_hdi[1]:+.4f}] | {fit.p_positive:.3f} | {fit.p_in_rope:.3f} | "
            f"{fit.tau_median:.4f} | [{fit.tau_hdi[0]:.4f}, {fit.tau_hdi[1]:.4f}] |"
        )

    if fits:
        lines += ["", "### P(|mu| < r): read your own ROPE off this curve", "", "| arm | " + " | ".join(f"r={r:g}" for r in radii) + " |", "|---|" + "---|" * len(radii)]
        for arm in fits:
            curve = rope_curve(scenario_effects(rows, arm=arm), radii=radii)
            lines.append(f"| `{arm}` | " + " | ".join(f"{mass:.3f}" for _, mass in curve) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Read a results JSONL and print the hierarchical report for one (model, K)."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Hierarchical posterior over the pooled advantage")
    parser.add_argument("results", help="path to a results JSONL")
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--k-label", default="k10")
    parser.add_argument("--rope", type=float, default=DEFAULT_ROPE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(args.results, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    print(hierarchical_report(records, model=args.model, k_label=args.k_label, rope=args.rope))


if __name__ == "__main__":
    main()
