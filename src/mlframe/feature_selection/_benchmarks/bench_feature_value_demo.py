"""Synthetic demonstration of each new pair-FE feature's business value.

For every feature added in Phases A-D, this bench includes a target
where the feature is expected to clearly beat the polynomial-only
baseline. Real numbers, real wins, no marketing.

Targets (each n=2000, binary classification, MI in nats):
* ``periodic_a``        -- ``y = sign(sin(2*pi*x_a))`` (single-feature
                            periodic) -> **Fourier** should win
* ``threshold``         -- ``y = (x_a > 0.5) AND (x_b > -0.3)``
                            (sharp 2D step) -> **Sigmoid / RBF** should win
* ``rbf_bump``          -- ``y = sign(exp(-(x_a-1)**2) - 0.5)``
                            (local Gaussian bump) -> **RBF** should win
* ``ratio_pole``        -- ``y = sign(x_a / (x_b + 0.5) - 1)`` -> **Pade
                            or div-bf** should win
* ``multiplicative``    -- ``y = sign(x_a * x_b)`` (XOR) -> **trivial mul**
                            beats all polynomials
* ``radial``            -- ``y = sign(x_a**2 + x_b**2 - 1)`` ->
                            **trivial sum_sq** beats all polynomials
* ``angular``           -- ``y = sign(arctan2(x_a, x_b) - 0.5)`` ->
                            **atan2 bin-func** wins
* ``log_multiplicative`` -- ``y = sign(log(|x_a|) + log(|x_b|) - c)``
                            -> **logabs bin-func** wins
* ``poly_3_terms``      -- ``y = sign(0.7*x_a**2 - 0.5*x_b**2 + 0.3*x_a*x_b)``
                            -> **Hermite / Chebyshev** wins
* ``triplet_xor``       -- ``y = sign(x_a*x_b*x_c)`` (3-way XOR) ->
                            **triplet abc_mul** wins (pair-FE cannot)
* ``log_separable``     -- ``y = sign(log|x_a| + log|x_b|)`` (after
                            auto_unary log_abs, this becomes linear) ->
                            **auto unary pre-transform** wins

For each target we compute:
  1. Best polynomial single-basis (Hermite/Chebyshev/Laguerre/Legendre)
     using CMA-ES + warm-start
  2. Best non-polynomial basis (Fourier/RBF/Sigmoid/Pade)
  3. Best trivial pair baseline (mul, ratio, sum_sq, atan2, logabs, ...)
  4. Best triplet baseline (when applicable)
  5. Best unary-pre-transformed pair (when applicable)

Run::

    python -m mlframe.feature_selection._benchmarks.bench_feature_value_demo
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

# Wave 87 (2026-05-21): module-level filter mutation removed; bench scripts
# now gate via __main__ to avoid poisoning the process-global filter on import.

from mlframe.feature_selection.filters.hermite_fe import (
    optimise_hermite_pair,
    basis_route_by_moments,
    detect_pair_symmetry,
)
from mlframe.feature_selection.filters.fe_baselines import (
    best_trivial_pair,
    score_triplet_baselines,
    auto_unary_transforms,
    best_unary_transform,
    _mi_1d,
)


def _make_periodic_a(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    x_a = rng.uniform(-1, 1, n)
    x_b = rng.normal(size=n)
    y = (np.sin(2 * np.pi * x_a) > 0).astype(np.int64)
    return x_a, x_b, y


def _make_threshold(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    x_a = rng.normal(size=n)
    x_b = rng.normal(size=n)
    y = ((x_a > 0.5) & (x_b > -0.3)).astype(np.int64)
    return x_a, x_b, y


def _make_rbf_bump(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    x_a = rng.normal(loc=1.0, scale=1.5, size=n)
    x_b = rng.normal(size=n)
    y = (np.exp(-((x_a - 1.0) ** 2)) > 0.5).astype(np.int64)
    return x_a, x_b, y


def _make_ratio_pole(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    x_a = rng.uniform(-2, 2, n)
    x_b = rng.uniform(-2, 2, n)
    y = ((x_a / (x_b + 0.5 + np.sign(x_b) * 1e-9)) > 1).astype(np.int64)
    return x_a, x_b, y


def _make_multiplicative(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    x_a = rng.normal(size=n)
    x_b = rng.normal(size=n)
    y = (np.sign(x_a * x_b) > 0).astype(np.int64)
    return x_a, x_b, y


def _make_radial(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    x_a = rng.normal(size=n)
    x_b = rng.normal(size=n)
    y = (x_a**2 + x_b**2 > 1.0).astype(np.int64)
    return x_a, x_b, y


def _make_angular(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    x_a = rng.normal(size=n)
    x_b = rng.normal(size=n)
    y = (np.arctan2(x_a, x_b) > 0.5).astype(np.int64)
    return x_a, x_b, y


def _make_log_mult(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    x_a = rng.lognormal(size=n)
    x_b = rng.lognormal(size=n)
    score = np.log(x_a) + np.log(x_b)
    y = (score > np.median(score)).astype(np.int64)
    return x_a, x_b, y


def _make_poly_3(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    x_a = rng.normal(size=n)
    x_b = rng.normal(size=n)
    score = 0.7 * x_a**2 - 0.5 * x_b**2 + 0.3 * x_a * x_b
    y = (score > np.median(score)).astype(np.int64)
    return x_a, x_b, y


def _make_log_separable(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    x_a = rng.lognormal(size=n) - rng.lognormal(size=n)  # signed lognormal-like
    x_b = rng.lognormal(size=n) - rng.lognormal(size=n)
    score = np.log(np.abs(x_a) + 1e-9) + np.log(np.abs(x_b) + 1e-9)
    y = (score > np.median(score)).astype(np.int64)
    return x_a, x_b, y


def _make_triplet(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    x_a = rng.normal(size=n)
    x_b = rng.normal(size=n)
    x_c = rng.normal(size=n)
    y = (np.sign(x_a * x_b * x_c) > 0).astype(np.int64)
    return x_a, x_b, x_c, y


PAIR_SCENARIOS = {
    "periodic_a (sin(2pi x_a))": _make_periodic_a,
    "threshold ((x_a>.5)&(x_b>-.3))": _make_threshold,
    "rbf_bump (exp(-(x_a-1)^2)>.5)": _make_rbf_bump,
    "ratio_pole (x_a/(x_b+.5)>1)": _make_ratio_pole,
    "multiplicative (XOR)": _make_multiplicative,
    "radial (x_a^2+x_b^2>1)": _make_radial,
    "angular (atan2>.5)": _make_angular,
    "log_mult (log(x_a)+log(x_b))": _make_log_mult,
    "poly_3 (0.7a^2-0.5b^2+0.3ab)": _make_poly_3,
    "log_separable": _make_log_separable,
}


def _run_polynomial_panel(x_a, x_b, y, *, n_trials=30):
    """Score each of the 8 bases on the same target. Returns
    ``{basis: mi}``."""
    scores = {}
    for basis in ["hermite", "legendre", "chebyshev", "laguerre", "fourier", "rbf", "sigmoid", "pade"]:
        try:
            res = optimise_hermite_pair(
                x_a, x_b, y,
                discrete_target=True,
                n_trials=n_trials, max_degree=3, basis=basis,
                baseline_uplift_threshold=0.0,
                use_trivial_baseline=False,  # we want raw poly mi
                optimizer="cma",
                warm_start=True,
            )
            scores[basis] = (res.mi if res else 0.0, res.bin_func_name if res else "-")
        except Exception:
            scores[basis] = (0.0, "ERR")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    print(f"\n=== Pair-FE feature-value demo, n={args.n}, n_trials={args.n_trials} ===\n")

    print(f"  {'scenario':>32s}  {'best_basis':>12s}  {'mi':>6s}  {'best_trivial':>12s}  {'mi':>6s}  {'verdict':>20s}")
    print("  " + "-" * 110)

    pair_summary = []
    for sc_name, mk in PAIR_SCENARIOS.items():
        x_a, x_b, y = mk(n=args.n)
        # 1. All bases via CMA + warm-start.
        basis_scores = _run_polynomial_panel(x_a, x_b, y, n_trials=args.n_trials)
        best_basis, (best_basis_mi, best_basis_bf) = max(basis_scores.items(), key=lambda kv: kv[1][0])
        # 2. Best trivial pair feature.
        trivial = best_trivial_pair(x_a, x_b, y, discrete_target=True)
        if trivial is not None:
            triv_name, _, triv_mi = trivial
        else:
            triv_name, triv_mi = "-", 0.0
        # Verdict.
        if triv_mi > best_basis_mi * 1.05:
            verdict = f"trivial wins (+{(triv_mi/max(best_basis_mi,1e-9)-1)*100:.0f}%)"
        elif best_basis_mi > triv_mi * 1.05:
            verdict = f"basis wins (+{(best_basis_mi/max(triv_mi,1e-9)-1)*100:.0f}%)"
        else:
            verdict = "tie"
        bf_str = f"({best_basis_bf})"
        print(f"  {sc_name:>32s}  {best_basis:>10s}{bf_str:>3s}  {best_basis_mi:6.3f}  "
              f"{triv_name:>12s}  {triv_mi:6.3f}  {verdict:>20s}")
        pair_summary.append({
            "scenario": sc_name,
            "best_basis": best_basis, "best_basis_mi": best_basis_mi,
            "best_basis_bf": best_basis_bf,
            "trivial_name": triv_name, "trivial_mi": triv_mi,
            "verdict": verdict,
        })

    # Auto unary pre-transform demo (log-separable scenario)
    print("\n  --- Phase B3: auto unary pre-transforms ---")
    print("  Scenario: log_separable (y = sign(log|x_a| + log|x_b| - median))")
    x_a, x_b, y = _make_log_separable(n=args.n)
    name_a, x_a_t, mi_a = best_unary_transform(x_a, y, discrete_target=True)
    name_b, x_b_t, mi_b = best_unary_transform(x_b, y, discrete_target=True)
    print(f"    best unary x_a: {name_a:>15s}  mi={mi_a:.4f}  (identity mi={_mi_1d(x_a, y, discrete_target=True):.4f})")
    print(f"    best unary x_b: {name_b:>15s}  mi={mi_b:.4f}  (identity mi={_mi_1d(x_b, y, discrete_target=True):.4f})")
    # After unary transform, run pair-FE.
    res_post = optimise_hermite_pair(
        x_a_t, x_b_t, y, n_trials=args.n_trials, max_degree=3, basis="chebyshev", baseline_uplift_threshold=0.0, use_trivial_baseline=False
    )
    res_pre = optimise_hermite_pair(
        x_a, x_b, y, n_trials=args.n_trials, max_degree=3, basis="chebyshev", baseline_uplift_threshold=0.0, use_trivial_baseline=False
    )
    print(f"    pair-FE WITHOUT unary: mi={res_pre.mi:.4f}")
    print(f"    pair-FE WITH unary:    mi={res_post.mi:.4f}  " f"(uplift x{res_post.mi/max(res_pre.mi, 1e-9):.2f})")

    # Triplet demo (Phase D1)
    print("\n  --- Phase D1: triplet (3-way) interactions ---")
    print("  Scenario: triplet_xor (y = sign(x_a * x_b * x_c))")
    x_a, x_b, x_c, y = _make_triplet(n=args.n)
    pair_best = best_trivial_pair(x_a, x_b, y, discrete_target=True)
    pair_mi = pair_best[2] if pair_best else 0.0
    triplet_scores = score_triplet_baselines(x_a, x_b, x_c, y, discrete_target=True)
    triplet_top = list(triplet_scores.items())[0]
    print(f"    best PAIR trivial:    {pair_best[0] if pair_best else '-':>12s}  mi={pair_mi:.4f}")
    print(f"    best TRIPLET trivial: {triplet_top[0]:>12s}  mi={triplet_top[1]:.4f}  " f"(uplift x{triplet_top[1]/max(pair_mi, 1e-9):.2f})")
    # Show top-3
    print(f"    top-3 triplets:")
    for k, v in list(triplet_scores.items())[:3]:
        print(f"      {k:>15s}  {v:.4f}")

    # CMA-ES vs Optuna timing
    print("\n  --- Phase A1: CMA-ES vs Optuna speedup ---")
    x_a, x_b, y = _make_multiplicative(n=args.n)
    for opt in ["optuna", "cma"]:
        t0 = time.perf_counter()
        for basis in ["hermite", "chebyshev"]:
            optimise_hermite_pair(x_a, x_b, y, n_trials=40, max_degree=4, basis=basis, baseline_uplift_threshold=0.0, use_trivial_baseline=False, optimizer=opt)
        dt = time.perf_counter() - t0
        print(f"    {opt:>10s} 2 bases x degree 4: {dt:6.2f}s")

    # Basis routing demo (Phase B1)
    print("\n  --- Phase B1: basis routing by moment fingerprint ---")
    rng = np.random.default_rng(0)
    examples = [
        ("standard normal", rng.normal(size=2000)),
        ("uniform [-1,1]", rng.uniform(-1, 1, 2000)),
        ("lognormal", rng.lognormal(size=2000)),
        ("exponential", rng.exponential(size=2000)),
        ("heavy-tail t", rng.standard_t(df=2, size=2000)),
    ]
    for name, x in examples:
        suggested = basis_route_by_moments(x)
        print(f"    {name:>20s}: skew={float(np.mean(((x-x.mean())/x.std())**3)):+.2f}  " f"-> {suggested}")

    # Symmetry detection demo (Phase B2)
    print("\n  --- Phase B2: pair symmetry detection ---")
    rng = np.random.default_rng(0)
    x_a = rng.normal(size=2000)
    x_b = rng.normal(size=2000)
    # Symmetric: y = sign(x_a * x_b)
    y_sym = (x_a * x_b > 0).astype(np.int64)
    # Asymmetric: y = sign(x_a - 2 x_b)
    y_asym = (x_a - 2 * x_b > 0).astype(np.int64)
    print(f"    y = sign(x_a*x_b)       symmetry={detect_pair_symmetry(x_a, x_b, y_sym):.3f}  (expect ~1.0)")
    print(f"    y = sign(x_a - 2*x_b)   symmetry={detect_pair_symmetry(x_a, x_b, y_asym):.3f}  (expect <1.0)")

    print("\n  === Summary ===")
    wins_basis = sum(1 for r in pair_summary if "basis wins" in r["verdict"])
    wins_trivial = sum(1 for r in pair_summary if "trivial wins" in r["verdict"])
    ties = sum(1 for r in pair_summary if r["verdict"] == "tie")
    print(f"    basis wins: {wins_basis}/{len(pair_summary)}")
    print(f"    trivial wins: {wins_trivial}/{len(pair_summary)}")
    print(f"    ties: {ties}/{len(pair_summary)}")
    print()
    print("    Key takeaway: a polynomial-only FE module would output the")
    print("    polynomial result on EVERY scenario; this would be SUBOPTIMAL on")
    print(f"    {wins_trivial} of {len(pair_summary)} cases here. The honest non-poly")
    print("    baseline + bin-function discovery + new bases capture more value.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
