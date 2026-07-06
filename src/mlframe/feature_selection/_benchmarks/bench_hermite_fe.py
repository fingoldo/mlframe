"""Benchmark improved Hermite-polynomial pair Feature Engineering on
synthetic targets where Hermite SHOULD win over identity.

Three synthetic regimes (each n=2000):

* ``xor``       -- ``y = sign(x1 * x2)`` (bilinear non-linearity, 2D-XOR)
* ``circle``    -- ``y = sign(x1**2 + x2**2 - r**2)`` (radial decision)
* ``saddle``    -- ``y = sign(x1**2 - x2**2)`` (saddle surface)

For each: compute MI of (x1, x2) joint with y (KSG baseline), then run
the legacy random-degree implementation, then the improved
``optimise_hermite_pair``. Compare uplift over baseline.

Run::

    python -m mlframe.feature_selection._benchmarks.bench_hermite_fe
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.feature_selection import mutual_info_classif


def _make_xor(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (np.sign(x1 * x2) > 0).astype(np.int64)
    return x1, x2, y


def _make_circle(n=2000, seed=42, r=1.0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = ((x1**2 + x2**2) > r**2).astype(np.int64)
    return x1, x2, y


def _make_saddle(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (np.sign(x1**2 - x2**2) > 0).astype(np.int64)
    return x1, x2, y


def _make_polynomial(n=2000, seed=42):
    """Smooth polynomial target -- explicit Hermite-friendly case."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    score = x1**2 - 2 * x1 * x2 + 0.5 * x2**3
    y = (score > np.median(score)).astype(np.int64)
    return x1, x2, y


# ---------------------------------------------------------------------------
# Realistic regimes -- mostly-linear targets with monotonic noise. These
# are what production tabular data looks like; XOR-like patterns are rare.
# ---------------------------------------------------------------------------


def _make_linear_dominant(n=2000, seed=42):
    """Pure linear: y = sign(0.7*x1 + 0.3*x2 + noise). Identity baseline
    should be near-optimal; Hermite should NOT recommend an engineered
    feature (the strict baseline check should return None)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    score = 0.7 * x1 + 0.3 * x2 + 0.3 * rng.normal(size=n)
    y = (score > np.median(score)).astype(np.int64)
    return x1, x2, y


def _make_monotonic(n=2000, seed=42):
    """Smooth monotonic: y depends on log(1+x1**2) - x2 + noise. Hermite
    can in principle approximate the log; comparable to identity on
    rank-correlated KSG."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    score = np.log(1 + x1**2) - x2 + 0.3 * rng.normal(size=n)
    y = (score > np.median(score)).astype(np.int64)
    return x1, x2, y


def _make_threshold(n=2000, seed=42):
    """Hard threshold: y = (x1 > x2). Identity already reveals the
    decision boundary; Hermite uplift small."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 > x2).astype(np.int64)
    return x1, x2, y


def _make_small_synergy(n=2000, seed=42):
    """Mostly linear with small bilinear synergy:
    y = sign(0.7*x1 + 0.3*x2 + 0.4*x1*x2 + noise). Realistic FE
    candidate; Hermite should give modest uplift over identity."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    score = 0.7 * x1 + 0.3 * x2 + 0.4 * x1 * x2 + 0.3 * rng.normal(size=n)
    y = (score > np.median(score)).astype(np.int64)
    return x1, x2, y


def _make_complex_polynomial(n=2000, seed=42):
    """High-degree target tanh(x1) * sin(x2*pi) needs degree >= 6 to
    approximate via Hermite expansion."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    score = np.tanh(x1) * np.sin(x2 * np.pi / 2)
    y = (score > 0).astype(np.int64)
    return x1, x2, y


def _make_uniform_xor(n=2000, seed=42):
    """Same XOR but inputs drawn from Uniform[-1, 1]. Matches Legendre /
    Chebyshev's natural domain; should highlight basis-mismatch effects."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, size=n)
    x2 = rng.uniform(-1, 1, size=n)
    y = (np.sign(x1 * x2) > 0).astype(np.int64)
    return x1, x2, y


def _make_california_pair(n=None, seed=42):
    """Realistic regression-derived target from California housing.
    Take MedInc + AveOccup as the pair; binary y = (MedHouseVal >
    median). Both inputs heavily skewed (incomes / occupancies);
    expect Laguerre to do well on raw, Hermite on z-scored."""
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    X, y_cont = data.data, data.target
    if n is not None and n < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=n, replace=False)
        X = X[idx]
        y_cont = y_cont[idx]
    x1 = X[:, 0]  # MedInc
    x2 = X[:, 5]  # AveOccup
    y = (y_cont > np.median(y_cont)).astype(np.int64)
    return x1, x2, y


def _make_diabetes_pair(n=None, seed=42):
    """Diabetes regression dataset, BMI + s5 (lamotrigine) as the pair.
    Smaller sample (n=442 default) -- exercises the small-N branch of
    the n_neighbors auto-pick."""
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    X, y_cont = data.data, data.target
    if n is not None and n < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=n, replace=False)
        X = X[idx]
        y_cont = y_cont[idx]
    x1 = X[:, 2]  # BMI
    x2 = X[:, 8]  # s5
    y = (y_cont > np.median(y_cont)).astype(np.int64)
    return x1, x2, y


_REGIMES = {
    # Synthetic Hermite-friendly:
    "xor": _make_xor,
    "circle": _make_circle,
    "saddle": _make_saddle,
    "polynomial": _make_polynomial,
    # Realistic (mostly linear + monotonic):
    "linear_dominant": _make_linear_dominant,
    "monotonic": _make_monotonic,
    "threshold": _make_threshold,
    "small_synergy": _make_small_synergy,
    # High-degree:
    "complex_polynomial": _make_complex_polynomial,
    # Different distributions / real datasets:
    "uniform_xor": _make_uniform_xor,
    "california_housing": _make_california_pair,
    "diabetes": _make_diabetes_pair,
}


def _ksg_baseline_pair(x1, x2, y) -> float:
    """KSG MI of (x1, x2) joint with discrete y."""
    Xn = np.column_stack([x1, x2])
    return float(mutual_info_classif(Xn, y, n_neighbors=3, random_state=42, discrete_features=False).max())


def _legacy_hermite(x1, x2, y, n_iters=2, n_trials_per_iter=100):
    """Legacy implementation reproduction: random length per trial,
    coef_range [-10, 10], physicist's hermval, no standardisation,
    no regularisation."""
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        return None
    from numpy.polynomial.hermite import hermval

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bin_funcs = {"add": np.add, "sub": np.subtract, "mul": np.multiply}

    best_mi = 0.0
    for _ in range(n_iters):
        length_a = np.random.randint(3, 8)
        length_b = np.random.randint(3, 8)

        def objective(trial):
            coef_a = np.array(
                [trial.suggest_float(f"a_{i}", -10, 10) for i in range(length_a)],
                dtype=np.float64,
            )
            coef_b = np.array(
                [trial.suggest_float(f"b_{i}", -10, 10) for i in range(length_b)],
                dtype=np.float64,
            )
            h_a = hermval(x1, coef_a)
            h_b = hermval(x2, coef_b)
            if not (np.all(np.isfinite(h_a)) and np.all(np.isfinite(h_b))):
                return -np.inf
            local_best = -np.inf
            for bf in bin_funcs.values():
                try:
                    combined = bf(h_a, h_b)
                except Exception:  # nosec B112 - best-effort path
                    continue
                if not np.all(np.isfinite(combined)):
                    continue
                mi = float(mutual_info_classif(
                    combined.reshape(-1, 1), y, n_neighbors=3, random_state=42,
                    discrete_features=False,
                )[0])
                if mi > local_best:
                    local_best = mi
            return local_best

        sampler = TPESampler(multivariate=True, seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials_per_iter, show_progress_bar=False)
        if study.best_value > best_mi:
            best_mi = study.best_value

    return best_mi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--n-trials", type=int, default=200, help="trials for improved Hermite (per degree)")
    parser.add_argument("--regime", default="all", help="comma-separated subset of: " + ",".join(_REGIMES))
    args = parser.parse_args()

    regimes = list(_REGIMES) if args.regime == "all" else args.regime.split(",")

    from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair

    print(f"\n=== Hermite FE bench, n={args.n}, n_trials/degree={args.n_trials} ===\n")
    print(f"  {'regime':12s}  {'baseline':>10s}  {'legacy':>10s}  {'improved':>10s}  {'uplift':>8s}  {'time-imp':>8s}")
    print("  " + "-" * 78)

    for regime_name in regimes:
        if regime_name not in _REGIMES:
            print(f"  unknown regime {regime_name!r}, skipping")
            continue
        x1, x2, y = _REGIMES[regime_name](n=args.n)
        baseline = _ksg_baseline_pair(x1, x2, y)

        t0 = time.perf_counter()
        legacy = _legacy_hermite(x1, x2, y, n_iters=2, n_trials_per_iter=100)
        t_legacy = time.perf_counter() - t0

        t0 = time.perf_counter()
        # Pin basis="hermite" -- this bench specifically compares the
        # legacy Hermite implementation to the improved one; switching
        # to the package default ("chebyshev") would make the
        # comparison apples-to-oranges. For cross-basis comparison,
        # use ``bench_polynomial_bases``.
        res = optimise_hermite_pair(
            x1, x2, y,
            discrete_target=True,
            max_degree=4,
            n_trials=args.n_trials,
            seed=42,
            basis="hermite",
        )
        t_imp = time.perf_counter() - t0

        improved_mi = res.mi if res is not None else 0.0
        uplift = improved_mi / max(baseline, 1e-12)
        print(f"  {regime_name:12s}  {baseline:10.4f}  {legacy or 0:10.4f}  " f"{improved_mi:10.4f}  {uplift:7.2f}x  {t_imp:7.2f}s")
        if res is not None:
            print(
                f"    improved best: degree={res.degree_a}, bf={res.bin_func_name}, "
                f"|c_a|_2={np.linalg.norm(res.coef_a):.2f}, |c_b|_2={np.linalg.norm(res.coef_b):.2f}"
            )


if __name__ == "__main__":
    main()
