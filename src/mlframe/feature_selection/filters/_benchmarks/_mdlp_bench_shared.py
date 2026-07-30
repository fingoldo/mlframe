"""Scenario generators + small eval helpers shared between ``bench_mdlp_validated_split_suite.py``
and ``bench_mdlp_robustness.py``. Extracted here (not left in either sibling) because the two
files need each other's names in both directions -- the robustness sweep needs ``SCENARIOS`` /
``_oos_mse`` / ``_split`` from the validated-split suite, and the suite's ``__main__`` block needs
the robustness sweep's runner functions -- which closed a two-module import cycle
(test_no_import_cycles) when both were top-level imports. A shared module both siblings import
from (rather than each other) breaks the cycle without changing either file's public API.
"""

from __future__ import annotations

import numpy as np

# -----------------------------------------------------------------------------
# Scenario generators. Each returns (x, y) as float64 1-D arrays, n rows.
# -----------------------------------------------------------------------------


def scen_pure_noise(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n), rng.standard_normal(n) * 1000.0


def scen_step_k_breakpoints(n: int, k: int, seed: int = 0):
    """Step function with exactly ``k`` true breakpoints in [-5, 5], noise sigma=2."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, n)
    cuts = np.linspace(-5, 5, k + 2)[1:-1]
    levels = rng.uniform(5, 40, k + 1)
    y = np.select([x < c for c in cuts] + [np.ones_like(x, dtype=bool)], [levels[i] for i in range(k)] + [levels[-1]])
    y = y + rng.standard_normal(n) * 2.0
    return x, y


def scen_non_monotonic_sine(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-10, 10, n)
    y = 20.0 * np.sin(x) + rng.standard_normal(n) * 3.0
    return x, y


def scen_multimodal_target(n: int, seed: int = 0):
    """y bimodal REGARDLESS of x (x carries no signal) -- a distractor multimodal target."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    mode = rng.integers(0, 2, n)
    y = np.where(mode == 0, rng.normal(-50, 5, n), rng.normal(50, 5, n))
    return x, y


def scen_interaction_only(n: int, seed: int = 0):
    """x1 alone carries ZERO marginal signal -- y depends on x1*x2 (XOR-family synergy). A valid
    per-column MDLP criterion should treat x1 like pure noise (only the JOINT with x2 predicts y,
    which is out of scope for a single-column binner)."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.choice([-1.0, 1.0], n)
    y = x1 * x2 * 10.0 + rng.standard_normal(n) * 1.0
    return x1, y  # only x1 passed to the binner -- x2 is the hidden confounder


def scen_lognormal_x(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.lognormal(0, 1.5, n)
    y = np.where(x < 2.0, 10.0, 30.0) + rng.standard_normal(n) * 2.0
    return x, y


def scen_cauchy_x(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_cauchy(n)
    y = np.where(x < 0.0, 10.0, 30.0) + rng.standard_normal(n) * 2.0
    return x, y


def scen_extreme_scale(n: int, seed: int = 0):
    """x spans 1e-3 to 1e6 -- numeric-stability probe."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(1e-3, 1e6, n)
    y = np.where(x < 3e5, 10.0, 30.0) + rng.standard_normal(n) * 2.0
    return x, y


def scen_with_nan(n: int, nan_frac: float, seed: int = 0):
    x, y = scen_step_k_breakpoints(n, k=2, seed=seed)
    rng = np.random.default_rng(seed + 1)
    mask = rng.random(n) < nan_frac
    x = x.copy()
    x[mask] = np.nan
    return x, y


SCENARIOS = {
    "pure_noise": lambda n, seed: scen_pure_noise(n, seed),
    "step_2bp": lambda n, seed: scen_step_k_breakpoints(n, 2, seed),
    "step_5bp": lambda n, seed: scen_step_k_breakpoints(n, 5, seed),
    "step_10bp": lambda n, seed: scen_step_k_breakpoints(n, 10, seed),
    "non_monotonic_sine": lambda n, seed: scen_non_monotonic_sine(n, seed),
    "multimodal_target": lambda n, seed: scen_multimodal_target(n, seed),
    "interaction_only": lambda n, seed: scen_interaction_only(n, seed),
    "lognormal_x": lambda n, seed: scen_lognormal_x(n, seed),
    "cauchy_x": lambda n, seed: scen_cauchy_x(n, seed),
    "extreme_scale_x": lambda n, seed: scen_extreme_scale(n, seed),
    "nan_1pct": lambda n, seed: scen_with_nan(n, 0.01, seed),
    "nan_10pct": lambda n, seed: scen_with_nan(n, 0.10, seed),
    "nan_30pct": lambda n, seed: scen_with_nan(n, 0.30, seed),
}


def _oos_mse(x_train, y_train, x_test, y_test, edges):
    inner = edges[1:-1] if edges.size >= 2 else edges
    inner = inner[np.isfinite(inner)]
    codes_train = np.searchsorted(inner, x_train, side="right")
    codes_test = np.searchsorted(inner, x_test, side="right")
    n_bins = int(inner.size) + 1
    means = np.full(n_bins, float(np.mean(y_train)) if y_train.size else 0.0)
    for b in range(n_bins):
        m = codes_train == b
        if m.any():
            means[b] = float(np.mean(y_train[m]))
    pred = means[np.clip(codes_test, 0, n_bins - 1)]
    valid = np.isfinite(pred) & np.isfinite(y_test)
    if not valid.any():
        return float("nan"), n_bins
    return float(np.mean((pred[valid] - y_test[valid]) ** 2)), n_bins


def _split(x, y, seed=0, test_frac=0.25):
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    return idx[n_test:], idx[:n_test]
