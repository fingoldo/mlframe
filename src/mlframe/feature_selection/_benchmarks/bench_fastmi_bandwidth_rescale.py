"""fastmi bias on a Gaussian pair (true MI = -0.5*log(1-rho^2)) with and without the n^(-1/6) MISE-bandwidth rescale."""
from __future__ import annotations

import math

import numpy as np

from mlframe.feature_selection.filters._fastmi import fastmi


def gaussian_mi(rho: float) -> float:
    """Closed-form MI (nats) of a bivariate Gaussian with correlation ``rho``."""
    return -0.5 * math.log(1.0 - rho * rho)


def mean_abs_rel_error(n: int, rhos=(0.3, 0.6, 0.9), seeds: int = 3, rescale: bool = True) -> float:
    """Mean over rho and seeds of |fastmi - true| / true."""
    errs = []
    for rho in rhos:
        for s in range(seeds):
            r = np.random.default_rng(100 * s + 7)
            x = r.standard_normal(n)
            y = rho * x + math.sqrt(1 - rho * rho) * r.standard_normal(n)
            est = fastmi(x, y, random_seed=s, mise_rescale=rescale)
            errs.append(abs(est - gaussian_mi(rho)) / gaussian_mi(rho))
    return float(np.mean(errs))


def main() -> None:
    """Print the relative error per n for both modes."""
    for n in (1_000, 10_000, 100_000, 1_000_000):
        old = mean_abs_rel_error(n, rescale=False)
        new = mean_abs_rel_error(n, rescale=True)
        print(f"n={n}: mean_rel_err legacy={old:.4f} rescaled={new:.4f}", flush=True)


if __name__ == "__main__":
    main()
