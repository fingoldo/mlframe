"""Columns and targets built for each of the eight bases: the basis expansion of the stated degree reproduces the target better than any other basis."""

import numpy as np

N = 6000
DEGREES = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def _case(basis: str):
    """``(x, y, expected degree)``: a column and a target that the ``basis`` expansion of that degree reproduces better than any other basis."""
    rng = np.random.default_rng(11)
    if basis == "hermite":
        x = rng.normal(size=N)
        return x, x**2 - 1.0, 2
    if basis == "legendre":
        x = rng.uniform(-1, 1, N)
        return x, (3 * x**2 - 1) / 2, 2
    if basis == "chebyshev":
        x = rng.uniform(-1, 1, N)
        return x, 4 * x**3 - 3 * x, 3
    if basis == "laguerre":
        x = rng.exponential(size=N)
        return x, (x**2 - 4 * x + 2) / 2, 2
    if basis == "fourier":
        x = rng.uniform(0, 1, N)
        return x, np.sin(2 * np.pi * x), 1
    if basis == "rbf":
        x = rng.normal(size=N)
        centre = np.quantile(x, 0.8)
        return x, (np.abs(x - centre) < 0.12).astype(float), 8
    if basis == "sigmoid":
        x = rng.uniform(0, 1, N)
        return x, (x > np.quantile(x, 0.9)).astype(float), 9
    if basis == "pade":
        x = rng.normal(size=N)
        z = (x - x.mean()) / x.std()
        return x, 1.0 / (1.0 + z**2), 2
    raise KeyError(basis)
