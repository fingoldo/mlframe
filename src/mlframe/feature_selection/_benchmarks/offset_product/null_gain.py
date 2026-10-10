"""Held-out gain of the best offset-product candidate on targets with NO shifted-product structure (pure noise and a ratio target), scaled by n.

The acceptance margin ``ACCEPT_MARGIN_C / n`` of ``_offset_product_fe`` must sit above the largest value printed here. Run: ``python -m mlframe.feature_selection._benchmarks.offset_product.null_gain``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _offset_product_fe as m

SIZES = (5000, 20000, 100000, 300000)
SEEDS = 6
NO_MARGIN = 1e9


def _best_gain(X: pd.DataFrame, y: np.ndarray) -> float:
    """Largest held-out gain over the column pairs (every pair is rejected under ``NO_MARGIN`` and reported to the sink)."""
    got = []
    m.hybrid_offset_product_fe(X, y, top_k=50, reject_sink=lambda **k: got.append(k["observed"]))
    return max(got)


def main() -> None:
    """Print, per size, the per-seed maximum null gain times n for a noise target and for a ratio target."""
    m.ACCEPT_MARGIN_C = NO_MARGIN
    for n in SIZES:
        noise, ratio = [], []
        for seed in range(SEEDS):
            r = np.random.default_rng(seed)
            X = pd.DataFrame({k: r.random(n) for k in "abcdef"})
            noise.append(_best_gain(X, r.random(n)))
            a, b, f = X["a"].to_numpy(), X["b"].to_numpy(), X["f"].to_numpy()
            ratio.append(_best_gain(X, 0.2 * a**2 / b + f / 5.0 + 0.1 * r.standard_normal(n)))
        print(n, "noise gain*n:", np.round(np.array(noise) * n, 1), "ratio-target gain*n:", np.round(np.array(ratio) * n, 1), flush=True)


if __name__ == "__main__":
    main()
