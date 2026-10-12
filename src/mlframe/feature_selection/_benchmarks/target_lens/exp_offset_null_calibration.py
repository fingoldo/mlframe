"""Null calibration of the mix-offset held-out gate: ``y`` is independent of ``X``, so the gate should accept the offset no more often than its level.

Each draw builds a frame of independent columns and a target that is pure noise, runs the same ``TargetOffsetRegressor`` decision
(selector with the mix on, HGB gate), and counts how often an offset is accepted. The false-accept rate is reported with its
binomial standard error against ``offset_alpha``.

Usage: ``python -m mlframe.feature_selection._benchmarks.target_lens.exp_offset_null_calibration [draws] [n_train]``.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.simplefilter("ignore")


def one(seed: int, n_train: int) -> bool:
    """True when the gate accepts an offset on a pure-noise target."""
    from mlframe.feature_selection.filters._target_offset_regressor import TargetOffsetRegressor
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(5000 + seed)
    n = n_train
    X = pd.DataFrame({f"c{j}": rng.standard_normal(n) for j in range(4)})
    y = rng.standard_normal(n)
    w = TargetOffsetRegressor(
        selector=MRMR(verbose=0, n_jobs=1, random_seed=seed, fe_target_mix_enable=True, fe_target_mix_max_bases=1, fe_max_steps=1),
        regressor=HistGradientBoostingRegressor(max_iter=100, random_state=0),
    ).fit(X, y)
    return w.offset_base_ is not None


def main() -> None:
    """Run the draws and print the false-accept rate against the level."""
    from scipy.stats import norm

    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    from mlframe.feature_selection.filters._target_offset_regressor import TargetOffsetRegressor

    alpha = float(TargetOffsetRegressor(selector=None, regressor=None).offset_alpha)
    accepted = sum(one(s, n_train) for s in range(draws))
    p = accepted / draws
    se = float(np.sqrt(alpha * (1 - alpha) / draws))
    print(f"draws {draws}, n_train {n_train}: offset accepted on pure noise {accepted} times ({p:.3f}); level {alpha}; binomial se {se:.3f}; z vs level {(p - alpha) / se:+.2f}")
    print(f"upper 95% bound on the accept rate: {p + norm.isf(0.05) * np.sqrt(max(p * (1 - p), 1e-12) / draws):.3f}")


if __name__ == "__main__":
    main()
