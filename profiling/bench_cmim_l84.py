"""Layer 84 profiling: CMIM hotspot identification on a realistic fixture.

Fixture: n=2500, ~10 raw + 20 engineered candidates (so p_eng=20 against
the raw_X redundancy reference). Profiles a full call to
``score_features_by_cmim`` end-to-end and prints the cProfile tottime
ranking truncated to functions inside the mlframe / numpy.
"""
from __future__ import annotations

import cProfile
import pstats
import io
import time

import numpy as np
import pandas as pd


def _build_fixture(n: int = 2500, seed: int = 0):
    """Realistic ~10 raw / ~20 engineered candidate fixture."""
    rng = np.random.default_rng(seed)
    raw_cols = {}
    for k in range(10):
        raw_cols[f"x{k}"] = rng.standard_normal(n)
    X_raw = pd.DataFrame(raw_cols)
    # build "engineered" candidates as He_2(x_k), He_3(x_k) so the eng
    # pool is ~20 columns (10 sources * 2 degrees).
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        generate_univariate_basis_features,
    )
    engineered = generate_univariate_basis_features(
        X_raw, degrees=(2, 3), basis="hermite",
    )
    # 30+ engineered cols -> trim to 20 to match the brief.
    if engineered.shape[1] > 20:
        engineered = engineered.iloc[:, :20]
    signal = (
        0.8 * (X_raw["x0"] ** 2)
        + 0.6 * (X_raw["x1"] ** 2)
        + 0.4 * (X_raw["x2"] ** 2)
    )
    thr = float(np.median(signal))
    y = ((signal + 0.05 * rng.standard_normal(n)) > thr).astype(int).to_numpy()
    return X_raw, engineered, y


def main():
    from mlframe.feature_selection.filters._orthogonal_cmim_fe import (
        score_features_by_cmim,
    )
    raw_X, engineered, y = _build_fixture(n=2500, seed=0)
    print(f"raw_X: {raw_X.shape}, engineered: {engineered.shape}, y: {y.shape}")

    # warm-up
    _ = score_features_by_cmim(raw_X, engineered, y, n_bins=10)

    # timing
    t0 = time.perf_counter()
    for _ in range(5):
        _ = score_features_by_cmim(raw_X, engineered, y, n_bins=10)
    t1 = time.perf_counter()
    mean_s = (t1 - t0) / 5
    print(f"Mean over 5 runs: {mean_s * 1000:.1f} ms")

    # cProfile
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        _ = score_features_by_cmim(raw_X, engineered, y, n_bins=10)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(40)
    print(s.getvalue())


if __name__ == "__main__":
    main()
