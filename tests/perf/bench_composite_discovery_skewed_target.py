"""iter69 profiling harness: CompositeTargetDiscovery on a SKEWED / heavy-tail
target so the y-transform fits (Yeo-Johnson Brent scan, quantile_normal sort,
signed_power skew-grid) get exercised heavily inside discovery + stability
bootstrap. Stays under a minute via n=4000 + a handful of bootstrap runs.
"""

from __future__ import annotations

import cProfile
import pstats
import sys
import time

import numpy as np
import pandas as pd


def build_df(n: int = 4000, seed: int = 11) -> pd.DataFrame:
    """Builds seeded synthetic test data; returns ``pd.DataFrame(cols)``."""
    rng = np.random.default_rng(seed)
    x_a = rng.normal(100.0, 20.0, n)
    x_b = rng.normal(50.0, 10.0, n)
    x_c = rng.normal(0.0, 1.0, n)
    x_d = rng.uniform(-3.0, 3.0, n)
    # Heavy-tailed, strongly right-skewed target -- the regime where YJ /
    # quantile_normal / signed_power transforms get selected + refit per fold.
    raw = 1.5 * x_a + 2.5 * x_b - 0.8 * (x_c**2) + np.sin(x_d) + rng.normal(0.0, 1.5, n)
    y = np.exp(raw / raw.std() * 1.3)
    cols = {
        "x_a": x_a,
        "x_b": x_b,
        "x_c": x_c,
        "x_d": x_d,
        "n0": rng.standard_normal(n),
        "n1": rng.standard_normal(n),
        "n2": rng.standard_normal(n),
        "n3": rng.standard_normal(n),
        "y": y,
    }
    return pd.DataFrame(cols)


def run_once(df: pd.DataFrame):
    """Fits ``CompositeTargetDiscovery.fit_with_stability_check`` on ``df`` and returns the fitted instance."""
    from mlframe.training.composite.discovery import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    n = len(df)
    feature_cols = [c for c in df.columns if c != "y"]
    cfg = CompositeTargetDiscoveryConfig(enabled=True, mi_sample_n=3000)
    disc = CompositeTargetDiscovery(config=cfg)
    disc.fit_with_stability_check(
        df=df,
        target_col="y",
        feature_cols=feature_cols,
        train_idx=np.arange(int(0.8 * n)),
        val_idx=np.arange(int(0.8 * n), n),
        n_bootstrap_runs=5,
        min_keep_fraction=0.5,
    )
    return disc


def main() -> None:
    """Runs one warm-up call, then profiles 3 further calls under ``--profile`` and prints the top-cumtime frames."""
    df = build_df()
    run_once(df)  # warm
    if "--profile" in sys.argv:
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(3):
            run_once(df)
        pr.disable()
        st = pstats.Stats(pr)
        st.sort_stats("tottime")
        st.print_stats(40)
    else:
        t = time.perf_counter()
        runs = 8
        for _ in range(runs):
            run_once(df)
        dt = time.perf_counter() - t
        print(f"e2e mean over {runs}: {dt / runs * 1000:.2f} ms")


if __name__ == "__main__":
    main()
