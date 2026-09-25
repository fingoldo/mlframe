"""Wall time of the tiny-model rerank: serial vs threads vs worker processes, on one discovery fit.

Run: ``python -m mlframe.training.composite._benchmarks.bench_rerank_backends [n_jobs] [sample_n]``

Processes are the default for a parallel rerank because the threaded one took a production kernel down three times out
of three with a heap corruption. This prices that choice: worker start-up and the memory-mapped hand-off of the screen
matrix against the per-spec LightGBM fits they wrap.

Verdict (4 physical cores, sample_n=20000, 12 specs x 1 family): serial 32.4 s, threads 20.1 s (with LightGBM's native
calls serialised, see ``mlframe._lightgbm_thread_safety``), processes 27.6 s on the first fit -- worker start-up and imports
-- and 14.5 s once the workers exist, which is every target after the first in a suite. Processes are the default: the
fastest mode once warm, and the one where a native fault cannot take the kernel down.

The same harness reproduced the production crash locally before the serialisation: the threaded run died with an access
violation in 3 of 9 fresh processes, inside the WAIC tie-break's LightGBM calls; 0 of 12 with it.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from mlframe.training.composite.discovery import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _data(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {f"f{i}": rng.normal(size=n) for i in range(20)}
    cols["base1"] = rng.normal(size=n)
    cols["base2"] = np.abs(rng.normal(size=n)) + 0.5
    y = 2.0 * cols["base1"] + cols["f0"] - 0.5 * cols["f1"] + rng.normal(size=n)
    df = pd.DataFrame(cols)
    df["y"] = y
    return df


def _run(df: pd.DataFrame, n_jobs: int, backend: str, sample_n: int) -> float:
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, transforms=["linear_residual", "diff", "ratio", "logratio", "additive_residual", "median_residual"],
        mi_nbins=8, mi_estimator="bin", top_k_after_mi=12, eps_mi_gain=-1.0, random_state=11, discovery_n_jobs=1,
        mi_gain_bootstrap_n=0, base_candidates=["base1", "base2"], screening="tiny_model", tiny_screening_models="single_lgbm",
        tiny_model_sample_n=sample_n, tiny_model_n_estimators=60, tiny_model_cv_folds=3, tiny_model_n_seed_repeats=1,
        deterministic_screening_models=True, require_beats_raw_baseline=False, tiny_rerank_n_jobs=n_jobs, tiny_rerank_backend=backend,
    )
    features = [c for c in df.columns if c != "y"]
    t0 = time.perf_counter()
    CompositeTargetDiscovery(config=cfg).fit(df=df, target_col="y", feature_cols=features, train_idx=np.arange(int(0.8 * len(df))))
    return time.perf_counter() - t0


if __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    sample_n = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
    df = _data(max(sample_n * 2, 40_000))
    _run(df, 1, "auto", 2_000)  # warm imports, numba and LightGBM so every mode below starts equally warm
    for label, jobs, backend in (("serial", 1, "auto"), ("threads", n_jobs, "threads"), ("processes", n_jobs, "processes"), ("processes (warm)", n_jobs, "processes")):
        print(f"{label:>17}: {_run(df, jobs, backend, sample_n):7.1f}s  (n_jobs={jobs}, sample_n={sample_n})", flush=True)
