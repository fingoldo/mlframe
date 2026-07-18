"""Benchmark: mi_correction='none' vs 'miller_madow' vs 'chao_shen' -- accuracy (selection quality on
a high-cardinality-noise-vs-low-cardinality-signal synthetic) and wall-time.

Run: PYTHONPATH=src python -m mlframe.feature_selection._benchmarks.bench_mi_correction_chao_shen_vs_miller_madow

Backs the default-choice decision for finding #7 (05_concurrency_and_statistics.md): the audit flagged
``mi_correction='chao_shen'`` as an accepted-but-silently-degraded API trap. Now that it's wired
(compute_mi_cs_from_classes / mi_chao_shen), this bench decides whether either correction should become
the new DEFAULT (currently 'none', legacy plug-in, bit-exact) per project convention ("most accurate on
the honest metric first, speed only breaks ties").
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import MRMR


def _make_fixture(seed: int, n: int, n_noise_highcard: int, noise_card: int):
    """One TRUE low-cardinality binary signal column + several high-cardinality pure-noise columns."""
    rng = np.random.default_rng(seed)
    signal = rng.integers(0, 2, n).astype(np.int64)
    y = signal.copy()
    cols = {"signal": signal}
    for i in range(n_noise_highcard):
        cols[f"noise_hc_{i}"] = rng.integers(0, noise_card, n).astype(np.int64)
    df = pd.DataFrame(cols)
    return df, pd.Series(y, name="y")


def _run_one(mi_correction: str, seed: int, n: int, n_noise_highcard: int, noise_card: int):
    df, y = _make_fixture(seed, n, n_noise_highcard, noise_card)
    m = MRMR(
        verbose=0,
        fe_max_steps=0,
        full_npermutations=1,
        baseline_npermutations=1,
        mi_correction=mi_correction,
        min_features_fallback=1,
    )
    t0 = time.perf_counter()
    m.fit(df, y)
    dt = time.perf_counter() - t0
    names = list(m.get_feature_names_out())
    signal_recovered = "signal" in names
    n_noise_leaked = sum(1 for c in names if c.startswith("noise_hc_"))
    return signal_recovered, n_noise_leaked, dt


def main():
    """Sweep small-n/high-cardinality-noise scenarios across mi_correction settings and report a
    signal-recovery / noise-rejection / wall-time table."""
    settings = ["none", "miller_madow", "chao_shen"]
    scenarios = [
        dict(n=300, n_noise_highcard=4, noise_card=60),
        dict(n=800, n_noise_highcard=4, noise_card=100),
        dict(n=2000, n_noise_highcard=4, noise_card=60),
    ]
    seeds = list(range(5))

    print(f"{'scenario':<28}{'setting':<16}{'signal_hit_rate':<18}{'avg_noise_leak':<16}{'avg_time_s':<12}")
    for scen in scenarios:
        for setting in settings:
            hits = 0
            leaks = []
            times = []
            for seed in seeds:
                hit, leak, dt = _run_one(setting, seed, **scen)
                hits += int(hit)
                leaks.append(leak)
                times.append(dt)
            scen_label = f"n={scen['n']},card={scen['noise_card']}"
            print(f"{scen_label:<28}{setting:<16}{hits}/{len(seeds):<17}{np.mean(leaks):<16.3f}{np.mean(times):<12.4f}")


if __name__ == "__main__":
    main()
