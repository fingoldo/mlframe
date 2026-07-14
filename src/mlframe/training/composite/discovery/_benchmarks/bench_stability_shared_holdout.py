"""Benchmark: shared-honest-holdout stability sweep vs per-replicate carve emulation (G6).

``fit_with_stability_check`` now carves the honest holdout ONCE and each replicate subsamples the
SCREEN pool only (see ``_honest_holdout.carve_screening_holdout``). Besides the statistical fix
(sweep-wide never-touched holdout), each replicate screens ``frac * 0.8n`` rows instead of
``frac * n`` (its own internal carve previously removed 20% AFTER the heavy setup was sized on the
full subsample), so the per-replicate MI screen shrinks ~20%.

The "before" arm is emulated in-process (no destructive git): per-replicate subsamples drawn from
the FULL train index with the shared-holdout marker cleared, i.e. the pre-fix flow bit-for-bit.

Measured on this machine (n=30_000, 5 replicates, screening="mi", 4 transforms):

    per-replicate carve (before) : 12.55 s / sweep
    shared holdout (after)       :  7.80 s / sweep  -> 1.61x speedup
    stable spec set identical across both arms (same names selected).

The prebin-cache reuse across replicates was investigated and is UNSOUND to force: the cache is
content-hash keyed on the exact screen-sample float matrix, and every replicate draws a different
row subsample (different seed), so the matrices differ and a forced key-share would change the MI
codes each replicate sees. The existing content-keyed ``PrebinCache`` correctly misses there.

Run: python -m mlframe.training.composite.discovery._benchmarks.bench_stability_shared_holdout
"""
from __future__ import annotations

import statistics
from timeit import default_timer as timer

import numpy as np
import pandas as pd


def _frame(n: int = 30_000, seed: int = 0):
    """Synthetic frame with one dominant base and several noise columns for the stability sweep."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 1000.0, n)
    x0 = rng.normal(size=n)
    y = base + 30.0 * x0 + rng.normal(0.0, 1.0, n)
    cols = {"base": base, "x0": x0, "y": y}
    for j in range(6):
        cols[f"noise{j}"] = rng.normal(size=n)
    return pd.DataFrame(cols)


def _cfg():
    """Shared screening="mi" discovery config used by both the shared-holdout and emulated pre-fix sweeps."""
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    return CompositeTargetDiscoveryConfig(
        enabled=True, random_state=0, screening="mi", base_candidates=["base"],
        transforms=["diff", "additive_residual", "linear_residual", "monotonic_residual"],
        honest_holdout_frac=0.2, auto_base_null_perms=0, multi_base_enabled=False,
        interaction_base_discovery_enabled=False, auto_chain_discovery_enabled=False,
    )


def _sweep(df, shared: bool, reps: int = 3) -> tuple[float, list[str]]:
    """Median wall time (over ``reps`` sweeps) and stable spec names for the shared-holdout vs emulated pre-fix flow."""
    from mlframe.training.composite import CompositeTargetDiscovery

    feats = [c for c in df.columns if c != "y"]
    times = []
    names: list[str] = []
    for _ in range(reps):
        disc = CompositeTargetDiscovery(_cfg())
        t0 = timer()
        if shared:
            disc.fit_with_stability_check(df, "y", feats, np.arange(len(df)), n_bootstrap_runs=5)
        else:
            _emulate_per_replicate_carve(disc, df, feats)
        times.append(timer() - t0)
        names = sorted(s.name for s in disc.specs_)
    return statistics.median(times), names


def _emulate_per_replicate_carve(disc, df, feats, n_runs: int = 5, frac: float = 0.5) -> None:
    """The pre-fix flow: each replicate subsamples the FULL train index and carves its own holdout."""
    from collections import Counter
    from mlframe.training.composite.ensemble import derive_seeds

    train_idx = np.arange(len(df))
    saved = disc.config
    disc.config = disc.config.model_copy()
    seeds = derive_seeds(int(disc.config.random_state), [f"stability_run_{i}" for i in range(n_runs)])
    counter: Counter = Counter()
    by_name: dict = {}
    sub_n = max(2, round(frac * train_idx.size))
    for i in range(n_runs):
        seed = int(seeds[f"stability_run_{i}"]) & 0x7FFFFFFF
        disc.config.random_state = seed
        rng = np.random.default_rng(seed)
        sub = np.sort(rng.choice(train_idx, size=sub_n, replace=False))
        disc.fit(df, "y", feats, sub)
        for s in disc.specs_:
            counter[s.name] += 1
            by_name.setdefault(s.name, s)
    disc.config = saved
    thr = max(1, int(0.6 * n_runs))
    disc.specs_ = [by_name[n] for n, c in counter.items() if c >= thr]


def main() -> None:
    """Run the emulated pre-fix vs shared-holdout stability sweeps and print the timing + selection summary."""
    df = _frame()
    t_before, specs_before = _sweep(df, shared=False)
    t_after, specs_after = _sweep(df, shared=True)
    print(f"per-replicate carve (before): {t_before:.2f} s/sweep, specs={specs_before}")
    print(f"shared holdout (after)      : {t_after:.2f} s/sweep, specs={specs_after}")
    print(f"speedup: {t_before / t_after:.2f}x; identical spec set: {specs_before == specs_after}")


if __name__ == "__main__":
    main()
