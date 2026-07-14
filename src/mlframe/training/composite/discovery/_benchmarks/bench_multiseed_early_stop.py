"""Benchmark: sequential multiseed early-stop in Phase B rerank (``enable_multiseed_early_stop``).

Measures a full ``CompositeTargetDiscovery.fit`` (screening="tiny_model") wall time with the
sequential multiseed early-stop OFF vs ON, on a synthetic frame with ~15-20 bases surviving Phase A
MI screening (so Phase B's per-candidate multiseed CV rerank -- ~80% of total discovery wall-time --
dominates). Most surviving candidates are noisy near-duplicates of the dominant base with decaying
relevance, so most of them get rejected by the raw-baseline gate: the scenario early-stop targets.

Run: python -m mlframe.training.composite.discovery._benchmarks.bench_multiseed_early_stop
"""
from __future__ import annotations

import statistics
from timeit import default_timer as timer

import numpy as np
import pandas as pd


def _frame(n: int = 4000, n_bases: int = 18, seed: int = 0):
    """One dominant base (``b0``) that composites cleanly, plus ``n_bases - 1`` progressively noisier
    copies whose composites mostly lose to the raw-y baseline gate -- the doomed-candidate population
    the early-stop is meant to skip seeds on."""
    rng = np.random.default_rng(seed)
    cols = {}
    b0 = rng.normal(loc=10.0, scale=3.0, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.95 * b0 + 0.4 * x1 - 0.2 * x2 + rng.normal(scale=0.2, size=n)
    cols["b0"] = b0
    for i in range(1, n_bases):
        cols[f"b{i}"] = b0 + rng.normal(0.0, 1.5 * i, n)  # decaying relevance -> mostly doomed composites
    cols["x1"] = x1
    cols["x2"] = x2
    cols["y"] = y
    return pd.DataFrame(cols), [c for c in cols if c != "y"]


def _run_fit(df, feats, bases, enable_early_stop: bool, reps: int = 3) -> tuple[float, list[str]]:
    """Median wall time (over ``reps`` fits) and the discovered spec names for one early-stop setting."""
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    times = []
    names: list[str] = []
    for _r in range(reps):
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, random_state=0, screening="tiny_model",
            tiny_screening_models="single_lgbm",
            tiny_model_n_estimators=15, tiny_model_cv_folds=3,
            tiny_model_sample_n=1500, tiny_model_n_seed_repeats=5,
            top_m_after_tiny=5, base_candidates=list(bases),
            require_beats_raw_baseline=True, raw_baseline_tolerance=1.02,
            honest_oof_selection=False, use_wilcoxon_gate=False,
            multi_base_enabled=False, interaction_base_discovery_enabled=False,
            auto_chain_discovery_enabled=False, auto_base_dedup_corr_threshold=1.0,
            enable_multiseed_early_stop=enable_early_stop,
            transforms=["diff", "additive_residual", "linear_residual"],
        )
        disc = CompositeTargetDiscovery(cfg)
        t0 = timer()
        disc.fit(df, "y", feats, np.arange(len(df)))
        times.append(timer() - t0)
        names = sorted(s.name for s in disc.specs_)
    return statistics.median(times), names


def main() -> None:
    """Run the OFF vs ON comparison and print the timing + selection summary."""
    df, feats = _frame()
    bases = [c for c in feats if c.startswith("b")]
    print(f"candidates: {len(bases)} bases x transforms surviving Phase A MI screen")
    t_off, specs_off = _run_fit(df, feats, bases, enable_early_stop=False)
    t_on, specs_on = _run_fit(df, feats, bases, enable_early_stop=True)
    print(f"early-stop OFF: {t_off:.2f} s/fit, {len(specs_off)} specs: {specs_off}")
    print(f"early-stop ON : {t_on:.2f} s/fit, {len(specs_on)} specs: {specs_on}")
    print(f"selection match: {specs_off == specs_on}")
    if t_on > 0:
        print(f"speedup: {t_off / t_on:.2f}x")


if __name__ == "__main__":
    main()
