"""Benchmark: early pruning of the base x transform grid via ``max_base_candidates`` (G3).

Measures a full ``CompositeTargetDiscovery.fit`` (screening="mi", bin estimator) on a synthetic
frame with a LONG explicit base-candidate list, capped vs uncapped. Each extra base multiplies the
per-(base, transform) MI screen, so the cap's win scales with the pruned fraction.

Measured on this machine (n=20_000 rows, 12 explicit bases + 8 noise features, 8 transforms,
Windows / python 3.14), after fixing a truncation bug where the MI-ranked candidate list was
computed but never sliced to ``[:cap]`` on the success path (only the fallback path truncated,
so the cap silently did nothing until this fix):

    uncapped (12 bases)            : 19.99 s / fit, 3 specs
    max_base_candidates=3          :  5.62 s / fit, 3 specs   -> 3.56x speedup
    selection: the capped run kept the same top specs (the signal-carrying bases rank first).

Why the within-pair-loop early-abort / streaming top-k (G3(b)) is NOT implemented (unsound):
* ``apply_fdr_control_to_candidates`` is a Benjamini-Hochberg pass over the FULL family of per-spec
  bootstrap p-values -- dropping evaluations once ``top_k_after_mi`` is "filled" changes the family
  size and therefore every adjusted p-value, altering selection.
* ``mi_gain`` of a pair is unknown before its evaluation and admits no cheap admissible upper bound
  (it depends on the fitted transform's T), so there is no provably-safe streaming cutoff: any
  early stop can discard a pair that would have ranked into the top-k.
The sound early pruning is therefore at the BASE level, before the pair loop -- which this cap does.

Run: python -m mlframe.training.composite.discovery._benchmarks.bench_base_candidate_cap
"""
from __future__ import annotations

import statistics
from timeit import default_timer as timer

import numpy as np
import pandas as pd


def _frame(n: int = 20_000, n_bases: int = 12, n_noise: int = 8, seed: int = 0):
    """Synthetic frame with one dominant base (``b0``), decaying-relevance copies, and pure noise columns."""
    rng = np.random.default_rng(seed)
    cols = {}
    # Two signal bases (the rest are progressively noisier copies + pure noise columns).
    b0 = rng.uniform(0.0, 1000.0, n)
    x0 = rng.normal(size=n)
    y = b0 + 30.0 * x0 + rng.normal(0.0, 1.0, n)
    cols["b0"] = b0
    for i in range(1, n_bases):
        cols[f"b{i}"] = b0 + rng.normal(0.0, 50.0 * i, n)  # decaying relevance
    cols["x0"] = x0
    for j in range(n_noise):
        cols[f"noise{j}"] = rng.normal(size=n)
    cols["y"] = y
    return pd.DataFrame(cols), [c for c in cols if c != "y"]


def _run_fit(df, feats, bases, cap, reps: int = 3) -> tuple[float, list[str]]:
    """Median wall time (over ``reps`` fits) and the discovered spec names for one ``max_base_candidates`` setting."""
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    times = []
    names: list[str] = []
    for _r in range(reps):
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, random_state=0, screening="mi", base_candidates=list(bases),
            max_base_candidates=cap, honest_holdout_frac=0.2, auto_base_null_perms=0,
            multi_base_enabled=False, interaction_base_discovery_enabled=False,
            auto_chain_discovery_enabled=False, auto_base_dedup_corr_threshold=1.0,
            transforms=["diff", "additive_residual", "median_residual", "ratio",
                        "linear_residual", "quantile_residual", "monotonic_residual", "asinh_residual"],
        )
        disc = CompositeTargetDiscovery(cfg)
        t0 = timer()
        disc.fit(df, "y", feats, np.arange(len(df)))
        times.append(timer() - t0)
        names = sorted(s.name for s in disc.specs_)
    return statistics.median(times), names


def main() -> None:
    """Run the capped vs uncapped comparison and print the timing + selection summary."""
    df, feats = _frame()
    bases = [c for c in feats if c.startswith("b")]
    t_off, specs_off = _run_fit(df, feats, bases, cap=None)
    t_on, specs_on = _run_fit(df, feats, bases, cap=3)
    print(f"uncapped ({len(bases)} bases): {t_off:.2f} s/fit, {len(specs_off)} specs")
    print(f"max_base_candidates=3      : {t_on:.2f} s/fit, {len(specs_on)} specs")
    print(f"speedup: {t_off / t_on:.2f}x")
    kept_capped_bases = {n.split('-')[-1] for n in specs_on}
    print(f"capped run bases: {sorted(kept_capped_bases)}")


if __name__ == "__main__":
    main()
