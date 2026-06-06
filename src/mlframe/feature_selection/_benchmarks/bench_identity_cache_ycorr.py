"""Bench: cross-target identity-cache y-correlation gate threshold (A1-06).

The cross-target identity cache short-circuits the FE pipeline when a prior fit on the SAME X-fingerprint
returned an identity result (all columns kept, nothing engineered). Without a y-correlation gate, a
genuinely-DIFFERENT target on the same X also hits the cache and gets the (possibly wrong) identity result.
The gate (mrmr_identity_cache_ycorr_threshold) refuses the short-circuit when |corr(new_y, prior_y)| is below
the threshold, forcing a correct full fit for the distinct target.

This bench sweeps the threshold and measures, on a fixed X with TWO targets (one highly correlated with the
caching target, one independent):
  - correlated target  : should still HIT the cache (perf preserved)
  - independent target : should be REFUSED below the threshold (correctness preserved)

Run:
    python -m mlframe.feature_selection._benchmarks.bench_identity_cache_ycorr

Verdict (this machine): default threshold 0.5. At 0.5 the correlated composite/residual target (|corr|~0.9)
still short-circuits while the independent target (|corr|~0.0) is refused and re-fit honestly. 0.0 disables the
gate (legacy: any X-fingerprint hit short-circuits, including the wrong independent target).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fingerprints import _mrmr_y_corr, _mrmr_y_corr_sample


def _make(seed=0, n=2000):
    rng = np.random.RandomState(seed)
    z = rng.randn(n)
    X = pd.DataFrame({f"f{i}": rng.randn(n) for i in range(6)})
    X["f0"] = z
    y_cache = (z + 0.2 * rng.randn(n) > 0).astype(int)        # caching target
    y_corr = (z + 0.4 * rng.randn(n) > 0).astype(int)         # correlated composite/residual proxy
    y_indep = rng.randint(0, 2, n)                            # independent target
    return X, y_cache, y_corr, y_indep


def main():
    """Run the y-correlation gate-threshold sweep and print the result table."""
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]
    results = []
    for seed in range(5):
        X, y_cache, y_corr, y_indep = _make(seed)
        s_cache = _mrmr_y_corr_sample(y_cache)
        corr_corr = abs(_mrmr_y_corr(_mrmr_y_corr_sample(y_corr), s_cache) or 0.0)
        corr_indep = abs(_mrmr_y_corr(_mrmr_y_corr_sample(y_indep), s_cache) or 0.0)
        for thr in thresholds:
            results.append({
                "seed": seed, "threshold": thr,
                "corr_target_ycorr": corr_corr, "corr_target_hits": corr_corr >= thr,
                "indep_target_ycorr": corr_indep, "indep_target_hits": corr_indep >= thr,
            })
    df = pd.DataFrame(results)
    summary = df.groupby("threshold").agg(
        corr_hit_rate=("corr_target_hits", "mean"),
        indep_hit_rate=("indep_target_hits", "mean"),
        corr_ycorr=("corr_target_ycorr", "mean"),
        indep_ycorr=("indep_target_ycorr", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))
    print("\nInterpretation: want corr_hit_rate=1.0 (perf preserved) AND indep_hit_rate=0.0 (correctness).")
    out = Path(__file__).parent / "_results" / f"identity_cache_ycorr_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print("wrote", out)


if __name__ == "__main__":
    main()
