"""Bench: per-model-tier FS re-fit cost and the cheap-key support-cache win (A1-07).

Within ONE target the suite trains several model tiers (linear / tree / neural), each cloning the unfitted
selector and RE-FITTING it on the same logical feature set. A1-07 asks whether caching the selector's fitted
support across tiers (keyed on a CHEAP stable token -- target id + selector-params hash + a cheap
shape/columns signature, NEVER a full train_X content hash) saves meaningful time.

This bench measures the wall-clock of N consecutive selector fits on the SAME (X, y) -- the per-tier pattern --
both WITHOUT any cache (clone + fit each tier) and WITH a support reuse (fit once, reuse). It quantifies the
saved time so the default is set from the measured benefit.

Run:
    python -m mlframe.feature_selection._benchmarks.bench_per_tier_fs_cache

Verdict (this machine): NO dedicated per-tier support cache added (SKIPPED, not rejected-on-quality).
  cold first fit:                                       62.2 s
  per-tier refit total over 3 tiers (MRMR _FIT_CACHE):  0.0177 s  (5.9 ms / tier)
  additional saving from a dedicated cheap-key cache:   0.0177 s
MRMR already carries a process-wide CONTENT-keyed _FIT_CACHE (and the suite a content-keyed pre_pipeline cache),
so the 2nd..Nth tier fit on the same (X, y) already replays in ~6 ms. A dedicated cheap-key support cache would
save only ~18 ms/target on top of that -- not worth the added code + invalidation risk. Per the TB-frame rule we
do NOT add a frame-content-hash cache; the existing _FIT_CACHE (keyed on a cheap content SIGNATURE + params, not
a full byte hash) is the right mechanism and already in place. RFECV/BorutaShap lack an analogous fitted-support
replay cache, but their per-tier cost is dominated by the inner CV refits that a support cache could legitimately
skip -- a future item, tracked, not implemented here because no cheap key spans their polars/pandas tier inputs
without a content hash.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification

from mlframe.feature_selection.filters import MRMR


def _make(seed=0, n=3000, p=30):
    X, y = make_classification(n_samples=n, n_features=p, n_informative=10, n_redundant=8, random_state=seed)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), pd.Series(y, name="t")


def _cheap_token(X, params_hash, target_name):
    """The cheap-and-safe A1-07 key: NO full content hash; just target + params + columns + n_rows."""
    return (target_name, params_hash, X.shape[0], frozenset(map(str, X.columns)))


def main():
    X, y = _make()
    n_tiers = 4
    results = {}

    # Cold first fit (JIT warm + real work).
    m0 = MRMR(full_npermutations=3, cv=3, random_seed=0)
    t0 = time.perf_counter()
    m0.fit(X, y)
    results["cold_first_fit_s"] = time.perf_counter() - t0
    support0 = list(getattr(m0, "support_", []))

    # Per-tier WITHOUT explicit support cache: clone + fit each tier (relies on MRMR _FIT_CACHE replay).
    per_tier_times = []
    for _ in range(n_tiers - 1):
        m = clone(m0)
        t = time.perf_counter()
        m.fit(X, y)
        per_tier_times.append(time.perf_counter() - t)
    results["per_tier_refit_mean_s"] = float(np.mean(per_tier_times))
    results["per_tier_refit_total_s"] = float(np.sum(per_tier_times))

    # WITH cheap-key support cache: fit once, reuse support for the remaining tiers (cost ~ token build).
    cache = {}
    params_hash = hash(tuple(sorted((k, str(v)) for k, v in m0.get_params(deep=False).items())))
    t = time.perf_counter()
    for _ in range(n_tiers - 1):
        tok = _cheap_token(X, params_hash, "t")
        if tok in cache:
            _ = cache[tok]
        else:
            cache[tok] = support0
    results["per_tier_cheapcache_total_s"] = time.perf_counter() - t

    print(json.dumps(results, indent=2, sort_keys=True))
    saved = results["per_tier_refit_total_s"] - results["per_tier_cheapcache_total_s"]
    print(f"\ncold first fit: {results['cold_first_fit_s']:.3f}s")
    print(f"per-tier refit (MRMR _FIT_CACHE replay) total over {n_tiers-1} tiers: {results['per_tier_refit_total_s']:.4f}s")
    print(f"cheap-key support-cache total: {results['per_tier_cheapcache_total_s']:.6f}s")
    print(f"additional saving from a dedicated cheap-key cache: {saved:.4f}s")
    out = Path(__file__).parent / "_results" / f"per_tier_fs_cache_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print("wrote", out)


if __name__ == "__main__":
    main()
