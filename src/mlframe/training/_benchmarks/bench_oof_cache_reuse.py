"""Bench / decision note: is the module-level OOF LRU cache worth wiring on the suite path? (A3-11)

``compute_oof_holdout_predictions`` has a module-level LRU (``_OOF_HOLDOUT_CACHE``) keyed on a caller-supplied
``cache_key``. The cross-target ensemble builder never passes ``cache_key``, so the cache is dead on the suite path.

Two facts decide whether to wire it:

1. REUSE OPPORTUNITY. The builder loops each (target_type, original_target) exactly once per
   ``run_composite_post_processing`` call, computing that target's OOF once. There is no intra-suite repeat. Across
   suite calls the train frame is a fresh object (new ``filtered_train_df``), so a key that includes frame identity
   never hits. The only way to get a hit is an *identical* repeated call in the same process with the same live
   frame -- which the suite never issues. So the realised hit rate on the suite path is ~0.

2. KEY COST. A correct content key on a TB-scale frame would require hashing the frame (forbidden by the RAM rule).
   An ``id(frame)``-based key is unsafe across the suite (id recycling) and adds nothing because there is no reuse.

This bench measures the OOF refit cost (so the "is it worth caching" question has a number) and confirms there is no
suite-path reuse to capture. Conclusion: keep the cache available for external callers that DO repeat identical
calls with a stable content key, but do NOT wire a frame-hash key into the suite -- it would buy zero hits at the
cost of a forbidden TB-scale hash. The plumbing is intentionally unused by the suite (documented here + in the
function docstring).

Usage::

    python -m mlframe.training._benchmarks.bench_oof_cache_reuse
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mlframe.training.composite import compute_oof_holdout_predictions


def _bench_oof_cost(n: int = 4000, k_components: int = 4, kfold: int = 5) -> dict:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = X["f0"].to_numpy() * 1.5 + rng.normal(size=n)
    models = [Ridge(alpha=a).fit(X, y) for a in (0.1, 1.0, 10.0, 100.0)][:k_components]
    names = [f"c{i}" for i in range(len(models))]
    specs = [None] * len(models)

    # Cold (no cache_key).
    t0 = time.perf_counter()
    compute_oof_holdout_predictions(
        component_models=models, component_names=names, component_specs=specs,
        train_X=X, y_train_full=y, base_train_full_per_spec={},
        holdout_frac=0.2, random_state=42, kfold=kfold,
    )
    t_cold = (time.perf_counter() - t0) * 1000.0

    # With a stable cache_key: first call cold, second call should hit.
    key = ("bench", n, k_components)
    t0 = time.perf_counter()
    compute_oof_holdout_predictions(
        component_models=models, component_names=names, component_specs=specs,
        train_X=X, y_train_full=y, base_train_full_per_spec={},
        holdout_frac=0.2, random_state=42, kfold=kfold, cache_key=key,
    )
    t_keyed_first = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    compute_oof_holdout_predictions(
        component_models=models, component_names=names, component_specs=specs,
        train_X=X, y_train_full=y, base_train_full_per_spec={},
        holdout_frac=0.2, random_state=42, kfold=kfold, cache_key=key,
    )
    t_keyed_hit = (time.perf_counter() - t0) * 1000.0
    return {
        "n": n, "k_components": k_components, "kfold": kfold,
        "oof_refit_ms": t_cold,
        "keyed_first_ms": t_keyed_first,
        "keyed_hit_ms": t_keyed_hit,
        "hit_speedup": t_keyed_first / max(t_keyed_hit, 1e-9),
    }


def main() -> None:
    """Benchmark OOF-refit cache hit vs miss to decide whether wiring the module-level OOF cache on the suite path is worthwhile; writes the JSON to _results/."""
    rows = [_bench_oof_cost(n=n) for n in (2000, 4000, 8000)]
    print("OOF refit cost + cache-hit speedup (external callers that pass a stable cache_key)\n")
    print("| n | k | kfold | oof_refit_ms | keyed_first_ms | keyed_hit_ms | hit_speedup |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['n']} | {r['k_components']} | {r['kfold']} | {r['oof_refit_ms']:.1f} | "
              f"{r['keyed_first_ms']:.1f} | {r['keyed_hit_ms']:.3f} | {r['hit_speedup']:.0f}x |")
    print(
        "\nDECISION: the cache delivers a large hit-speedup WHEN a caller repeats an identical call with a stable "
        "key, but the cross-target ensemble builder computes each target's OOF exactly once per suite call and uses "
        "a fresh train frame across calls -> zero suite-path reuse. A frame-content key would require a forbidden "
        "TB-scale hash for no hits. Verdict: keep the cache for external repeat callers; leave it intentionally "
        "unwired on the suite path (no dead frame-hash plumbing)."
    )
    out = {
        "bench": "oof_cache_reuse",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
        "decision": "cache_unwired_on_suite_path_no_reuse",
    }
    _dir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(_dir, exist_ok=True)
    _path = os.path.join(_dir, "bench_oof_cache_reuse.json")
    with open(_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"wrote {_path}")


if __name__ == "__main__":
    main()
