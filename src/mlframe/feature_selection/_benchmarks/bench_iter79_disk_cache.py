"""iter79 bench: disk cache hit/miss A/B at compact-C3 scale.

Measures the OOF-SHAP wall-clock saving on the second fit when the cache is enabled. Compact-C3
(width=2000, n_rows=4000) keeps the bench under the 180 s budget while still putting OOF-SHAP at
several seconds per fit so the hit/miss delta is unambiguous. The chosen-subset bit-identity check
is the strictest gate: if any post-cache stage drifts the cache is wrong.

Run:  python -m mlframe.feature_selection._benchmarks.bench_iter79_disk_cache
"""
from __future__ import annotations

import os
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _make_data(n=4000, width=2000, n_inf=8, seed=0):
    rng = np.random.default_rng(seed)
    inf = rng.standard_normal((n, n_inf))
    noise = rng.standard_normal((n, width - n_inf))
    X = pd.DataFrame(
        np.column_stack([inf, noise]),
        columns=[f"inf{i}" for i in range(n_inf)]
        + [f"noise{i}" for i in range(width - n_inf)],
    )
    coefs = np.linspace(1.0, 0.4, n_inf)
    logits = inf @ coefs
    y = (logits + 0.3 * rng.standard_normal(n) > 0).astype(int)
    return X, y


def _run_one(X, y, cache_dir):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS(
        classification=True,
        metric="brier",
        n_splits=3,
        n_models=1,
        max_features=8,
        prefilter_top=200,
        prefilter_method="univariate",
        revalidate=False,
        trust_guard=False,
        cluster_features=False,
        random_state=0,
        verbose=False,
        cache_dir=cache_dir,
    )
    # Per-stage timings: opt-in via ``_stage_timings`` (no overhead on production fits).
    sel._stage_timings = {}
    t0 = time.perf_counter()
    sel.fit(X, y)
    t_total = time.perf_counter() - t0
    return dict(
        total=t_total,
        oof_shap=float(sel._stage_timings.get("oof_shap", 0.0)),
        chosen=tuple(sorted(getattr(sel, "selected_features_", []))),
    )


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    X, y = _make_data()
    print(f"# data: n={len(X)} width={X.shape[1]}")

    with tempfile.TemporaryDirectory() as td:
        # Variant 1: cache_dir=None (current default), single fit.
        r_off = _run_one(X, y, cache_dir=None)
        print(f"cache=OFF        total={r_off['total']:.2f}s oof_shap={r_off['oof_shap']:.2f}s")

        # Variant 2: cache_dir=tmp (first fit -- miss).
        cache_path = os.path.join(td, "shap_cache")
        r_miss = _run_one(X, y, cache_dir=cache_path)
        print(f"cache=ON  (miss) total={r_miss['total']:.2f}s oof_shap={r_miss['oof_shap']:.2f}s")

        # Variant 3: cache_dir=tmp (second fit -- hit).
        r_hit = _run_one(X, y, cache_dir=cache_path)
        print(f"cache=ON  (hit)  total={r_hit['total']:.2f}s oof_shap={r_hit['oof_shap']:.2f}s")

        oof_speedup = r_miss["oof_shap"] / max(r_hit["oof_shap"], 1e-9)
        e2e_speedup = r_miss["total"] / max(r_hit["total"], 1e-9)
        bit_identical = r_off["chosen"] == r_miss["chosen"] == r_hit["chosen"]
        print()
        print(f"OOF-SHAP speedup (miss -> hit): {oof_speedup:.2f}x")
        print(f"E2E      speedup (miss -> hit): {e2e_speedup:.2f}x")
        print(f"chosen subset bit-identical across off/miss/hit: {bit_identical}")
        print(f"chosen (off): {r_off['chosen']}")
        print(f"chosen (hit): {r_hit['chosen']}")


if __name__ == "__main__":
    main()
