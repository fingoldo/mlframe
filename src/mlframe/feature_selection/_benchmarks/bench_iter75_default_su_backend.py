"""iter75 bench: ShapProxiedFS default ``cluster_backend='auto'`` vs explicit ``'pearson'`` at
two regimes.

Regime A: width=2000 / n_rows=5000 - auto picks SU (under default auto cap 2000).
Regime B: width=10000 / n_rows=10000 (C3-style) - auto falls back to Pearson (above auto cap).

Reports per regime: chosen backend, e2e wall, recall on planted informatives, and the chosen
feature set so any cluster-boundary differences are inspectable side-by-side.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np


def _make(width, n_rows, n_informative=5, seed=0):
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    return make_regime_dataset(
        n_samples=n_rows,
        n_informative=n_informative,
        n_redundant=10,
        redundancy_rho=0.8,
        n_noise=max(0, width - n_informative - 10),
        snr=8.0,
        task="binary",
        seed=seed,
    )


def _recall(selected, roles):
    informative = {n for n, role in roles.items() if role == "informative"}
    return len(set(selected) & informative) / max(len(informative), 1)


def run_regime(name, width, n_rows, max_features=10, **kw):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    print(f"\n[{name}] width={width} n_rows={n_rows} max_features={max_features}", flush=True)
    X, y, roles = _make(width, n_rows)

    common = dict(
        random_state=0, verbose=False,
        prefilter_top=200,
        max_features=max_features,
        n_models=1, n_splits=3,
        out_of_fold=True,
        revalidate=False, trust_guard=False, run_importance_ablation=False,
        cluster_features=True,
        cluster_auto_threshold=10,
        brute_force_max_features=12,
        shap_prefilter_enabled=False,
    )
    common.update(kw)

    rows = []
    for backend in ("auto", "pearson"):
        sps = ShapProxiedFS(cluster_backend=backend, **common)
        t0 = time.perf_counter()
        sps.fit(X, y)
        dt = time.perf_counter() - t0
        rep = sps.shap_proxy_report_
        chosen = list(sps.selected_features_)
        b = rep["clustering"]["backend"]
        src = rep["clustering"].get("bins_source", "n/a")
        n_mc = rep["clustering"].get("n_multi_clusters", "?")
        recall = _recall(chosen, roles)
        print(
            f"  backend_arg={backend:<8} -> chosen={b:<8} bins_src={src:<12} "
            f"n_mc={n_mc:>3} e2e={dt:6.2f}s  recall={recall:.3f}  "
            f"selected={sorted(chosen)}",
            flush=True,
        )
        rows.append(dict(backend_arg=backend, chosen=b, bins_src=src, n_mc=n_mc,
                         e2e=dt, recall=recall, selected=sorted(chosen)))
    return rows


def main():
    print("iter75 bench: SU-as-default vs Pearson opt-in", flush=True)
    run_regime("width=2000/n=5000 (auto -> SU expected)", width=2000, n_rows=5000, max_features=10)
    run_regime("width=10000/n=10000 (auto -> Pearson expected)", width=10000, n_rows=10000,
               max_features=10)


if __name__ == "__main__":
    main()
