"""Iter95 prefilter_n_estimators sweep at C3.

Sweep of ``ShapProxiedFS(prefilter_n_estimators=...)`` over {100, 50, 25} at C3
(width=10000, n_rows=10000, n_informative=20, n_redundant=20, snr=8.0, seed=0).

The prefilter stage fits a cloned ranking booster on the full-width search frame and ranks
features by ``feature_importances_``; the top-``prefilter_top`` survive into the heavy pipeline.
The prefilter is a pure rank-consumer: only the ORDER of importance matters, not absolute
loss numbers. Importance rankings stabilise well below the default 300 trees, and the
``min(current, cap)`` clamp can only ever LOWER the budget. Gate (combined iter90 + iter94
lessons): chosen subset BIT-IDENTICAL across all three OR same recall AND chosen-subset
honest_loss within +-5%; prefilter wall measurably faster at the smallest value tested.

Run::

    $env:PYTHONPATH = '<worktree>/src'
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_iter95_prefilter_n_estimators_sweep
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
import warnings

warnings.filterwarnings("ignore")


C3 = dict(width=10000, n_rows=10000, n_informative=20, n_redundant=20, redundancy_rho=0.8, snr=8.0, seed=0)


def _make_dataset(cfg):
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    n_noise = max(0, cfg["width"] - cfg["n_informative"] - cfg["n_redundant"])
    X, y, roles = make_regime_dataset(
        n_samples=cfg["n_rows"], n_informative=cfg["n_informative"],
        n_redundant=cfg["n_redundant"], redundancy_rho=cfg["redundancy_rho"],
        n_noise=n_noise, snr=cfg["snr"], task="binary", seed=cfg["seed"])
    return X, y, roles


def _build_selector(seed, *, prefilter_n_estimators):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    return ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=20, n_splits=4, n_revalidation_models=3, trust_guard=True,
        n_anchors=30,
        prefilter_n_estimators=prefilter_n_estimators,
        run_importance_ablation=True, within_cluster_refine=True,
        revalidation_ucb_enabled=True,
        revalidation_adaptive_n_models=True,
        random_state=seed, verbose=False)


def _informative_recall(chosen, roles):
    """Fraction of true-informative columns retained in the chosen subset.

    ``roles`` is a ``{column_name: role_str}`` mapping; ``informative`` columns are the ground truth.
    """
    inf_cols = {name for name, role in roles.items() if role == "informative"}
    if not inf_cols:
        return float("nan")
    sel = set(chosen)
    return len(sel & inf_cols) / len(inf_cols)


def run_one(value, cfg, X, y, roles):
    print(f"\n[prefilter_n_estimators={value}] starting", flush=True)
    sel = _build_selector(cfg["seed"], prefilter_n_estimators=value)
    sel._stage_timings = {}
    t0 = time.perf_counter()
    sel.fit(X, y)
    total = time.perf_counter() - t0
    report = sel.shap_proxy_report_
    rv = report.get("revalidation", {}) or {}
    ranked = rv.get("ranked", []) or []
    # Chosen-subset honest_loss is the FIRST ranked entry after the stable-score sort
    # AND the winner_full_loss patch (when cap is set). When ranking is empty (no revalidation),
    # this stays None and the gate falls back to chosen-subset identity.
    chosen_honest_loss = None
    if ranked:
        chosen_honest_loss = ranked[0].get("honest_loss")
    chosen = tuple(sorted(sel.selected_features_))
    return dict(
        value=value,
        total=total,
        prefilter_wall=sel._stage_timings.get("prefilter"),
        chosen=chosen,
        n_selected=len(chosen),
        informative_recall=_informative_recall(chosen, roles),
        chosen_honest_loss=chosen_honest_loss,
    )


def main():
    print(f"[iter95] cfg={C3}", flush=True)
    t0 = time.perf_counter()
    X, y, roles = _make_dataset(C3)
    print(f"[iter95] dataset shape={X.shape} in {time.perf_counter()-t0:.1f}s", flush=True)

    results = []
    for value in (100, 50, 25):
        r = run_one(value, C3, X, y, roles)
        loss_str = f"{r['chosen_honest_loss']:.6f}" if r['chosen_honest_loss'] is not None else "n/a"
        print(f"[prefilter_n_estimators={value}] total={r['total']:.2f}s "
              f"pf_wall={r['prefilter_wall']:.3f}s "
              f"recall={r['informative_recall']:.4f} "
              f"chosen_loss={loss_str} "
              f"n_sel={r['n_selected']}", flush=True)
        results.append(r)

    print("\n=== iter95 prefilter_n_estimators sweep ===")
    print(f"{'value':>6} {'pf_wall':>10} {'e2e':>8} {'recall':>8} {'chosen_loss':>14} {'n_sel':>6}")
    for r in results:
        loss_str = f"{r['chosen_honest_loss']:.6f}" if r["chosen_honest_loss"] is not None else "n/a"
        print(f"{r['value']:>6} {r['prefilter_wall']:>10.3f} {r['total']:>8.2f} " f"{r['informative_recall']:>8.4f} {loss_str:>14} {r['n_selected']:>6}")

    base = results[0]
    print("\n=== chosen-subset comparison vs baseline value=100 ===")
    for r in results:
        ident = "IDENTICAL" if r["chosen"] == base["chosen"] else "DIFFER"
        jac = len(set(r["chosen"]) & set(base["chosen"])) / max(1, len(set(r["chosen"]) | set(base["chosen"])))
        symdiff = set(base["chosen"]) ^ set(r["chosen"])
        print(f"  value={r['value']:>3}: {ident}  jaccard={jac:.3f}  symdiff={sorted(symdiff)}")

    base_loss = base["chosen_honest_loss"]
    if base_loss is not None:
        print("\n=== chosen-subset honest_loss delta vs baseline ===")
        for r in results:
            loss = r["chosen_honest_loss"]
            if loss is None:
                print(f"  value={r['value']:>3}: chosen_loss=n/a")
                continue
            delta = (loss - base_loss) / abs(base_loss) if base_loss else 0.0
            print(f"  value={r['value']:>3}: chosen_loss={loss:.6f}  delta={delta*100:+.2f}%")

    print("\nbaseline (value=100) chosen:", sorted(base["chosen"]))


if __name__ == "__main__":
    main()
