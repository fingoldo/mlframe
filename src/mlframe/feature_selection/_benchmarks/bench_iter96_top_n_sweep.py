"""Iter96 top_n sweep at C3.

Sweep of ``ShapProxiedFS(top_n=...)`` over {20, 16, 12, 8} at C3
(width=10000, n_rows=10000, n_informative=20, n_redundant=20, snr=8.0, seed=0).

``top_n`` is the number of candidate subsets the search heuristic forwards for honest
revalidation. Each candidate gets ``n_revalidation_models`` (default 3) fits, so per-search
budget is ``top_n * n_models``. iter50 MMR drops near-duplicates from the evaluated set and
iter77 adaptive early-stops models when the winner is stable: with those guards on, the
static ceiling top_n=20 (current default 30 reduced to 20 in this bench harness via
explicit kw) may be too generous. Gate (post-iter95 lessons): chosen subset BIT-IDENTICAL
across all four OR same recall AND chosen-subset honest_loss within +-5%; revalidation
wall measurably faster at the smallest value tested.

Run::

    $env:PYTHONPATH = '<worktree>/src'
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_iter96_top_n_sweep
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


def _build_selector(seed, *, top_n):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    return ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=500, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=top_n, n_splits=4, n_revalidation_models=3, trust_guard=True,
        n_anchors=30,
        prefilter_n_estimators=100,
        run_importance_ablation=True, within_cluster_refine=True,
        revalidation_ucb_enabled=True,
        revalidation_adaptive_n_models=True,
        random_state=seed, verbose=False)


def _informative_recall(chosen, roles):
    inf_cols = {name for name, role in roles.items() if role == "informative"}
    if not inf_cols:
        return float("nan")
    sel = set(chosen)
    return len(sel & inf_cols) / len(inf_cols)


def run_one(value, cfg, X, y, roles):
    print(f"\n[top_n={value}] starting", flush=True)
    sel = _build_selector(cfg["seed"], top_n=value)
    sel._stage_timings = {}
    t0 = time.perf_counter()
    sel.fit(X, y)
    total = time.perf_counter() - t0
    report = sel.shap_proxy_report_
    rv = report.get("revalidation", {}) or {}
    ranked = rv.get("ranked", []) or []
    chosen_honest_loss = ranked[0].get("honest_loss") if ranked else None
    # n_models_run: max across ranked candidates (adaptive early-stop applies per-candidate).
    n_models_run_vals = [int(r.get("ucb", {}).get("n_models_run", 0)) for r in ranked if isinstance(r.get("ucb"), dict)]
    n_models_run_max = max(n_models_run_vals) if n_models_run_vals else None
    n_models_run_sum = sum(n_models_run_vals) if n_models_run_vals else None
    chosen = tuple(sorted(sel.selected_features_))
    return dict(
        value=value,
        total=total,
        reval_wall=sel._stage_timings.get("revalidation"),
        chosen=chosen,
        n_selected=len(chosen),
        n_evaluated=len(ranked),
        informative_recall=_informative_recall(chosen, roles),
        chosen_honest_loss=chosen_honest_loss,
        n_models_run_max=n_models_run_max,
        n_models_run_sum=n_models_run_sum,
    )


def main():
    print(f"[iter96] cfg={C3}", flush=True)
    t0 = time.perf_counter()
    X, y, roles = _make_dataset(C3)
    print(f"[iter96] dataset shape={X.shape} in {time.perf_counter()-t0:.1f}s", flush=True)

    results = []
    for value in (20, 16, 12, 8):
        r = run_one(value, C3, X, y, roles)
        loss_str = f"{r['chosen_honest_loss']:.6f}" if r['chosen_honest_loss'] is not None else "n/a"
        rv_str = f"{r['reval_wall']:.3f}s" if r['reval_wall'] is not None else "n/a"
        print(f"[top_n={value}] total={r['total']:.2f}s reval_wall={rv_str} "
              f"recall={r['informative_recall']:.4f} chosen_loss={loss_str} "
              f"n_sel={r['n_selected']} n_eval={r['n_evaluated']} "
              f"n_models_run(max/sum)={r['n_models_run_max']}/{r['n_models_run_sum']}",
              flush=True)
        results.append(r)

    print("\n=== iter96 top_n sweep ===")
    print(f"{'value':>6} {'reval_wall':>11} {'e2e':>8} {'recall':>8} {'chosen_loss':>14} " f"{'n_sel':>6} {'n_eval':>7} {'mdl_sum':>8}")
    for r in results:
        loss_str = f"{r['chosen_honest_loss']:.6f}" if r['chosen_honest_loss'] is not None else "n/a"
        rv_str = f"{r['reval_wall']:.3f}" if r['reval_wall'] is not None else "n/a"
        print(f"{r['value']:>6} {rv_str:>11} {r['total']:>8.2f} "
              f"{r['informative_recall']:>8.4f} {loss_str:>14} {r['n_selected']:>6} "
              f"{r['n_evaluated']:>7} {str(r['n_models_run_sum']):>8}")

    base = results[0]
    print("\n=== chosen-subset comparison vs baseline value=20 ===")
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

    print("\nbaseline (value=20) chosen:", sorted(base["chosen"]))


if __name__ == "__main__":
    main()
