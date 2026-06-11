"""Speed/quality tradeoff bench for the ShapProxiedFS native-importance pre-filter methods.

Iteration-4 profiling proved the pre-filter (one model fit on ALL columns to rank importances and keep
``prefilter_top``) is the DOMINANT wide-data fit cost (~66% at 10k features). This bench characterises
the four ``prefilter_method`` options against EACH OTHER on BOTH axes that matter:

  - speed  : the pre-filter stage wall-clock in isolation (``prefilter_columns`` on all columns), plus
             the end-to-end ``ShapProxiedFS.fit`` wall-clock, swept at 2k / 5k / 10k features;
  - quality: informative-recovery -- (a) PRE-FILTER recall: did the planted informatives survive the
             top-K cut (if the prefilter drops an informative, SHAP never sees it); (b) END-TO-END
             recall: how many informatives the FINAL selected subset keeps.

Data is ``make_regime_dataset`` (a few informatives + correlated redundant copies + a heavy noise
flood -- the user's real wide regime), with ground-truth roles + ``oracle_subset`` to score recovery.

Run (PowerShell)::

    $env:PYTHONPATH = '<worktree>\\src'
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_shap_proxy_prefilter

Args: ``--widths 2000,5000,10000``, ``--rows 4000``, ``--prefilter-top 500``, ``--methods model,univariate,fast_model,gpu_model``,
``--end-to-end`` (also run the full fit per method, not just the isolated prefilter stage; slower).
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def make_wide(n_features, *, n_rows, n_informative=8, n_redundant=12, seed=0):
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    n_noise = max(0, n_features - n_informative - n_redundant)
    return make_regime_dataset(
        n_samples=n_rows, n_informative=n_informative, n_redundant=n_redundant,
        redundancy_rho=0.9, n_noise=n_noise, snr=5.0, task="binary", seed=seed)


def _informative_names(roles):
    return {name for name, r in roles.items() if r == "informative"}


def time_prefilter_stage(method, X, y, roles, *, prefilter_top, seed=0):
    """Time ``prefilter_columns`` (the isolated stage) and score PRE-FILTER recall of informatives."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import prefilter_columns

    model = make_default_estimator(classification=True, random_state=seed)
    yf = np.asarray(y, dtype=np.float64)
    t0 = time.perf_counter()
    working_cols, info = prefilter_columns(
        model, X, yf, method=method, prefilter_top=prefilter_top,
        classification=True, n_features=X.shape[1])
    secs = time.perf_counter() - t0

    kept_names = {str(X.columns[i]) for i in working_cols}
    informative = _informative_names(roles)
    pre_recall = len(informative & kept_names)
    return dict(secs=secs, resolved=info["method"], kept=info["kept"],
                pre_recall=pre_recall, n_informative=len(informative))


def time_end_to_end(method, X, y, roles, *, prefilter_top, seed=0):
    """Full ``ShapProxiedFS.fit`` wall-clock + END-TO-END informative recall for one prefilter method."""
    import pandas as pd

    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto",
        prefilter_top=prefilter_top, prefilter_method=method,
        cluster_features=True, cluster_corr_threshold=0.7,
        top_n=20, n_splits=4, n_revalidation_models=3, trust_guard=True, n_anchors=24,
        run_importance_ablation=True, within_cluster_refine=True,
        random_state=seed, verbose=False)
    sel._stage_timings = {}
    t0 = time.perf_counter()
    sel.fit(X, pd.Series(y))
    total = time.perf_counter() - t0
    informative = _informative_names(roles)
    selected = set(sel.selected_features_)
    return dict(total=total, prefilter_secs=sel._stage_timings.get("prefilter", 0.0),
                e2e_recall=len(informative & selected), n_selected=len(selected),
                n_informative=len(informative))


def run(widths, *, rows, prefilter_top, methods, end_to_end):
    print(f"=== ShapProxiedFS prefilter speed/quality tradeoff (rows={rows}, prefilter_top={prefilter_top}) ===")
    # ---- Stage-isolated table (always): prefilter wall-clock + pre-filter informative recall. -------
    stage = {}  # (method, width) -> dict
    for w in widths:
        X, y, roles = make_wide(w, n_rows=rows)
        for m in methods:
            r = time_prefilter_stage(m, X, y, roles, prefilter_top=prefilter_top)
            stage[(m, w)] = r
            print(f"  [stage] width={w:>6} method={m:<11} -> {r['secs']:7.3f}s  "
                  f"resolved={r['resolved']:<11} pre_recall={r['pre_recall']}/{r['n_informative']}", flush=True)

    print("\n=== PRE-FILTER STAGE: wall-clock seconds (lower is better) ===")
    _print_grid(stage, widths, methods, key="secs", fmt="{:>9.3f}")
    print("\n=== PRE-FILTER STAGE: informative recall (kept / total; higher is better) ===")
    _print_recall_grid(stage, widths, methods, recall_key="pre_recall")
    # Speedup of each method vs the baseline "model" prefilter, per width.
    if "model" in methods:
        print("\n=== PRE-FILTER STAGE: speedup vs 'model' (x; higher = faster) ===")
        for w in widths:
            base = stage[("model", w)]["secs"]
            cells = "  ".join(f"{m}={base / stage[(m, w)]['secs']:.2f}x" for m in methods if (m, w) in stage)
            print(f"  width={w:>6}: {cells}")

    if not end_to_end:
        print("\n(skipped end-to-end fits; pass --end-to-end for full-pipeline wall-clock + final recall)")
        return

    # ---- End-to-end table: full fit wall-clock + final informative recall. --------------------------
    e2e = {}
    for w in widths:
        X, y, roles = make_wide(w, n_rows=rows)
        for m in methods:
            r = time_end_to_end(m, X, y, roles, prefilter_top=prefilter_top)
            e2e[(m, w)] = r
            print(f"  [e2e]   width={w:>6} method={m:<11} -> total={r['total']:8.2f}s "
                  f"(prefilter {r['prefilter_secs']:6.2f}s)  e2e_recall={r['e2e_recall']}/{r['n_informative']} "
                  f"nsel={r['n_selected']}", flush=True)

    print("\n=== END-TO-END fit: total seconds (lower is better) ===")
    _print_grid(e2e, widths, methods, key="total", fmt="{:>9.2f}")
    print("\n=== END-TO-END fit: final informative recall (kept / total) ===")
    _print_recall_grid(e2e, widths, methods, recall_key="e2e_recall")
    if "model" in methods:
        print("\n=== END-TO-END speedup vs 'model' prefilter (x) ===")
        for w in widths:
            base = e2e[("model", w)]["total"]
            cells = "  ".join(f"{m}={base / e2e[(m, w)]['total']:.2f}x" for m in methods if (m, w) in e2e)
            print(f"  width={w:>6}: {cells}")


def _print_grid(table, widths, methods, *, key, fmt):
    header = f"{'method':<13}" + "".join(f"{w:>10}" for w in widths)
    print(header)
    print("-" * len(header))
    for m in methods:
        row = f"{m:<13}"
        for w in widths:
            row += fmt.format(table[(m, w)][key]) if (m, w) in table else f"{'-':>10}"
        print(row)


def _print_recall_grid(table, widths, methods, *, recall_key):
    header = f"{'method':<13}" + "".join(f"{w:>10}" for w in widths)
    print(header)
    print("-" * len(header))
    for m in methods:
        row = f"{m:<13}"
        for w in widths:
            if (m, w) in table:
                d = table[(m, w)]
                row += f"{str(d[recall_key]) + '/' + str(d['n_informative']):>10}"
            else:
                row += f"{'-':>10}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="2000,5000,10000")
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--prefilter-top", type=int, default=500)
    ap.add_argument("--methods", default="model,univariate,fast_model,gpu_model")
    ap.add_argument("--end-to-end", action="store_true")
    args = ap.parse_args()
    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    run(widths, rows=args.rows, prefilter_top=args.prefilter_top, methods=methods, end_to_end=args.end_to_end)


if __name__ == "__main__":
    main()
