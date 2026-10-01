"""Multi-dataset replay bench for the RFECV futility stop (``_futility_stop.futility_verdict``).

Phase 1 (``--collect``): ONE full RFECV search per (dataset, seed, estimator) with the stop OFF and ``max_refits`` iterations; the per-iteration
``(N, fold scores)`` trace (``eval_trace_``) and the held-out OOS score of the subset behind every evaluated N (fresh 1/3 split RFECV never saw) are recorded.
The search path up to any iteration does not depend on the stop, so a stop at iteration t is exactly "the first t iterations of the recorded search".

Phase 2 (``--replay``): for each setting of (min_iters, alpha, patience_frac) walk the trace, call ``futility_verdict`` on every prefix (``remaining`` = iterations
left under ``max_refits``/p, as the live loop computes it) and record the stop iteration; the pick (``one_se_max``, SE band) of the truncated curve is compared with
the pick of the full curve: same N (selection-equivalence), iterations saved, time saved (iteration cost proxied by its feature count N), and dOOS when different.

Run (background; JSON to the git-ignored ``_results/``):
    python -m mlframe.feature_selection.wrappers.rfecv._benchmarks.bench_rfecv_futility_stop --collect [--seeds 6] [--jobs 3] [--max-refits 30]
    python -m mlframe.feature_selection.wrappers.rfecv._benchmarks.bench_rfecv_futility_stop --replay <collect.json>[,<more.json>]

Verdict: see the table in CHANGELOG.md (Unreleased) and the class docstring of ``_futility_stop``; raw numbers are re-derivable with ``--replay``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np

from mlframe.feature_selection.wrappers.rfecv._futility_stop import futility_verdict, winners_from_trace
from mlframe.feature_selection.wrappers.rfecv._benchmarks.bench_rfecv_one_se_rule import DATASETS, ESTIMATORS, load_dataset, make_estimator, oos_score

logger = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).parent / "_results"
GRID = tuple(itertools.product(('full', 'pick'), (3, 5, 7), (0.01, 0.05, 0.2), (0.0, 0.1, 0.25)))  # (anchor, min_iters, alpha, patience_frac)


# Production-like arm: many rows (tiny SE), every column carries signal or is a weak decoy, tree ensemble -- the regime where the full set is the pick.
PROD_DATASETS = ("big_dense_clf", "big_weakdecoy_clf", "big_dense_reg", "big_friedman_reg", "mid_dense_clf")
# Long-horizon arm (p=80, max_refits=50): the only place the patience scaling (window = patience_frac * iterations left) is exercised beyond its floor.
LONG_DATASETS = ("long_dense_clf", "long_weakdecoy_clf", "long_dense_reg")


def _load(name: str):
    """Dataset loader: the production-like arm is generated here, everything else comes from the one-SE bench."""
    import pandas as pd
    from sklearn.datasets import make_classification, make_friedman1, make_regression

    if name == "big_dense_clf":
        X, y = make_classification(20000, 30, n_informative=30, n_redundant=0, n_clusters_per_class=2, class_sep=0.8, random_state=0)
        task = "binary"
    elif name == "big_weakdecoy_clf":
        X, y = make_classification(20000, 40, n_informative=24, n_redundant=0, class_sep=0.7, random_state=0, shuffle=False)
        task = "binary"
    elif name == "big_dense_reg":
        X, y = make_regression(20000, 30, n_informative=30, noise=50.0, random_state=0)
        task = "regression"
    elif name == "big_friedman_reg":
        X, y = make_friedman1(20000, n_features=30, noise=1.0, random_state=0)
        task = "regression"
    elif name == "long_dense_clf":
        X, y = make_classification(8000, 80, n_informative=60, n_redundant=0, class_sep=0.8, random_state=0, shuffle=False)
        task = "binary"
    elif name == "long_weakdecoy_clf":
        X, y = make_classification(8000, 80, n_informative=40, n_redundant=0, class_sep=0.6, random_state=0, shuffle=False)
        task = "binary"
    elif name == "long_dense_reg":
        X, y = make_regression(8000, 80, n_informative=80, noise=50.0, random_state=0)
        task = "regression"
    elif name == "mid_dense_clf":
        X, y = make_classification(5000, 25, n_informative=25, n_redundant=0, n_clusters_per_class=2, class_sep=0.8, random_state=0)
        task = "binary"
    else:
        return load_dataset(name)
    X = np.asarray(X, dtype=float)
    return pd.DataFrame((X - X.mean(0)) / X.std(0), columns=[f"f{i}" for i in range(X.shape[1])]), np.asarray(y), task


def pick_one_se_max(curve: dict) -> int:
    """``one_se_max`` (SE band) pick on ``{N: fold scores}`` -- the largest N whose mean >= best mean - std/sqrt(k) of the best N."""
    means = {n: float(np.nanmean(a)) for n, a in curve.items() if n > 0}
    best = max(means, key=lambda n: (means[n], n))
    a = curve[best]
    thr = means[best] - float(np.nanstd(a)) / np.sqrt(max(int(np.isfinite(a).sum()), 1))
    return int(max(n for n, m in means.items() if m >= thr))


def collect_one(dataset: str, seed: int, est_kind: str, max_refits: int) -> dict:
    """Run one un-stopped RFECV and record its trace plus the OOS score of every evaluated subset size."""
    from sklearn.model_selection import train_test_split

    from mlframe.feature_selection.wrappers import RFECV

    X, y, task = _load(dataset)
    strat = y if task != "regression" else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=seed, stratify=strat)
    t0 = time.process_time()
    sel = RFECV(estimator=make_estimator(est_kind, task, seed), n_features_selection_rule="one_se_max", cv=3, random_state=seed, max_refits=max_refits, verbose=0)
    sel.fit(Xtr, ytr)
    fit_s = time.process_time() - t0
    trace = [(int(n), [float(v) for v in s]) for n, s in sel.eval_trace_]
    oos = {}
    for n in {n for n, _ in trace}:
        kept = list(sel.selected_features_.get(n, [])) or list(X.columns)
        est = make_estimator(est_kind, task, seed)
        est.fit(Xtr[kept], ytr)
        oos[str(n)] = oos_score(est, Xte[kept], yte, task)
    curve, _ = winners_from_trace(trace)
    return {"dataset": dataset, "seed": seed, "est": est_kind, "p": int(X.shape[1]), "max_refits": max_refits, "trace": trace, "oos": oos, "fit_s": fit_s,
            "n_kept_live": int(sel.n_features_), "pick_replayed": pick_one_se_max(curve)}


def _job(args, sink: str):
    """Run one unit and append its JSON line to ``sink`` at once, so a partial sweep is always replayable."""
    try:
        rec = collect_one(*args)
    except Exception as exc:  # the sweep stays resilient; failures are reported, never silently dropped
        logger.warning("bench job %s failed: %s: %s", args[:3], type(exc).__name__, exc)
        rec = {"dataset": args[0], "seed": args[1], "est": args[2], "error": f"{type(exc).__name__}: {exc}"}
    with open(sink, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def load_units(paths: str) -> list:
    """Units from comma-separated ``.json`` (list) and ``.jsonl`` (one unit per line) result files."""
    units: list = []
    for f in paths.split(","):
        txt = Path(f).read_text()
        units += [json.loads(ln) for ln in txt.splitlines() if ln.strip()] if f.endswith(".jsonl") else json.loads(txt)
    return units


def replay_unit(u: dict, anchor: str, min_iters: int, alpha: float, patience_frac: float) -> dict:
    """Stop iteration, pick and savings of one recorded search under one setting."""
    trace = [(n, s) for n, s in u["trace"]]
    T, p = len(trace), u["p"]
    stop_t = T
    for t in range(2, T + 1):
        rem = max(min(p - t, u["max_refits"] - t), 0)
        if futility_verdict(trace[:t], min_iters=min_iters, alpha=alpha, patience_frac=patience_frac, remaining=rem, full_n=p, anchor=anchor).stop:
            stop_t = t
            break
    full_pick = pick_one_se_max(winners_from_trace(trace)[0])
    cut_pick = pick_one_se_max(winners_from_trace(trace[:stop_t])[0])
    cost = np.array([n for n, _ in trace], dtype=float)
    oos = u["oos"]
    return {
        "stopped": stop_t < T, "iters_saved": (T - stop_t) / T, "time_saved": float(cost[stop_t:].sum() / cost.sum()), "same_n": cut_pick == full_pick,
        "d_oos": oos[str(cut_pick)] - oos[str(full_pick)], "full_pick": full_pick, "cut_pick": cut_pick, "T": T,
    }


def summarize(units: list) -> str:
    """Per-setting table over all units plus the per-unit detail of the non-equivalent cases of the recommended setting."""
    units = [u for u in units if "error" not in u]
    out = [
        (
            f"units: {len(units)}; mean trace length {np.mean([len(u['trace']) for u in units]):.1f}; live-vs-replayed pick agreement "
            f"{np.mean([u['n_kept_live'] == u['pick_replayed'] for u in units]):.3f}"
        )
    ]
    out.append("anchor min_iters alpha patience | stop_rate  iters_saved(all)  time_saved(all)  same_N  d_OOS(all)  n_diff  d_OOS(diff only)")
    for an, mi, al, pf in GRID:
        r = [replay_unit(u, an, mi, al, pf) for u in units]
        diff = [x for x in r if not x["same_n"]]
        out.append(
            f"{an:>6s} {mi:>9d} {al:>5.2f} {pf:>8.2f} | {np.mean([x['stopped'] for x in r]):8.3f}  {np.mean([x['iters_saved'] for x in r]):16.3f}  "
            f"{np.mean([x['time_saved'] for x in r]):15.3f}  {np.mean([x['same_n'] for x in r]):6.3f}  {np.mean([x['d_oos'] for x in r]):+10.5f}  {len(diff):6d}  "
            f"{(np.mean([x['d_oos'] for x in diff]) if diff else 0.0):+10.5f}"
        )
    return "\n".join(out)


def breakdown(units: list, anchor: str = "full", min_iters: int = 5, alpha: float = 0.05, patience_frac: float = 0.1) -> str:
    """Per (dataset, estimator) stop rate / savings / equivalence of one setting, to see WHERE the stop fires."""
    import collections

    groups: dict = collections.defaultdict(list)
    for u in units:
        if "error" not in u:
            groups[(u["dataset"], u["est"])].append(replay_unit(u, anchor, min_iters, alpha, patience_frac))
    out = [f"setting anchor={anchor} min_iters={min_iters} alpha={alpha} patience_frac={patience_frac}", "dataset              est        n  stop  iters_saved  same_N  pick==p"]
    for (d, e), r in sorted(groups.items()):
        out.append(
            f"{d:20s} {e:9s} {len(r):2d}  {np.mean([x['stopped'] for x in r]):4.2f}  {np.mean([x['iters_saved'] for x in r]):11.3f}  "
            f"{np.mean([x['same_n'] for x in r]):6.3f}  {np.mean([x['full_pick'] == u['p'] for x, u in zip(r, [q for q in units if (q['dataset'], q.get('est')) == (d, e) and 'error' not in q])]):6.2f}"
        )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--replay", type=str, default=None)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--max-refits", type=int, default=30)
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--arm", choices=("small", "prod", "long"), default="small", help="'prod': many-row dense datasets where the full set tends to be the pick")
    a = ap.parse_args()
    if a.replay:
        units = load_units(a.replay)
        print(summarize(units))
        print()
        print(breakdown(units))
        return
    from joblib import Parallel, delayed

    datasets: tuple = DATASETS
    ests: tuple = ESTIMATORS
    seeds = range(a.seeds)
    if a.fast:
        datasets, seeds, ests = DATASETS[:2], range(2), ("lightgbm",)
    if a.arm == "prod":
        datasets, ests = PROD_DATASETS, ("lightgbm", "linear")
    elif a.arm == "long":
        datasets, ests = LONG_DATASETS, ("lightgbm", "linear")
    if a.datasets:
        datasets = tuple(a.datasets.split(","))
    jobs = [(d, s, e, a.max_refits) for s, d, e in itertools.product(seeds, datasets, ests)]  # seed-major: any prefix covers every dataset
    t0 = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"rfecv_futility_collect_{a.arm}_{time.strftime('%Y%m%d_%H%M%S')}{'_fast' if a.fast else ''}.jsonl"
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(_job)(j, str(out)) for j in jobs)
    print("wrote", out, f"wall={time.time() - t0:.0f}s")
    print(summarize(res))


if __name__ == "__main__":
    main()
