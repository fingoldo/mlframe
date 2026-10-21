"""Paired A/B of the shipped target-mix stage on the processes of ``exp_target_mix``: the same ``MRMR`` with ``fe_target_mix_enable`` off and on.

Prints, per process, the median held-out RMSE of a ridge and a gradient-boosting model on the selected output, the paired change with its t in standard errors, the extra columns, the recovery of the
planted weak term (max |corr| of any output column with it on the held-out rows) and the fit time of both arms.

Usage: ``python -m mlframe.feature_selection._benchmarks.target_lens.exp_target_mix_stage [seeds] [n_train] [processes comma separated]``.
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mlframe.feature_selection._benchmarks.target_lens.exp_target_mix import N_TEST, PROCESSES, _models, _rmse

warnings.simplefilter("ignore")


def one(process: str, n_train: int, seed: int) -> dict:
    """Both arms on one draw."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(1000 + seed)
    X_all, y_all, _truth, weak_all = PROCESSES[process](n_train + N_TEST, rng)
    Xtr, ytr = X_all.iloc[:n_train].reset_index(drop=True), np.asarray(y_all)[:n_train]
    Xte, yte = X_all.iloc[n_train:].reset_index(drop=True), np.asarray(y_all)[n_train:]
    weak_te = None if weak_all is None else np.asarray(weak_all)[n_train:]
    out = {}
    for flag in (False, True):
        t0 = time.perf_counter()
        fs = MRMR(verbose=0, n_jobs=1, random_seed=seed, fe_target_mix_enable=flag, fe_target_mix_max_bases=int(__import__('os').environ.get('MAX_BASES', '2'))).fit(Xtr, pd.Series(ytr, name="y"))
        wall = time.perf_counter() - t0
        tr, te = fs.transform(Xtr), fs.transform(Xte)
        row = {"time": wall, "n_feat": tr.shape[1]}
        sc = StandardScaler().fit(tr)
        for name, make in _models().items():
            row[name] = _rmse(make().fit(sc.transform(tr), ytr).predict(sc.transform(te)), yte)
        row["recovery"] = float("nan") if weak_te is None else max(_cs if (_cs := [abs(np.corrcoef(te[c], weak_te)[0, 1]) for c in te.columns if te[c].std() > 0]) else [0.0])
        out[flag] = row
    return out


def main() -> None:
    """Run and print."""
    import os

    if os.environ.get("COMPOUND_VIEWS_LINEAR"):
        import mlframe.feature_selection.filters._mrmr_fit_impl._residual_rescue_stage as _S

        _S._COMPOUND_VIEWS_LINEAR = True
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    which = sys.argv[3].split(",") if len(sys.argv) > 3 else list(PROCESSES)
    for process in which:
        runs = []
        for s in range(seeds):
            runs.append(one(process, n_train, s))
            print(f"  {process} seed {s} done", flush=True)
        print(f"\n{process}  (n_train={n_train}, {seeds} seeds)")
        for model in ("ridge", "hgb"):
            off = np.array([r[False][model] for r in runs])
            on = np.array([r[True][model] for r in runs])
            d = on - off
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            print(f"  {model:5} off {np.median(off):.3f}  on {np.median(on):.3f}  paired {np.median(d) / np.median(off):+.1%} ({np.median(d) / se if se else float('nan'):+.1f} se)")
        print(f"  extra columns {np.median([r[True]['n_feat'] - r[False]['n_feat'] for r in runs]):.0f};  fit time off {np.median([r[False]['time'] for r in runs]):.1f}s on {np.median([r[True]['time'] for r in runs]):.1f}s;  recovery off {np.median([r[False]['recovery'] for r in runs]):.2f} on {np.median([r[True]['recovery'] for r in runs]):.2f}", flush=True)


if __name__ == "__main__":
    main()
