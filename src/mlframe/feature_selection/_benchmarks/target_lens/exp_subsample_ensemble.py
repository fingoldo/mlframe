"""L-23: a row-subsample ensemble of MRMR as a cheap source of usable columns, against one MRMR on all rows.

Arm A: one MRMR on all training rows. Arm E: K MRMR runs, each on a random subsample of the training rows (fraction ``frac``);
the output is the union of the selected output columns (duplicates by name merged). Both feed the same ridge and HGB; held-out RMSE
on ``N_TEST`` rows, paired over seeds, plus the fit time of each arm.

Usage: ``python -m mlframe.feature_selection._benchmarks.target_lens.exp_subsample_ensemble [seeds] [n_train] [K] [frac]``.
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


def _union_output(selectors, X: pd.DataFrame) -> pd.DataFrame:
    """Column union of the selectors' outputs on ``X``, by column name."""
    frames = [s.transform(X) for s in selectors]
    out = pd.concat(frames, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def one(process: str, n_train: int, seed: int, K: int, frac: float) -> dict:
    """Both arms on one draw."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(2000 + seed)
    X_all, y_all, _truth, _weak = PROCESSES[process](n_train + N_TEST, rng)
    Xtr, ytr = X_all.iloc[:n_train].reset_index(drop=True), np.asarray(y_all)[:n_train]
    Xte, yte = X_all.iloc[n_train:].reset_index(drop=True), np.asarray(y_all)[n_train:]
    out = {}
    t0 = time.perf_counter()
    single = MRMR(verbose=0, n_jobs=1, random_seed=seed).fit(Xtr, pd.Series(ytr, name="y"))
    t_single = time.perf_counter() - t0
    arms = {"A": ([single], t_single)}
    t0 = time.perf_counter()
    sub_rng = np.random.default_rng(seed)
    subs = []
    for k in range(K):
        idx = np.sort(sub_rng.choice(n_train, size=int(frac * n_train), replace=False))
        subs.append(MRMR(verbose=0, n_jobs=1, random_seed=seed + k).fit(Xtr.iloc[idx].reset_index(drop=True), pd.Series(ytr[idx], name="y")))
    arms["E"] = (subs, time.perf_counter() - t0)
    for arm, (sels, t) in arms.items():
        tr, te = _union_output(sels, Xtr), _union_output(sels, Xte)
        row = {"time": t, "n_feat": tr.shape[1]}
        sc = StandardScaler().fit(tr)
        for name, make in _models().items():
            row[name] = _rmse(make().fit(sc.transform(tr), ytr).predict(sc.transform(te)), yte)
        out[arm] = row
    return out


def main() -> None:
    """Run every process and print the paired table."""
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    print(f"n_train={n_train}, {seeds} seeds, K={K}, frac={frac}: arm E (ensemble union) against arm A (one MRMR), held-out RMSE", flush=True)
    for process in PROCESSES:
        runs = [one(process, n_train, s, K, frac) for s in range(seeds)]
        print(f"\n{process}", flush=True)
        for model in _models():
            a = np.array([r["A"][model] for r in runs])
            e = np.array([r["E"][model] for r in runs])
            d = e - a
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            print(f"  {model:5} A {np.median(a):.3f}  E {np.median(e):.3f}  paired {np.median(d) / np.median(a):+.1%} ({np.median(d) / se if se else float('nan'):+.1f} se)", flush=True)
        print(f"  fit time A {np.median([r['A']['time'] for r in runs]):.1f}s  E {np.median([r['E']['time'] for r in runs]):.1f}s;  columns A {np.median([r['A']['n_feat'] for r in runs]):.0f} E {np.median([r['E']['n_feat'] for r in runs]):.0f}", flush=True)


if __name__ == "__main__":
    main()
