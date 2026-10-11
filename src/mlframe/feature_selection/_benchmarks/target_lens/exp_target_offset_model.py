"""Paired OOS check of the final model trained on the target the feature-base mix found (``TargetOffsetRegressor``) against the same model without the offset.

Both arms use the same selector and the same features; arm A has the mix off (no base, zero offset), arm B has it on. Held-out RMSE on ``N_TEST`` rows, paired over seeds, for ridge and HGB.

Usage: ``python -m mlframe.feature_selection._benchmarks.target_lens.exp_target_offset_model [seeds] [n_train] [processes comma separated]``.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

from mlframe.feature_selection._benchmarks.target_lens.exp_target_mix import N_TEST, PROCESSES, _models, _rmse

warnings.simplefilter("ignore")


def one(process: str, n_train: int, seed: int) -> dict:
    """Three arms on one draw: A mix off (plain), B mix on with the offset, C mix on without the offset (features of the mix only); the model names are those of ``exp_target_mix._models``."""
    from mlframe.feature_selection.filters._target_offset_regressor import TargetOffsetRegressor
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(1000 + seed)
    X_all, y_all, _truth, _weak = PROCESSES[process](n_train + N_TEST, rng)
    Xtr, ytr = X_all.iloc[:n_train].reset_index(drop=True), np.asarray(y_all)[:n_train]
    Xte, yte = X_all.iloc[n_train:].reset_index(drop=True), np.asarray(y_all)[n_train:]
    out = {}
    for arm, mix, use_offset in (("A", False, True), ("B", True, True), ("C", True, False)):
        t0 = time.perf_counter()
        sel = MRMR(verbose=0, n_jobs=1, random_seed=seed, fe_target_mix_enable=mix, fe_target_mix_max_bases=int(os.environ.get("MAX_BASES", "1")))
        row = {"time": 0.0}
        for name, make in _models().items():
            m = TargetOffsetRegressor(selector=sel, regressor=make(), use_offset=use_offset).fit(Xtr, ytr)
            row[name] = _rmse(m.predict(Xte), yte)
        row["time"] = time.perf_counter() - t0
        row["offset"] = m.offset_base_ is not None
        out[arm] = row
    return out


def main() -> None:
    """Run every process and print the paired table."""
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    which = sys.argv[3].split(",") if len(sys.argv) > 3 else list(PROCESSES)
    for process in which:
        runs = [one(process, n_train, s) for s in range(seeds)]
        print(f"\n{process}  (n_train={n_train}, {seeds} seeds, offset found in {sum(r['B']['offset'] for r in runs)} of {seeds})")
        for model in _models():
            a = np.array([r["A"][model] for r in runs])
            b = np.array([r["B"][model] for r in runs])
            d = b - a
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            print(f"  {model:5} plain {np.median(a):.3f}  offset {np.median(b):.3f}  paired {np.median(d) / np.median(a):+.1%} ({np.median(d) / se if se else float('nan'):+.1f} se)", flush=True)
        for model in _models():
            a = np.array([r["A"][model] for r in runs])
            c = np.array([r["C"][model] for r in runs])
            d = c - a
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            print(f"  {model:5} features-only (C vs plain) {np.median(d) / np.median(a):+.1%} ({np.median(d) / se if se else float('nan'):+.1f} se)", flush=True)
        print(f"  fit time plain {np.median([r['A']['time'] for r in runs]):.1f}s  offset {np.median([r['B']['time'] for r in runs]):.1f}s", flush=True)


if __name__ == "__main__":
    main()
