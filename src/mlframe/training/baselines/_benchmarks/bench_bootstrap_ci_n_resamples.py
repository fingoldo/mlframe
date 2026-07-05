"""Bench: bootstrap-CI n_resamples default for the strongest-baseline CI.

Lever: ``DummyBaselinesConfig.bootstrap_ci_n_resamples`` (default 1000),
used by ``_compute_bootstrap_ci`` to report a percentile 95% CI (2.5/97.5)
on the strongest baseline's primary metric, fired only at small n
(min(n_val, n_test) < bootstrap_ci_threshold=2000).

Question: is 1000 over- or under-kill? A percentile-bootstrap CI bound is
itself a Monte-Carlo estimate; its only n_resamples-dependent error is the
MC jitter of the 2.5/97.5 percentile of the resample distribution (the
DATA-sampling error is fixed by n, independent of n_resamples). So the
honest metric is: how much does the REPORTED (lo, hi) move when only the
bootstrap seed changes, at each n_resamples? Plus bias vs a 50k-resample
reference (the n_resamples->inf bound).

HONEST metric per (scenario, data-seed):
  * reference (lo*, hi*) = percentile bound at B=50000 resamples.
  * for each candidate B in {200,500,1000,2000}: over 40 bootstrap-seeds,
    measure MC std of lo and hi (jitter) and |mean(lo)-lo*| bias, expressed
    as a FRACTION of the CI half-width (hi*-lo*)/2 -- a scale-free error.

Decision rule: a challenger B flips the default only if it wins the MAJORITY
of (scenario x data-seed) cells on the honest metric AND the win is
material (lower B is preferred at a tie since it is cheaper).

Run:
  python src/mlframe/training/baselines/_benchmarks/bench_bootstrap_ci_n_resamples.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MLFRAME_NO_CUDA_AUTOCONFIG", "1")
os.environ.setdefault("MLFRAME_KEEP_BROKEN_CUPY", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

import numpy as np  # noqa: E402

from mlframe.training.baselines._dummy_bootstrap import (  # noqa: E402
    _numba_bootstrap_rmse_samples,
    _numba_bootstrap_logloss_binary_samples,
)

CANDIDATES = (200, 500, 1000, 2000)
REFERENCE_B = 50_000
N_SEEDS = 40


def _ci_rmse(y, p, B, seed):
    s = _numba_bootstrap_rmse_samples(
        np.ascontiguousarray(y, np.float64),
        np.ascontiguousarray(p, np.float64),
        int(B), int(seed),
    )
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def _ci_logloss(y, p, B, seed):
    s = _numba_bootstrap_logloss_binary_samples(
        np.ascontiguousarray(y, np.int64),
        np.ascontiguousarray(p, np.float64),
        int(B), int(seed),
    )
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def _make_scenario(name, n, dseed):
    rng = np.random.default_rng(dseed)
    if name == "rmse_small":
        y = rng.normal(0, 1, n)
        p = y + rng.normal(0, 0.7, n)
        return "rmse", y, p
    if name == "rmse_heavytail":
        y = rng.standard_t(3, n)
        p = y + rng.standard_t(3, n) * 0.6
        return "rmse", y, p
    if name == "logloss_balanced":
        y = rng.integers(0, 2, n)
        base = rng.random(n)
        p = np.clip(0.5 + (y - 0.5) * 0.5 + (base - 0.5) * 0.4, 1e-3, 1 - 1e-3)
        return "logloss", y, p
    if name == "logloss_imbalanced":
        y = (rng.random(n) < 0.12).astype(np.int64)
        p = np.clip(0.12 + (y - 0.12) * 0.4 + rng.normal(0, 0.15, n), 1e-3, 1 - 1e-3)
        return "logloss", y, p
    if name == "rmse_tiny":
        y = rng.normal(0, 1, n)
        p = y + rng.normal(0, 0.9, n)
        return "rmse", y, p
    raise ValueError(name)


SCENARIOS = [
    ("rmse_small", 800),
    ("rmse_heavytail", 1200),
    ("logloss_balanced", 600),
    ("logloss_imbalanced", 1500),
    ("rmse_tiny", 120),
]


def main():
    # warm JIT
    _ci_rmse(np.zeros(20), np.ones(20), 50, 0)
    _ci_logloss(np.zeros(20, int), np.full(20, 0.5), 50, 0)

    # per-candidate aggregate of scale-free total error (jitter+bias)
    cell_winner_counts = {B: 0 for B in CANDIDATES}
    rows = []
    total_cells = 0

    for sc_name, n in SCENARIOS:
        for dseed in (1, 2, 3):
            kind, y, p = _make_scenario(sc_name, n, dseed)
            ci = _ci_rmse if kind == "rmse" else _ci_logloss
            lo_ref, hi_ref = ci(y, p, REFERENCE_B, 777)
            half = max((hi_ref - lo_ref) / 2.0, 1e-9)

            cand_err = {}
            for B in CANDIDATES:
                los = np.empty(N_SEEDS)
                his = np.empty(N_SEEDS)
                for k in range(N_SEEDS):
                    los[k], his[k] = ci(y, p, B, 1000 + k)
                jitter = (los.std() + his.std()) / 2.0
                bias = (abs(los.mean() - lo_ref) + abs(his.mean() - hi_ref)) / 2.0
                # total scale-free error (RMS of jitter & bias) / half-width
                err = float(np.sqrt(jitter**2 + bias**2) / half)
                cand_err[B] = err
                rows.append((sc_name, dseed, B, round(err, 5), round(jitter / half, 5), round(bias / half, 5)))

            # the default 1000 is the incumbent; a challenger "wins" the cell
            # only if its error is materially (>=10% relative) lower than 1000.
            best_B = min(cand_err, key=cand_err.get)
            cell_winner_counts[best_B] += 1
            total_cells += 1

    print("scenario              dseed   B     err     jitter   bias")
    for r in rows:
        print(f"{r[0]:<20} {r[1]:>3}  {r[2]:>5}  {r[3]:<7} {r[4]:<7} {r[5]}")

    print("\nPer-candidate mean scale-free error (lower=better):")
    by_B = {B: [] for B in CANDIDATES}
    for r in rows:
        by_B[r[2]].append(r[3])
    for B in CANDIDATES:
        print(f"  B={B:>5}: mean_err={np.mean(by_B[B]):.5f}  max_err={np.max(by_B[B]):.5f}")

    print(f"\nCell winners (lowest err per cell, {total_cells} cells):")
    for B in CANDIDATES:
        print(f"  B={B:>5}: {cell_winner_counts[B]} cells")

    # Decision: does a challenger beat the incumbent 1000 in the MAJORITY of cells?
    inc = 1000
    chal_wins = {B: 0 for B in CANDIDATES if B != inc}
    for r in rows:
        pass
    # rebuild per-cell err to compare each challenger vs incumbent directly
    per_cell = {}
    for r in rows:
        per_cell.setdefault((r[0], r[1]), {})[r[2]] = r[3]
    for B in chal_wins:
        for cell, errs in per_cell.items():
            if errs[B] < errs[inc] * 0.90:  # material 10% better
                chal_wins[B] += 1
    print(f"\nChallengers materially (>=10%) better than incumbent B={inc}, per cell (of {total_cells}):")
    for B, w in chal_wins.items():
        print(f"  B={B:>5}: {w} cells")

    maj = total_cells // 2 + 1
    flips = [B for B, w in chal_wins.items() if w >= maj]
    print(f"\nMajority threshold = {maj}. Flip candidates: {flips or 'NONE -> KEEP 1000'}")


if __name__ == "__main__":
    main()
