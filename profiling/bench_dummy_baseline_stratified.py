"""Bench: stratified dummy-baseline accumulator (iter117, 2026-05-21).

Compares the pre-refactor per-rep-zeros-list + np.mean implementation against
the shipped accumulator (one (N, K) zeros + per-rep +=1.0 + final divide).
At 100k val + 100k test rows / 3 classes / 40 repeats:

    old : ~520 ms (allocates ~200 MB of throwaway (N, K) one-hot arrays)
    new : ~390 ms (one (N, K) accumulator + per-rep fancy-index increments)
    speedup ~1.3x; bit-identical output for the same seed.

Run: ``python profiling/bench_dummy_baseline_stratified.py``
"""

import time
import numpy as np


def old_strat(n_val, n_test, n_classes, n_repeats, seed, train_prior):
    classes = np.arange(n_classes)
    val_strat_runs = []
    test_strat_runs = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        if n_val > 0:
            val_classes = rng.choice(classes, size=n_val, p=train_prior)
            val_strat = np.zeros((n_val, n_classes))
            val_strat[np.arange(n_val), val_classes] = 1.0
            val_strat_runs.append(val_strat)
        if n_test > 0:
            test_classes = rng.choice(classes, size=n_test, p=train_prior)
            test_strat = np.zeros((n_test, n_classes))
            test_strat[np.arange(n_test), test_classes] = 1.0
            test_strat_runs.append(test_strat)
    val_mean = np.mean(val_strat_runs, axis=0) if val_strat_runs else None
    test_mean = np.mean(test_strat_runs, axis=0) if test_strat_runs else None
    return val_mean, test_mean


def new_strat(n_val, n_test, n_classes, n_repeats, seed, train_prior):
    classes = np.arange(n_classes)
    val_acc = np.zeros((n_val, n_classes)) if n_val > 0 else None
    test_acc = np.zeros((n_test, n_classes)) if n_test > 0 else None
    val_row_idx = np.arange(n_val) if n_val > 0 else None
    test_row_idx = np.arange(n_test) if n_test > 0 else None
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        if val_acc is not None:
            val_classes = rng.choice(classes, size=n_val, p=train_prior)
            val_acc[val_row_idx, val_classes] += 1.0
        if test_acc is not None:
            test_classes = rng.choice(classes, size=n_test, p=train_prior)
            test_acc[test_row_idx, test_classes] += 1.0
    val_mean = val_acc / n_repeats if val_acc is not None else None
    test_mean = test_acc / n_repeats if test_acc is not None else None
    return val_mean, test_mean


train_prior = np.array([0.33, 0.34, 0.33])
n_val = n_test = 100_000
n_classes = 3
n_repeats = 40

for name, fn in (("old", old_strat), ("new", new_strat)):
    for _ in range(3):
        t = time.perf_counter()
        v, t2 = fn(n_val, n_test, n_classes, n_repeats, 42, train_prior)
        print(f'{name:>4}: {(time.perf_counter()-t)*1000:7.1f}ms  val_shape={v.shape}')

# Correctness: outputs should be identical for same seed
v_old, t_old = old_strat(n_val, n_test, n_classes, n_repeats, 42, train_prior)
v_new, t_new = new_strat(n_val, n_test, n_classes, n_repeats, 42, train_prior)
print(f'val identical: {np.array_equal(v_old, v_new)}, max diff: {np.abs(v_old - v_new).max()}')
print(f'test identical: {np.array_equal(t_old, t_new)}')
