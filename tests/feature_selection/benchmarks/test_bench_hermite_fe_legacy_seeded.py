"""FS_BENCHMARKS_A-2: _legacy_hermite drew length_a/length_b via the unseeded global np.random.randint,
while every other RNG use in the same file (data gen, TPESampler seed=42, optimise_hermite_pair seed=42)
is fully seeded, making the printed 'legacy' baseline MI non-deterministic across runs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("optuna")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.feature_selection._benchmarks.bench_hermite_fe import _legacy_hermite, _make_xor


def test_legacy_hermite_deterministic_across_runs_regardless_of_global_random_state():
    """Two calls with the same seed must produce bit-identical results even when the GLOBAL
    np.random state differs between them -- proving the length draw no longer depends on it."""
    x1, x2, y = _make_xor(n=200, seed=1)

    np.random.seed(0)
    result_a = _legacy_hermite(x1, x2, y, n_iters=2, n_trials_per_iter=5, seed=7)

    np.random.seed(999)  # deliberately different global state before the second call
    result_b = _legacy_hermite(x1, x2, y, n_iters=2, n_trials_per_iter=5, seed=7)

    assert result_a == result_b


def test_legacy_hermite_different_seeds_can_differ():
    """Sanity: the seed argument actually controls the draw (not a no-op constant)."""
    x1, x2, y = _make_xor(n=200, seed=1)
    results = {_legacy_hermite(x1, x2, y, n_iters=2, n_trials_per_iter=5, seed=s) for s in range(5)}
    assert len(results) > 1, "expected at least some variation across different seeds"
