"""Permutation filling and the permutation cache's byte budget.

The fill is checked as behaviour: each row is a permutation of ``range(n)``, not the identity, and reproducible for a base seed. The byte budget is read when the module is imported,
so it is checked in a fresh interpreter with the environment set, not by reloading the module in this process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import mlframe.feature_selection.filters._family_maxt_gate as gate


def test_each_filled_row_is_a_permutation_and_not_the_identity():
    """Every row holds each index exactly once, and the shuffle moved it away from the identity."""
    n = 40
    idx = np.empty((5, n), dtype=np.int64)
    gate._fill_permutations(idx, 1234)
    for r in range(idx.shape[0]):
        np.testing.assert_array_equal(np.sort(idx[r]), np.arange(n))
        assert not np.array_equal(idx[r], np.arange(n))


def test_the_fill_is_uniform_over_the_permutations_of_four():
    """Over 24000 rows of four, each of the 24 permutations appears about 1000 times: a shuffle that skips a swap biases the counts."""
    n = 4
    rows = 24000
    idx = np.empty((rows, n), dtype=np.int64)
    gate._fill_permutations(idx, 99)
    keys = idx @ (n ** np.arange(n))
    counts = np.bincount(keys, minlength=n**n)
    perms = [p for p in range(n**n) if sorted(np.array([(p // n**k) % n for k in range(n)]).tolist()) == list(range(n))]
    observed = counts[perms]
    assert len(perms) == 24
    assert observed.min() > 0.85 * 1000 and observed.max() < 1.15 * 1000, observed


def test_the_last_position_is_not_fixed_by_the_shuffle():
    """The last index moves: across rows it lands on every position, not only its own."""
    n = 6
    idx = np.empty((600, n), dtype=np.int64)
    gate._fill_permutations(idx, 5)
    last_values = np.unique(idx[:, n - 1])
    assert last_values.size == n


def test_the_fill_is_reproducible_for_a_base_seed_and_differs_across_seeds():
    """The same base seed fills the same rows; another seed fills different ones."""
    a = np.empty((3, 30), dtype=np.int64)
    b = np.empty((3, 30), dtype=np.int64)
    c = np.empty((3, 30), dtype=np.int64)
    gate._fill_permutations(a, 7)
    gate._fill_permutations(b, 7)
    gate._fill_permutations(c, 8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


_PROBE = "import mlframe.feature_selection.filters._family_maxt_gate as g; print(g._PERM_BLOCK_CACHE_BYTES, g._PERM_BLOCK_BUDGET_BYTES)"


def _probe(env_overrides: dict) -> tuple[int, int]:
    """The cache and block budgets (bytes) of a fresh interpreter started with *env_overrides* applied."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLFRAME_PERM_BLOCK_")}
    env.update(env_overrides)
    # The probe must import the same source tree as this process (a mutated copy included), not whatever PYTHONPATH names.
    src_root = str(Path(gate.__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join([src_root, env.get("PYTHONPATH", "")])
    out = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, check=True, timeout=600)
    cache, block = out.stdout.split()
    return int(cache), int(block)


def test_default_budgets_are_1024_and_192_mib():
    """With no overrides the cache holds 1024 MiB and a permutation block 192 MiB."""
    cache, block = _probe({})
    assert cache == 1024 * 1024 * 1024
    assert block == 192 * 1024 * 1024


def test_a_zero_cache_budget_is_honoured_and_not_raised_to_one_megabyte():
    """``MLFRAME_PERM_BLOCK_CACHE_MB=0`` turns the cache off: the budget is zero, not the one-megabyte minimum of the block budget."""
    cache, _ = _probe({"MLFRAME_PERM_BLOCK_CACHE_MB": "0"})
    assert cache == 0


def test_the_cache_budget_follows_its_environment_variable():
    """An explicit ``MLFRAME_PERM_BLOCK_CACHE_MB`` sets the cache budget in MiB, and the block budget is unaffected."""
    cache, block = _probe({"MLFRAME_PERM_BLOCK_CACHE_MB": "64"})
    assert cache == 64 * 1024 * 1024
    assert block == 192 * 1024 * 1024
