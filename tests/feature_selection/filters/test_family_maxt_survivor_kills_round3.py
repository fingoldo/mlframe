"""Behavioural pins for the family max-T gate's plug-in MI kernel and the block budget defaults, which the mutation sweep found unpinned.

The MI kernel must skip a code with no rows without ending the scan over the codes after it. The block budget's env var name and its
default are the contract a deployment sets against, so they are checked in a child interpreter: the value is read once at import.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import mlframe.feature_selection.filters._family_maxt_gate as gate


def _reference_mi(x, n_x, y, n_y):
    """Plug-in mutual information in nats, computed from a dense joint table."""
    n = len(x)
    joint = np.zeros((n_x, n_y))
    for a, b in zip(x, y):
        joint[a, b] += 1
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)
    mi = 0.0
    for a in range(n_x):
        for b in range(n_y):
            if joint[a, b] > 0:
                mi += joint[a, b] / n * math.log(joint[a, b] * n / (pa[a] * pb[b]))
    return mi


def test_mi_one_column_into_keeps_scanning_past_an_unused_middle_code():
    """A code with no rows is skipped, and the codes after it still contribute to the MI."""
    x = np.array([0, 0, 2, 2, 2, 0, 2, 0], dtype=np.int64)  # code 1 is unused
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0], dtype=np.int64)
    joint = np.zeros(3 * 2, dtype=np.int64)
    got = gate._mi_one_column_into(x, 3, y, 2, joint)
    assert got == pytest.approx(_reference_mi(x, 3, y, 2), rel=1e-12)
    assert got > 0.0


def test_mi_one_column_into_zeroes_a_dirty_buffer_before_counting():
    """The caller's buffer may hold counts from a previous column; the kernel must clear it first."""
    x = np.array([0, 1, 1, 0, 1], dtype=np.int64)
    y = np.array([1, 0, 0, 1, 1], dtype=np.int64)
    dirty = np.full(2 * 2, 99, dtype=np.int64)
    clean = np.zeros(2 * 2, dtype=np.int64)
    assert gate._mi_one_column_into(x, 2, y, 2, dirty) == gate._mi_one_column_into(x, 2, y, 2, clean)


_PROBE = "from mlframe.feature_selection.filters import _family_maxt_gate as g; print(g.{name})"


def _child_value(name, env_name, env_value):
    """Import the gate in a child interpreter with ``env_name`` set (or unset when ``env_value`` is None) and return the named constant."""
    env = {k: v for k, v in os.environ.items() if k != env_name}
    if env_value is not None:
        env[env_name] = env_value
    env["PYTHONPATH"] = str(Path(gate.__file__).resolve().parents[3])
    out = subprocess.run([sys.executable, "-c", _PROBE.format(name=name)], env=env, capture_output=True, text=True, check=True, timeout=600)
    return int(out.stdout.strip().splitlines()[-1])


def test_block_budget_defaults_to_192_mib():
    """Without the env var one block of permutation indices may take 192 MiB."""
    assert _child_value("_PERM_BLOCK_BUDGET_BYTES", "MLFRAME_PERM_BLOCK_BUDGET_MB", None) == 192 * 1024 * 1024


@pytest.mark.parametrize("mb", ["5", "1"])
def test_block_budget_env_var_is_named_and_read_in_mib(mb):
    """The budget is read from ``MLFRAME_PERM_BLOCK_BUDGET_MB`` as MiB, so a deployment's setting takes effect."""
    assert _child_value("_PERM_BLOCK_BUDGET_BYTES", "MLFRAME_PERM_BLOCK_BUDGET_MB", mb) == int(mb) * 1024 * 1024


def test_large_n_cache_default_is_384_mib():
    """Beyond the full-cache row count the cache holds at most 384 MiB of permutation indices."""
    assert _child_value("_PERM_CACHE_LARGE_N_BYTES", "UNUSED_ENV", None) == 384 * 1024 * 1024


def test_row_count_threshold_switches_to_the_large_n_cap(monkeypatch):
    """At exactly 200000 rows the full byte cap applies and two blocks stay cached; one row more drops to the large-n cap and keeps one."""
    from collections import OrderedDict

    monkeypatch.setattr(gate, "_PERM_BLOCK_CACHE_BYTES", 1_700_000)
    monkeypatch.setattr(gate, "_PERM_CACHE_LARGE_N_BYTES", 1_000_000)

    def _cached_blocks(n):
        """Number of blocks the cache keeps after two single-block draws over ``n`` rows, starting from an empty cache."""
        monkeypatch.setattr(gate, "_perm_block_cache", OrderedDict())
        monkeypatch.setattr(gate, "_perm_block_cache_bytes", 0)
        y = np.zeros(n, dtype=np.int64)
        for seed in (1, 2):
            gate._permutation_block(np.random.default_rng(seed), y, 1)
        return len(gate._perm_block_cache)

    assert _cached_blocks(200_000) == 2
    assert _cached_blocks(200_001) == 1


def _family_inputs(seed=3, k=4, n=60, n_x_max=4, n_y=3):
    """A small random family of ``k`` columns with ``n_x_max`` codes each, and ``n_y`` label classes."""
    rng = np.random.default_rng(seed)
    cols = rng.integers(0, n_x_max, (k, n)).astype(np.int64)
    n_x = np.full(k, n_x_max, dtype=np.int64)
    y = rng.integers(0, n_y, n).astype(np.int64)
    return cols, n_x, y, n_y


@pytest.mark.parametrize("block", [3, 7])
def test_family_null_max_splits_the_stream_into_blocks_in_order(block):
    """Each block draws its own seed, so a block split is the concatenation of separate calls that share one generator, in the same chunk sizes."""
    cols, n_x, y, n_y = _family_inputs()
    n_perm = 10
    chunks = [min(block, n_perm - s) for s in range(0, n_perm, block)]
    rng_ref = np.random.default_rng(11)
    reference = np.concatenate([gate.family_null_max(cols, n_x, y, n_y, m, rng_ref, block=m) for m in chunks])
    split = gate.family_null_max(cols, n_x, y, n_y, n_perm, np.random.default_rng(11), block=block)
    assert split.shape == (n_perm,)
    np.testing.assert_array_equal(split, reference)


def test_family_bit_prep_reuses_the_cached_bit_sets_on_the_host_path():
    """A caller that scores the same columns again gets the bit sets from its cache, not a rebuild."""
    cols, n_x, y, n_y = _family_inputs()
    cache: dict = {}
    first, dev_first = gate._family_bit_prep(cols, n_x, y, n_y, False, cache)
    second, dev_second = gate._family_bit_prep(cols, n_x, y, n_y, False, cache)
    assert first is not None and dev_first is None and dev_second is None
    assert second is first
    assert cache["bit_prep"] is first


def test_family_bit_prep_declines_when_the_labels_have_too_many_classes():
    """Above the class limit the family keeps the histogram kernels: no bit sets and no device copy."""
    cols, n_x, y, _ = _family_inputs()
    too_many = gate.BITSET_MAX_CLASSES + 1
    assert gate._family_bit_prep(cols, n_x, y, too_many, False, None) == (None, None)


def test_block_cache_starts_with_no_bytes_counted():
    """A fresh process counts no cached bytes, so the first block is admitted against the full budget."""
    assert _child_value("_perm_block_cache_bytes", "UNUSED_ENV", None) == 0
