"""Behavioural pins for the ``_family_maxt_gate`` paths that the mutation sweep found unpinned.

Each test fails for a specific class of mutant: block slicing across several permutation blocks, the device dispatch decision, the
cache release, the quantile-code block layout and fallback, and the exact early stop of the permutation test.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _family_maxt_gate as gate


def _family(seed: int, k: int = 4, n: int = 60, n_x_max: int = 5, n_y: int = 3):
    """Random small family: candidate codes, bin counts, labels and the label class count."""
    rng = np.random.default_rng(seed)
    cols = rng.integers(0, n_x_max, size=(k, n)).astype(np.int16)
    n_x = np.array([int(cols[j].max()) + 1 for j in range(k)], dtype=np.int64)
    y = rng.integers(0, n_y, size=n).astype(np.int64)
    return cols, n_x, y, n_y


def _sequential_reference(cols, n_x, y, n_y, sizes, seed):
    """The null maxima of consecutive single-block calls sharing one generator: the value a block-split call must reproduce."""
    rng = np.random.default_rng(seed)
    parts = [gate.family_null_max(cols, n_x, y, n_y, m, rng, block=256) for m in sizes]
    return np.concatenate(parts)


@pytest.mark.parametrize("n_perm, block", [(10, 3), (7, 2), (9, 4)])
def test_family_null_max_split_into_blocks_matches_sequential_blocks(n_perm, block):
    """A permutation count larger than the block is served block by block; each block continues the same generator stream.

    Kills slice-bound and argument-order mutants in the block loop of ``family_null_max``: any wrong slice writes a block to the wrong rows.
    """
    cols, n_x, y, n_y = _family(seed=11)
    sizes = [min(block, n_perm - s) for s in range(0, n_perm, block)]
    expected = _sequential_reference(cols, n_x, y, n_y, sizes, seed=5)
    got = gate.family_null_max(cols, n_x, y, n_y, n_perm, np.random.default_rng(5), block=block)
    np.testing.assert_array_equal(got, expected)


def test_family_null_max_fills_every_row_of_a_multi_block_null():
    """No row of the null vector is left unwritten when the permutations span several blocks."""
    cols, n_x, y, n_y = _family(seed=12)
    out = gate.family_null_max(cols, n_x, y, n_y, 11, np.random.default_rng(3), block=4)
    assert out.shape == (11,)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


def test_block_budget_narrows_the_block_below_the_row_limit():
    """The block is cut to the byte budget: 2 bytes a row up to 65535 rows, 4 bytes above, never below one permutation."""
    n_small = 1000
    assert gate._budgeted_perm_block(256, n_small) == min(256, gate._PERM_BLOCK_BUDGET_BYTES // (2 * n_small))
    assert gate._budgeted_perm_block(10**9, 10) == min(10**9, gate._PERM_BLOCK_BUDGET_BYTES // 20)
    assert gate._budgeted_perm_block(256, 10**12) == 1


def test_device_dispatch_passes_work_as_permutations_times_columns_times_rows(monkeypatch):
    """``_null_device_wanted`` asks the per-host backend about ``(rows, columns, n_perm * columns * rows)`` in that order.

    Distinct sizes make a transposed argument or a wrong operator visible in the recorded call.
    """
    seen = {}

    def _record(n_rows, n_cols, work):
        """Record the arguments of the dispatch call and report the host backend."""
        seen["args"] = (n_rows, n_cols, work)
        return "host"

    monkeypatch.setattr(gate, "_family_null_backend", _record)
    monkeypatch.setattr("mlframe.feature_selection.filters._gpu_policy.gpu_globally_disabled", lambda: False)
    monkeypatch.setattr("mlframe.feature_selection.filters._fe_gpu_strict.fe_gpu_strict_enabled", lambda: False)
    cols = np.zeros((3, 7), dtype=np.int16)
    y = np.zeros(7, dtype=np.int64)
    gate._null_device_wanted(cols, y, 5)
    assert seen["args"] == (7, 3, 5 * 3 * 7)


@pytest.mark.parametrize("backend, expected", [("device", True), ("host", False)])
def test_device_dispatch_follows_the_backend_choice(monkeypatch, backend, expected):
    """The device decision is the backend table's answer when the GPU is not disabled and strict mode is off."""
    monkeypatch.setattr(gate, "_family_null_backend", lambda *_: backend)
    monkeypatch.setattr("mlframe.feature_selection.filters._gpu_policy.gpu_globally_disabled", lambda: False)
    monkeypatch.setattr("mlframe.feature_selection.filters._fe_gpu_strict.fe_gpu_strict_enabled", lambda: False)
    assert gate._null_device_wanted(np.zeros((2, 5), dtype=np.int16), np.zeros(5, dtype=np.int64), 4) is expected


def test_global_gpu_disable_wins_over_strict_mode(monkeypatch):
    """MLFRAME_DISABLE_GPU wins over STRICT: the null stays on the host even when strict mode asks for the device."""
    monkeypatch.setattr("mlframe.feature_selection.filters._gpu_policy.gpu_globally_disabled", lambda: True)
    monkeypatch.setattr("mlframe.feature_selection.filters._fe_gpu_strict.fe_gpu_strict_enabled", lambda: True)
    assert gate._null_device_wanted(np.zeros((2, 5), dtype=np.int16), np.zeros(5, dtype=np.int64), 4) is False


def test_strict_mode_forces_the_device_when_not_disabled(monkeypatch):
    """Strict mode puts the null on the device without consulting the backend table."""
    monkeypatch.setattr("mlframe.feature_selection.filters._gpu_policy.gpu_globally_disabled", lambda: False)
    monkeypatch.setattr("mlframe.feature_selection.filters._fe_gpu_strict.fe_gpu_strict_enabled", lambda: True)
    monkeypatch.setattr(gate, "_family_null_backend", lambda *_: pytest.fail("strict mode must not consult the backend table"))
    assert gate._null_device_wanted(np.zeros((2, 5), dtype=np.int16), np.zeros(5, dtype=np.int64), 4) is True


def test_release_empties_the_permutation_block_cache_and_reports_the_bytes():
    """Releasing the cache drops every cached block and returns the bytes it held."""
    gate._release_perm_block_cache()
    rng = np.random.default_rng(1)
    y = np.arange(50, dtype=np.int64) % 3
    gate._permutation_block(rng, y, 4)
    assert len(gate._perm_block_cache) >= 1
    held = gate._perm_block_cache_bytes
    assert held > 0
    freed = gate._release_perm_block_cache()
    assert freed == held
    assert len(gate._perm_block_cache) == 0
    assert gate._perm_block_cache_bytes == 0


def test_permutation_block_cache_hit_restores_the_generator_stream():
    """A cache hit must leave the generator where a fresh draw would have left it."""
    gate._release_perm_block_cache()
    y = np.arange(40, dtype=np.int64) % 4
    rng_fresh = np.random.default_rng(9)
    fresh = gate._permutation_block(rng_fresh, y, 3)
    after_fresh = rng_fresh.integers(0, 10**9)
    rng_hit = np.random.default_rng(9)
    hit = gate._permutation_block(rng_hit, y, 3)
    np.testing.assert_array_equal(fresh, hit)
    assert rng_hit.integers(0, 10**9) == after_fresh


def test_release_is_registered_with_the_cache_registry():
    """The permutation-block cache releaser is in the global release registry."""
    from mlframe.feature_selection.filters import _cache_release

    assert gate._release_perm_block_cache in _cache_release._RELEASERS


def test_device_oom_fallback_logs_at_debug_when_cupy_error_types_are_missing(monkeypatch, caplog):
    """Without cupy's error types the device OOM fallback is disabled, and the reason is logged under the MRMR logger."""
    import builtins

    real_import = builtins.__import__

    def _no_cupy(name, *args, **kwargs):
        """Make any cupy import fail, as on a machine without cupy."""
        if name.startswith("cupy"):
            raise ImportError("no cupy in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_cupy)
    with caplog.at_level(logging.DEBUG, logger="mlframe.feature_selection.filters.mrmr"):
        assert gate._device_null_oom_errors() == ()
    messages = [r.getMessage() for r in caplog.records if r.name == "mlframe.feature_selection.filters.mrmr"]
    assert any("cupy error types unavailable" in m and "no cupy in this test" in m for m in messages)


def test_family_codes_block_layout_does_not_change_the_codes():
    """Coding the candidate columns in blocks of 2 gives the same codes and counts as one block, for a pandas frame and an array."""
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(120, 7)))
    whole_codes, whole_nx = gate._family_codes(X, 8)
    split_codes, split_nx = gate._family_codes(X, 8, block=2)
    np.testing.assert_array_equal(split_codes, whole_codes)
    np.testing.assert_array_equal(split_nx, whole_nx)
    arr_codes, arr_nx = gate._family_codes(X.to_numpy(), 8, block=3)
    np.testing.assert_array_equal(arr_codes, whole_codes)
    np.testing.assert_array_equal(arr_nx, whole_nx)


def test_family_codes_fills_an_all_non_finite_column_with_zero_codes():
    """A column with no finite value gets the 0.0 fallback, so every one of its codes is 0 and its count is 1."""
    rng = np.random.default_rng(6)
    X = pd.DataFrame(rng.normal(size=(50, 3)))
    X.iloc[:, 1] = np.nan
    codes, nx = gate._family_codes(X, 6, block=1)
    assert np.all(codes[1] == 0)
    assert nx[1] == 1
    assert codes[0].max() > 0 and codes[2].max() > 0


def test_family_codes_pandas_and_numpy_paths_agree_on_infinite_values():
    """A column with inf goes through the median fix on both the pandas block path and the numpy path."""
    rng = np.random.default_rng(8)
    base = rng.normal(size=(60, 4))
    base[5, 2] = np.inf
    base[9, 0] = -np.inf
    X_pd = pd.DataFrame(base)
    pd_codes, pd_nx = gate._family_codes(X_pd, 5, block=2)
    np_codes, np_nx = gate._family_codes(base, 5, block=2)
    np.testing.assert_array_equal(pd_codes, np_codes)
    np.testing.assert_array_equal(pd_nx, np_nx)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_permutation_early_stop_gives_the_brute_force_flags(seed):
    """The early stop returns exactly the flags of the full permutation test, for batches on both sides of alpha.

    The null is a deterministic function of the generator, so the brute-force reference is one full draw of ``B`` rows from the same seed.
    """
    B = 512
    alpha = 0.05

    def null_max(m, r):
        """Null maxima stub: one value per permutation, drawn from the generator."""
        return r.random(m)

    observed = np.array([0.80, 0.97, 0.9995, 0.99, 0.2, 0.9999])
    got = gate._permutation_pass_flags_with(null_max, observed, alpha, B, seed)
    full = np.random.default_rng(seed).random(B)
    R = (full[:, None] >= observed[None, :]).sum(axis=0)
    brute = (1.0 + R) / (B + 1.0) <= alpha
    np.testing.assert_array_equal(got, brute)


def test_permutation_pass_flags_returns_all_false_when_the_probe_already_shows_no_signal():
    """If every observed maximum sits at or below the median of the probe, every batch fails without a full draw."""
    calls = []

    def null_max(m, r):
        """Null maxima stub: one value per permutation, drawn from the generator."""
        calls.append(m)
        return np.full(m, 10.0)

    got = gate._permutation_pass_flags_with(null_max, np.array([0.1, 0.2]), 0.05, 512, 0)
    assert got.tolist() == [False, False]
    assert calls == [min(512, gate.FAMILY_PROBE_BLOCK)]
