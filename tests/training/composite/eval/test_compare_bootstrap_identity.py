"""CPX26: row-chunked paired-bootstrap CI must be bit-identical to the OLD
monolithic (n_boot, n) draw, and must cap peak temp memory by the block, not
n_boot*n.

The fix row-chunks the bootstrap to avoid materialising two n_boot*n
temporaries (~16 GB at n_boot=1000, n=1e6). numpy fills rng.integers
row-major, so drawing the resamples in contiguous blocks consumes the RNG
stream in the exact same order -> identical resamples -> identical CI.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from mlframe.training.composite.compare import _paired_bootstrap_ci, compare_models


def _old_monolithic(diff, n_boot, alpha, rng):
    """Pre-CPX26 reference: single (n_boot, n) draw + gather."""
    n = diff.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    obs = float(diff.mean())
    tail = float(np.mean(boot_means <= 0.0)) if obs >= 0 else float(np.mean(boot_means >= 0.0))
    return lo, hi, min(1.0, 2.0 * tail)


@pytest.mark.parametrize("seed", [0, 7, 123])
@pytest.mark.parametrize("n,n_boot", [(500, 200), (10_000, 1000), (33_333, 257)])
def test_chunked_bootstrap_ci_bit_identical(seed, n, n_boot):
    rng_d = np.random.default_rng(seed + 1000)
    diff = rng_d.standard_normal(n) * 0.1 + 0.005
    alpha = 0.05
    old = _old_monolithic(diff, n_boot, alpha, np.random.default_rng(seed))
    new = _paired_bootstrap_ci(diff, n_boot, alpha, np.random.default_rng(seed))
    assert old == new, f"CI not bit-identical: OLD={old} NEW={new}"


def test_chunked_draws_same_index_stream():
    """Block-concatenated index draws must equal the monolithic draw."""
    n, n_boot, block = 257, 100, 64
    mono = np.random.default_rng(42).integers(0, n, size=(n_boot, n))
    rng = np.random.default_rng(42)
    chunks = [rng.integers(0, n, size=(min(s + block, n_boot) - s, n)) for s in range(0, n_boot, block)]
    cat = np.concatenate(chunks, axis=0)
    assert np.array_equal(mono, cat)


def test_peak_temp_bounded_by_block_not_nboot():
    """NEW peak temp must be far below the monolithic n_boot*n*8*2 bytes."""
    n, n_boot, block = 50_000, 1000, 64
    diff = np.random.default_rng(3).standard_normal(n) * 0.1

    tracemalloc.start()
    tracemalloc.reset_peak()
    _paired_bootstrap_ci(diff, n_boot, alpha=0.05, rng=np.random.default_rng(0))
    _, new_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    monolithic_temp = n_boot * n * 8 * 2  # idx int64 + float64 gather
    block_bound = block * n * 8 * 2
    # Allow generous headroom (quantiles/intermediate) but must be << monolithic.
    assert new_peak < 4 * block_bound, f"peak {new_peak} exceeds 4x block bound {4 * block_bound}"
    assert new_peak < monolithic_temp / 4, f"peak {new_peak} not << monolithic {monolithic_temp}"


def test_compare_models_end_to_end_bootstrap_unchanged():
    """Full compare_models bootstrap path stays bit-identical across the fix."""
    rng = np.random.default_rng(11)
    n = 5000
    y = rng.standard_normal(n)

    class _P:
        def __init__(self, err):
            self.err = err

        def predict(self, X):
            return X + self.err

    champ = _P(rng.standard_normal(n) * 0.5)
    chall = _P(rng.standard_normal(n) * 0.3)
    res = compare_models(champ, chall, y, y, metric="rmse", n_boot=1000, random_state=0)
    # Recompute CI directly via OLD reference on the same diff to pin identity.
    pc = (y - champ.predict(y)) ** 2
    ph = (y - chall.predict(y)) ** 2
    diff = pc - ph
    lo, hi, p = _old_monolithic(diff, 1000, 0.05, np.random.default_rng(0))
    assert (res["ci_low"], res["ci_high"], res["p_value"]) == (lo, hi, p)
