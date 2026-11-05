"""The threshold bootstrap draws its resamples in row chunks, and the interval does not depend on the chunk size.

The full ``(n_boot, n)`` index matrix used to be built in one allocation, which is gigabytes at production sizes. The chunked draw consumes
the same generator stream, so a tiny chunk budget must give the same interval as a single chunk.
"""

from __future__ import annotations

import numpy as np

from mlframe.metrics.classification import _threshold_optimization as to


def _data(n: int = 300, seed: int = 0):
    """A noisy binary problem where the optimal threshold is well inside the score range."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.int64)
    score = np.clip(0.35 * y + rng.normal(scale=0.3, size=n) + 0.3, 0.0, 1.0)
    return y, score


def test_chunked_resample_draw_gives_the_same_interval_as_one_chunk(monkeypatch):
    """A byte budget that forces many small chunks returns exactly the interval a single chunk returns."""
    y, score = _data()
    monkeypatch.setattr(to, "_BOOT_INDEX_MAX_BYTES", 1 << 30)
    whole = to.optimal_threshold_bootstrap_ci(y, score, metric="f1", n_boot=200, alpha=0.1, random_state=7)

    calls = {"n": 0}
    real_kernel = to._bootstrap_threshold_kernel

    def counting_kernel(*args, **kwargs):
        """Count the kernel invocations so the chunking is observable."""
        calls["n"] += 1
        return real_kernel(*args, **kwargs)

    monkeypatch.setattr(to, "_bootstrap_threshold_kernel", counting_kernel)
    monkeypatch.setattr(to, "_BOOT_INDEX_MAX_BYTES", 8 * len(y) * 7)
    chunked = to.optimal_threshold_bootstrap_ci(y, score, metric="f1", n_boot=200, alpha=0.1, random_state=7)

    assert calls["n"] > 1
    assert chunked == whole
