"""The tiny rerank holds a bounded number of per-base X-without-base matrices, and a synthetic base drops its parents.

The rerank built a ``np.delete`` copy of the screen matrix for every distinct base before scoring and held all of them to
the end: 16 bases at 60k x 199 were 760 MB of a 958 MB rerank peak. The copies are now gathered on first use and at most
two stay cached (``_per_base_x.PerBaseMatrices``); the specs are dispatched grouped by base so each is gathered once.
"""

from __future__ import annotations

import tracemalloc
import warnings

import numpy as np
import pandas as pd


def test_the_map_gathers_on_demand_bounds_its_cache_and_drops_a_synthetic_bases_parents():
    from mlframe.training.composite.discovery._per_base_x import PerBaseMatrices, base_ordered

    x = np.arange(40, dtype=np.float32).reshape(5, 8)
    cols = [f"c{i}" for i in range(8)]
    bases = {c: x[:, i] for i, c in enumerate(cols[:4])} | {"c1__mul__c5": x[:, 1] * x[:, 5], "": np.zeros(5), "outside": np.ones(5)}
    m = PerBaseMatrices(x, cols, bases, capacity=2)
    for b in ("c0", "c1", "c2", "c3"):
        np.testing.assert_array_equal(m[b][1], np.delete(x, cols.index(b), axis=1))
    assert len(m._built) == 2
    np.testing.assert_array_equal(m["c1__mul__c5"][1], np.delete(x, [1, 5], axis=1))
    assert m[""][1] is x and m["outside"][1] is x
    assert m.get("absent") is None and "c0" in m and m.base_screen("c2") is bases["c2"]
    assert base_ordered([0, 1, 2, 3, 4], lambda i: "ab"[i % 2]) == [0, 2, 4, 1, 3]


def _rerank_peak(n_bases: int, monkeypatch) -> tuple:
    """Traced peak inside the rerank and the number of distinct bases it scored."""
    from mlframe.training.composite.discovery import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    n, f = 4000, 240
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"f{i}": rng.normal(size=n).astype(np.float32) for i in range(f)})
    X["y"] = 2 * X["f0"] + np.sin(X["f1"]) + rng.normal(0, 0.3, n)
    peaks = []
    real = CompositeTargetDiscovery._tiny_model_rerank

    def spy(self, kept_specs, *a, **k):
        tracemalloc.start()
        try:
            return real(self, kept_specs, *a, **k)
        finally:
            peaks.append((tracemalloc.get_traced_memory()[1], len({s.base_column for s in kept_specs})))
            tracemalloc.stop()

    monkeypatch.setattr(CompositeTargetDiscovery, "_tiny_model_rerank", spy)
    cfg = CompositeTargetDiscoveryConfig(enabled=True, random_state=0, mi_sample_n=n, eps_mi_gain=-1.0, base_candidates=[f"f{i}" for i in range(n_bases)],
                                         transforms=["linear_residual"], interaction_base_discovery_enabled=False, tiny_rerank_n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CompositeTargetDiscovery(cfg).fit(X, "y", [f"f{i}" for i in range(f)], np.arange(n))
    assert peaks, "the rerank did not run"
    return peaks[0]


def test_the_rerank_peak_stays_below_one_copy_per_base(monkeypatch):
    """With twelve bases, one float32 X-without-base and one float64 WAIC copy per base alone are 36 screen matrices;
    the bounded caches keep the whole rerank under 32 (176 MB before, 106 MB after at 4k x 240)."""
    peak, n_bases = _rerank_peak(12, monkeypatch)
    assert n_bases >= 10, n_bases
    screen_matrix = 4000 * 240 * 4
    assert peak < 32 * screen_matrix, f"rerank peak {peak / 1e6:.1f} MB, {peak / screen_matrix:.1f} screen matrices, with {n_bases} bases"
