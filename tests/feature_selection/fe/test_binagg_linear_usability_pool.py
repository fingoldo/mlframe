"""The pair pool of ``_keep_linearly_usable`` keeps the same columns, in the same order, as the serial run."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _adaptive_nbins
from mlframe.feature_selection.filters._binned_numeric_agg_redundancy import _keep_linearly_usable


class _Cands:
    """The minimal candidate store: names and per-name bin codes."""

    def __init__(self, codes):
        self._codes = codes
        self.names = list(codes)

    def codes(self, name, nbins):
        """Codes of ``name``."""
        return self._codes[name]


def test_pool_equals_serial(monkeypatch):
    """One worker and four workers keep identical column lists."""
    rng = np.random.default_rng(11)
    n = 4000
    X = pd.DataFrame({f"g{i}": rng.integers(0, 12, n).astype(float) for i in range(6)} | {f"a{i}": rng.normal(size=n) for i in range(6)})
    cell0 = X["g0"].to_numpy().astype(int)
    y = ((cell0 % 2) ^ (rng.random(n) < 0.1)).astype(int)  # a non-linear function of the cell
    raw, rejected, codes = {}, {}, {}
    for i in range(6):
        cell = X[f"g{i}"].to_numpy().astype(int)
        for stat, signal in (("mean", y * 0.0), ("std", rng.normal(size=n) * 0.0)):
            name = f"bagg_{i}_{stat}"
            raw[name] = {"group_col": f"g{i}", "agg_col": f"a{i}"}
            rejected[name] = ([f"g{i}", f"a{i}"], 0.0)
            means = np.bincount(cell, weights=y + signal) / np.maximum(np.bincount(cell), 1)
            codes[name] = means[cell]
    cands = _Cands(codes)

    def run(jobs):
        """``kept_cols`` with the pool sized to ``jobs``."""
        monkeypatch.setattr(_adaptive_nbins, "_resolve_edge_jobs", lambda *a, **k: jobs)
        kept: list = []
        _keep_linearly_usable(cands, raw, X, lambda g: X[g].to_numpy(), 10, y, rejected, kept)
        return kept

    serial = run(1)
    assert serial, "the fixture must keep something, else the comparison is empty"
    assert serial == run(4)
