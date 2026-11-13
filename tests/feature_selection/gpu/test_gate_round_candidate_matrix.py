"""The CMI gate's round-level candidate matrix: the device build equals the host build."""

from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from mlframe.feature_selection.filters import _fe_batched_mi, _fe_cmi_redundancy_gate as G, _mi_greedy_cmi_fe

pytestmark = pytest.mark.gpu


def _capture(monkeypatch, g, names):
    """Run ``_gate_round_batched`` and return the matrix it hands to ``batched_cmi_gpu``."""
    seen = []

    def fake(x_cols, *a, **k):
        """Record the candidate matrix and stop the round."""
        seen.append(x_cols)
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(_mi_greedy_cmi_fe, "_cmi_gpu_enabled", lambda **k: True)
    monkeypatch.setattr(_fe_batched_mi, "batched_cmi_gpu", fake)
    G._gate_round_batched(g, names, None, 0.95)
    return seen[0]


def test_resident_build_equals_host_build(monkeypatch):
    """Stacked-and-transposed device codes are the same (n, K) int64 matrix as the host column-by-column build."""
    rng = np.random.default_rng(0)
    n, k = 5000, 7
    names = [f"c{i}" for i in range(k)]
    host = {nm: rng.integers(0, 10, n).astype(np.int8) for nm in names}
    y = rng.integers(0, 2, n).astype(np.int64)
    dev = {nm: cp.asarray(v) for nm, v in host.items()}
    on_dev = _capture(monkeypatch, SimpleNamespace(y_dense=y, n_rows=n, cand_bins=host, cand_bins_dev=dev), names)
    on_host = _capture(monkeypatch, SimpleNamespace(y_dense=y, n_rows=n, cand_bins=host, cand_bins_dev={}), names)
    assert isinstance(on_dev, cp.ndarray) and isinstance(on_host, np.ndarray)
    assert on_dev.dtype == cp.int64 and on_dev.shape == (n, k)
    assert on_dev.flags.c_contiguous
    assert np.array_equal(cp.asnumpy(on_dev), on_host)


def test_the_whole_round_runs_on_the_device_without_dropping_to_empty_tables(monkeypatch):
    """The round-level batched CMI and analytic floor complete on resident codes: the tables hold every candidate and no 'round CMI/floor computation failed' warning is logged (a leaked loop variable used to make this fail silently)."""
    import logging

    rng = np.random.default_rng(3)
    n, k = 6000, 5
    names = [f"c{i}" for i in range(k)]
    host = {nm: rng.integers(0, 8, n).astype(np.int8) for nm in names}
    y = rng.integers(0, 2, n).astype(np.int64)
    dev = {nm: cp.asarray(v) for nm, v in host.items()}
    g = SimpleNamespace(y_dense=y, n_rows=n, cand_bins=host, cand_bins_dev=dev)
    monkeypatch.setattr(_mi_greedy_cmi_fe, "_cmi_gpu_enabled", lambda **kw: True)
    records = []

    class _Catch(logging.Handler):
        """Collects the warnings of the gate logger."""

        def emit(self, record):
            """Keep the formatted message."""
            records.append(record.getMessage())

    handler = _Catch(level=logging.WARNING)
    log = logging.getLogger("mlframe.feature_selection.filters.mrmr")
    log.addHandler(handler)
    try:
        z = cp.asarray(rng.integers(0, 4, n).astype(np.int64))  # the conditioning support of a round (never None once a feature is admitted)
        cmi, _floor, cards = G._gate_round_batched(g, names, z, 0.95)
    finally:
        log.removeHandler(handler)
    assert set(cmi) == set(names) and set(cards) == set(names)
    assert all(np.isfinite(v) for v in cmi.values())
    assert not [m for m in records if "round CMI/floor computation failed" in m]
