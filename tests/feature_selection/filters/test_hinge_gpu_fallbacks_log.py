"""Every hinge / honest-null device fallback is logged at WARNING, and the batch path never drops a column on a device fault."""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pytest

from mlframe.feature_selection.filters import _hinge_basis_fe as B
from mlframe.feature_selection.filters import _hinge_detect_gpu_resident as R
from mlframe.feature_selection.filters import _hinge_detect_gpu_resident_batch as RB
from mlframe.feature_selection.filters import _hinge_precheck_gpu as H


def _kinked(n: int = 3000):
    """A column with one slope change at 0.2 and its target."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=n)
    y = 1.5 * np.maximum(x - 0.2, 0.0) + 0.2 * rng.normal(size=n)
    return x, y


def test_device_detector_failure_is_logged_and_host_detector_runs(monkeypatch, caplog):
    """A fault in the device detector warns and the host detector still finds the kink."""
    x, y = _kinked()
    monkeypatch.setattr(R, "hinge_gpu_enabled", lambda: True)

    def boom(*a, **k):
        """Stand-in device detector that faults."""
        raise RuntimeError("cuda fault")

    monkeypatch.setattr(R, "detect_hinge_breakpoints_gpu", boom)
    with caplog.at_level(logging.WARNING, logger=B.logger.name):
        taus = B._detect_hinge_breakpoints(x, y)
    assert taus
    assert any("device breakpoint detector failed" in r.getMessage() for r in caplog.records)


def test_batched_detector_failure_is_logged(monkeypatch, caplog):
    """A fault in the batched device detector warns and every column reaches the per-column path."""
    x, y = _kinked()
    monkeypatch.setattr(R, "hinge_gpu_enabled", lambda: True)

    def boom(*a, **k):
        """Stand-in batched detector that faults."""
        raise RuntimeError("cuda fault")

    monkeypatch.setattr(RB, "detect_hinge_breakpoints_gpu_batch", boom)
    monkeypatch.setattr(R, "detect_hinge_breakpoints_gpu", lambda *a, **k: None)
    with caplog.at_level(logging.WARNING, logger=B.logger.name):
        out = B._detect_hinge_breakpoints_for_columns([("a", x), ("b", x.copy())], y, max_breakpoints=2, min_heldout_r2_uplift=0.02)
    assert any("batched device detector failed" in r.getMessage() for r in caplog.records)
    assert out.get("a")


def test_batch_detector_does_not_turn_a_column_fault_into_no_hinge(monkeypatch, caplog):
    """A column whose device detector returns None makes the batch return None (per-column fallback), not an empty tau list."""
    x, y = _kinked()
    fake_cp = types.SimpleNamespace(asarray=lambda a: a)
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    from mlframe.feature_selection.filters import _fe_resident_operands as RO

    monkeypatch.setattr(RO, "resident_operand", lambda arr, name, dtype=None: arr)
    monkeypatch.setattr(RB, "_batched_precheck_gpu", lambda *a, **k: np.array([True, True]))
    monkeypatch.setattr(R, "detect_hinge_breakpoints_gpu", lambda *a, **k: None)
    kw = dict(
        max_breakpoints=2, min_heldout_r2_uplift=0.02, precheck_qs=B._HINGE_PRECHECK_QS, precheck_min_sse_drop=0.005,
        cand_q_lo=0.1, cand_q_hi=0.9, n_candidates=24, min_rows=200, min_seg_rows=30,
    )
    with caplog.at_level(logging.WARNING, logger=RB.logger.name):
        got = RB.detect_hinge_breakpoints_gpu_batch([x, x.copy()], y, **kw)
    assert got is None
    assert any("faulted on column" in r.getMessage() for r in caplog.records)


def test_precheck_device_unavailable_is_logged_and_host_result_returned(monkeypatch, caplog):
    """A device precheck that cannot run warns and the host kernel gives the drops."""
    rng = np.random.default_rng(1)
    xs = rng.normal(size=(4, 2000))
    y = np.maximum(xs[0] - 0.2, 0.0) + 0.1 * rng.normal(size=2000)
    monkeypatch.setenv("MLFRAME_HINGE_PRECHECK_DEVICE", "1")
    monkeypatch.setitem(sys.modules, "cupy", None)
    with caplog.at_level(logging.WARNING, logger=H.logger.name):
        got = H.hinge_precheck_drops(xs, y, B._HINGE_PRECHECK_QS, 30)
    assert any("device form is unavailable" in r.getMessage() for r in caplog.records)
    assert np.allclose(got, H._drops_host(xs, y, B._HINGE_PRECHECK_QS, 30, H.VARIANCE_FLOOR))


def test_precheck_survivors_streams_column_blocks(monkeypatch):
    """The batched precheck never stacks more than one column block of the frame at a time."""
    rng = np.random.default_rng(2)
    n = 1000
    cols = [(f"c{i}", rng.normal(size=n)) for i in range(5)]
    y = rng.normal(size=n)
    seen: list = []

    def fake_drops(xs, yy, qs, min_seg, *a):
        """Records the block width and reports every column as a survivor."""
        seen.append(xs.shape[0])
        return np.ones(xs.shape[0])

    monkeypatch.setattr(H, "column_block_width", lambda n_rows: 2)
    monkeypatch.setattr(H, "hinge_precheck_drops", fake_drops)
    out = B._precheck_survivors(cols, y)
    assert [c for c, _ in out] == [c for c, _ in cols]
    assert max(seen) <= 2 and sum(seen) == 5


def test_replay_fills_a_non_finite_value_with_the_fit_mean():
    """A hinge recipe with a stored fit mean replays a NaN row from it, not from the replay batch's own mean."""
    import pandas as pd

    rec = B.build_hinge_basis_recipe(name="f", src_name="a", tau=0.0, side="gt", fill=10.0)
    batch = pd.DataFrame({"a": [np.nan, -5.0, -7.0]})
    out = B._apply_hinge_basis(rec, batch)
    assert out[0] == pytest.approx(10.0)


def test_generate_hinge_features_stores_the_fit_mean_for_replay():
    """The meta of every emitted hinge column carries the source column's fit mean."""
    import pandas as pd

    x, y = _kinked()
    _eng, meta = B.generate_hinge_features(pd.DataFrame({"a": x}), y=y)
    assert meta
    assert all(m["fill"] == pytest.approx(float(np.mean(x))) for m in meta.values())


def test_bitset_accept_decline_is_logged(monkeypatch, caplog):
    """A bitset acceptance scan that does not apply logs that the per-step kernel runs."""
    from mlframe.feature_selection.filters import _honest_null_accept as A
    from mlframe.feature_selection.filters import _honest_null_accept_bits as AB

    monkeypatch.setattr(AB, "root_split_accept_gpu", lambda *a, **k: False)
    cp = types.SimpleNamespace(asarray=lambda a: a)
    with caplog.at_level(logging.INFO, logger=A.logger.name):
        got = A._bitset_accept(cp, None, None, None, 100, 8, 1, 5, 0.05, 0.5, None, None, None, None)
    assert got is False
    assert any("does not apply" in r.getMessage() for r in caplog.records)
