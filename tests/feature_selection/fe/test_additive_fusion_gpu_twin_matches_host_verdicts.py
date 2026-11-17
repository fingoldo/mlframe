"""The GPU-resident additive-fusion twin decides like the host proposal: half scales, MI-grid snapping, rejection ledger and the tail-concentration override (run on numpy stand-ins for cupy)."""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _fe_additive_fusion as H
from mlframe.feature_selection.filters import _fe_additive_fusion_gpu_resident as G
from mlframe.feature_selection.filters import _fe_mi_contract as C


def test_raw_half_scales_ignore_nan_rows_and_fall_back_to_one_like_the_host():
    """The twin's scale of a half is the host ``_half_scale``: NaN rows are skipped (not zero-filled) and a constant or infinite half gives 1.0."""
    rng = np.random.default_rng(0)
    nan_col = rng.standard_normal(500) * 5.0
    nan_col[::7] = np.nan
    const = np.full(500, 3.0)
    inf_col = rng.standard_normal(500)
    inf_col[3] = np.inf
    cols = [nan_col, const, inf_col]
    got = G._raw_half_scales(np, cols)
    want = [H._half_scale({"vals": c}) for c in cols]
    np.testing.assert_allclose(got, want, rtol=1e-12)
    assert got[1] == 1.0 and got[2] == 1.0


def _pair_setup(monkeypatch, mi_add: float, mi_sub: float, strong_mi: float):
    """Patch the device pieces of ``_gpu_evaluate_pair_weighted`` with host stand-ins and return the call inputs and the recorded rejections."""
    from mlframe.feature_selection.filters import _fe_rejection_ledger as L
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as M
    from mlframe.feature_selection.filters import _usability_njit_pool as P

    n = 64
    mis = iter([mi_add, mi_sub])
    monkeypatch.setattr(P, "_gpu_quantile_bin_codes", lambda arr, qs: (np.zeros((arr.shape[0], n), dtype=np.int64), 4))
    monkeypatch.setattr(M, "_cmi_from_binned", lambda *a, **k: next(mis))
    monkeypatch.setattr(G, "_pair_multiple_r_gpu", lambda *a, **k: (0.0, 0.0, 0.0, 0.0))
    rejected: list = []
    monkeypatch.setattr(L, "record_fe_rejection", lambda *a, **k: rejected.append(k))
    vals = np.random.default_rng(0).standard_normal((n, 2))
    ha = {"name": "a", "col": 0, "mi": strong_mi, "floor": 0.0}
    hb = {"name": "b", "col": 1, "mi": strong_mi / 2, "floor": 0.0}
    return vals, ha, hb, rejected, n


def test_gpu_pair_verdict_snaps_mi_to_the_shared_grid_like_the_host(monkeypatch):
    """A fused MI that clears the strong half's bar only by less than the MI grid quantum is refused, as on the host (unsnapped it was admitted)."""
    q = C._MI_TIEBREAK_QUANTUM
    bar = C.quantize_mi_tiebreak(0.4)
    vals, ha, hb, rejected, n = _pair_setup(monkeypatch, bar + 0.3 * q, 0.0, bar)
    out = G._gpu_evaluate_pair_weighted(SimpleNamespace(), np, ha, hb, vals, np.zeros(n, dtype=np.int64), None, vals[:, 0], 1.0, n, 1.0, (1.0, 1.0))
    assert out is None
    assert len(rejected) == 1 and rejected[0]["gate"] == "additive_fusion"


def test_gpu_pair_verdict_admits_a_fused_mi_a_full_quantum_above_the_bar(monkeypatch):
    """The snapped comparison still admits a genuine uplift."""
    q = C._MI_TIEBREAK_QUANTUM
    bar = C.quantize_mi_tiebreak(0.4)
    vals, ha, hb, rejected, n = _pair_setup(monkeypatch, bar + 3 * q, 0.0, bar)
    out = G._gpu_evaluate_pair_weighted(SimpleNamespace(), np, ha, hb, vals, np.zeros(n, dtype=np.int64), None, vals[:, 0], 1.0, n, 1.0, (1.0, 1.0))
    assert out is not None and out[0] == "add" and not rejected


def _raw_probe(monkeypatch, tail_verdict: bool, y_cont):
    """Run the GPU raw-operand check with every device call replaced and return the subsumed raw names."""
    from mlframe.feature_selection.filters import _fe_raw_redundancy_drop as R
    from mlframe.feature_selection.filters import _fe_raw_redundancy_helpers as RH
    from mlframe.feature_selection.filters import _fe_resident_operands as O
    from mlframe.feature_selection.filters import _usability_njit_pool as P

    n = 50
    monkeypatch.setattr(P, "_gpu_quantile_bin_codes", lambda arr, qs: (np.zeros((1, n), dtype=np.int64), 2))
    monkeypatch.setattr(O, "resident_operand", lambda arr, key: arr)
    monkeypatch.setattr(R, "raw_retains_signal_given_genuine_children", lambda **k: True)
    seen: list = []

    def _tail(raw, y, kids, **k):
        """Record the call and return the scripted verdict."""
        seen.append((raw.shape[0], len(kids)))
        return tail_verdict

    monkeypatch.setattr(RH, "raw_is_tail_concentrated_subsumed", _tail)
    X = pd.DataFrame({"a": np.arange(n, dtype=float)})
    ha = {"tokens": {"a"}}
    hb = {"tokens": set()}
    subsumed: set = set()
    fit = SimpleNamespace(_fe_prewarp_y_continuous_=y_cont)
    G._subsumed_raw_operand_check_gpu(ha, hb, subsumed, X, n, {}, 1, None, np.zeros(n, dtype=np.int64), 0, np.zeros(n, dtype=np.int64), fit, np.arange(n, dtype=float))
    return subsumed, seen


def test_gpu_raw_operand_check_applies_the_tail_concentration_override_of_the_host(monkeypatch):
    """A raw operand the binned probe keeps but that is tail-concentrated and subsumed by the fused compound is dropped, as on the host."""
    subsumed, seen = _raw_probe(monkeypatch, True, np.arange(50, dtype=float))
    assert subsumed == {"a"} and seen == [(50, 1)]


def test_gpu_raw_operand_check_keeps_the_raw_without_a_continuous_target(monkeypatch):
    """Without a continuous target the override is not consulted and the binned verdict stands."""
    subsumed, seen = _raw_probe(monkeypatch, True, None)
    assert subsumed == set() and seen == []


def test_gpu_resident_fault_in_the_proposal_is_logged_at_warning(monkeypatch, caplog):
    """A device fault that sends the proposal to the host path says so."""
    from mlframe.utils.log_throttle import reset_throttle_counts

    class _Fault(RuntimeError):
        """Stand-in for a cupy runtime fault."""

    def _boom(*a, **k):
        """Twin that fails on the device."""
        raise _Fault("device lost")

    reset_throttle_counts()
    monkeypatch.setattr(G, "propose_additive_fusions_gpu", _boom)
    monkeypatch.setattr(np.linalg, "LinAlgError", _Fault)
    import mlframe.feature_selection.filters._gpu_strict_fe._entry as E

    monkeypatch.setattr(E, "fe_gpu_strict_resident_enabled", lambda: True)
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters.mrmr"):
        got = H._try_gpu_resident_fusions(SimpleNamespace(), {}, {}, [], set(), [], np.zeros(3, dtype=np.int64), None, 10, 0, 0)
    assert got is None
    assert any("host proposal runs" in r.getMessage() for r in caplog.records)
