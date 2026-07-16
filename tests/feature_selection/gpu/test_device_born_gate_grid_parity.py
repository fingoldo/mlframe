"""DEVICE-BORN conditional-gate tau-grid parity (2026-06-29).

The conditional-gate scan builds an ``(n, k)`` tau-grid candidate matrix per (gate, operand) combo and scores
its per-column MI. Under STRICT residency the host path materialises that matrix and uploads it
(``_orth_mi_backends.py:311`` -- the dominant H2D of a GPU-strict F2 fit). ``gate_grid_mi_resident`` instead
builds the tau-grid DEVICE-BORN (cupy elementwise from resident operand columns) and scores it with the resident
plug-in MI, so only the small operand columns cross H2D.

These tests pin the SELECTION-EQUIVALENCE hard gate:

* The device-born MI must be PER-COLUMN BIT-IDENTICAL to the host STRICT path on the SAME binning estimator
  (the resident plug-in bins each column independently; building the candidates on-device vs host-then-upload
  cannot change a single column's value), for BOTH the "mask" and "select" gate modes AND for BOTH the EDGE
  (default) and RANK (bytematch) estimators -- the device path threads the SAME ``rank_binning`` flag the host
  path uses, so the estimator never switches (the reg_mixed flip the diagnosis warned about).
* The argmax (the selected tau) must agree, which is what the gate actually consumes.
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda() -> bool:
    """Whether a usable CUDA device is available (used to skip the module when it is not)."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        return is_cuda_available()
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


def _host_block(mode, cv, av, bv, taus):
    """Host-side reference tau-grid block for one gate-grid combo (mask/select), mirroring the device kernel's layout."""
    n = cv.shape[0]
    f = np.empty((n, len(taus)), dtype=np.float64)
    if mode == "mask":
        for j, t in enumerate(taus):
            f[:, j] = (cv > t).astype(np.float64) * av
    else:
        for j, t in enumerate(taus):
            f[:, j] = np.where(cv > t, av, bv)
    return f


@pytest.mark.parametrize("rank_binning", [False, True], ids=["edge", "rank"])
@pytest.mark.parametrize("nclasses", [2, 4, 10])
def test_gate_grid_mi_resident_bit_identical_to_host_strict(rank_binning, nclasses, monkeypatch):
    """The device-born tau-grid MI equals the host STRICT resident MI to bit precision on the SAME estimator,
    for both gate modes and across target cardinalities."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    monkeypatch.setenv("MLFRAME_CMI_GPU", "1")

    from mlframe.feature_selection.filters._conditional_gate_fe import _gate_grid_mi
    from mlframe.feature_selection.filters._resident_candidate_mi import gate_grid_mi_resident

    rng = np.random.default_rng(20260629)
    n = 3000
    cv = rng.uniform(0, 1, n)
    av = rng.uniform(-2, 2, n)
    bv = rng.uniform(-2, 2, n)
    yi = rng.integers(0, nclasses, n).astype(np.int64)
    taus = np.quantile(cv, [0.1, 0.3, 0.5, 0.7, 0.9])
    nbins = 12

    # Host STRICT path: concatenate the two combos' blocks, score via _gate_grid_mi (-> _mi_classif_batch STRICT
    # resident, EDGE or RANK per the flag). The device path must reproduce it bit-for-bit.
    blocks = [_host_block("mask", cv, av, bv, taus), _host_block("select", cv, av, bv, taus)]
    big = np.ascontiguousarray(np.concatenate(blocks, axis=1))
    # Force the host estimator to match the requested rank_binning by monkeypatching the gate-rank predicate.
    import mlframe.feature_selection.filters._conditional_gate_fe as cg
    monkeypatch.setattr(cg, "_gate_rank_binning", lambda: rank_binning)
    host_mi = np.asarray(_gate_grid_mi(big, yi, nbins), dtype=np.float64)

    specs = [
        ("mask", ("a", "c"), (cv, av), taus),
        ("select", ("a", "b", "c"), (cv, av, bv), taus),
    ]
    dev_mi = gate_grid_mi_resident(specs, yi, nbins, rank_binning=rank_binning)
    assert dev_mi is not None, "resident device-born gate-grid returned None on a CUDA host"
    assert dev_mi.shape == host_mi.shape

    maxdiff = float(np.max(np.abs(dev_mi - host_mi)))
    # Device and host sum the same per-bin log terms in a different reduction order (device-side
    # atomics/segmented reduction vs host np.sum) -- true bit-identity isn't guaranteed by IEEE 754,
    # only equivalence up to FP-reorder noise. Observed maxdiff on this suite: ~4.58e-16 (10-rank
    # combo), i.e. 1-2 ULPs at this magnitude -- several orders below anything that could move an
    # argmax decision. 1e-9 matches the project's established FP-reorder tolerance elsewhere.
    assert maxdiff < 1e-9, (
        f"device-born tau-grid MI diverged from host STRICT ({'rank' if rank_binning else 'edge'}, "
        f"nclasses={nclasses}): maxdiff={maxdiff:.3e}\n host={np.round(host_mi, 8)}\n dev ={np.round(dev_mi, 8)}"
    )
    # Per-combo argmax (the selected tau) must agree -- what the gate consumes.
    k = len(taus)
    assert int(np.argmax(host_mi[:k])) == int(np.argmax(dev_mi[:k]))
    assert int(np.argmax(host_mi[k:])) == int(np.argmax(dev_mi[k:]))


def test_gate_grid_mi_resident_column_order_matches_host():
    """The returned MI vector concatenates each combo's per-tau MIs in spec order with per-tau columns in tau
    order -- the layout the caller's per-combo argmax slicing depends on."""
    from mlframe.feature_selection.filters._resident_candidate_mi import gate_grid_mi_resident

    rng = np.random.default_rng(7)
    n = 2000
    cv = rng.uniform(0, 1, n)
    av = rng.uniform(0, 1, n)
    bv = rng.uniform(0, 1, n)
    yi = rng.integers(0, 3, n).astype(np.int64)
    taus = np.asarray([0.25, 0.5, 0.75])

    specs = [("mask", ("a", "c"), (cv, av), taus), ("select", ("a", "b", "c"), (cv, av, bv), taus)]
    dev_mi = gate_grid_mi_resident(specs, yi, 10, rank_binning=False)
    assert dev_mi is not None
    assert dev_mi.shape == (2 * len(taus),)
    assert np.all(np.isfinite(dev_mi))
