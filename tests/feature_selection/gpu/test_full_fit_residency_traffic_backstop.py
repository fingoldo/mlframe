"""Full-fit residency_audit() regression backstop (mrmr_audit_2026-07-20 gpu_residency.md #5).

Every existing byte-traffic audit test covers exactly ONE FE family in isolation
(``test_cmi_residency_traffic.py`` -> ``greedy_cmi_fe_construct``, which is itself opt-in and OFF by
default; ``test_device_born_flags_parity_and_traffic.py`` -> one flag at a time). None audits a REAL
multi-family ``MRMR.fit()`` call, so a cross-family or cache-eviction-order leak (e.g. a future edit to
``assemble_resident_matrix``'s fallback branch reintroducing a bulk upload) would pass every existing
single-family test and still balloon the fit-level H2D/D2H volume.

BASELINE, measured on this file's own fixture (2026-07-21, single quiet GPU, 40000 rows, F2 pair
search + 2 FE rounds under ``MLFRAME_FE_GPU_STRICT=1`` / ``MLFRAME_CMI_GPU=1`` / ``MLFRAME_FE_VRAM_F32=1``):
several hundred bulk H2D/D2H ops (run-to-run variance observed in the 350-620 range across seeds/JIT-
warm state). This is FAR from "a handful of per-device operand uploads" the individual device_born_*
mechanisms each claim in isolation -- confirming exactly the cross-family gap this proposal names:
per-family selection-equivalence has been checked (see the sibling parity tests), but nothing
previously audited the AGGREGATE fit-level byte traffic. This test does NOT assert that gap is already
closed (it is not -- see the module list at the bottom, a concrete follow-up item, not a silent cap);
it locks in a budget with headroom over the measured range as a regression backstop, so a NEW leak
(not yet audited, not yet fixed) trips this test immediately instead of silently regressing the
GPU-strict fit further.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
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

# Measured baseline (see module docstring) with ~2x headroom over the observed 350-620 range: catches a
# NEW regression (e.g. a reintroduced bulk upload doubling the count) without false-failing on ordinary
# run-to-run variance (candidate-pool ordering / KTC cache state / which FE families' MI gates happen to
# admit a column this seed).
_BULK_H2D_BUDGET = 1200
_BULK_D2H_BUDGET = 1200


def _run_strict_fit(n: int = 40000, seed: int = 0):
    """Run one real multi-family MRMR.fit() under the 3 STRICT-residency flags, wrapped in
    residency_audit(). Explicit MLFRAME_FE_GPU_STRICT=1 bypasses the AUTO row-count gate (default
    100k), so this fixture does not need to pay a 100k-row fit's wall time to exercise the STRICT path
    -- the flag forces it regardless of n."""
    from mlframe.feature_selection.filters._gpu_strict_fe import residency_audit
    from mlframe.feature_selection.filters.mrmr import MRMR

    saved = {k: os.environ.get(k) for k in ("MLFRAME_FE_GPU_STRICT", "MLFRAME_CMI_GPU", "MLFRAME_FE_VRAM_F32")}
    os.environ["MLFRAME_FE_GPU_STRICT"] = "1"
    os.environ["MLFRAME_CMI_GPU"] = "1"
    os.environ["MLFRAME_FE_VRAM_F32"] = "1"
    try:
        rng = np.random.default_rng(seed)
        a, b, c, d, e = (rng.uniform(0.1, 1.1, n) for _ in range(5))
        X = pd.DataFrame({k: v.astype(np.float64) for k, v in zip("abcde", (a, b, c, d, e))})
        y = a**2 / b + e / 5.0 + np.log(np.abs(c) + 1e-9) * np.sin(d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with residency_audit() as rep:
                m = MRMR(full_npermutations=10, baseline_npermutations=20, fe_max_steps=2, verbose=0, n_jobs=1)
                m.fit(X, y)
        return rep, m
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def strict_fit_audit():
    """Run the full-fit STRICT-residency audit ONCE for the whole module. Skips (not fails) if the GPU
    path never actually engaged (contended/shared CUDA context on this host -- an environmental
    condition documented as the same skip class ``test_cmi_residency_traffic.py``'s pair_search_audit
    fixture already uses, not a residency regression)."""
    rep, m = _run_strict_fit()
    print("\nFULL-FIT STRICT residency: " + rep.summary())
    if not rep.h2d and not rep.d2h:
        pytest.skip("no device traffic recorded at all -- STRICT path likely did not engage on this host (contended/unavailable CUDA context)")
    return rep, m


def test_full_fit_completes_and_selects_features(strict_fit_audit):
    """Sanity gate: the audited fit actually ran the real pipeline and produced a non-degenerate
    support_ (a crashed/empty fit would make the byte-traffic numbers below meaningless)."""
    _rep, m = strict_fit_audit
    assert len(getattr(m, "support_", [])) >= 1, "the STRICT fit selected no features -- traffic audit below would be meaningless"


def test_full_fit_bulk_h2d_within_regression_budget(strict_fit_audit):
    """PRIMARY GATE: aggregate bulk H2D across a real multi-family MRMR.fit() must stay within a fixed
    budget with headroom over the measured baseline (360 bulk H2D on this exact fixture, 2026-07-21) --
    catches a NEW cross-family or cache-eviction-order leak, not yet audited by any single-family test."""
    rep, _m = strict_fit_audit
    assert len(rep.bulk_h2d) <= _BULK_H2D_BUDGET, f"bulk H2D count {len(rep.bulk_h2d)} exceeds the regression budget {_BULK_H2D_BUDGET}; {rep.summary()}"


def test_full_fit_bulk_d2h_within_regression_budget(strict_fit_audit):
    """Same regression backstop for aggregate bulk D2H (the read-back half of the same cross-family
    leak class)."""
    rep, _m = strict_fit_audit
    assert len(rep.bulk_d2h) <= _BULK_D2H_BUDGET, f"bulk D2H count {len(rep.bulk_d2h)} exceeds the regression budget {_BULK_D2H_BUDGET}; {rep.summary()}"


# KNOWN GAP (not silently capped -- a concrete follow-up, mrmr_audit_2026-07-20 gpu_residency.md #5's
# own honest finding): several hundred bulk H2D/D2H ops is far above "a handful of per-device operand
# uploads" every individual device_born_* mechanism claims in isolation. The per-family parity tests
# (test_device_born_flags_parity_and_traffic.py, test_resident_311_residual_parity.py,
# test_cmi_residency_traffic.py) already pin SELECTION-equivalence per family; nothing yet attributes
# WHICH family/site contributes the bulk of these 360+353 transfers at fit level, which would be the
# next step toward actually closing this gap (likely candidates given the batch_pair_mi CUDA-fallback
# warning seen on this fixture: the F2 pair-search CPU-kernel fallback path, which is NOT under the
# resident-codes contract test_cmi_residency_traffic.py's pair-search block audits).
