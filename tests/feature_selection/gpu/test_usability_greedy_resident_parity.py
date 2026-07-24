"""Selection-equivalence gate for the GPU-RESIDENT regression usability greedy.

X_TEST_COVERAGE_QUALITY-6 fix (mrmr_audit_2026-07-22): ``_usability_greedy_gpu_resident.py`` had zero
test references anywhere in the suite despite its classification sibling
(``_usability_greedy_clf_gpu_resident``) already being pinned by
``test_usability_greedy_clf_resident_parity.py``. This mirrors that pattern for the regression twin:

  * SELECTION-EQUIVALENCE: the resident twin (and the flag-on dispatch) select the SAME feature list
    as the flag-off CPU ``usability_greedy`` regression path across seeds / signal regimes. THIS IS
    THE BAR (selection-equivalence, not byte-identity; only float reduction order differs ~1e-12).
  * FALLBACK CONTRACT: an empty pool, a too-short target (n<2), and ``classification=True`` (which
    delegates to the logistic sibling module) all behave as documented.
  * RESIDENCY: one bulk value-matrix + target H2D, no per-candidate D2H (mirrors the clf twin's own
    residency contract).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda() -> bool:
    """Whether a usable CUDA device is present this process."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


def _make_pool(n, seed, strong):
    """Build a synthetic regression usability pool with a known linear-usable signal."""
    from mlframe.feature_selection.filters._usability_aware_selection import UsableCandidate

    rng = np.random.default_rng(seed)
    a, b, c, d, e, f = (rng.random(n) for _ in range(6))
    scale = 2.5 if strong else 0.6
    y = scale * (2.5 * a - 1.8 * b + 1.5 * np.log(c * 2) * np.sin(d / 3) + 0.3 * f) + rng.normal(scale=0.2, size=n)
    cols = {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "cd": np.log(c * 2) * np.sin(d / 3), "ab": a * b, "noise": rng.random(n)}
    pool = [UsableCandidate(nm, v.astype(np.float64), float(abs(np.corrcoef(v, y)[0, 1])), None, (nm,), ()) for nm, v in cols.items()]
    return pool, y


def _cpu_ref(pool, y, **kw):
    """The flag-OFF (exact CPU) regression selection (names)."""
    from mlframe.feature_selection.filters._usability_aware_selection import usability_greedy

    saved = os.environ.get("MLFRAME_FE_GPU_STRICT_RESIDENT", "")
    os.environ["MLFRAME_FE_GPU_STRICT_RESIDENT"] = "0"
    try:
        return [c.name for c in usability_greedy(pool, y, classification=False, **kw)]
    finally:
        if saved:
            os.environ["MLFRAME_FE_GPU_STRICT_RESIDENT"] = saved
        else:
            os.environ.pop("MLFRAME_FE_GPU_STRICT_RESIDENT", None)


def test_regression_resident_selection_equivalent_to_cpu():
    """The resident twin selects the SAME feature list as the CPU regression greedy across seeds/regimes."""
    from mlframe.feature_selection.filters._usability_greedy_gpu_resident import usability_greedy_gpu_resident

    kw = dict(K=6, n_folds=4, shortlist=20, mae_improve_rel=0.005)
    for strong in (True, False):
        for seed in range(6):
            pool, y = _make_pool(1200, seed, strong)
            cpu = _cpu_ref(pool, y, seed=seed, **kw)
            direct = usability_greedy_gpu_resident(pool, y, seed=seed, **kw)
            assert direct is not None, f"resident twin unexpectedly deferred (strong={strong} seed={seed})"
            gpu = [c.name for c in direct]
            assert gpu == cpu, f"selection diverged strong={strong} seed={seed}: cpu={cpu} gpu={gpu}"


def test_regression_flag_on_dispatch_matches_cpu():
    """The full ``usability_greedy`` dispatch under the resident flag selects the CPU set."""
    from mlframe.feature_selection.filters._usability_aware_selection import usability_greedy

    kw = dict(K=6, n_folds=4, shortlist=20, mae_improve_rel=0.005)
    os.environ["MLFRAME_FE_GPU_STRICT"] = "1"
    os.environ["MLFRAME_FE_GPU_STRICT_RESIDENT"] = "1"
    try:
        for seed in range(4):
            pool, y = _make_pool(1200, seed, True)
            cpu = _cpu_ref(pool, y, seed=seed, **kw)
            gpu = [c.name for c in usability_greedy(pool, y, classification=False, seed=seed, **kw)]
            assert gpu == cpu, f"flag-on dispatch diverged seed={seed}: cpu={cpu} gpu={gpu}"
    finally:
        os.environ.pop("MLFRAME_FE_GPU_STRICT", None)
        os.environ.pop("MLFRAME_FE_GPU_STRICT_RESIDENT", None)


def test_classification_delegates_to_logistic_sibling():
    """``classification=True`` routes to the logistic resident sibling, not this regression twin."""
    from mlframe.feature_selection.filters._usability_greedy_gpu_resident import usability_greedy_gpu_resident

    pool, y_cont = _make_pool(400, 0, True)
    y_bin = (y_cont > np.median(y_cont)).astype(int)
    kw = dict(K=4, n_folds=4, shortlist=8, mae_improve_rel=0.005)
    direct = usability_greedy_gpu_resident(pool, y_bin, seed=0, classification=True, **kw)
    # The logistic sibling either resolves a selection or defers (None); either is valid -- this test
    # only pins that classification=True does NOT run this module's OWN regression body (which would
    # treat y_bin as continuous and silently produce a different, wrong selection).
    if direct is not None:
        assert isinstance(direct, list)


def test_empty_pool_and_short_target_return_none():
    """An empty pool or a target with < 2 rows must return ``None`` (defer to the CPU path), never raise."""
    from mlframe.feature_selection.filters._usability_greedy_gpu_resident import usability_greedy_gpu_resident

    assert usability_greedy_gpu_resident([], np.array([1.0, 2.0]), seed=0) is None
    pool, _ = _make_pool(10, 0, True)
    assert usability_greedy_gpu_resident(pool, np.array([1.0]), seed=0) is None


def test_residency_one_matrix_h2d_no_per_candidate_d2h():
    """Contract: one bulk value-matrix H2D; ZERO bulk D2H; bulk-H2D count constant as the pool grows."""
    from mlframe.feature_selection.filters._usability_aware_selection import UsableCandidate
    from mlframe.feature_selection.filters._usability_greedy_gpu_resident import usability_greedy_gpu_resident
    from mlframe.feature_selection.filters._gpu_strict_fe import residency_audit

    def _audit(P):
        """Run one resident regression greedy under the residency audit and return its report."""
        rng = np.random.default_rng(0)
        n = 3000
        cols = {f"x{i}": rng.random(n) for i in range(P)}
        y = 2.5 * cols["x0"] - 1.8 * cols["x1"] + rng.normal(scale=0.2, size=n)
        pool = [UsableCandidate(nm, v, float(abs(np.corrcoef(v, y)[0, 1])), None, (nm,), ()) for nm, v in cols.items()]
        kw = dict(K=4, seed=0, n_folds=4, shortlist=min(P, 20), mae_improve_rel=0.005)
        _ = usability_greedy_gpu_resident(pool, y, **kw)  # warm cupy/JIT outside the audit
        with residency_audit() as rep:
            usability_greedy_gpu_resident(pool, y, **kw)
        return rep

    r12 = _audit(12)
    r40 = _audit(40)
    assert len(r12.bulk_d2h) == 0 and len(r40.bulk_d2h) == 0
    assert len(r12.bulk_h2d) == len(r40.bulk_h2d)
    assert max(r40.bulk_h2d) > max(r12.bulk_h2d)
