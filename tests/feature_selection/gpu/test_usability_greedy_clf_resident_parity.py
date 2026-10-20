"""Selection-equivalence gate for the GPU-RESIDENT classification usability greedy.

The CPU classification branch of ``usability_greedy`` scores K-fold CV-logloss of a
``StandardScaler -> LogisticRegression(max_iter=200)``. ``_usability_greedy_clf_gpu_resident``
ports that resident: a strictly-convex L2 Newton (C=1.0, intercept unpenalised) on the
per-column standardised design reaches the SAME unique optimum sklearn's lbfgs finds, so
the per-fold CV-logloss -- and the committed feature set -- match. This pins:

  * BINARY selection-equivalence: the resident twin (and the flag-on dispatch) select the
    SAME feature list as the flag-off CPU LogisticRegression greedy across seeds / signal
    regimes. THIS IS THE BAR (selection-equivalence, not byte-identity).
  * MULTICLASS deferral: sklearn's default multinomial is fit by lbfgs in the symmetric
    (rank-deficient-Hessian) coefficient space; a resident Newton there hits a singular
    Hessian and a reduced re-parametrisation flips near-tie CV-logloss selections, so the
    twin returns ``None`` for >2 classes and the dispatch falls back to the EXACT CPU path
    (identical selection).
  * RESIDENCY: under a flag-on greedy the candidate value matrix uploads ONCE and there is
    NO per-candidate bulk D2H (the contract; audited by transfer size).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda() -> bool:
    """Need cuda."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


def _make_pool(n, seed, strong, n_classes=2):
    """Make pool."""
    from mlframe.feature_selection.filters._usability_aware_selection import UsableCandidate

    rng = np.random.default_rng(seed)
    a, b, c, d, e, f = (rng.random(n) for _ in range(6))
    if strong:
        score = 2.5 * a + 2.0 * np.log(c * 2) * np.sin(d / 3) - 1.8 * b + 1.5 * f
        scale = 3.0
    else:
        score = 1.5 * a + 0.8 * np.log(c * 2) * np.sin(d / 3) - 0.5 * b + 0.3 * f
        scale = 1.0
    if n_classes == 2:
        pr = 1.0 / (1.0 + np.exp(-scale * (score - score.mean())))
        y = (rng.random(n) < pr).astype(int)
    else:
        qs = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
        y = np.digitize(score, qs)
    cols = {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "cd": np.log(c * 2) * np.sin(d / 3), "ab": a * b, "noise": rng.random(n)}
    pool = [UsableCandidate(nm, v.astype(np.float64), float(abs(np.corrcoef(v, y)[0, 1])), None, (nm,), ()) for nm, v in cols.items()]
    return pool, y


def _cpu_ref(pool, y, **kw):
    """The flag-OFF (CPU LogisticRegression) selection (names)."""
    from mlframe.feature_selection.filters._usability_aware_selection import usability_greedy

    saved = os.environ.get("MLFRAME_FE_GPU_STRICT_RESIDENT", "")
    os.environ["MLFRAME_FE_GPU_STRICT_RESIDENT"] = "0"
    try:
        return [c.name for c in usability_greedy(pool, y, classification=True, **kw)]
    finally:
        if saved:
            os.environ["MLFRAME_FE_GPU_STRICT_RESIDENT"] = saved
        else:
            os.environ.pop("MLFRAME_FE_GPU_STRICT_RESIDENT", None)


def test_binary_clf_resident_selection_equivalent_to_cpu():
    """Binary clf resident selection equivalent to cpu."""
    from mlframe.feature_selection.filters._usability_greedy_clf_gpu_resident import (
        usability_greedy_clf_gpu_resident,
    )

    kw = dict(K=6, n_folds=4, shortlist=20, mae_improve_rel=0.005)
    for strong in (True, False):
        for seed in range(6):
            pool, y = _make_pool(1200, seed, strong)
            cpu = _cpu_ref(pool, y, seed=seed, **kw)
            direct = usability_greedy_clf_gpu_resident(pool, y, seed=seed, **kw)
            assert direct is not None, f"resident twin unexpectedly deferred (strong={strong} seed={seed})"
            gpu = [c.name for c in direct]
            assert gpu == cpu, f"selection diverged strong={strong} seed={seed}: cpu={cpu} gpu={gpu}"


def test_binary_clf_flag_on_dispatch_matches_cpu():
    """The full ``usability_greedy`` dispatch under the resident flag selects the CPU set."""
    from mlframe.feature_selection.filters._usability_aware_selection import usability_greedy

    kw = dict(K=6, n_folds=4, shortlist=20, mae_improve_rel=0.005)
    os.environ["MLFRAME_FE_GPU_STRICT"] = "1"
    os.environ["MLFRAME_FE_GPU_STRICT_RESIDENT"] = "1"
    try:
        for seed in range(4):
            pool, y = _make_pool(1200, seed, True)
            cpu = _cpu_ref(pool, y, seed=seed, **kw)
            gpu = [c.name for c in usability_greedy(pool, y, classification=True, seed=seed, **kw)]
            assert gpu == cpu, f"flag-on dispatch diverged seed={seed}: cpu={cpu} gpu={gpu}"
    finally:
        os.environ.pop("MLFRAME_FE_GPU_STRICT", None)
        os.environ.pop("MLFRAME_FE_GPU_STRICT_RESIDENT", None)


def test_multiclass_defers_to_cpu():
    """>2 classes: the resident twin returns None and the dispatch falls back to the exact CPU selection.

    Re-confirmed with fresh evidence (2026-06-28, sklearn 1.8.0): sklearn's symmetric multinomial basis
    yields a singular Newton Hessian (unpenalised-intercept null -> NaN), and a non-singular reduced
    (C-1) basis converges to a different L2 optimum that flips 9/24 multiclass selections by the gauge
    alone. See the >2-class guard's evidence block in ``_usability_greedy_clf_gpu_resident``.
    """
    from mlframe.feature_selection.filters._usability_aware_selection import usability_greedy
    from mlframe.feature_selection.filters._usability_greedy_clf_gpu_resident import (
        usability_greedy_clf_gpu_resident,
    )

    kw = dict(K=4, n_folds=4, shortlist=8, mae_improve_rel=0.005)
    pool, y = _make_pool(1000, 0, True, n_classes=3)
    assert usability_greedy_clf_gpu_resident(pool, y, seed=0, **kw) is None
    cpu = _cpu_ref(pool, y, seed=0, **kw)
    os.environ["MLFRAME_FE_GPU_STRICT"] = "1"
    os.environ["MLFRAME_FE_GPU_STRICT_RESIDENT"] = "1"
    try:
        disp = [c.name for c in usability_greedy(pool, y, classification=True, seed=0, **kw)]
    finally:
        os.environ.pop("MLFRAME_FE_GPU_STRICT", None)
        os.environ.pop("MLFRAME_FE_GPU_STRICT_RESIDENT", None)
    assert disp == cpu


def test_residency_one_matrix_h2d_no_per_candidate_d2h():
    """Contract: one bulk value-matrix H2D + bounded one-time setup uploads; ZERO bulk D2H; nothing
    scales per-candidate (the bulk-H2D count is constant as the pool grows)."""
    from mlframe.feature_selection.filters._usability_aware_selection import UsableCandidate
    from mlframe.feature_selection.filters._usability_greedy_clf_gpu_resident import (
        usability_greedy_clf_gpu_resident,
    )
    from mlframe.feature_selection.filters._gpu_strict_fe import residency_audit

    def _audit(P):
        """Helper that audit."""
        rng = np.random.default_rng(0)
        n = 3000
        cols = {f"x{i}": rng.random(n) for i in range(P)}
        score = 2.5 * cols["x0"] - 1.8 * cols["x1"]
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-3 * (score - score.mean())))).astype(int)
        pool = [UsableCandidate(nm, v, float(abs(np.corrcoef(v, y)[0, 1])), None, (nm,), ()) for nm, v in cols.items()]
        kw = dict(K=4, seed=0, n_folds=4, shortlist=min(P, 20), mae_improve_rel=0.005)
        _ = usability_greedy_clf_gpu_resident(pool, y, **kw)  # warm cupy/JIT outside the audit
        with residency_audit() as rep:
            usability_greedy_clf_gpu_resident(pool, y, **kw)
        return rep

    r12 = _audit(12)
    r40 = _audit(40)
    # ZERO per-candidate bulk D2H in both.
    assert len(r12.bulk_d2h) == 0 and len(r40.bulk_d2h) == 0
    # Bulk H2D is constant w.r.t. pool size (one value matrix + the same one-time setup uploads),
    # i.e. nothing scales per-candidate.
    assert len(r12.bulk_h2d) == len(r40.bulk_h2d)
    # The single largest bulk H2D is the value matrix and it GROWS with P (12 -> 40 cols).
    assert max(r40.bulk_h2d) > max(r12.bulk_h2d)


def test_float32_newton_converges_at_realistic_size():
    """A float32 Newton step cannot reach the float64 tolerance, so the twin used to defer to the CPU on every realistic-size pool; it now converges and selects the signal columns."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._usability_aware_selection import UsableCandidate
    from mlframe.feature_selection.filters._usability_greedy_clf_gpu_resident import usability_greedy_clf_gpu_resident

    rng = np.random.default_rng(0)
    n = 20000
    base = rng.random((n, 3))
    score = 2.0 * base[:, 0] - 1.5 * base[:, 1] + base[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-3 * (score - score.mean())))).astype(int)
    cols = {f"x{i}": rng.standard_normal(n) for i in range(20)}
    cols.update({"a": base[:, 0], "b": base[:, 1], "c": base[:, 2]})
    pool = [UsableCandidate(nm, v.astype(np.float64), float(abs(np.corrcoef(v, y)[0, 1])), None, (nm,), ()) for nm, v in cols.items()]
    got = usability_greedy_clf_gpu_resident(pool, y)
    assert got is not None
    assert {c.name for c in got} == {"a", "b", "c"}


def test_batched_candidate_logloss_matches_per_candidate_sklearn_fit():
    """The batched Newton fit of all candidates of a fold gives each candidate's validation logloss of the per-candidate logistic fit (sklearn, L2 C=1 on standardized columns)."""
    cp = pytest.importorskip("cupy")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    from mlframe.feature_selection.filters._usability_greedy_clf_batched import cv_candidates_binary
    from mlframe.feature_selection.filters._usability_greedy_clf_gpu_resident import _ResidentClfFallbackError

    rng = np.random.default_rng(3)
    n = 6000
    V = rng.standard_normal((n, 9)).astype(np.float32)
    y = (rng.random(n) < 1 / (1 + np.exp(-(V[:, 0] - 0.7 * V[:, 3])))).astype(np.int64)
    folds_host = rng.permutation(n) % 3
    specs = [(cp.asarray(folds_host != fo), cp.asarray(folds_host == fo), True) for fo in range(3)]
    selected, cands = [0], [3, 5, 8, 1]
    got = cv_candidates_binary(
        cp, cp.asarray(V), selected, cands, specs, cp.asarray(y), dtype=cp.float32, newton_max_iter=200, tol=2e-4, lam=1.0, logloss_eps=1e-15, fallback_error=_ResidentClfFallbackError,
    )
    for ci, c in enumerate(cands):
        cols = [*selected, c]
        for fo in range(3):
            tr, va = folds_host != fo, folds_host == fo
            mu, sd = V[tr][:, cols].mean(0), V[tr][:, cols].std(0)
            clf = LogisticRegression(C=1.0, tol=1e-10, max_iter=1000).fit((V[tr][:, cols] - mu) / sd, y[tr])
            want = log_loss(y[va], clf.predict_proba((V[va][:, cols] - mu) / sd)[:, 1], labels=[0, 1])
            assert abs(got[ci, fo] - want) < 2e-4
