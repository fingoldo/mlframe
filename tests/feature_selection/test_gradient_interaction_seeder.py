"""Triad for the gradient-interaction (mixed second partials) seeder -- backlog idea #21.

unit       -- mixed-partial energy correct on a known saddle vs additive; permutation-null
              rejects noise; OOF-gate blocks a non-learning surrogate; analytic == finite-diff.
biz_value  -- proposing the smooth saddle pair improves downstream model accuracy on the
              sin(x5)*x31 fixture vs not proposing it.
cProfile   -- surrogate fit + finite-diff hotspot; the routed-default single-fit cost.

All fixtures are n<=2000 (idea #21 is the heaviest -- kept lean). Single-process; no xdist.
"""
import cProfile
import io
import pstats
import time
from itertools import combinations

import numpy as np
import pytest

sk = pytest.importorskip("sklearn")

from mlframe.feature_selection.filters._gradient_interaction_seeder import (  # noqa: E402
    _finite_diff_mixed_partial_energy,
    _fit_rff_ridge,
    _rff_analytic_mixed_partial_energy,
    _rff_analytic_mixed_partial_energy_loop,
    _route_gradient_seeder,
    _standardize,
    rank_gradient_interaction_pairs,
)

P = 60
N = 2000
FOCUS = (5, 31)


def _make(kind, n=N, p=P, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    a, b = X[:, 5], X[:, 31]
    if kind == "sin_prod":
        y = np.sin(a) * b + 0.3 * rng.standard_normal(n)
    elif kind == "product":
        y = a * b + 0.3 * rng.standard_normal(n)
    elif kind == "additive":
        y = X[:, :8].sum(1) + 0.3 * rng.standard_normal(n)
    elif kind == "additive_nonlin":
        y = np.sin(X[:, 0]) + X[:, 1] ** 2 + np.abs(X[:, 2]) + X[:, 3:8].sum(1) + 0.3 * rng.standard_normal(n)
    elif kind == "noise":
        y = rng.standard_normal(n)
    else:
        raise ValueError(kind)
    return X, y


def _rank(energies, focus):
    ranked = sorted(energies.items(), key=lambda kv: -kv[1])
    return next((i for i, (k, v) in enumerate(ranked) if k == focus), None)


# ============================ UNIT ============================

def test_mixed_partial_high_on_product_saddle():
    """Analytic mixed partial concentrates on the (5,31) product saddle (rank 0)."""
    X, y = _make("product")
    Xs = _standardize(X.astype(np.float64))
    pairs = list(combinations(range(P), 2))
    rbf, ridge = _fit_rff_ridge(Xs, y, n_components=400, gamma=1.0 / P, alpha=1.0, seed=0)
    en = _rff_analytic_mixed_partial_energy(rbf, ridge, Xs, pairs)
    assert _rank(en, FOCUS) == 0
    # the saddle pair's energy is far above the median pair energy
    vals = np.array(list(en.values()))
    assert en[FOCUS] > 5 * np.median(vals)


def test_mixed_partial_near_zero_on_additive():
    """A purely additive surface has ~zero mixed partials -- (5,31) must NOT rank near the top
    relative to the no-interaction baseline (its energy is not a clear outlier)."""
    X, y = _make("additive")
    Xs = _standardize(X.astype(np.float64))
    pairs = list(combinations(range(P), 2))
    rbf, ridge = _fit_rff_ridge(Xs, y, n_components=400, gamma=1.0 / P, alpha=1.0, seed=0)
    en = _rff_analytic_mixed_partial_energy(rbf, ridge, Xs, pairs)
    vals = np.array(list(en.values()))
    # no pair should be a strong outlier: max energy is not many-fold the median (additive => flat)
    assert vals.max() < 4 * np.median(vals)


def test_analytic_matches_finite_diff():
    """The exact RFF mixed partial and central finite differences agree (same ranking + ~values)."""
    X, y = _make("product")
    Xs = _standardize(X.astype(np.float64))
    # restrict to a small pair set for speed
    cols = [5, 31, 0, 1, 2, 10, 20, 40]
    Xsub = Xs[:, cols]
    pairs = list(combinations(range(len(cols)), 2))
    rbf, ridge = _fit_rff_ridge(Xsub, y, n_components=400, gamma=1.0 / len(cols), alpha=1.0, seed=0)
    en_a = _rff_analytic_mixed_partial_energy(rbf, ridge, Xsub, pairs)

    def predict(Z):
        return ridge.predict(rbf.transform(Z))

    en_fd = _finite_diff_mixed_partial_energy(predict, Xsub, pairs, h=0.05, max_rows=2000, seed=0)
    # same top pair
    top_a = max(en_a, key=en_a.get)
    top_fd = max(en_fd, key=en_fd.get)
    assert top_a == top_fd == (0, 1)  # (5,31) -> local indices (0,1)
    # values correlate strongly
    ka = np.array([en_a[p] for p in pairs])
    kf = np.array([en_fd[p] for p in pairs])
    assert np.corrcoef(ka, kf)[0, 1] > 0.97


def test_batched_kernel_matches_loop_kernel():
    """The batched analytic kernel (routed default) is byte-comparable to the per-pair loop kernel
    (keep-all-kernels rule: both versions kept, dispatcher uses the fast one)."""
    X, y = _make("product")
    Xs = _standardize(X.astype(np.float64))
    pairs = list(combinations(range(P), 2))
    rbf, ridge = _fit_rff_ridge(Xs, y, n_components=400, gamma=1.0 / P, alpha=1.0, seed=0)
    en_loop = _rff_analytic_mixed_partial_energy_loop(rbf, ridge, Xs, pairs)
    en_batch = _rff_analytic_mixed_partial_energy(rbf, ridge, Xs, pairs, chunk=128)
    for p in pairs:
        assert abs(en_loop[p] - en_batch[p]) <= 1e-9 * (1 + abs(en_loop[p]))


def test_oof_gate_blocks_noise():
    """Pure noise: the OOF self-gate fires (surrogate does not learn) -> 0 proposals."""
    X, y = _make("noise")
    proposed, energies, diag = rank_gradient_interaction_pairs(X, y, list(range(P)), seed=0)
    # either it didn't learn, or it learned by fluke but the null rail rejects everything
    assert len(proposed) == 0
    if diag.get("learned"):
        assert diag["n_proposed"] == 0


def test_permutation_null_rejects_additive():
    """Additive (no interaction): surrogate learns the linear signal, but the additive-residual
    baseline + permutation null reject all chance saddles -> 0 proposals."""
    X, y = _make("additive")
    proposed, energies, diag = rank_gradient_interaction_pairs(X, y, list(range(P)), seed=0)
    assert diag["learned"] is True   # it DOES learn the additive signal
    assert len(proposed) == 0        # but proposes nothing (no interaction)


def test_additive_nonlinear_also_clean():
    """Additive but per-feature NONLINEAR (sin(x0)+x1^2+...): still 0 proposals (no cross term)."""
    X, y = _make("additive_nonlin")
    proposed, _, diag = rank_gradient_interaction_pairs(X, y, list(range(P)), seed=0)
    assert diag["learned"] is True
    assert len(proposed) == 0


def test_dispatcher_routes_by_size():
    """The size dispatcher routes ON only in the [min_p, max_p] regime."""
    assert _route_gradient_seeder(2000, 60)[0] is True
    assert _route_gradient_seeder(2000, 5)[0] is False     # trivial pool
    assert _route_gradient_seeder(2000, 300)[0] is False   # cost wall


# ============================ WIN / complementarity ============================

def test_win_proposes_saddle_pair():
    """WIN fixture y=sin(x5)*x31+noise: the saddle pair (5,31) is proposed and ranks #1."""
    X, y = _make("sin_prod")
    proposed, energies, diag = rank_gradient_interaction_pairs(X, y, list(range(P)), seed=0)
    assert diag["learned"] is True
    assert FOCUS in proposed
    assert _rank(energies, FOCUS) == 0


# ============================ end-to-end integration ============================

def test_end_to_end_regression_proposes_saddle():
    """End-to-end in the live MRMR FE pipeline (REGRESSION target -> multi-bin ordinal): the
    gradient seeder proposes the (x5,x31) saddle operands when enabled. The proposer feeds the
    pool from the RAW continuous X (not the discretised ``data``); the existing gates decide."""
    pd = pytest.importorskip("pandas")
    from mlframe.feature_selection.filters import _gradient_interaction_seeder as G
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _make("sin_prod")
    Xdf = pd.DataFrame(X, columns=[f"x{i}" for i in range(P)])

    captured = {}
    orig = G.rank_gradient_interaction_pairs

    def traced(*a, **k):
        r = orig(*a, **k)
        captured["diag"] = r[2]
        captured["proposed"] = r[0]
        return r

    G.rank_gradient_interaction_pairs = traced
    try:
        m = MRMR(max_runtime_mins=3, verbose=1, random_seed=0,
                 fe_gradient_interaction_enable=True, fe_synergy_screen_max_features=80)
        m.fit(Xdf, pd.Series(y, name="target"))   # continuous -> regression -> multi-bin target
    finally:
        G.rank_gradient_interaction_pairs = orig

    assert "proposed" in captured, "gradient seeder was not invoked"
    assert captured["diag"]["learned"] is True
    assert FOCUS in captured["proposed"], f"saddle not proposed; got {captured['proposed']}"


def test_end_to_end_default_off_is_noop():
    """Default OFF: the seeder is not invoked at all (byte-identical legacy path)."""
    pd = pytest.importorskip("pandas")
    from mlframe.feature_selection.filters import _gradient_interaction_seeder as G
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _make("sin_prod")
    Xdf = pd.DataFrame(X, columns=[f"x{i}" for i in range(P)])
    invoked = {"n": 0}
    orig = G.rank_gradient_interaction_pairs

    def traced(*a, **k):
        invoked["n"] += 1
        return orig(*a, **k)

    G.rank_gradient_interaction_pairs = traced
    try:
        m = MRMR(max_runtime_mins=2, verbose=1, random_seed=0)  # default: flag OFF
        assert m.fe_gradient_interaction_enable is False
        m.fit(Xdf, pd.Series(y, name="target"))
    finally:
        G.rank_gradient_interaction_pairs = orig
    assert invoked["n"] == 0


# ============================ biz_value ============================

def test_biz_value_proposing_saddle_improves_accuracy():
    """Proposing the (5,31) product feature improves downstream Ridge OOS R2 vs raw-only,
    on the sin(x5)*x31 fixture. This is the value of surfacing the smooth saddle pair."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import r2_score

    X, y = _make("sin_prod")
    proposed, _, diag = rank_gradient_interaction_pairs(X, y, list(range(P)), seed=0)
    assert FOCUS in proposed

    Xs = _standardize(X.astype(np.float64))
    cv = KFold(5, shuffle=True, random_state=0)
    # baseline: raw features only
    r2_raw = r2_score(y, cross_val_predict(Ridge(alpha=1.0), Xs, y, cv=cv))
    # +engineered product of the proposed saddle pair
    a, b = proposed[0]
    eng = (Xs[:, a] * Xs[:, b]).reshape(-1, 1)
    Xe = np.hstack([Xs, eng])
    r2_eng = r2_score(y, cross_val_predict(Ridge(alpha=1.0), Xe, y, cv=cv))
    assert r2_eng > r2_raw + 0.05, f"engineered R2 {r2_eng:.3f} not > raw {r2_raw:.3f}+0.05"


# ============================ cProfile ============================

def test_cprofile_core_cost_hotspot(capsys):
    """Profile the routed-default core (surrogate fit + analytic energy, NO null) on the
    n=2000/p=60 fixture; report the cost the dispatcher pays per FE step and the top hotspots."""
    X, y = _make("sin_prod")
    Xs = _standardize(X.astype(np.float64))
    pairs = list(combinations(range(P), 2))

    def core():
        rbf, ridge = _fit_rff_ridge(Xs, y, n_components=400, gamma=1.0 / P, alpha=1.0, seed=0)
        return _rff_analytic_mixed_partial_energy(rbf, ridge, Xs, pairs)

    # warm + time
    t0 = time.perf_counter()
    en = core()
    dt_ms = (time.perf_counter() - t0) * 1000

    pr = cProfile.Profile()
    pr.enable()
    core()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(8)
    with capsys.disabled():
        print(f"\n[gradient-seeder core] surrogate+analytic-energy cost = {dt_ms:.1f} ms "
              f"({len(pairs)} pairs, n={N}, p={P})")
        print(s.getvalue().split("\n\n", 1)[-1][:900])
    # core must be cheap (this is what the dispatcher routes); not the heavy full null
    assert dt_ms < 3000
    assert len(en) == len(pairs)
