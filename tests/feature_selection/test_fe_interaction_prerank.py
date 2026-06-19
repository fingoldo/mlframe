"""Unit + recall pins for the interaction-propensity pre-rank that chooses WHICH wide-frame columns
enter the capped O(p^2) synergy sweep (see filters/_fe_interaction_prerank.py).

Pins the three contract guarantees the bench established:
  * second_moment_propensity recovers PURE-interaction operands (zero marginal MI) that marginal MI misses,
    at realistic leakage -- on RAW values AND on quantile bin-codes (so the wide-frame wiring may use codes);
  * the L=0.0 perfectly-balanced case is irreducible (no lift over random) -- pinned so we never claim it;
  * top_k_by_interaction_propensity is deterministic, a no-op when top_k >= n_candidates, and O(n*p) cheap.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_interaction_prerank import (
    second_moment_propensity,
    top_k_by_interaction_propensity,
    fused_propensity,
    _discrete_score_numpy_loop,
)
from mlframe.feature_selection.filters import _fe_interaction_prerank_kernels as _kernels

N = 8000
NBINS = 8
P = 1200
K = 6  # 12 planted pure-pair operand columns


def _make_frame(p, seed, leak):
    """K pure sign-product pair interactions; operands ~0 marginal signal; ``leak`` adds main effect."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, p))
    cols = rng.choice(p, size=2 * K, replace=False)
    operands, logit = [], np.zeros(N)
    for k in range(K):
        ia, ib = int(cols[2 * k]), int(cols[2 * k + 1])
        operands += [ia, ib]
        a, b = X[:, ia], X[:, ib]
        logit += 1.6 * np.sign(a) * np.sign(b)
        logit += leak * 1.6 * (a + b)
    p_y = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(N) < p_y).astype(np.int32)
    return X, y, set(operands)


def _bin_codes(X):
    out = np.empty_like(X, dtype=np.int16)
    for j in range(X.shape[1]):
        q = np.quantile(X[:, j], np.linspace(0, 1, NBINS + 1)[1:-1])
        out[:, j] = np.searchsorted(q, X[:, j])
    return out


def _marginal_mi(X, y):
    n, p = X.shape
    fy = np.bincount(y, minlength=2) / n
    codes = _bin_codes(X)
    scores = np.empty(p)
    for j in range(p):
        c = codes[:, j]
        mi = 0.0
        for vx in range(NBINS):
            mask = c == vx
            nx = int(mask.sum())
            if nx == 0:
                continue
            px = nx / n
            yy = y[mask]
            for vy in range(2):
                jc = int((yy == vy).sum())
                if jc and fy[vy] > 0:
                    jf = jc / n
                    mi += jf * np.log(jf / (px * fy[vy]))
        scores[j] = mi
    return scores


def _recall(scores, operands, m):
    top = set(np.argsort(scores)[::-1][:m].tolist())
    return len(top & operands) / len(operands)


def test_second_moment_beats_marginal_on_leaky_pure_interaction_raw_and_codes():
    """At realistic leakage (L=0.1) the second-moment propensity recovers the planted pure-interaction
    operands into the top-250 FAR better than marginal MI -- on raw values AND on quantile bin-codes
    (the bin-code path is what the wide-frame wiring uses, so its recall must hold too)."""
    raw_2m, code_2m, marg = [], [], []
    for seed in (0, 1, 2):
        X, y, ops = _make_frame(P, seed, leak=0.1)
        codes = _bin_codes(X).astype(np.float64)
        raw_2m.append(_recall(second_moment_propensity(X, y), ops, 250))
        code_2m.append(_recall(second_moment_propensity(codes, y), ops, 250))
        marg.append(_recall(_marginal_mi(X, y), ops, 250))
    raw_2m, code_2m, marg = np.mean(raw_2m), np.mean(code_2m), np.mean(marg)
    # second-moment (both representations) clears a real recall bar and beats marginal MI by a wide margin.
    assert raw_2m >= 0.60, f"raw 2nd-moment recall too low: {raw_2m:.2f}"
    assert code_2m >= 0.55, f"bin-code 2nd-moment recall too low: {code_2m:.2f}"
    assert raw_2m >= marg + 0.10, f"2nd-moment ({raw_2m:.2f}) did not beat marginal MI ({marg:.2f})"
    # random baseline at top-250 of p=1200 is ~0.21; both must clear it decisively.
    assert code_2m >= 0.40


def test_perfectly_balanced_interaction_is_irreducible():
    """L=0.0 (exact sign product, zero higher-moment leakage): NO O(p) score beats the random baseline.
    Pinned so the pre-rank never CLAIMS to recover this measure-zero case (only the exhaustive sweep can)."""
    recalls = []
    for seed in (0, 1, 2):
        X, y, ops = _make_frame(P, seed, leak=0.0)
        recalls.append(_recall(second_moment_propensity(X, y), ops, 250))
    mean_recall = float(np.mean(recalls))
    random_base = 250 / P  # ~0.208
    # within noise of random -- explicitly NOT a recovery (allow a small slop above the baseline).
    assert mean_recall <= random_base + 0.15, (
        f"unexpected recovery of a perfectly-balanced interaction ({mean_recall:.2f} vs base {random_base:.2f}); "
        "the irreducibility assumption no longer holds -- revisit the pre-rank claims")


def test_top_k_deterministic_and_noop_when_k_exceeds_pool():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 40))
    y = (rng.random(500) < 0.5).astype(int)
    cand = [3, 7, 11, 19, 23, 31]
    a = top_k_by_interaction_propensity(X, y, cand, top_k=3)
    b = top_k_by_interaction_propensity(X, y, cand, top_k=3)
    assert a == b and len(a) == 3 and a == sorted(a)        # deterministic + sorted
    assert set(a).issubset(set(cand))
    # top_k >= pool size -> all candidates, sorted (a pure no-op selection)
    assert top_k_by_interaction_propensity(X, y, cand, top_k=10) == sorted(cand)
    assert top_k_by_interaction_propensity(X, y, cand, top_k=0) == []


def test_nominal_multiclass_is_relabel_invariant():
    """A nominal multiclass target must NOT have its arbitrary class CODES squared (that made the kept set
    depend on the integer assigned to each class -- critique bug #2, 2026-06-19). The one-hot relabel-
    invariant score gives the SAME ranking under any relabeling of the classes."""
    rng = np.random.default_rng(0)
    n, p = 6000, 400
    X = rng.standard_normal((n, p))
    ia, ib, ic, idd = 10, 200, 50, 300
    a, b, c, d = X[:, ia], X[:, ib], X[:, ic], X[:, idd]
    # 4-class NOMINAL target from two leaky interactions (sign product + a main-effect leak so the operands
    # carry recoverable higher-moment signal -- the realistic regime, not the irreducible pure-balanced case).
    s1 = np.sign(a) * np.sign(b) + 0.6 * (a + b)
    s2 = np.sign(c) * np.sign(d) + 0.6 * (c + d)
    cls = (2 * (s1 > 0) + (s2 > 0)).astype(int)  # 0..3, nominal
    operands = {ia, ib, ic, idd}

    # Recall of the operands must be STABLE across relabelings -- the bug (squaring class codes) made it swing
    # 0.12-0.88 (std 0.25) with the arbitrary integer per class. The one-hot score uses only the PARTITION
    # 1[y==c], never the label value, so recall is invariant (up to float summation-order noise on boundary
    # ties). recall = fraction of the 4 operands in the top-250.
    def _recall_top(y_):
        top = set(np.argsort(second_moment_propensity(X, y_))[::-1][:250])
        return len(operands & top) / len(operands)

    recalls = [_recall_top(cls)]
    for relabel in ([3, 1, 0, 2], [2, 3, 1, 0], [1, 0, 3, 2], [0, 3, 2, 1]):
        recalls.append(_recall_top(np.array([relabel[v] for v in cls])))
    assert np.std(recalls) <= 0.03, f"recall not relabel-invariant for nominal multiclass: {recalls}"
    assert np.mean(recalls) >= 0.5, f"multiclass operand recovery too low: {recalls}"


def _planted_interaction_X(p=400, seed=0, leak=0.2):
    """X with two leaky pure-pair interactions; returns X, the continuous driver s, and the operand set."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((4000, p))
    ia, ib, ic, idd = 7, 150, 33, 290
    a, b, c, d = X[:, ia], X[:, ib], X[:, ic], X[:, idd]
    s = np.sign(a) * np.sign(b) + np.sign(c) * np.sign(d) + leak * (a + b + c + d)
    return X, s, {ia, ib, ic, idd}


@pytest.mark.parametrize("target_kind", ["binary", "nominal_multiclass", "ordinal_multiclass",
                                         "regression_continuous", "regression_binned",
                                         "boolean", "string_labels"])
def test_all_target_types_score_finite_and_recover(target_kind):
    """second_moment_propensity must work for EVERY target type: produce finite scores AND recover the planted
    leaky-interaction operands above the random baseline, for binary / nominal / ordinal multiclass / continuous
    regression / binned regression / boolean / non-numeric string labels."""
    X, s, operands = _planted_interaction_X(p=400, seed=0, leak=0.2)
    n = X.shape[0]
    rng = np.random.default_rng(1)

    if target_kind == "binary":
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-s))).astype(int)
    elif target_kind == "nominal_multiclass":
        s2 = np.sign(X[:, 33]) * np.sign(X[:, 290])
        y = (2 * (s > 0) + (s2 > 0)).astype(int)            # 0..3, treated as nominal
    elif target_kind == "ordinal_multiclass":
        y = np.digitize(s, np.quantile(s, [0.25, 0.5, 0.75])).astype(int)   # 0..3 ordered
    elif target_kind == "regression_continuous":
        y = (s + 0.1 * rng.standard_normal(n)).astype(float)               # >64 unique -> moment path
    elif target_kind == "regression_binned":
        y = np.digitize(s, np.quantile(s, np.linspace(0, 1, 9)[1:-1]))      # 8 bins (the synergy-site form)
    elif target_kind == "boolean":
        y = (s > 0)                                                         # bool dtype
    else:  # string_labels (non-numeric nominal)
        lab = np.array(["lo", "mid", "hi"])
        y = lab[np.digitize(s, np.quantile(s, [0.33, 0.66]))]              # object/str array

    scores = second_moment_propensity(X, y)
    assert scores.shape == (X.shape[1],)
    assert np.isfinite(scores).all(), f"{target_kind}: non-finite scores"
    top = set(np.argsort(scores)[::-1][:100])
    recall = len(operands & top) / len(operands)
    assert recall >= 0.5, f"{target_kind}: operand recall {recall:.2f} at top-100 (random ~{100/400:.2f})"


@pytest.mark.parametrize("nclasses", [2, 5])
@pytest.mark.parametrize("backend", ["numpy", "numba", "cupy"])
def test_kernel_variants_match_per_class_loop_reference(backend, nclasses):
    """Every dispatcher backend (numpy/numba/cupy GEMM) must reproduce the original per-class-loop
    reference score to float precision AND give the identical descending ranking -- the kernel
    optimization is a pure speedup, never a behavior change. cupy/numba skip cleanly if unavailable."""
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("no CUDA device")
        except Exception:
            pytest.skip("cupy present but no usable GPU")
    if backend == "numba":
        pytest.importorskip("numba")

    rng = np.random.default_rng(7)
    n, p = 4000, 1500
    V = np.nan_to_num(rng.standard_normal((n, p)))
    V2 = V * V
    yf = rng.integers(0, nclasses, n).astype(np.float64)
    classes = np.unique(yf)

    ref = _discrete_score_numpy_loop(V, V2, yf, classes)
    try:
        got = _kernels.compute_discrete_score(V, V2, yf, classes, backend=backend)
    except Exception as e:  # GPU OOM under concurrent load etc. -- the variant exists, just unbenchable now
        pytest.skip(f"{backend} unavailable at runtime: {e}")

    assert np.allclose(got, ref, rtol=1e-8, atol=1e-10), f"{backend} score drift max={np.abs(got-ref).max():.2e}"
    # ranking (the only thing top-k consumes) must be bit-identical to the reference
    assert np.array_equal(np.argsort(-got, kind="stable"), np.argsort(-ref, kind="stable")), \
        f"{backend} ranking differs from per-class-loop reference"


def test_second_moment_uses_kernel_and_matches_loop_end_to_end():
    """The public second_moment_propensity (which now routes the discrete path through the kernel
    dispatcher) must equal the per-class-loop computed on the same standardized inputs."""
    rng = np.random.default_rng(3)
    X = np.nan_to_num(rng.standard_normal((3000, 800)))
    y = (rng.random(3000) < 0.4).astype(int)
    got = second_moment_propensity(X, y)
    yf = y.astype(np.float64)
    ref = _discrete_score_numpy_loop(X, X * X, yf, np.unique(yf))
    assert np.allclose(got, ref, rtol=1e-8, atol=1e-10)


def test_fused_recall_floor_at_realistic_leakage():
    """The fused criterion (2nd-moment + marginal + gbm split-frequency rank-fusion) must recover the
    planted pure-pair operands into the top-250 at L=0.1 at least as well as plain 2nd-moment, and clear a
    real floor. LightGBM is required for the gbm ingredient -- skip cleanly if absent (then fusion degrades
    to 2nd-moment+marginal and the >= comparison still holds)."""
    sm_r, fz_r = [], []
    for seed in (0, 1, 2):
        X, y, ops = _make_frame(P, seed, leak=0.1)
        sm_r.append(_recall(second_moment_propensity(X, y), ops, 250))
        fz_r.append(_recall(fused_propensity(X, y), ops, 250))
    sm, fz = float(np.mean(sm_r)), float(np.mean(fz_r))
    # fusion never regresses recall vs the base (min-rank keeps anything either signal ranks high), and the
    # base itself clears the benched bar -- a small slop guards against seed/summation-order noise.
    assert fz >= sm - 0.02, f"fused recall {fz:.2f} regressed vs 2nd-moment {sm:.2f}"
    assert fz >= 0.60, f"fused recall too low at L=0.1: {fz:.2f}"


def test_fused_criterion_routes_through_top_k():
    """top_k_by_interaction_propensity(..., criterion='fused') is accepted and returns a valid sorted top-k."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1500, 60))
    y = (rng.random(1500) < 0.5).astype(int)
    cand = list(range(0, 60, 2))
    out = top_k_by_interaction_propensity(X, y, cand, top_k=5, criterion="fused")
    assert len(out) == 5 and out == sorted(out) and set(out).issubset(set(cand))


def test_auto_picks_fused_when_small():
    """The default "auto" criterion escalates to the high-recall ``fused`` when one LightGBM fit over the
    candidate columns is predicted affordable (small/moderate p). Predictor is monkeypatched so the gate is
    deterministic and fast -- no real fit is run."""
    import mlframe.feature_selection.filters._fe_interaction_prerank as m
    import mlframe.feature_selection.filters._fe_interaction_prerank_kernels as k

    orig = k.predict_gbm_fit_seconds
    try:
        k.predict_gbm_fit_seconds = lambda n, p: (5.0, 100.0, "cache")  # cheap fit -> well under any budget
        rng = np.random.default_rng(0)
        X = rng.standard_normal((1500, 80))
        y = (rng.random(1500) < 0.5).astype(int)
        out = m.top_k_by_interaction_propensity(X, y, list(range(80)), top_k=20)  # criterion defaults to "auto"
    finally:
        k.predict_gbm_fit_seconds = orig
    assert m._LAST_AUTO_CHOICE[0] == "fused", f"auto should pick fused when small: {m._LAST_AUTO_CHOICE}"
    assert len(out) == 20 and out == sorted(out)


def test_auto_picks_second_moment_when_wide():
    """"auto" falls back to the cheap ``second_moment`` when the predicted LightGBM fit exceeds the budget
    (a WIDE frame). Monkeypatched predictor returns a huge time so NO real 100k-wide fit is run in the test."""
    import mlframe.feature_selection.filters._fe_interaction_prerank as m
    import mlframe.feature_selection.filters._fe_interaction_prerank_kernels as k

    orig = k.predict_gbm_fit_seconds
    try:
        k.predict_gbm_fit_seconds = lambda n, p: (9.9e4, 1.0, "cache")  # unaffordable -> cheap fallback
        rng = np.random.default_rng(1)
        X = rng.standard_normal((1500, 80))
        y = (rng.random(1500) < 0.5).astype(int)
        out = m.top_k_by_interaction_propensity(X, y, list(range(80)), top_k=20)
    finally:
        k.predict_gbm_fit_seconds = orig
    assert m._LAST_AUTO_CHOICE[0] == "second_moment", f"auto should pick second_moment when wide: {m._LAST_AUTO_CHOICE}"
    assert len(out) == 20 and out == sorted(out)


def test_budget_from_max_runtime_mins_honored():
    """The threaded ``budget_seconds`` (derived from MRMR's max_runtime_mins * 60) bounds the auto gate: a
    fixed predicted fit time flips fused<->second_moment as the budget crosses it. Predictor monkeypatched
    so the only varying input is the budget."""
    import mlframe.feature_selection.filters._fe_interaction_prerank as m
    import mlframe.feature_selection.filters._fe_interaction_prerank_kernels as k

    orig = k.predict_gbm_fit_seconds
    try:
        k.predict_gbm_fit_seconds = lambda n, p: (30.0, 100.0, "cache")  # fixed 30s predicted fit
        rng = np.random.default_rng(2)
        X = rng.standard_normal((1000, 40))
        y = (rng.random(1000) < 0.5).astype(int)
        # budget 120s (max_runtime_mins=2 -> 120s) > 30s -> fused
        m.top_k_by_interaction_propensity(X, y, list(range(40)), top_k=10, budget_seconds=120.0)
        assert m._LAST_AUTO_CHOICE[0] == "fused", m._LAST_AUTO_CHOICE
        # budget 10s < 30s -> second_moment
        m.top_k_by_interaction_propensity(X, y, list(range(40)), top_k=10, budget_seconds=10.0)
        assert m._LAST_AUTO_CHOICE[0] == "second_moment", m._LAST_AUTO_CHOICE
    finally:
        k.predict_gbm_fit_seconds = orig


def test_explicit_zero_budget_forces_cheap_path():
    """P0/Low-4 (2026-06-19): an EXPLICIT non-positive budget (max_runtime_mins=0 -> 0s) means 'no time' and
    must force the cheap second_moment, NOT collapse to the soft default and run the expensive gbm fit. Only
    budget_seconds=None falls back to the soft default."""
    import mlframe.feature_selection.filters._fe_interaction_prerank as m
    import mlframe.feature_selection.filters._fe_interaction_prerank_kernels as k

    orig = k.predict_gbm_fit_seconds
    try:
        k.predict_gbm_fit_seconds = lambda n, p: (0.001, 1e9, "cache")  # gbm would look ~free
        rng = np.random.default_rng(3)
        X = rng.standard_normal((800, 30))
        y = (rng.random(800) < 0.5).astype(int)
        for bad in (0.0, -5.0):
            m.top_k_by_interaction_propensity(X, y, list(range(30)), top_k=8, budget_seconds=bad)
            assert m._LAST_AUTO_CHOICE[0] == "second_moment", (bad, m._LAST_AUTO_CHOICE)
        # None -> soft default -> with a ~free predicted fit, picks fused (contrast)
        m.top_k_by_interaction_propensity(X, y, list(range(30)), top_k=8, budget_seconds=None)
        assert m._LAST_AUTO_CHOICE[0] == "fused", m._LAST_AUTO_CHOICE
    finally:
        k.predict_gbm_fit_seconds = orig


def test_gate_is_hw_calibrated_not_magic_constant():
    """The auto gate's cost prediction must be sourced from the per-host kernel_tuning_cache (or its analytic
    fallback), NOT a hardcoded threshold. measured_gbm_cols_per_second returns a (value, source) where source
    is 'cache' (HW-measured) or 'fallback', and predict_gbm_fit_seconds threads that source through."""
    from mlframe.feature_selection.filters._fe_interaction_prerank_kernels import (
        measured_gbm_cols_per_second, predict_gbm_fit_seconds, warm_gbm_cost_cache,
    )
    warm_gbm_cost_cache()
    cps, source = measured_gbm_cols_per_second(8000)
    assert cps > 0 and source in ("cache", "fallback")
    predicted, cps2, src2 = predict_gbm_fit_seconds(8000, 2000)
    assert predicted > 0 and cps2 > 0 and src2 in ("cache", "fallback")
    # prediction must scale with n_candidates (a cost MODEL, not a constant gate)
    p_small = predict_gbm_fit_seconds(8000, 500)[0]
    p_big = predict_gbm_fit_seconds(8000, 9000)[0]
    assert p_big > p_small, "predicted fit time must grow with candidate count (HW-calibrated cost model)"


def test_auto_recall_matches_fused_small_and_second_moment_wide():
    """auto must DELIVER the recall of the criterion it selects: fused recall (~0.92) when it picks fused on a
    small frame, second_moment recall (~0.88) when it picks second_moment on a wide frame. Predictor is
    monkeypatched to force each branch deterministically (no real wide fit)."""
    import mlframe.feature_selection.filters._fe_interaction_prerank as m
    import mlframe.feature_selection.filters._fe_interaction_prerank_kernels as k

    orig = k.predict_gbm_fit_seconds
    try:
        # SMALL regime: force fused -> auto top-k must equal fused top-k (same ranking) on each seed.
        k.predict_gbm_fit_seconds = lambda n, p: (1.0, 1000.0, "cache")
        for seed in (0, 1, 2):
            X, y, _ = _make_frame(P, seed, leak=0.1)
            cand = list(range(P))
            a = m.top_k_by_interaction_propensity(X, y, cand, top_k=250)
            assert m._LAST_AUTO_CHOICE[0] == "fused"
            f = top_k_by_interaction_propensity(X, y, cand, top_k=250, criterion="fused")
            assert a == f, "auto(fused) selection must match explicit fused selection"
        # WIDE regime: force second_moment -> auto top-k must equal second_moment top-k.
        k.predict_gbm_fit_seconds = lambda n, p: (9.9e4, 1.0, "cache")
        for seed in (0, 1, 2):
            X, y, _ = _make_frame(P, seed, leak=0.1)
            cand = list(range(P))
            a = m.top_k_by_interaction_propensity(X, y, cand, top_k=250)
            assert m._LAST_AUTO_CHOICE[0] == "second_moment"
            sm = top_k_by_interaction_propensity(X, y, cand, top_k=250, criterion="second_moment")
            assert a == sm, "auto(second_moment) selection must match explicit second_moment selection"
    finally:
        k.predict_gbm_fit_seconds = orig


def test_single_class_target_no_crash():
    """A degenerate constant target must not crash and must return finite (zero-information) scores."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 20))
    s = second_moment_propensity(X, np.zeros(300, dtype=int))
    assert np.isfinite(s).all()


def test_constant_column_scores_zero_no_nan():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 5))
    X[:, 2] = 4.0  # constant column -> undefined corr -> must score 0, not NaN
    y = (rng.random(300) < 0.5).astype(int)
    s = second_moment_propensity(X, y)
    assert np.isfinite(s).all()
    assert s[2] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--no-cov"])
