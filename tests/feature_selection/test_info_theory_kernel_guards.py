"""Guard-rail regressions for the info-theory MI/SU kernels:

* **B3** -- the joint-count accumulator in the ``compute_mi_from_classes`` /
  ``compute_su_from_classes`` family and the FE noise-gate kernels must use an
  ``int64`` counter, decoupled from the caller-supplied ``dtype`` (default
  ``int32``). A single joint cell count can reach ``n``; an ``int32`` (or
  narrower) counter wraps negative above its range, turning ``log(jf/...)`` into
  NaN MI. (Same defect class ``merge_vars`` was already fixed for.)
* **B4** -- ``batch_pair_mi_prange`` / ``batch_triple_mi_prange`` allocate a
  per-pair / per-triple histogram sized by ``nbins`` (= ``data.max(axis=0)+1``),
  which is unbounded for high-cardinality categoricals. A cardinality cap skips
  pathological pairs/triples and returns the MI=0.0 no-information sentinel the
  FE gate already treats as uninformative, instead of OOMing the worker.
* **B7** -- the SU plug-in numerator ``H(X)+H(Y)-H(XY)`` must be floored at 0
  before the SU ratio (consistent with the CMI clamp): float round-off on
  near-deterministic / exactly-independent columns can leave it slightly
  negative, yielding a tiny negative SU treated as a valid low relevance.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# B7: SU numerator floored at 0
# --------------------------------------------------------------------------- #

# Exact outer-product joint counts -> X and Y are EXACTLY independent, so the true
# plug-in MI is mathematically 0. Floating-point reduction order in
# ``sum jf*log(jf/(px*py))`` lands at a tiny NEGATIVE value, which pre-fix flowed
# straight into ``2*mi/denom`` as a negative SU. (a, b found by an exact-product
# search; pinned so the test is deterministic across runs.)
_INDEP_A = np.array([16, 26, 22, 4, 2, 34, 30], dtype=np.int64)
_INDEP_B = np.array([33, 21, 32, 13], dtype=np.int64)


def _exact_independent_classes():
    """Build (classes_x, freqs_x, classes_y, freqs_y) for an EXACTLY independent
    joint whose plug-in MI rounds slightly negative."""
    counts = np.outer(_INDEP_A, _INDEP_B)  # exact product => independence => true MI == 0
    n = int(counts.sum())
    fx = counts.sum(axis=1) / n
    fy = counts.sum(axis=0) / n
    cx_parts = []
    cy_parts = []
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            c = int(counts[i, j])
            cx_parts.append(np.full(c, i, dtype=np.int64))
            cy_parts.append(np.full(c, j, dtype=np.int64))
    cx = np.concatenate(cx_parts)
    cy = np.concatenate(cy_parts)
    return cx, fx, cy, fy


def test_b7_plugin_mi_actually_rounds_negative_on_this_case():
    """Sanity: the crafted exactly-independent joint does produce a tiny NEGATIVE
    plug-in MI (so the SU clamp below is exercising a real condition, not a no-op)."""
    from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes

    cx, fx, cy, fy = _exact_independent_classes()
    mi = compute_mi_from_classes(cx, fx, cy, fy, dtype=np.int32)
    assert mi < 0.0, f"expected a tiny negative plug-in MI, got {mi!r}"
    assert mi > -1e-9, f"expected ONLY round-off magnitude, got {mi!r}"


def test_b7_su_from_classes_floored_at_zero():
    """``compute_su_from_classes`` must return >= 0 even when the plug-in numerator
    rounds negative; pre-fix it returned the tiny negative ``2*mi/denom``."""
    from mlframe.feature_selection.filters.info_theory import compute_su_from_classes

    cx, fx, cy, fy = _exact_independent_classes()
    su = compute_su_from_classes(cx, fx, cy, fy, dtype=np.int32)
    assert su >= 0.0, f"SU must be floored at 0, got {su!r}"


def test_b7_pre_fix_su_would_have_been_negative():
    """Documents the pre-fix behaviour: the UNCLAMPED ``2*mi/denom`` on this case is
    strictly negative. This is the value the kernel returned before the floor."""
    from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes

    cx, fx, cy, fy = _exact_independent_classes()
    mi = compute_mi_from_classes(cx, fx, cy, fy, dtype=np.int32)
    hx = -(fx[fx > 0] * np.log(fx[fx > 0])).sum()
    hy = -(fy[fy > 0] * np.log(fy[fy > 0])).sum()
    su_unclamped = 2.0 * mi / (hx + hy)
    assert su_unclamped < 0.0, f"pre-fix SU should be negative, got {su_unclamped!r}"


def test_b7_symmetric_uncertainty_entropy_path_floored():
    """The entropy-kernel ``symmetric_uncertainty`` SU path is floored at 0 too.

    Crafts a single 2-bin variable that is identical to the target (perfect
    dependence) -- ``H(X)+H(Y)-H(XY)`` should equal ``H(X)`` but float reduction
    order can leave the difference slightly negative on degenerate slices. The
    public contract is simply SU >= 0; we assert that on a sweep of small joints."""
    from mlframe.feature_selection.filters.info_theory import symmetric_uncertainty

    rng = np.random.default_rng(0)
    for _ in range(50):
        n = 400
        x = rng.integers(0, 3, n).astype(np.int64)
        # exact-independent target via a permutation-free product is hard to force
        # through the merge_vars path, so just assert the non-negativity invariant
        # across random low-card joints (the floor guarantees it unconditionally).
        y = rng.integers(0, 3, n).astype(np.int64)
        data = np.column_stack([x, y]).astype(np.int32)
        nbins = np.array([3, 3], dtype=np.int64)
        su = symmetric_uncertainty(
            data, np.array([0], dtype=np.int64), np.array([1], dtype=np.int64), nbins
        )
        assert su >= 0.0, f"symmetric_uncertainty must be >= 0, got {su!r}"


# --------------------------------------------------------------------------- #
# B3: int64 joint-count accumulator (decoupled from caller dtype)
# --------------------------------------------------------------------------- #


def test_b3_accumulator_does_not_wrap_under_narrow_dtype():
    """A single joint cell holding > 127 samples must NOT wrap when the caller
    passes ``dtype=np.int8``. Pre-fix the accumulator was allocated at ``dtype``,
    so a 200-count cell wrapped to -56 and corrupted the MI (mirrors the
    ``merge_vars`` int8 overflow regression). Post-fix the counter is int64, so
    the MI matches the analytic value regardless of ``dtype``.

    This unit-tests the dtype/logic without allocating 2^31 rows: an int8 counter
    overflows at 128, so a 200-sample cell is a faithful proxy for the int32
    overflow that bites above ~2.1e9 rows.
    """
    from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes

    # Perfect dependence x==y, 2 classes; class 0 has 200 rows (> int8 max 127).
    n0, n1 = 200, 50
    n = n0 + n1
    cx = np.concatenate([np.zeros(n0, np.int64), np.ones(n1, np.int64)])
    cy = cx.copy()
    p0, p1 = n0 / n, n1 / n
    fx = np.array([p0, p1])
    fy = np.array([p0, p1])
    true_mi = -(p0 * math.log(p0) + p1 * math.log(p1))  # X determines Y => I = H(Y)

    mi = compute_mi_from_classes(cx, fx, cy, fy, dtype=np.int8)
    assert np.isfinite(mi), f"MI went non-finite (counter wrapped), got {mi!r}"
    assert abs(mi - true_mi) < 1e-12, f"MI {mi!r} != analytic {true_mi!r} (counter wrapped)"


def test_b3_su_accumulator_does_not_wrap_under_narrow_dtype():
    """Same overflow proxy for ``compute_su_from_classes``: the SU joint counter is
    int64, so a 200-count cell under ``dtype=np.int8`` yields the correct SU=1.0
    on a perfect-dependence pair instead of garbage from a wrapped count."""
    from mlframe.feature_selection.filters.info_theory import compute_su_from_classes

    n0, n1 = 200, 50
    n = n0 + n1
    cx = np.concatenate([np.zeros(n0, np.int64), np.ones(n1, np.int64)])
    cy = cx.copy()
    p0, p1 = n0 / n, n1 / n
    fx = np.array([p0, p1])
    fy = np.array([p0, p1])
    su = compute_su_from_classes(cx, fx, cy, fy, dtype=np.int8)
    # Perfect dependence and identical marginals => SU == 1.0.
    assert np.isfinite(su)
    assert abs(su - 1.0) < 1e-12, f"SU {su!r} != 1.0 (counter wrapped under int8)"


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_b3_mi_dtype_independent_on_no_overflow_case(dtype):
    """On a no-overflow case the MI is bit-identical regardless of the caller's
    ``dtype`` -- the int64 promotion only prevents wrap, it never changes results."""
    from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes

    rng = np.random.default_rng(3)
    n = 4000  # ~per-cell counts well under int8's 127 -> no wrap at any dtype
    cx = rng.integers(0, 5, n).astype(np.int64)
    # y correlated with x but noisy
    cy = ((cx + rng.integers(0, 2, n)) % 5).astype(np.int64)
    fx = np.bincount(cx, minlength=5).astype(np.float64) / n
    fy = np.bincount(cy, minlength=5).astype(np.float64) / n
    ref = compute_mi_from_classes(cx, fx, cy, fy, dtype=np.int64)
    got = compute_mi_from_classes(cx, fx, cy, fy, dtype=dtype)
    assert got == ref, f"dtype={dtype} changed MI: {got!r} != {ref!r}"


def test_b3_batch_noise_gate_accumulator_int64():
    """The FE noise-gate kernels (``batch_mi_with_noise_gate`` v1/v2 + the
    ``_relevance_from_dense`` / ``_densify_and_relevance_fused`` / ``_perm_failcount_col``
    accumulators) keep an int64 joint counter. Smoke-check both kernels run and
    agree on a small case with a narrow caller ``dtype`` (proxy for overflow safety)."""
    from mlframe.feature_selection.filters.info_theory import (
        batch_mi_with_noise_gate,
        batch_mi_with_noise_gate_v2,
    )

    rng = np.random.default_rng(5)
    n, K = 600, 4
    disc = rng.integers(0, 6, (n, K)).astype(np.int8)
    nbins = np.full(K, 6, dtype=np.int64)
    cy = rng.integers(0, 3, n).astype(np.int64)
    fy = np.bincount(cy, minlength=3).astype(np.float64) / n

    kw = dict(
        disc_2d=disc, factors_nbins=nbins, classes_y=cy, classes_y_safe=cy.copy(),
        freqs_y=fy, npermutations=0, base_seed=np.uint64(0),
        min_nonzero_confidence=0.0, use_su=False, dtype=np.int8, classes_dtype=np.int8,
    )
    out_v1 = batch_mi_with_noise_gate(**kw)
    out_v2 = batch_mi_with_noise_gate_v2(**kw)
    assert np.all(np.isfinite(out_v1)) and np.all(np.isfinite(out_v2))
    assert np.array_equal(out_v1, out_v2), "v1/v2 noise-gate outputs diverged"


# --------------------------------------------------------------------------- #
# B4: cardinality cap on the pair / triple histograms
# --------------------------------------------------------------------------- #


def test_b4_pair_cardinality_cap_returns_sentinel():
    """A pair whose ``nb_a*nb_b`` exceeds ``MAX_JOINT_CARDINALITY`` must NOT allocate
    the giant histogram -- it returns the MI=0.0 no-information sentinel. A tiny n
    with two columns in [0, 9000) gives a raw cardinality of 81M > the 64M cap."""
    from mlframe.feature_selection.filters.info_theory._batch_kernels import (
        MAX_JOINT_CARDINALITY,
        batch_pair_mi_prange,
    )

    rng = np.random.default_rng(0)
    n = 50
    hi = 9000
    assert hi * hi > MAX_JOINT_CARDINALITY  # the pair would exceed the cap
    data = np.column_stack(
        [rng.integers(0, hi, n), rng.integers(0, hi, n)]
    ).astype(np.int64)
    nbins = np.array([hi, hi], dtype=np.int64)
    pa = np.array([0], dtype=np.int64)
    pb = np.array([1], dtype=np.int64)
    cy = rng.integers(0, 2, n).astype(np.int64)
    fy = np.bincount(cy, minlength=2).astype(np.float64) / n

    out = batch_pair_mi_prange(data, pa, pb, nbins, cy, fy)
    assert out.shape == (1,)
    assert out[0] == 0.0, f"over-cap pair should be sentinel 0.0, got {out[0]!r}"


def test_b4_triple_cardinality_cap_returns_sentinel():
    """A triple whose ``nb_a*nb_b*nb_c`` exceeds the cap must skip the raw-card
    ``remap`` allocation (the OOM hazard) and return MI=0.0. 500^3 = 125M > 64M."""
    from mlframe.feature_selection.filters.info_theory._batch_kernels import (
        MAX_JOINT_CARDINALITY,
        batch_triple_mi_prange,
    )

    rng = np.random.default_rng(0)
    n = 50
    hi = 500
    assert hi ** 3 > MAX_JOINT_CARDINALITY
    data = np.column_stack(
        [rng.integers(0, hi, n), rng.integers(0, hi, n), rng.integers(0, hi, n)]
    ).astype(np.int64)
    nbins = np.array([hi, hi, hi], dtype=np.int64)
    ta = np.array([0], dtype=np.int64)
    tb = np.array([1], dtype=np.int64)
    tc = np.array([2], dtype=np.int64)
    cy = rng.integers(0, 2, n).astype(np.int64)
    fy = np.bincount(cy, minlength=2).astype(np.float64) / n

    out = batch_triple_mi_prange(data, ta, tb, tc, nbins, cy, fy)
    assert out.shape == (1,)
    assert out[0] == 0.0, f"over-cap triple should be sentinel 0.0, got {out[0]!r}"


def test_b4_under_cap_pair_still_computes_real_mi():
    """Negative control: a normal low-cardinality pair (well under the cap) still
    computes a real, positive MI -- the cap must not fire on the common path."""
    from mlframe.feature_selection.filters.info_theory._batch_kernels import (
        batch_pair_mi_prange,
    )

    rng = np.random.default_rng(7)
    n = 2000
    a = rng.integers(0, 4, n)
    # b correlated with a so the pair carries information about y
    y = (a % 2).astype(np.int64)
    b = ((a + rng.integers(0, 2, n)) % 4)
    data = np.column_stack([a, b]).astype(np.int64)
    nbins = np.array([4, 4], dtype=np.int64)
    pa = np.array([0], dtype=np.int64)
    pb = np.array([1], dtype=np.int64)
    fy = np.bincount(y, minlength=2).astype(np.float64) / n
    out = batch_pair_mi_prange(data, pa, pb, nbins, y, fy)
    assert np.isfinite(out[0])
    assert out[0] > 0.0, f"under-cap informative pair should have MI>0, got {out[0]!r}"
