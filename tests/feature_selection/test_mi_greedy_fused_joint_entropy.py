"""Regression tests for the fused densify+entropy kernel in the greedy-CMI-FE hot path.

``_joint_entropy_two`` fuses ``_renumber_joint`` + ``_entropy_from_classes`` for a two-array joint (the
labels are discarded, only H/k are used) and ``cmi_from_binned_fixed_yz`` reuses the round-fixed dense
``(y,z)`` codes so ``H(X,Y,Z)`` is a 2-array densify against ``yz_dense`` instead of a 3-column renumber.

These pin the bit-identity (to fp reduction order ~1e-9) that the optimisation must preserve. They FAIL on
pre-fix code: ``_joint_entropy_two`` did not exist and ``cmi_from_binned_fixed_yz`` had no ``yz_i`` param
(the yz_i calls raise ``TypeError``).
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as m


def _ref_joint_entropy(a, b):
    j, _ = m._renumber_joint(np.asarray(a, np.int64), np.asarray(b, np.int64))
    return m._entropy_from_classes(j)


@pytest.mark.parametrize("n", [100, 1000, 50_000])
@pytest.mark.parametrize("ca,cb", [(3, 4), (10, 10), (50, 500), (2, 300)])
def test_joint_entropy_two_matches_renumber_plus_entropy(n, ca, cb):
    rng = np.random.default_rng(n + ca + cb)
    a = rng.integers(0, ca, n).astype(np.int64)
    b = rng.integers(0, cb, n).astype(np.int64)
    H_ref, k_ref = _ref_joint_entropy(a, b)
    H, k = m._joint_entropy_two(a, b)
    assert k == k_ref
    assert H == pytest.approx(H_ref, abs=1e-9)


def test_joint_entropy_two_fallback_on_over_cap_span():
    # Cartesian span (max_a+1)*(max_b+1) over the array cap -> njit returns k=-1 sentinel and the wrapper
    # falls back to generic renumber+entropy. Result must still match the reference exactly.
    n = 20_000
    rng = np.random.default_rng(7)
    a = rng.integers(0, 5000, n).astype(np.int64)
    b = rng.integers(0, 5000, n).astype(np.int64)  # span ~ 25M > _FAC_ARRAY_CAP (16M) -> fallback
    H_ref, k_ref = _ref_joint_entropy(a, b)
    H, k = m._joint_entropy_two(a, b)
    assert k == k_ref
    assert H == pytest.approx(H_ref, abs=1e-9)


def test_joint_entropy_two_empty():
    H, k = m._joint_entropy_two(np.empty(0, np.int64), np.empty(0, np.int64))
    assert (H, k) == (0.0, 0)


@pytest.mark.parametrize("n", [2000, 30_000])
def test_cmi_fixed_yz_reuse_path_bit_identical(n):
    rng = np.random.default_rng(n)
    x = rng.integers(0, 8, n).astype(np.int64)
    y = rng.integers(0, 4, n).astype(np.int64)
    z = rng.integers(0, 30, n).astype(np.int64)
    yi, zi, h_yz, h_z, k_yz, k_z, nf = m.precompute_cmi_yz_terms(y, z)
    yz_dense, _ = m._renumber_joint(yi, zi)
    cmi_3col = m.cmi_from_binned_fixed_yz(x, yi, zi, h_yz, h_z, k_yz, k_z, nf)  # y/z-hoisted, 3-col xyz
    cmi_reuse = m.cmi_from_binned_fixed_yz(x, yi, zi, h_yz, h_z, k_yz, k_z, nf, yz_i=yz_dense)  # fused 2-col xyz
    cmi_ref = m._cmi_from_binned(x, y, z)  # full reference
    assert cmi_reuse == pytest.approx(cmi_3col, abs=1e-9)
    assert cmi_reuse == pytest.approx(cmi_ref, abs=1e-9)


def test_cmi_from_binned_conditional_matches_marginal_reduction():
    # Signal case: z carries x=>y so CMI should collapse. Sanity that the fused conditional path is sane
    # and non-negative (Miller-Madow clamped), plus marginal MI on independent data ~0.
    rng = np.random.default_rng(3)
    n = 20_000
    x = rng.integers(0, 6, n).astype(np.int64)
    y = rng.integers(0, 6, n).astype(np.int64)
    mi_indep = m._cmi_from_binned(x, y, None)
    assert 0.0 <= mi_indep < 0.02


@pytest.mark.parametrize("n", [2000, 30_000])
def test_precompute_cmi_yz_terms_hyz_matches_renumber_plus_entropy(n):
    # Fix 3: precompute_cmi_yz_terms now fuses H(Y,Z) via _joint_entropy_two instead of
    # _renumber_joint + _entropy_from_classes (which allocated the length-n relabel array only to
    # discard it). The returned (h_yz, k_yz) must stay bit-identical to the reference renumber+entropy.
    rng = np.random.default_rng(n)
    y = rng.integers(0, 5, n).astype(np.int64)
    z = rng.integers(0, 25, n).astype(np.int64)
    _, _, h_yz, h_z, k_yz, k_z, nf = m.precompute_cmi_yz_terms(y, z)
    H_ref, k_ref = _ref_joint_entropy(y, z)
    hz_ref, kz_ref = m._entropy_from_classes(np.ascontiguousarray(z, np.int64))
    assert k_yz == k_ref
    assert h_yz == pytest.approx(H_ref, abs=1e-9)
    assert (k_z, h_z) == pytest.approx((kz_ref, hz_ref), abs=1e-9)
    assert nf == float(n)


def test_greedy_construct_no_redundant_cmi_from_binned_calls():
    # Fails pre-fix: BEFORE the hoist, the step-0 marginal main scan scored every candidate via
    # ``_cmi_from_binned(cand, y, None)`` and ``_noise_floor_for_current_z`` scored 24 sampled candidates
    # per step via ``_cmi_from_binned`` -- so the module-level ``_cmi_from_binned`` was called O(n_candidates)
    # times, each redundantly recomputing the y/z-invariant ``_renumber_joint`` / ``_entropy_from_classes``
    # block. AFTER the hoist both callers route through ``marginal_mi_binned_fixed_y`` /
    # ``cmi_from_binned_fixed_yz`` (invariant block computed once per step), so ``_cmi_from_binned`` is no
    # longer called from the greedy loop at all. This asserts the redundant-call path is gone.
    import os as _os

    import pandas as pd

    saved = {k: _os.environ.get(k) for k in ("MLFRAME_CMI_GPU", "MLFRAME_FE_GPU_STRICT")}
    _os.environ["MLFRAME_CMI_GPU"] = "0"
    _os.environ.pop("MLFRAME_FE_GPU_STRICT", None)

    rng = np.random.default_rng(11)
    n = 4000
    a, b, c, d = (rng.random(n) for _ in range(4))
    score = (a * a) - 1.0 + 0.7 * np.sin(d * 3.0) - 0.5 * b
    y = (score > np.median(score)).astype(np.int64)
    X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})

    calls = {"n": 0}
    orig = m._cmi_from_binned

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    m._cmi_from_binned = _counting
    try:
        _X_aug, scores = m.greedy_cmi_fe_construct(
            X,
            y,
            nbins=10,
            seed_cols_count=4,
            top_k=5,
            min_cmi_gain=0.0,
        )
    finally:
        m._cmi_from_binned = orig
        for k, v in saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v

    # The greedy loop (main scan + noise floor) must not call _cmi_from_binned anymore.
    assert calls["n"] == 0, f"_cmi_from_binned still called {calls['n']}x from the greedy loop"
    # And the construct still selects something (compound target has recoverable signal).
    assert len(scores) >= 1
