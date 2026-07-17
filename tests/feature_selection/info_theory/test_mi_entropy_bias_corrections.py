"""Regression tests for confirmed MI / entropy-estimator BIAS fixes (SA4/SA10/SA11/SA12/SA14/SA15).

Each test pins the CORRECTED (less-biased) behaviour and would FAIL on the pre-fix code. The shared theme: a
plug-in / mis-corrected information estimator manufactured a deterministic offset on an INDEPENDENT (or
independent-given-Z) pair where the true quantity is ~0, or distorted a known-value case; the principled bias
correction removes the offset. Each test reproduces the pre-fix value inline (no destructive git) to prove the
gap is real, then asserts the post-fix value is near truth.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# SA4 -- interaction information: MM joint bias must use OCCUPIED, not design, bins
# ---------------------------------------------------------------------------
def test_sa4_ii_occupied_bins_removes_false_synergy_on_independent_heavy_tailed_pair():
    """Pre-fix the joint MM bias used DESIGN cardinality nbins_a*nbins_b; on a heavy-tailed pair whose joint
    occupies far fewer than nbins_a*nbins_b cells this over-corrects the joint term by the empty-cell count and
    leaves a large deterministic II offset (false synergy). Using occupied joint bins (entropy-kernel convention)
    drives the corrected II of an INDEPENDENT pair to ~0."""
    from mlframe.feature_selection.filters._interaction_information import (
        pair_interaction_information,
        _marginal_mi_codes,
    )

    rng = np.random.default_rng(0)
    n, NB = 600, 20
    a = np.clip(rng.geometric(0.5, n) - 1, 0, NB - 1).astype(np.int64)
    b = np.clip(rng.geometric(0.5, n) - 1, 0, NB - 1).astype(np.int64)
    y = np.clip(rng.geometric(0.5, n) - 1, 0, NB - 1).astype(np.int64)
    inv_n = 1.0 / n
    mi_a, k_a = _marginal_mi_codes(a, y, NB, NB, inv_n)
    mi_b, k_b = _marginal_mi_codes(b, y, NB, NB, inv_n)
    pair_mi, k_j = _marginal_mi_codes(a * NB + b, y, NB * NB, NB, inv_n)
    k_y = int((np.bincount(y, minlength=NB) > 0).sum())
    assert k_j < NB * NB, "test needs a SPARSE joint (occupied < design) to exercise the bug"

    # Pre-fix path: design-cardinality fallback (no occupied args). Large spurious offset.
    ii_design = pair_interaction_information(mi_a, mi_b, pair_mi, NB, NB, NB, n)
    # Post-fix path: occupied-bin correction.
    ii_occupied = pair_interaction_information(
        mi_a,
        mi_b,
        pair_mi,
        NB,
        NB,
        NB,
        n,
        k_a_occupied=k_a,
        k_b_occupied=k_b,
        k_joint_occupied=k_j,
        k_y_occupied=k_y,
    )
    assert abs(ii_design) > 1.0, f"pre-fix design-cardinality II should carry a large offset, got {ii_design}"
    assert abs(ii_occupied) < 0.3, f"occupied-bin II of an independent pair should be ~0, got {ii_occupied}"
    assert abs(ii_occupied) < abs(ii_design)


# ---------------------------------------------------------------------------
# SA10 -- Chao-Shen MI: the three entropies must share ONE coverage basis
# ---------------------------------------------------------------------------
def test_sa10_chao_shen_mi_shared_coverage_kills_false_mi_on_independent_sparse_pair():
    """Pre-fix H_x+H_y-H_xy used THREE independently-estimated Good-Turing coverages; on a sparse independent
    pair the mismatched rescalings leave a deterministic positive residual (false MI). A single shared coverage
    makes the three terms commensurable so MI -> 0 on independence."""
    from mlframe.feature_selection.filters._chao_shen import (
        chao_shen_mi,
        chao_shen_entropy_from_counts,
    )

    rng = np.random.default_rng(16)
    n = 200
    x = np.clip(rng.geometric(0.25, n) - 1, 0, 30).astype(np.int64)
    y = np.clip(rng.geometric(0.25, n) - 1, 0, 30).astype(np.int64)
    Kx, Ky = int(x.max()) + 1, int(y.max()) + 1
    j = np.zeros((Kx, Ky), dtype=np.int64)
    for i in range(n):
        j[x[i], y[i]] += 1
    flat = j.ravel().astype(np.int64)
    rs = j.sum(1).astype(np.int64)
    cs = j.sum(0).astype(np.int64)

    # Pre-fix: per-term coverage (default coverage=-1 estimates locally on EACH table).
    pre = max(0.0, chao_shen_entropy_from_counts(rs) + chao_shen_entropy_from_counts(cs) - chao_shen_entropy_from_counts(flat))
    # Post-fix path through the public MI (shared coverage).
    post = chao_shen_mi(x, y)

    assert pre > 0.1, f"pre-fix per-term-coverage CS-MI should show false MI, got {pre}"
    assert post < 0.05, f"shared-coverage CS-MI on independent pair should be ~0, got {post}"
    assert post < pre


def test_sa10_chao_shen_mi_preserves_known_mi_on_perfect_dependence():
    """Negative control: the shared-coverage fix must NOT damage a real signal. y=x over 4 classes -> MI ~ ln(4)."""
    from mlframe.feature_selection.filters._chao_shen import chao_shen_mi

    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, 800).astype(np.int64)
    mi = chao_shen_mi(x, x.copy())
    assert mi == pytest.approx(math.log(4.0), abs=0.1), f"y=x MI should be ~ln4, got {mi}"


# ---------------------------------------------------------------------------
# SA11 -- cat MM correction: the six II entropies must telescope
# ---------------------------------------------------------------------------
def test_sa11_cat_ii_mm_telescopes_no_false_synergy_keeps_real_synergy():
    """Pre-fix applied MM per-entropy with each term's own occupied k; those six occupied-k corrections do not
    telescope to the analytic II bias ``(a-1)(b-1)(c-1)/(2n)``. On an INDEPENDENT high-card pair at small n the
    residual is large and POSITIVE, manufacturing false synergy (a wrong-sign verdict vs the ~0 truth). The single
    telescoped occupied-k correction collapses it to ~0 while leaving a genuine synergy positive."""
    from mlframe.feature_selection.filters._cat_mm_correction import _compute_pair_ii_mm
    from mlframe.feature_selection.filters.info_theory import (
        entropy as _entropy,
        entropy_miller_madow as _entropy_mm,
        merge_vars,
    )

    K = 8

    def _setup(x1, x2, yv, n, dtype=np.int32):
        fd = np.column_stack([x1, x2, yv]).astype(dtype)
        nbins = np.array([K, K, K], dtype=np.int64)
        ti = np.array([2], dtype=np.int64)
        _, freqs_y, _ = merge_vars(
            factors_data=fd,
            vars_indices=ti,
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=dtype,
        )
        return fd, nbins, ti, freqs_y

    def _ii_post(x1, x2, yv, n):
        fd, nbins, ti, freqs_y = _setup(x1, x2, yv, n)
        h_y = _entropy(freqs=freqs_y)
        return _compute_pair_ii_mm(
            factors_data=fd,
            idx_a=0,
            idx_b=1,
            nbins=nbins,
            target_indices=ti,
            classes_y=yv.astype(np.int64),
            freqs_y=freqs_y,
            h_y=h_y,
            use_mm=True,
            dtype=np.int32,
        )

    def _ii_prefix_per_term(x1, x2, yv, n):
        fd, nbins, _ti, _ = _setup(x1, x2, yv, n)
        H = lambda idx: _entropy_mm(
            merge_vars(
                factors_data=fd,
                vars_indices=np.array(np.unique(idx), dtype=np.int64),
                var_is_nominal=None,
                factors_nbins=nbins,
                dtype=np.int32,
            )[1],
            n,
        )
        return H([0, 1]) + H([0, 2]) + H([1, 2]) - H([0, 1, 2]) - H([0]) - H([1]) - H([2])

    # Independent pair at small n: true II ~ 0. Pre-fix per-term MM manufactures a large POSITIVE (false synergy).
    n = 400
    rng = np.random.default_rng(3)
    x1 = rng.integers(0, K, n).astype(np.int64)
    x2 = rng.integers(0, K, n).astype(np.int64)
    y_ind = rng.integers(0, K, n).astype(np.int64)
    ii_pre = _ii_prefix_per_term(x1, x2, y_ind, n)
    ii_post = _ii_post(x1, x2, y_ind, n)
    assert ii_pre > 0.1, f"pre-fix per-term MM should fabricate positive synergy on independence, got {ii_pre}"
    assert abs(ii_post) < 0.1, f"telescoped II of an independent pair should be ~0, got {ii_post}"

    # Genuine synergy: y = (x1 + x2) mod K -> II must stay clearly positive after the correction.
    n2 = 2000
    rng2 = np.random.default_rng(7)
    x1s = rng2.integers(0, K, n2).astype(np.int64)
    x2s = rng2.integers(0, K, n2).astype(np.int64)
    y_syn = (x1s + x2s) % K
    assert _ii_post(x1s, x2s, y_syn, n2) > 0.05, "genuine modular-sum synergy must stay positive"


# ---------------------------------------------------------------------------
# SA12 -- PID synergy: MM-correct each MI term so independent high-card pairs report ~0 synergy
# ---------------------------------------------------------------------------
def test_sa12_pid_no_false_synergy_on_independent_high_cardinality_small_n():
    """Pre-fix synergy = plug-in total - U1 - U2 - R, dominated by the 3-D-joint over-binning bias in `total`,
    reports large positive synergy on an INDEPENDENT high-cardinality pair at small n. MM-correcting each MI term
    on its occupied bins (the composite (X1,X2) source uses occupied JOINT cells) drives synergy to ~0."""
    from mlframe.feature_selection.filters._pid_decomposition import pid_decomposition

    rng = np.random.default_rng(1)
    n, K = 400, 12
    x1 = rng.integers(0, K, n).astype(np.int64)
    x2 = rng.integers(0, K, n).astype(np.int64)
    y = rng.integers(0, K, n).astype(np.int64)  # all mutually independent

    res = pid_decomposition(x1, x2, y, K, K, K)
    assert res["synergistic"] < 0.05, f"independent high-card pair should yield ~0 synergy post-fix, got {res['synergistic']}"


def test_sa12_pid_preserves_xor_synergy():
    """Negative control: the MM correction must not erase a genuine 2-way synergy. XOR -> high synergy."""
    from mlframe.feature_selection.filters._pid_decomposition import pid_decomposition

    rng = np.random.default_rng(2)
    n = 500
    x1 = rng.integers(0, 2, n).astype(np.int64)
    x2 = rng.integers(0, 2, n).astype(np.int64)
    y = (x1 ^ x2).astype(np.int64)
    res = pid_decomposition(x1, x2, y, 2, 2, 2)
    assert res["synergistic"] > 0.5, f"XOR synergy should remain high, got {res['synergistic']}"


# ---------------------------------------------------------------------------
# SA14 -- fastMI: marginal + joint entropy must share the same estimator basis
# ---------------------------------------------------------------------------
def _fastmi_prefix_analytic_marginals(x, y):
    """Reproduce the PRE-FIX fastMI estimator: joint entropy from the binned KDE grid, but marginal entropy from
    the ANALYTIC standard-normal closed form 0.5*log(2 pi e) -- the asymmetric-basis bug. Used only to prove the
    fixed estimator is strictly closer to the analytic truth."""
    from mlframe.feature_selection.filters import _fastmi as F

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    u = F._rank_to_uniform(x)
    v = F._rank_to_uniform(y)
    zx = np.clip(F._probit(u), -4.0, 4.0)
    zy = np.clip(F._probit(v), -4.0, 4.0)
    M = 128
    edges = np.linspace(-4.0, 4.0, M + 1)
    counts, _, _ = np.histogram2d(zx, zy, bins=[edges, edges])
    samples = counts / n
    h = F._silverman_bandwidth(n, 1.0)
    kernel = F._gaussian_kernel_2d(M, h, 8.0)
    density = np.maximum(F._fft_conv_2d(samples, kernel), 1e-15)
    cell_area = (8.0 / M) ** 2
    p = np.maximum(density / (density.sum() * cell_area), 1e-15)
    H_joint = -float((p * np.log(p)).sum()) * cell_area
    H_marg = 0.5 * math.log(2.0 * math.pi * math.e)
    return max(0.0, 2.0 * H_marg - H_joint)


def test_sa14_fastmi_consistent_basis_closer_to_truth_on_known_mi():
    """Pre-fix MI = 2*H_marg(ANALYTIC standard-normal) - H_joint(binned-KDE): two different (n-, M-, h-dependent)
    bias scales that do NOT cancel. Integrating BOTH marginal entropies from the SAME KDE grid as the joint makes
    the bias cancel, so the corrected estimate is strictly closer to the analytic gaussian-copula truth, and an
    independent pair stays ~0."""
    from mlframe.feature_selection.filters._fastmi import fastmi

    rng = np.random.default_rng(4)
    n = 4000
    x = rng.standard_normal(n)
    y_indep = rng.standard_normal(n)
    assert fastmi(x, y_indep, bandwidth="silverman") < 0.05, "independent fastMI should be ~0"

    rho = 0.7
    y_dep = rho * x + math.sqrt(1.0 - rho**2) * rng.standard_normal(n)
    truth = -0.5 * math.log(1.0 - rho**2)  # 0.347
    mi_pre = _fastmi_prefix_analytic_marginals(x, y_dep)
    mi_post = fastmi(x, y_dep, bandwidth="silverman")
    assert abs(mi_post - truth) < abs(mi_pre - truth), f"consistent-basis fastMI must be closer to truth {truth:.3f}: pre={mi_pre:.4f} post={mi_post:.4f}"
    assert mi_post > 0.15, f"rho=0.7 fastMI should be clearly positive, got {mi_post}"


# ---------------------------------------------------------------------------
# SA15 -- conditional SU: normalizer must use MM-corrected conditional entropies
# ---------------------------------------------------------------------------
def test_sa15_conditional_su_mm_normalizer_near_zero_on_independent_given_z():
    """Pre-fix normalized plug-in CMI by plug-in conditional entropies; the joint (X,Z)/(Y,Z)/(X,Y,Z) terms carry
    steeper over-binning bias than H(Z), so the bare ratio sits well above 0 on an independent-given-Z pair.
    Routing every entropy through entropy_miller_madow debiases both numerator and denominator -> SU ~ 0."""
    from mlframe.feature_selection.filters.info_theory._entropy_kernels import (
        conditional_symmetric_uncertainty,
        entropy as _entropy,
        merge_vars,
    )

    rng = np.random.default_rng(5)
    n, K = 800, 8
    z = rng.integers(0, K, n).astype(np.int64)
    x = rng.integers(0, K, n).astype(np.int64)
    y = rng.integers(0, K, n).astype(np.int64)  # X _||_ Y | Z (all independent)
    fd = np.column_stack([x, y, z]).astype(np.int32)
    nb = np.array([K, K, K], dtype=np.int64)

    # Pre-fix value: plug-in entropies on the SAME merges.
    def _h(idx):
        _, f, _ = merge_vars(
            factors_data=fd,
            vars_indices=np.unique(np.array(idx, dtype=np.int64)),
            var_is_nominal=None,
            factors_nbins=nb,
            dtype=np.int32,
        )
        return _entropy(freqs=f)

    h_z, h_xz, h_yz, h_xyz = _h([2]), _h([0, 2]), _h([1, 2]), _h([0, 1, 2])
    su_plugin = 2.0 * (h_xz + h_yz - h_z - h_xyz) / (h_xz + h_yz - 2.0 * h_z)

    su_mm = conditional_symmetric_uncertainty(
        fd,
        np.array([0]),
        np.array([1]),
        np.array([2]),
        nb,
    )
    assert su_plugin > 0.1, f"pre-fix plug-in cond-SU should show false dependence, got {su_plugin}"
    assert su_mm < 0.1, f"MM-corrected cond-SU on independent-given-Z should be ~0, got {su_mm}"
    assert su_mm < su_plugin
