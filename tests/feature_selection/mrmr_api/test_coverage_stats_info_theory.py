"""Statistical-correctness coverage for the MRMR information-theory + null machinery.

These are UNIT tests on the kernels themselves (info_theory, _analytic_mi_null, permutation,
_synergy_detector, _pid_decomposition, DCD pair_su / discover_cluster_members), pinning the
statistical contracts the higher-level MRMR fit relies on:

  * plug-in MI is positively biased on independent variables; Miller-Madow materially reduces it.
  * Symmetric Uncertainty is bounded in [0, 1] (==1 on an identical pair, <1 cross-cardinality).
  * the analytic G-test null mean / p reproduce the permutation null on a DENSE large-n table.
  * the analytic gate is NOT applicable (KEEP, don't reject) on a SPARSE high-cardinality table
    where the chi-square approximation is invalid.
  * PID / synergy detection fire on XOR (pure synergy) and stay silent on a redundant/linear target.
  * DCD pair_su + discover_cluster_members prune a collinear duplicate group and keep an
    independent one.

All synthetics are n<=5000, fixed seeds, numba pre-warmed in a module fixture so a cold compile
does not blow a budget assertion.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory import (
    entropy,
    entropy_miller_madow,
    mi,
    mi_miller_madow,
    symmetric_uncertainty,
    conditional_mi,
    compute_mi_from_classes,
    merge_vars,
)
from mlframe.feature_selection.filters._analytic_mi_null import (
    analytic_mi_null,
    analytic_null_applicable,
)
from mlframe.feature_selection.filters._synergy_detector import detect_synergy
from mlframe.feature_selection.filters._pid_decomposition import pid_decomposition
from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
    DCDState,
    discover_cluster_members,
)
from mlframe.feature_selection.filters._dynamic_cluster_discovery._dcd_metrics import (
    pair_su,
    should_be_pruned,
)


@pytest.fixture(scope="module", autouse=True)
def _prewarm_numba():
    """Compile every njit kernel these tests hit once, so per-test timing budgets are not blown by a cold JIT."""
    n = 64
    rng = np.random.default_rng(0)
    fd = np.column_stack([rng.integers(0, 4, n), rng.integers(0, 2, n)]).astype(np.int32)
    fnb = np.array([4, 2], dtype=np.int64)
    _ = mi(fd, np.array([0]), np.array([1]), fnb)
    _ = mi_miller_madow(fd, np.array([0]), np.array([1]), fnb)
    _ = symmetric_uncertainty(fd, np.array([0]), np.array([1]), fnb)
    cx, fx, _ = merge_vars(fd, np.array([0]), None, fnb)
    cy, fy, _ = merge_vars(fd, np.array([1]), None, fnb)
    _ = compute_mi_from_classes(cx, fx, cy, fy)
    _ = conditional_mi(fd, np.array([0]), np.array([1]), np.array([1]),
                       np.zeros(2, dtype=np.int32), fnb)
    _ = entropy(fx)
    _ = entropy_miller_madow(fx, n)
    yield


def _two_col(a, b, nb_a, nb_b):
    fd = np.column_stack([a, b]).astype(np.int32)
    fnb = np.array([nb_a, nb_b], dtype=np.int64)
    return fd, fnb


# ----------------------------------------------------------------------------------------------------
# 1. Miller-Madow bias correction actually REDUCES plug-in MI bias on independent variables.
# ----------------------------------------------------------------------------------------------------

def test_miller_madow_reduces_plugin_mi_bias_on_independent_highcard():
    """On a high-cardinality independent pair (true MI = 0), plug-in MI is positively biased; the
    Miller-Madow estimate is at least 3x closer to zero. Measured: plugin~0.0257, mm~0.0020 (~13x)."""
    rng = np.random.default_rng(0)
    n, nbx = 400, 20
    x = rng.integers(0, nbx, n).astype(np.int32)
    y = rng.integers(0, 2, n).astype(np.int32)
    fd, fnb = _two_col(x, y, nbx, 2)
    mi_plug = mi(fd, np.array([0]), np.array([1]), fnb)
    mi_mm = mi_miller_madow(fd, np.array([0]), np.array([1]), fnb)
    assert mi_plug > 0.01, f"plug-in MI should be visibly biased upward, got {mi_plug}"
    # MM is bias-corrected: well below plug-in, and much closer to the true 0.
    assert mi_mm < mi_plug, f"MM ({mi_mm}) must be below plug-in ({mi_plug})"
    assert abs(mi_mm) <= abs(mi_plug) / 3.0, (
        f"MM should cut independent-pair bias >=3x: plugin={mi_plug:.5f} mm={mi_mm:.5f}"
    )


def test_miller_madow_entropy_is_above_plugin_and_bounded():
    """entropy_miller_madow >= plug-in entropy (adds (k-1)/(2n)) and stays finite; k<=1 falls back to plug-in exactly."""
    rng = np.random.default_rng(1)
    n = 500
    x = rng.integers(0, 8, n).astype(np.int32)
    _, freqs, _ = merge_vars(np.column_stack([x, x]).astype(np.int32),
                             np.array([0]), None, np.array([8, 8], dtype=np.int64))
    h_plug = entropy(freqs)
    h_mm = entropy_miller_madow(freqs, n)
    assert h_mm > h_plug, "MM entropy adds a positive correction for k>1"
    assert h_mm == pytest.approx(h_plug + (len(freqs) - 1) / (2.0 * n))
    # single-bin (k=1) -> plug-in exactly (0.0), no negative correction.
    single = np.array([1.0])
    assert entropy_miller_madow(single, n) == entropy(single)


def test_miller_madow_converges_to_plugin_as_n_grows():
    """The MM bias term (k_x-1)(k_y-1)/(2n) -> 0, so at large n MM and plug-in MI agree to <1e-3."""
    rng = np.random.default_rng(2)
    n, nbx = 5000, 6
    x = rng.integers(0, nbx, n).astype(np.int32)
    # build a genuine signal so MI is O(0.1), not near the bias floor
    y = ((x % 2) ^ (rng.random(n) < 0.1).astype(np.int32)).astype(np.int32)
    fd, fnb = _two_col(x, y, nbx, 2)
    mi_plug = mi(fd, np.array([0]), np.array([1]), fnb)
    mi_mm = mi_miller_madow(fd, np.array([0]), np.array([1]), fnb)
    assert abs(mi_plug - mi_mm) < 1e-3, f"large-n MM should track plug-in: {mi_plug} vs {mi_mm}"


# ----------------------------------------------------------------------------------------------------
# 2. Symmetric Uncertainty normalization bounds in [0, 1].
# ----------------------------------------------------------------------------------------------------

def test_su_is_one_on_identical_columns():
    """SU(X, X) == 1 exactly (perfect mutual dependence)."""
    rng = np.random.default_rng(3)
    x = rng.integers(0, 5, 1000).astype(np.int32)
    fd, fnb = _two_col(x, x, 5, 5)
    su = symmetric_uncertainty(fd, np.array([0]), np.array([1]), fnb)
    assert su == pytest.approx(1.0, abs=1e-9), f"SU of a column with itself must be 1.0, got {su}"


def test_su_bounded_and_zero_on_independent():
    """SU lies in [0, 1] and is ~0 (well under a strong-signal pair) on an independent pair."""
    rng = np.random.default_rng(4)
    n = 2000
    x = rng.integers(0, 5, n).astype(np.int32)
    y = rng.integers(0, 4, n).astype(np.int32)
    fd, fnb = _two_col(x, y, 5, 4)
    su = symmetric_uncertainty(fd, np.array([0]), np.array([1]), fnb)
    assert 0.0 <= su <= 1.0, f"SU must be in [0,1], got {su}"
    assert su < 0.05, f"independent pair SU should be near 0, got {su}"


def test_su_normalizes_away_cardinality_inflation():
    """A binary feature carrying the SAME signal as the target keeps a HIGH SU, while a high-cardinality
    pure-noise feature (which inflates RAW MI by sheer entropy) gets a near-zero SU. This is the
    cardinality-bias correction SU exists for."""
    rng = np.random.default_rng(5)
    n = 2000
    y = (rng.random(n) < 0.5).astype(np.int32)
    binary_signal = y.copy()
    binary_signal[: n // 10] ^= 1  # 90% agreement
    highcard_noise = rng.integers(0, 50, n).astype(np.int32)

    fd_sig, fnb_sig = _two_col(binary_signal, y, 2, 2)
    fd_noise, fnb_noise = _two_col(highcard_noise, y, 50, 2)
    su_sig = symmetric_uncertainty(fd_sig, np.array([0]), np.array([1]), fnb_sig)
    su_noise = symmetric_uncertainty(fd_noise, np.array([0]), np.array([1]), fnb_noise)
    mi_noise = mi(fd_noise, np.array([0]), np.array([1]), fnb_noise)

    assert 0.0 <= su_noise <= 1.0 and 0.0 <= su_sig <= 1.0
    assert mi_noise > 0.01, "raw MI of the 50-bin noise feature is inflated by entropy"
    assert su_sig > 0.3, f"binary true-signal SU should stay high, got {su_sig}"
    assert su_noise < 0.05, f"high-card noise SU should collapse near 0, got {su_noise}"
    assert su_sig > 6 * su_noise, "SU must rank the true binary signal far above the high-card noise"


# ----------------------------------------------------------------------------------------------------
# 3. Analytic G-test null vs permutation null AGREEMENT on a dense large-n table.
# ----------------------------------------------------------------------------------------------------

def _mi_and_classes(x, y, nbx, nby):
    fd, fnb = _two_col(x, y, nbx, nby)
    cx, fx, _ = merge_vars(fd, np.array([0]), None, fnb)
    cy, fy, _ = merge_vars(fd, np.array([1]), None, fnb)
    omi = compute_mi_from_classes(cx, fx, cy, fy)
    return omi, cx, fx, cy, fy


def test_analytic_null_mean_matches_permutation_on_dense_table():
    """On a dense independent table (n=60k, 8x4 cells), the analytic Miller-Madow null mean
    (Bx-1)(By-1)/(2N) matches the empirical permutation null mean to within 10% relative.
    Measured: analytic 0.000175 vs permutation 0.000176."""
    rng = np.random.default_rng(1)
    n, nbx, nby = 60_000, 8, 4
    x = rng.integers(0, nbx, n).astype(np.int32)
    y = rng.integers(0, nby, n).astype(np.int32)
    omi, cx, fx, cy, fy = _mi_and_classes(x, y, nbx, nby)
    null_mean, _ = analytic_mi_null(omi, n, len(fx), len(fy))

    cyp = cy.copy()
    perm_mis = np.empty(200)
    for s in range(200):
        rng.shuffle(cyp)
        perm_mis[s] = compute_mi_from_classes(cx, fx, cyp, fy)
    perm_mean = float(perm_mis.mean())

    assert perm_mean > 0, "sanity: permutation null mean is a small positive bias floor"
    rel = abs(null_mean - perm_mean) / perm_mean
    assert rel < 0.10, f"analytic null mean {null_mean:.6g} vs perm {perm_mean:.6g} rel-err {rel:.3f}"


def test_analytic_null_p_agrees_with_permutation_decision_on_dense_table():
    """The analytic G-test p-value AGREES with the empirical permutation p on a noise pair (a noise p
    is uniform-ish, so the contract is agreement |analytic - perm| small, NOT 'both high'), and the
    analytic p is decisively significant (~0) on a clear signal."""
    rng = np.random.default_rng(11)
    n, nbx, nby = 50_000, 6, 3

    # NOISE pair -> analytic p must track the permutation p (both non-tiny, close to each other).
    x = rng.integers(0, nbx, n).astype(np.int32)
    y = rng.integers(0, nby, n).astype(np.int32)
    omi, cx, fx, cy, fy = _mi_and_classes(x, y, nbx, nby)
    _, p_analytic = analytic_mi_null(omi, n, len(fx), len(fy))
    cyp = cy.copy()
    nfail = 0
    for s in range(400):
        rng.shuffle(cyp)
        if compute_mi_from_classes(cx, fx, cyp, fy) >= omi:
            nfail += 1
    p_perm = nfail / 400
    assert p_analytic > 0.02, f"noise must NOT read as a strong signal, p={p_analytic}"
    assert abs(p_analytic - p_perm) < 0.2, f"analytic p {p_analytic:.3f} must track perm p {p_perm:.3f}"

    # SIGNAL pair -> significant both ways.
    xs = rng.integers(0, nbx, n).astype(np.int32)
    ys = (xs % nby).astype(np.int32)
    ys[: n // 20] = rng.integers(0, nby, n // 20).astype(np.int32)  # 5% noise
    omi_s, cxs, fxs, cys, fys = _mi_and_classes(xs, ys, nbx, nby)
    _, p_analytic_s = analytic_mi_null(omi_s, n, len(fxs), len(fys))
    assert p_analytic_s < 0.01, f"strong signal must read significant analytically, p={p_analytic_s}"


def test_analytic_null_applicable_keeps_sparse_highcard_table():
    """The analytic gate must be INAPPLICABLE on a sparse high-cardinality contingency table (where
    the chi-square approximation is invalid) so the caller KEEPS the candidate / falls back to the
    permutation test, even though n is large. Dense low-card at the same n IS applicable."""
    n, nby = 60_000, 4
    # dense low-card -> applicable.
    assert analytic_null_applicable(n, 8, nby) is True
    # sparse high-card (5000 x 4 cells => avg expected count 3 < 5) -> not applicable.
    assert analytic_null_applicable(n, 5000, nby) is False
    # below the n floor -> never applicable regardless of cardinality.
    assert analytic_null_applicable(1000, 4, 2) is False


def test_analytic_batch_gate_keeps_sparse_rejects_dense_noise():
    """analytic_batch_noise_gate zeroes a DENSE-noise candidate (valid chi-square -> reject) but KEEPS
    a SPARSE high-cardinality candidate's observed MI untouched (invalid chi-square -> no reject)."""
    from mlframe.feature_selection.filters._analytic_mi_null import analytic_batch_noise_gate

    rng = np.random.default_rng(21)
    n = 60_000
    y = rng.integers(0, 4, n).astype(np.int64)
    dense_noise = rng.integers(0, 8, n)         # 8 x 4 cells -> avg 1875, valid chi-square
    sparse_noise = rng.integers(0, 5000, n)     # 5000 x 4 cells -> avg 3, invalid chi-square
    disc = np.column_stack([dense_noise, sparse_noise]).astype(np.int32)

    # observed MI per column (raw nats) from the same kernels.
    def _omi(col, nbx):
        omi, *_ = _mi_and_classes(col.astype(np.int32), y.astype(np.int32), nbx, 4)
        return omi
    obs = np.array([_omi(dense_noise, 8), _omi(sparse_noise, 5000)], dtype=np.float64)

    gated = analytic_batch_noise_gate(disc, obs, y, n, min_nonzero_confidence=0.99)
    assert gated[0] == 0.0, "dense noise candidate must be rejected by the valid G-test"
    assert gated[1] == obs[1], "sparse high-card candidate's observed MI must be kept (invalid test)"


# ----------------------------------------------------------------------------------------------------
# 4. Conditional-MI redundancy: drops a redundant copy, keeps a private (XOR) interaction.
# ----------------------------------------------------------------------------------------------------

def test_conditional_mi_drops_redundant_keeps_private_interaction():
    """I(X; Y | Z): collapses to ~0 when X is a redundant copy of Z (Z already explains Y), but stays
    LARGE for a private XOR interaction where Y = X xor Z (X carries information about Y only given Z)."""
    rng = np.random.default_rng(31)
    n = 3000
    z = rng.integers(0, 2, n).astype(np.int32)

    # Redundant case: X == Z, Y == Z. Given Z, X tells nothing new about Y.
    x_red = z.copy()
    y_red = z.copy()
    fd_r = np.column_stack([x_red, y_red, z]).astype(np.int32)
    fnb_r = np.array([2, 2, 2], dtype=np.int64)
    nominal = np.zeros(3, dtype=np.int32)
    cmi_red = conditional_mi(fd_r, np.array([0]), np.array([1]), np.array([2]), nominal, fnb_r)

    # Private interaction: Y = X xor Z. Marginally X tells little about Y, but GIVEN Z it tells everything.
    x_xor = rng.integers(0, 2, n).astype(np.int32)
    y_xor = (x_xor ^ z).astype(np.int32)
    fd_x = np.column_stack([x_xor, y_xor, z]).astype(np.int32)
    cmi_xor = conditional_mi(fd_x, np.array([0]), np.array([1]), np.array([2]), nominal, fnb_r)

    assert cmi_red < 0.02, f"redundant copy should have CMI ~0 given Z, got {cmi_red}"
    assert cmi_xor > 0.5, f"private XOR interaction should have large CMI given Z, got {cmi_xor}"
    assert cmi_xor > 20 * max(cmi_red, 1e-6), "CMI must rank the private interaction far above the redundant copy"


# ----------------------------------------------------------------------------------------------------
# 5. Synergy / PID on XOR.
# ----------------------------------------------------------------------------------------------------

def test_pid_decomposition_pure_synergy_on_xor():
    """PID of (X1, X2, Y=X1 xor X2): synergistic ~= ln(2), all of unique/redundant ~= 0."""
    rng = np.random.default_rng(41)
    n = 4000
    x1 = rng.integers(0, 2, n).astype(np.int64)
    x2 = rng.integers(0, 2, n).astype(np.int64)
    y = (x1 ^ x2).astype(np.int64)
    pid = pid_decomposition(x1, x2, y, 2, 2, 2)
    assert pid["synergistic"] > 0.6, f"XOR should be ~ln2 synergy, got {pid['synergistic']}"
    assert pid["redundant"] < 0.02, f"XOR has no redundancy, got {pid['redundant']}"
    assert pid["unique_x1"] < 0.02 and pid["unique_x2"] < 0.02, "XOR has no unique single-operand info"
    assert pid["synergistic"] > 10 * (pid["redundant"] + pid["unique_x1"] + pid["unique_x2"] + 1e-9)


def test_pid_decomposition_pure_redundancy_on_copies():
    """PID of (X1, X2=X1, Y=X1): redundant ~= ln(2), synergistic ~= 0 (the mirror of the XOR case)."""
    rng = np.random.default_rng(42)
    n = 4000
    x1 = rng.integers(0, 2, n).astype(np.int64)
    x2 = x1.copy()
    y = x1.copy()
    pid = pid_decomposition(x1, x2, y, 2, 2, 2)
    assert pid["redundant"] > 0.6, f"identical copies should be ~ln2 redundancy, got {pid['redundant']}"
    assert pid["synergistic"] < 0.02, f"copies have no synergy, got {pid['synergistic']}"


def test_pid_rejects_negative_sentinels():
    """pid_decomposition raises on negative (NaN-sentinel) indices rather than silently wrap-indexing."""
    x1 = np.array([0, 1, -1, 0], dtype=np.int64)
    x2 = np.array([0, 0, 1, 1], dtype=np.int64)
    y = np.array([0, 1, 1, 0], dtype=np.int64)
    with pytest.raises(ValueError):
        pid_decomposition(x1, x2, y, 2, 2, 2)


def test_detect_synergy_fires_on_xor_not_on_linear():
    """The cheap pre-fit synergy probe flags an XOR target (excess >> threshold) and stays silent on a
    linear additive target. Measured XOR excess ~0.69 vs threshold ~0.01."""
    rng = np.random.default_rng(43)
    n = 3000
    b1 = rng.integers(0, 2, n)
    b2 = rng.integers(0, 2, n)
    noise = np.column_stack([rng.integers(0, 2, n) for _ in range(8)])
    X = np.column_stack([b1, b2, noise]).astype(float)

    y_xor = (b1 ^ b2).astype(float)
    is_syn, info = detect_synergy(X, y_xor, max_rows=2000, nbins=4, random_seed=0)
    assert is_syn is True, f"XOR target must be flagged synergistic, info={info}"
    assert info["real_excess"] > info["threshold"], "XOR excess must clear the null threshold"

    y_lin = b1.astype(float)  # purely a single marginal -> no synergy
    is_syn_lin, _ = detect_synergy(X, y_lin, max_rows=2000, nbins=4, random_seed=0)
    assert is_syn_lin is False, "a single-marginal linear target must NOT be flagged synergistic"


# ----------------------------------------------------------------------------------------------------
# 6. DCD cluster pruning on a collinear group.
# ----------------------------------------------------------------------------------------------------

def _dcd_state(fd, fnb, tau=0.7):
    st = DCDState(factors_data=fd, factors_nbins=fnb, tau_cluster=tau)
    st.pool_pruned_mask = np.zeros(fd.shape[1], dtype=bool)
    return st


def test_dcd_pair_su_separates_duplicate_from_independent():
    """SU(anchor, exact-duplicate) == 1, SU(anchor, independent) ~ 0."""
    rng = np.random.default_rng(51)
    n = 600
    base = rng.integers(0, 4, n).astype(np.int32)
    indep = rng.integers(0, 4, n).astype(np.int32)
    fd = np.column_stack([base, base.copy(), indep]).astype(np.int32)
    fnb = np.array([4, 4, 4], dtype=np.int64)
    st = _dcd_state(fd, fnb)
    assert pair_su(st, 0, 1, factors_data=fd, factors_nbins=fnb) == pytest.approx(1.0, abs=1e-9)
    assert pair_su(st, 0, 2, factors_data=fd, factors_nbins=fnb) < 0.05


def test_dcd_prunes_collinear_group_keeps_private():
    """discover_cluster_members prunes the exact collinear duplicates of the anchor and keeps the two
    independent columns. should_be_pruned reflects the resulting mask."""
    rng = np.random.default_rng(52)
    n = 600
    base = rng.integers(0, 4, n).astype(np.int32)
    indep1 = rng.integers(0, 4, n).astype(np.int32)
    indep2 = rng.integers(0, 4, n).astype(np.int32)
    fd = np.column_stack([base, base.copy(), base.copy(), indep1, indep2]).astype(np.int32)
    fnb = np.array([4, 4, 4, 4, 4], dtype=np.int64)
    st = _dcd_state(fd, fnb)

    members = discover_cluster_members(st, 0, [1, 2, 3, 4], factors_data=fd, factors_nbins=fnb)
    assert members == {1, 2}, f"only the collinear duplicates should be pruned, got {members}"
    assert should_be_pruned(st, 1) and should_be_pruned(st, 2)
    assert not should_be_pruned(st, 3) and not should_be_pruned(st, 4)
    # tuple (interaction) candidate: pruned only when ALL components are pruned.
    assert should_be_pruned(st, (1, 2)) is True
    assert should_be_pruned(st, (1, 3)) is False
