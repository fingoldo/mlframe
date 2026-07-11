"""Numerical-equivalence test for ``batch_pair_usability_corr_gpu``'s CPU and CUDA backends vs the
existing ``_fe_usability_signal.abs_pearson``/``usability_form_corrs`` reference.

Both the njit_parallel and CUDA backends must reproduce ``_abs_pearson_njit``'s exact two-pass
mean-then-center reduction for every one of the 9 candidate forms, including its documented degenerate
cases (a near-constant column, NaN/inf rows). The CUDA test auto-skips when CUDA is unavailable.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_usability_signal import (
    _single_operand_usability_corr,
    abs_pearson,
    pair_is_tail_concentrated_rankaware,
)
from mlframe.feature_selection.filters.batch_pair_usability_corr_gpu import (
    ALL_FORM_IDS,
    FORM_X0,
    FORM_X0_DIV_X1,
    FORM_X0_MUL_X1,
    FORM_X0_SQ,
    FORM_X0SQ_DIV_X1,
    FORM_X1,
    FORM_X1_DIV_X0,
    FORM_X1_SQ,
    FORM_X1SQ_DIV_X0,
    _CUDA_AVAIL,
    batch_pair_tail_concentration_rankaware,
    batch_pair_usability_corr_cuda,
    batch_pair_usability_corr_njit_parallel,
    dispatch_batch_pair_usability_corr,
)


def _eval_form_numpy(form_id, x0, x1, eps=1e-12):
    """Reference form construction mirroring usability_form_corrs's own numpy form-building exactly."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if form_id == FORM_X0:
            return x0
        if form_id == FORM_X1:
            return x1
        if form_id == FORM_X0_SQ:
            return x0 * x0
        if form_id == FORM_X1_SQ:
            return x1 * x1
        x0f = np.where(np.abs(x0) < eps, np.nan, x0)
        x1f = np.where(np.abs(x1) < eps, np.nan, x1)
        if form_id == FORM_X0_DIV_X1:
            return x0 / x1f
        if form_id == FORM_X1_DIV_X0:
            return x1 / x0f
        if form_id == FORM_X0SQ_DIV_X1:
            return (x0 * x0) / x1f
        if form_id == FORM_X1SQ_DIV_X0:
            return (x1 * x1) / x0f
        return x0 * x1  # FORM_X0_MUL_X1


def _build_pairs(n_operands, n_pairs, n, seed):
    rng = np.random.default_rng(seed)
    operand_matrix = rng.standard_normal((n_operands, n)).astype(np.float64)
    y = rng.standard_normal(n).astype(np.float64)
    pair_a = rng.integers(0, n_operands, size=n_pairs).astype(np.int64)
    pair_b = rng.integers(0, n_operands, size=n_pairs).astype(np.int64)
    return y, operand_matrix, pair_a, pair_b


def test_cpu_backend_matches_abs_pearson_reference_all_forms():
    n, n_operands, n_pairs = 2000, 8, 15
    y, operand_matrix, pair_a, pair_b = _build_pairs(n_operands, n_pairs, n, seed=0)

    result = batch_pair_usability_corr_njit_parallel(y, operand_matrix, pair_a, pair_b, ALL_FORM_IDS)
    assert result.shape == (n_pairs, len(ALL_FORM_IDS))

    for p in range(n_pairs):
        x0 = operand_matrix[pair_a[p]]
        x1 = operand_matrix[pair_b[p]]
        for f_idx, form_id in enumerate(ALL_FORM_IDS):
            v = _eval_form_numpy(int(form_id), x0, x1)
            expected = abs_pearson(y, v)
            assert abs(result[p, f_idx] - expected) <= 1e-9, (p, form_id, result[p, f_idx], expected)


def test_cpu_backend_matches_reference_with_nan_rows():
    """NaN/inf rows (e.g. from a ratio form's floored denominator) must be dropped identically to the
    scalar abs_pearson reference -- not just on all-finite data."""
    n = 3000
    rng = np.random.default_rng(3)
    y, operand_matrix, pair_a, pair_b = _build_pairs(4, 6, n, seed=3)
    # Poison some rows of operand 0 with near-zero values (triggers the eps-floor -> NaN in ratio forms)
    # and operand 1 with actual NaN/inf.
    operand_matrix[0, rng.choice(n, 50, replace=False)] = 0.0
    operand_matrix[1, rng.choice(n, 30, replace=False)] = np.nan
    operand_matrix[1, rng.choice(n, 20, replace=False)] = np.inf

    result = batch_pair_usability_corr_njit_parallel(y, operand_matrix, pair_a, pair_b, ALL_FORM_IDS)
    for p in range(pair_a.shape[0]):
        x0 = operand_matrix[pair_a[p]]
        x1 = operand_matrix[pair_b[p]]
        for f_idx, form_id in enumerate(ALL_FORM_IDS):
            v = _eval_form_numpy(int(form_id), x0, x1)
            expected = abs_pearson(y, v)
            assert abs(result[p, f_idx] - expected) <= 1e-9, (p, form_id)


def test_degenerate_near_constant_column_returns_zero_not_cancellation_artifact():
    """The exact adversarial case _abs_pearson_njit's docstring documents: a near-constant column (std/mean
    below the CV floor) must return ~0.0 -- NOT a spurious nonzero from catastrophic cancellation. This is
    the numerical property that ruled out a naive one-pass GPU reduction during design."""
    n = 1000
    x_const = (1.0 + np.linspace(0, 1e-15, n)).astype(np.float64)
    rng = np.random.default_rng(9)
    x_other = rng.standard_normal(n)
    y = rng.standard_normal(n)

    operand_matrix = np.vstack([x_const, x_other])
    pair_a = np.array([0], dtype=np.int64)
    pair_b = np.array([1], dtype=np.int64)

    result = batch_pair_usability_corr_njit_parallel(y, operand_matrix, pair_a, pair_b, np.array([FORM_X0], dtype=np.int64))
    assert result[0, 0] == 0.0, f"expected exact 0.0 for a near-constant column, got {result[0, 0]!r}"


def test_dispatcher_cpu_force_matches_direct_call():
    n, n_operands, n_pairs = 1500, 5, 10
    y, operand_matrix, pair_a, pair_b = _build_pairs(n_operands, n_pairs, n, seed=5)
    direct = batch_pair_usability_corr_njit_parallel(y, operand_matrix, pair_a, pair_b, ALL_FORM_IDS)
    dispatched, backend = dispatch_batch_pair_usability_corr(y, operand_matrix, pair_a, pair_b, force_backend="cpu")
    assert backend == "cpu"
    assert np.array_equal(direct, dispatched)


def test_dispatcher_defaults_to_all_forms_when_omitted():
    n, n_operands, n_pairs = 800, 4, 5
    y, operand_matrix, pair_a, pair_b = _build_pairs(n_operands, n_pairs, n, seed=6)
    result, _ = dispatch_batch_pair_usability_corr(y, operand_matrix, pair_a, pair_b, force_backend="cpu")
    assert result.shape == (n_pairs, len(ALL_FORM_IDS))


def test_dispatcher_rejects_unknown_backend():
    n, n_operands, n_pairs = 200, 3, 3
    y, operand_matrix, pair_a, pair_b = _build_pairs(n_operands, n_pairs, n, seed=7)
    with pytest.raises(ValueError, match="force_backend"):
        dispatch_batch_pair_usability_corr(y, operand_matrix, pair_a, pair_b, force_backend="tpu")


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAIL, reason="CUDA not available on this host")
def test_cuda_backend_matches_cpu_backend():
    n, n_operands, n_pairs = 4000, 10, 25
    y, operand_matrix, pair_a, pair_b = _build_pairs(n_operands, n_pairs, n, seed=11)

    cpu_result = batch_pair_usability_corr_njit_parallel(y, operand_matrix, pair_a, pair_b, ALL_FORM_IDS)
    cuda_result = batch_pair_usability_corr_cuda(y, operand_matrix, pair_a, pair_b, ALL_FORM_IDS)

    assert cuda_result.shape == cpu_result.shape
    np.testing.assert_allclose(cuda_result, cpu_result, atol=1e-9, rtol=1e-9)


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAIL, reason="CUDA not available on this host")
def test_cuda_backend_degenerate_case_matches_cpu():
    """The catastrophic-cancellation adversarial case must ALSO hold on the CUDA backend, not just CPU --
    this is the one part of the design that genuinely needed device-side validation, not a copy-paste."""
    n = 1000
    x_const = (1.0 + np.linspace(0, 1e-15, n)).astype(np.float64)
    rng = np.random.default_rng(12)
    x_other = rng.standard_normal(n)
    y = rng.standard_normal(n)
    operand_matrix = np.vstack([x_const, x_other])
    pair_a = np.array([0], dtype=np.int64)
    pair_b = np.array([1], dtype=np.int64)
    form_ids = np.array([FORM_X0], dtype=np.int64)

    cuda_result = batch_pair_usability_corr_cuda(y, operand_matrix, pair_a, pair_b, form_ids)
    assert cuda_result[0, 0] == 0.0, f"CUDA backend must ALSO return exact 0.0, got {cuda_result[0, 0]!r}"


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAIL, reason="CUDA not available on this host")
def test_dispatcher_cuda_force_matches_cpu_result():
    n, n_operands, n_pairs = 3000, 8, 20
    y, operand_matrix, pair_a, pair_b = _build_pairs(n_operands, n_pairs, n, seed=13)
    cpu_result, cpu_backend = dispatch_batch_pair_usability_corr(y, operand_matrix, pair_a, pair_b, force_backend="cpu")
    cuda_result, cuda_backend = dispatch_batch_pair_usability_corr(y, operand_matrix, pair_a, pair_b, force_backend="cuda")
    assert cpu_backend == "cpu"
    assert cuda_backend == "cuda"
    np.testing.assert_allclose(cuda_result, cpu_result, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# batch_pair_tail_concentration_rankaware -- mirrors the main-loop usability-admission gate
# (score_prospective_pairs, 2026-07-11 batching fix) against the serial per-pair reference.
# ---------------------------------------------------------------------------


def _outlier_ratio_fixture(seed=7, n=6000, w_tail=0.055):
    """SAME construction as the adversarial fixture in
    test_score_prospective_pairs_usability_admission_nondominant.py: a balanced (non-tail-concentrated,
    dominant) pair and an outlier-driven ratio (tail-concentrated, non-dominant) pair sharing one target."""
    rng = np.random.default_rng(seed)
    a_dom = rng.standard_normal(n)
    b_dom = rng.standard_normal(n)
    dom_term = a_dom * b_dom

    a_tail = rng.standard_normal(n)
    b_tail = rng.uniform(0.5, 2.0, n)
    outlier_mask = rng.random(n) < 0.02
    b_tail[outlier_mask] = rng.uniform(0.01, 0.03, int(outlier_mask.sum()))
    ratio_tail = (a_tail**2) / b_tail

    noise = rng.standard_normal(n) * 0.01
    y = 1.0 * dom_term + w_tail * ratio_tail + noise
    return y, {0: a_dom, 1: b_dom, 2: a_tail, 3: b_tail}


def _serial_verdicts(y, operands, pair_a, pair_b, min_corr, pairness_margin, max_rank_frac):
    return np.array([
        pair_is_tail_concentrated_rankaware(
            y, operands[int(a)], operands[int(b)], min_corr=min_corr, pairness_margin=pairness_margin, max_rank_frac=max_rank_frac,
        )
        for a, b in zip(pair_a, pair_b)
    ], dtype=bool)


def test_batch_tail_concentration_matches_serial_on_random_noise_pairs():
    """Random noise pairs: neither the min_corr gate nor the rank-collapse signature should fire -- every
    verdict should be False on BOTH paths, and they must agree pair-for-pair."""
    n, n_operands, n_pairs = 4000, 10, 30
    y, operand_matrix, pair_a, pair_b = _build_pairs(n_operands, n_pairs, n, seed=21)
    operands = {i: operand_matrix[i] for i in range(n_operands)}

    batched = batch_pair_tail_concentration_rankaware(y, operand_matrix, pair_a, pair_b, min_corr=0.6, pairness_margin=1.05, max_rank_frac=0.7)
    serial = _serial_verdicts(y, operands, pair_a, pair_b, 0.6, 1.05, 0.7)
    assert np.array_equal(batched, serial)
    assert not serial.any(), "sanity: random noise pairs should not be tail-concentrated"


def test_batch_tail_concentration_detects_genuine_nondominant_tail_pair():
    """THE adversarial case this batching fix targets: a non-dominant, outlier-driven ratio pair passes,
    a balanced dominant pair does not -- both paths must agree, and the genuine pair must be True."""
    y, operands = _outlier_ratio_fixture()
    pair_a = np.array([0, 2], dtype=np.int64)
    pair_b = np.array([1, 3], dtype=np.int64)
    operand_matrix = np.vstack([operands[0], operands[1], operands[2], operands[3]])

    batched = batch_pair_tail_concentration_rankaware(y, operand_matrix, pair_a, pair_b, min_corr=0.6, pairness_margin=1.05, max_rank_frac=0.7)
    serial = _serial_verdicts(y, operands, pair_a, pair_b, 0.6, 1.05, 0.7)
    assert np.array_equal(batched, serial)
    assert batched[0] == False, "pair (0,1) balanced -- must NOT be tail-concentrated"  # noqa: E712
    assert batched[1] == True, "pair (2,3) outlier-driven ratio -- must BE tail-concentrated"  # noqa: E712


def test_batch_tail_concentration_precomputed_single_corr_matches_internal():
    """The ``single_corr`` fast path (per-operand cache, mirrors score_prospective_pairs' own
    ``_cached_single_corr``) must give the IDENTICAL verdicts as letting the function compute cs itself."""
    y, operands = _outlier_ratio_fixture()
    pair_a = np.array([0, 2], dtype=np.int64)
    pair_b = np.array([1, 3], dtype=np.int64)
    operand_matrix = np.vstack([operands[0], operands[1], operands[2], operands[3]])

    without_precomp = batch_pair_tail_concentration_rankaware(y, operand_matrix, pair_a, pair_b, min_corr=0.6, pairness_margin=1.05, max_rank_frac=0.7)

    single_corr = np.array([float(_single_operand_usability_corr(y, operand_matrix[i])) for i in range(4)])
    with_precomp = batch_pair_tail_concentration_rankaware(
        y, operand_matrix, pair_a, pair_b, min_corr=0.6, pairness_margin=1.05, max_rank_frac=0.7, single_corr=single_corr,
    )
    assert np.array_equal(without_precomp, with_precomp)


def test_batch_tail_concentration_empty_input_returns_empty_array():
    y = np.zeros(10)
    operand_matrix = np.zeros((2, 10))
    pair_a = np.array([], dtype=np.int64)
    pair_b = np.array([], dtype=np.int64)
    result = batch_pair_tail_concentration_rankaware(y, operand_matrix, pair_a, pair_b, min_corr=0.6, pairness_margin=1.05, max_rank_frac=0.7)
    assert result.shape == (0,)
    assert result.dtype == bool


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
