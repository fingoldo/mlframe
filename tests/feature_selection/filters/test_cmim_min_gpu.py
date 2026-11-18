"""The fused CMIM scorer (numba kernel, CUDA kernel) gives the scores of the host loop, and ``score_features_by_cmim`` ranks the same with it."""

from __future__ import annotations

import numpy as np
from tests.feature_selection.gpu._gpu_support import require_device
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _cmim_min_gpu as C
from mlframe.feature_selection.filters import _orthogonal_cmim_fe as M

NO_REF = 1e299


def _case(n=6000, k=24, s=5, nb=10):
    """Binned candidates (one informative), references (one a noisy copy of it), a mask with a candidate that has no allowed reference."""
    rng = np.random.default_rng(3)
    y = (rng.random(n) < 0.4).astype(np.int64)
    x = rng.integers(0, nb, size=(k, n)).astype(np.int16)
    x[0] = np.where(y == 1, rng.integers(5, 10, n), rng.integers(0, 5, n))
    zs = [rng.integers(0, nb, size=n) for _ in range(s)]
    zs[1] = np.where(rng.random(n) < 0.8, x[0], zs[1]).astype(np.int64)
    allowed = rng.random((s, k)) < 0.8
    allowed[:, 5 % k] = False
    return x, y, zs, allowed, nb


def _reference(x, y, zs, allowed):
    """The host loop of ``score_features_by_cmim`` (cached z terms, early exit at 0); ``1e300`` for a candidate with no allowed reference."""
    cache = M._build_cmi_yz_cache(y, zs)
    out = np.full(x.shape[0], 1e300)
    for c in range(x.shape[0]):
        entries = [e for j, e in enumerate(cache) if allowed[j, c]]
        if entries:
            out[c] = M._cmim_score_cached(x[c].astype(np.int64), y, entries, float(y.size))
    return out


def test_the_numba_kernel_gives_the_host_loop_scores():
    """All candidates agree with the host loop to 1e-12; the candidate with no allowed reference reads as none."""
    x, y, zs, allowed, nb = _case()
    ref = _reference(x, y, zs, allowed)
    got = C._scores_host(x, y, zs, allowed, nb)
    assert got is not None
    assert (got[5] >= NO_REF) and (ref[5] >= NO_REF)
    ok = ref < NO_REF
    assert np.abs(got - ref)[ok].max() < 1e-12
    assert ref[0] > 1e-3


def test_the_cuda_kernel_gives_the_host_loop_scores():
    """The CUDA kernel agrees with the host loop (one launch for the whole block)."""
    require_device()
    x, y, zs, allowed, nb = _case()
    ref = _reference(x, y, zs, allowed)
    got = C._scores_device(x, y, zs, allowed, nb)
    assert got is not None, "the device path returned nothing on a host with a usable device"
    ok = ref < NO_REF
    assert got[5] >= NO_REF
    assert np.abs(got - ref)[ok].max() < 1e-12


def test_codes_out_of_range_or_a_huge_table_are_declined():
    """A code beyond ``n_bins`` and a joint table beyond ``MAX_CELLS`` return ``None`` (the caller keeps its older paths) instead of indexing out of the table."""
    x, y, zs, allowed, nb = _case(n=500, k=4, s=2)
    bad = x.copy()
    bad[0, 0] = nb
    assert C._scores_host(bad, y, zs, allowed, nb) is None
    assert C._scores_host(x, y, zs, allowed, 10_000) is None


def test_a_narrow_block_budget_gives_the_same_scores(monkeypatch):
    """With a candidate block of three candidates the scores equal the one-block ones."""
    x, y, zs, allowed, nb = _case(n=2000, k=10, s=3)
    monkeypatch.setenv("MLFRAME_CMIM_MIN_DEVICE", "0")
    whole = C.cmim_min_scores(x, y, zs, allowed, nb)
    monkeypatch.setattr(C, "_BLOCK_BYTES", 3 * 2 * x.shape[1])
    assert C.candidate_block_width(x.shape[1]) == 3
    assert np.array_equal(C.cmim_min_scores(x, y, zs, allowed, nb), whole)


@pytest.mark.parametrize("forced", ["0", "1"])
def test_score_features_by_cmim_ranks_the_same_with_the_fused_scorer(monkeypatch, forced):
    """``score_features_by_cmim`` returns the columns and the scores of the older paths with the fused scorer on, on either backend."""
    rng = np.random.default_rng(0)
    n = 8000
    raw = pd.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(6)})
    eng = pd.DataFrame({"x0__sq": raw["x0"] ** 2, "x1__cube": raw["x1"] ** 3, "x2__copy3": raw["x3"] + 0.01 * rng.standard_normal(n), "x4__noise": rng.standard_normal(n)})
    y = ((raw["x0"] ** 2 + raw["x1"] + 0.5 * raw["x3"] + 0.5 * rng.standard_normal(n)) > 1.0).astype(int).to_numpy()
    monkeypatch.setenv("MLFRAME_CMIM_MIN", "0")
    monkeypatch.setenv("MLFRAME_CMIM_DEVICE", "0")
    want = M.score_features_by_cmim(raw, eng, y)
    monkeypatch.setenv("MLFRAME_CMIM_MIN", "1")
    monkeypatch.setenv("MLFRAME_CMIM_MIN_DEVICE", forced)
    got = M.score_features_by_cmim(raw, eng, y)
    assert list(got["engineered_col"]) == list(want["engineered_col"])
    np.testing.assert_allclose(got["engineered_mi"].to_numpy(), want["engineered_mi"].to_numpy(), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("forced", ["0", "1"])
def test_a_candidate_with_no_reference_of_another_source_gets_its_marginal_mi(monkeypatch, forced):
    """With one raw column every candidate of that source has no reference: the score is the Miller-Madow marginal MI, the same through the kernels (a constant reference) as through the host ``_cmi_from_binned``."""
    rng = np.random.default_rng(1)
    n = 8000
    raw = pd.DataFrame({"x0": rng.standard_normal(n)})
    eng = pd.DataFrame({"x0__sq": raw["x0"] ** 2, "x0__cube": raw["x0"] ** 3})
    y = ((raw["x0"] ** 2 + 0.5 * rng.standard_normal(n)) > 1.0).astype(int).to_numpy()
    monkeypatch.setenv("MLFRAME_CMIM_MIN", "0")
    want = M.score_features_by_cmim(raw, eng, y)
    monkeypatch.setenv("MLFRAME_CMIM_MIN", "1")
    monkeypatch.setenv("MLFRAME_CMIM_MIN_DEVICE", forced)
    got = M.score_features_by_cmim(raw, eng, y)
    assert list(got["engineered_col"]) == list(want["engineered_col"])
    np.testing.assert_allclose(got["engineered_mi"].to_numpy(), want["engineered_mi"].to_numpy(), rtol=1e-9, atol=1e-12)
    assert (want["engineered_mi"] > 0).any()
