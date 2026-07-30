"""Regression tests for the mrmr audit fix wave (see audits/mrmr_audit_2026-07-25/).

Fast, unit-level pins for the bug fixes whose inputs can be constructed directly (no full MRMR fit); the
fit-level and GPU-parity pins live alongside the existing group-aware / FE tests. Each test fails on the
pre-fix code and passes post-fix.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_mm_rerank_ii_matches_standalone_mm_ii():
    """The MM-rerank path must not double-correct the Miller-Madow II bias: its MM II must equal the standalone _compute_pair_ii_mm(use_mm=True) for the same pair."""
    from mlframe.feature_selection.filters._cat_mm_correction import _compute_pair_ii_mm, _maybe_rerank_with_mm
    from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig
    from mlframe.feature_selection.filters.info_theory import entropy

    rng = np.random.default_rng(0)
    n = 3000
    a = rng.integers(0, 12, size=n)  # high-cardinality categorical operands
    b = rng.integers(0, 10, size=n)
    y = ((a + b) % 3).astype(np.int64)  # 3-class target the pair jointly determines
    factors_data = np.column_stack([a, b, y]).astype(np.int32)
    nbins = np.array([12, 10, 3], dtype=np.int64)
    target_indices = np.array([2], dtype=np.int64)
    classes_y = y.astype(np.int32)
    freqs_y = np.bincount(y, minlength=3).astype(np.float64) / n
    h_y = entropy(freqs=freqs_y)

    ii_plugin = _compute_pair_ii_mm(factors_data, 0, 1, nbins, target_indices, classes_y, freqs_y, h_y, use_mm=False, dtype=np.int32)
    ii_direct_mm = _compute_pair_ii_mm(factors_data, 0, 1, nbins, target_indices, classes_y, freqs_y, h_y, use_mm=True, dtype=np.int32)
    # A high-card pair with only 3 y-classes carries a non-trivial telescoped bias, so MM and plug-in must
    # actually differ, otherwise the test cannot distinguish the double-correction.
    assert not np.isclose(ii_direct_mm, ii_plugin), "fixture too weak: MM and plug-in II are identical"

    cfg = CatFEConfig(use_miller_madow=True)  # force MM on the single survivor pair
    ii_mm_arr, _ = _maybe_rerank_with_mm(
        factors_data,
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([ii_plugin], dtype=np.float64),
        nbins,
        target_indices,
        classes_y,
        freqs_y,
        cfg,
        np.int32,
        verbose=0,
    )
    assert np.isclose(
        ii_mm_arr[0], ii_direct_mm, atol=1e-9
    ), f"rerank MM II {ii_mm_arr[0]} != standalone MM II {ii_direct_mm} (plug-in was {ii_plugin}); double-correction reintroduced"


class _FakeCudaErr(Exception):
    """Stand-in for numba's CudaAPIError/CudaDriverError (Exception subclasses, NOT RuntimeError)."""


def test_friend_graph_gpu_forced_cuda_fault_falls_back_to_cpu(monkeypatch):
    """A forced-cuda friend-graph dispatch must fall back to CPU (return None) on a non-RuntimeError CUDA fault, not propagate it."""
    import mlframe.feature_selection.filters.friend_graph_gpu as fgg

    monkeypatch.setattr(fgg, "_CUDA_AVAIL", True, raising=False)
    monkeypatch.setattr(fgg, "friend_graph_stats_cuda", lambda *a, **k: (_ for _ in ()).throw(_FakeCudaErr("driver fault")))
    sel = [0, 1]
    factors_data = np.zeros((10, 3), dtype=np.int32)
    factors_nbins = np.array([2, 2, 2], dtype=np.int64)
    target_indices = np.array([2], dtype=np.int64)
    out = fgg.dispatch_friend_graph_stats(sel, factors_data, factors_nbins, target_indices, force_backend="cuda")
    assert out is None, "a forced-cuda CudaAPIError must fall back to CPU (None), not propagate"


def test_friend_graph_gpu_auto_cuda_fault_falls_back_to_cpu(monkeypatch):
    """The auto (non-forced) friend-graph dispatch must also fall back to CPU on a non-RuntimeError CUDA fault."""
    import mlframe.feature_selection.filters.friend_graph_gpu as fgg

    monkeypatch.setattr(fgg, "_CUDA_AVAIL", True, raising=False)
    monkeypatch.setattr(fgg, "_friend_graph_backend_choice", lambda n, k: "cuda")
    monkeypatch.setattr(fgg, "friend_graph_stats_cuda", lambda *a, **k: (_ for _ in ()).throw(_FakeCudaErr("driver fault")))
    out = fgg.dispatch_friend_graph_stats(
        [0, 1],
        np.zeros((10, 3), dtype=np.int32),
        np.array([2, 2, 2], dtype=np.int64),
        np.array([2], dtype=np.int64),
    )
    assert out is None


def test_usability_corr_dispatch_honors_gpu_optout(monkeypatch):
    """dispatch_batch_pair_usability_corr must honor MLFRAME_DISABLE_GPU even on a CUDA-capable host where cuda_warp would otherwise be the default backend."""
    import mlframe.feature_selection.filters.batch_pair_usability_corr_gpu as bpu

    monkeypatch.setattr(bpu, "_CUDA_AVAIL", True, raising=False)
    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    # Tripwire: if the opt-out is defeated and a GPU kernel is entered, fail loudly instead of maybe-crashing.
    monkeypatch.setattr(bpu, "batch_pair_usability_corr_cuda_warp", lambda *a, **k: pytest.fail("GPU kernel entered despite MLFRAME_DISABLE_GPU=1"))
    monkeypatch.setattr(bpu, "batch_pair_usability_corr_cuda", lambda *a, **k: pytest.fail("GPU kernel entered despite MLFRAME_DISABLE_GPU=1"))

    rng = np.random.default_rng(0)
    n = 400
    y = rng.normal(size=n).astype(np.float64)
    operand_matrix = rng.normal(size=(4, n)).astype(np.float64)
    pair_a = np.array([0, 1], dtype=np.int64)
    pair_b = np.array([2, 3], dtype=np.int64)
    _result, backend = bpu.dispatch_batch_pair_usability_corr(y, operand_matrix, pair_a, pair_b)
    assert backend not in ("cuda", "cuda_warp"), f"opt-out defeated: backend={backend}"


def test_bayesian_blocks_dispatch_defaults_bounded_subsample(monkeypatch):
    """The Bayesian-Blocks dispatch layer must default a bounded subsample_threshold so the O(N^2) DP never runs unbounded by default."""
    import mlframe.feature_selection.filters._adaptive_nbins as an

    captured = {}

    def _spy(col, **kwargs):
        """Record the subsample_threshold the dispatch layer forwards, then return a trivial edge array."""
        captured["subsample_threshold"] = kwargs.get("subsample_threshold")
        return np.array([-np.inf, 0.0, np.inf], dtype=np.float64)

    monkeypatch.setattr(an, "edges_bayesian_blocks", _spy)
    col = np.random.default_rng(0).normal(size=50000)
    an.per_feature_edges(col.reshape(-1, 1), method="bayesian_blocks", n_jobs=1)
    assert captured.get("subsample_threshold"), "BB dispatch must forward a bounded subsample_threshold, not 0"
    assert 0 < captured["subsample_threshold"] <= 50000
    assert captured["subsample_threshold"] == an._BB_DEFAULT_SUBSAMPLE_THRESHOLD


class _AllMedoidsStub:
    """Minimal sklearn-cloneable inner selector for GroupAwareMRMR: selects every medoid it is fit on."""

    def __init__(self, dummy: int = 0):
        """Store the single dummy hyperparameter so get_params/set_params round-trip under sklearn clone."""
        self.dummy = dummy

    def get_params(self, deep: bool = True):
        """Return the single dummy hyperparameter (sklearn estimator protocol)."""
        return {"dummy": self.dummy}

    def set_params(self, **p):
        """Set hyperparameters in place and return self (sklearn estimator protocol)."""
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y, **kw):
        """Select every column presented (support_ = all medoid indices) and return self."""
        self.support_ = np.arange(X.shape[1], dtype=np.int64)
        return self


def test_group_aware_polars_bridge_matches_pandas_clustering():
    """GroupAwareMRMR's polars->pandas bridge must preserve per-column dtypes so a polars frame clusters identically to its pandas equivalent (no object-dtype collapse)."""
    pl = pytest.importorskip("polars")
    import pandas as pd

    from mlframe.feature_selection.filters.group_aware import GroupAwareMRMR

    rng = np.random.default_rng(0)
    n = 300
    a = rng.normal(size=n)
    b = a * 2.0 + 0.01 * rng.normal(size=n)  # a,b ~ perfectly correlated -> one cluster
    c = rng.normal(size=n)
    labels = np.array(["x", "y", "z", "w"])[rng.integers(0, 4, size=n)]  # a non-numeric column
    y = (a + c > 0).astype(int)

    pdf = pd.DataFrame({"a": a, "b": b, "c": c, "cat": labels})
    poldf = pl.DataFrame({"a": a, "b": b, "c": c, "cat": labels})

    m_pd = GroupAwareMRMR(estimator=_AllMedoidsStub(), corr_threshold=0.9).fit(pdf.copy(), y)
    m_pl = GroupAwareMRMR(estimator=_AllMedoidsStub(), corr_threshold=0.9).fit(poldf, y)
    assert list(m_pl.cluster_assignments_) == list(
        m_pd.cluster_assignments_
    ), f"polars bridge diverged from pandas clustering: pl={list(m_pl.cluster_assignments_)} pd={list(m_pd.cluster_assignments_)}"
    # a & b must share a cluster (their real corr ~1); the object-dtype collapse bug broke exactly this.
    ca = list(m_pl.cluster_assignments_)
    assert ca[0] == ca[1], f"correlated a,b should cluster together under the polars path; got {ca}"


def test_adaptive_arity_multileg_recipes_freeze_preprocess_params():
    """Adaptive-arity arity>=2 recipes must freeze per-leg preprocess params so transform() replays (never refits) the basis axis."""
    import pandas as pd

    from mlframe.feature_selection.filters._orthogonal_adaptive_arity_fe import (
        hybrid_orth_mi_adaptive_arity_fe_with_recipes,
    )

    rng = np.random.default_rng(1)
    n = 1500
    a = rng.normal(3.0, 2.0, size=n)  # non-standard mean/scale so preprocess params are non-trivial
    b = rng.normal(-1.0, 0.5, size=n)
    c = rng.normal(10.0, 5.0, size=n)
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    # A strong even-symmetric 2-way cross ((a-mean)^2 * (b-mean)^2) that no single leg captures, so the
    # adaptive-arity scan should emit an arity-2 orth_pair_cross recipe.
    az = (a - a.mean()) ** 2
    bz = (b - b.mean()) ** 2
    y = (az * bz > np.median(az * bz)).astype(np.int64)

    _X_aug, _uni, _adaptive, recipes = hybrid_orth_mi_adaptive_arity_fe_with_recipes(
        df,
        y,
        cols=["a", "b", "c"],
        max_arity=2,
        degrees=(2, 3),
        top_k=6,
        seed_k=3,
        top_count=4,
        min_uplift=1.0,
        min_abs_mi_frac=0.0,
        adaptive_min_uplift=1.0,
        adaptive_min_abs_mi_frac=0.0,
    )
    multileg = [r for r in recipes if getattr(r, "kind", "") in ("orth_pair_cross", "orth_triplet_cross", "orth_quadruplet_cross")]
    if not multileg:
        pytest.skip("fixture produced no arity>=2 adaptive-arity recipe to check")
    for r in multileg:
        extra = dict(getattr(r, "extra", {}) or {})
        n_legs = len(getattr(r, "src_names", ()) or ())
        want = ["preprocess_params_i", "preprocess_params_j", "preprocess_params_k", "preprocess_params_l"][:n_legs]
        present = [k for k in want if extra.get(k) is not None]
        assert (
            present
        ), f"arity-{n_legs} recipe {getattr(r, 'name', '?')} froze NO preprocess params (keys={list(extra)}) -- transform() will refit the basis axis"
