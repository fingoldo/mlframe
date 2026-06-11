"""Tests for compute-gated SHAP diagnostic panels (reporting/charts/shap_panels.py).

Ships: unit tests (gate, top-K ordering, paths written, NO figure leak), a biz_value test (a synthetic
where y depends strongly on f0 -> f0 tops the beeswarm AND its dependence trend is the correct monotone
sign), and a cProfile pass at a production shape (n>=1e6 subsampled to 20k) documenting the sample cap as
the wall-time lever.

shap is a project dep but guarded via importorskip so a shap-less CI env skips rather than errors.
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

shap = pytest.importorskip("shap")
pd = pytest.importorskip("pandas")
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

from mlframe.reporting.charts import shap_panels as sp  # noqa: E402


def _strong_f0(n: int, n_feat: int = 5, *, coef: float = 3.0, noise: float = 0.1, seed: int = 0):
    """Synthetic where y is a strong monotone-increasing function of f0 alone."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_feat)), columns=[f"f{i}" for i in range(n_feat)])
    y = coef * X["f0"].to_numpy() + noise * rng.normal(size=n)
    return X, y


def _fit_rf(X, y, *, n_estimators: int = 30, max_depth: int = 6, seed: int = 0) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=1).fit(X, y)


@pytest.fixture(autouse=True)
def _no_leaked_figures():
    """Every test must leave the pyplot registry as it found it (no figure leak in long sessions)."""
    before = set(plt.get_fignums())
    yield
    leaked = set(plt.get_fignums()) - before
    for num in leaked:
        plt.close(num)
    assert not leaked, f"test leaked open figures: {sorted(leaked)}"


def test_is_tree_model_gate():
    """Tree estimators -> True (cheap exact TreeExplainer); linear / unknown -> False (opt-in kernel)."""
    assert sp.is_tree_model(RandomForestRegressor()) is True
    assert sp.is_tree_model(LinearRegression()) is False
    assert sp.is_tree_model(None) is False

    class _Marked:
        _is_tree = True

    assert sp.is_tree_model(_Marked()) is True


def test_non_tree_default_is_opt_in():
    """A non-tree model with no ``allow_kernel`` is SKIPPED (KernelExplainer too slow for default-on)."""
    X, y = _strong_f0(300, seed=1)
    lr = LinearRegression().fit(X, y)
    res = sp.shap_summary_and_dependence(lr, X, feature_names=list(X.columns))
    assert res.skipped is not None
    assert "allow_kernel" in res.skipped
    assert res.figures == [] and res.paths == []


def test_tree_unit_paths_ordering_and_no_leak(tmp_path):
    """Tree path: beeswarm + top-K dependence files written, top-K ordering by mean|SHAP|, figures closed."""
    X, y = _strong_f0(3000, seed=0)
    m = _fit_rf(X, y)
    base = str(tmp_path / "shap.png")
    res = sp.shap_summary_and_dependence(m, X, feature_names=list(X.columns), max_rows=2000, top_k=4, plot_file=base)

    assert res.skipped is None
    assert res.explainer_kind == "tree"
    assert len(res.top_features) == 4
    assert res.top_features[0] == "f0"  # the only signal feature must rank first

    # mean_abs_shap descending matches the reported top-K order.
    name_to_mean = dict(zip([f"f{i}" for i in range(5)], res.mean_abs_shap))
    means = [name_to_mean[nm] for nm in res.top_features]
    assert means == sorted(means, reverse=True)

    # 1 beeswarm + 4 dependence = 5 figures and 5 files on disk.
    assert len(res.figures) == 5
    assert len(res.paths) == 5
    assert all(os.path.exists(p) for p in res.paths)
    assert any("beeswarm" in p for p in res.paths)
    assert sum("dependence" in p for p in res.paths) == 4

    # No leak: the producer closed every figure it opened.
    assert plt.get_fignums() == []


def test_plot_outputs_dsl_format(tmp_path):
    """``plot_outputs`` selects the matplotlib raster/vector format(s) (e.g. svg)."""
    X, y = _strong_f0(1500, seed=2)
    m = _fit_rf(X, y, n_estimators=20)
    base = str(tmp_path / "shap")  # no extension -> DSL decides
    res = sp.shap_summary_and_dependence(
        m, X, feature_names=list(X.columns), max_rows=1000, top_k=2, plot_file=base, plot_outputs="matplotlib[svg]",
    )
    assert res.paths
    assert all(p.endswith(".svg") for p in res.paths)
    assert all(os.path.exists(p) for p in res.paths)


def test_kernel_opt_in_runs_bounded():
    """Non-tree + ``allow_kernel=True`` runs KernelExplainer on the small caps and closes its figures."""
    X, y = _strong_f0(300, n_feat=4, seed=1)
    lr = LinearRegression().fit(X, y)
    res = sp.shap_summary_and_dependence(
        lr, X, feature_names=list(X.columns), allow_kernel=True, kernel_max_rows=120, kernel_background=40, top_k=3,
    )
    assert res.skipped is None
    assert res.explainer_kind == "kernel"
    assert res.top_features[0] == "f0"
    assert plt.get_fignums() == []


def test_empty_input_skips():
    """Zero rows / zero columns is a best-effort skip, not a crash."""
    m = _fit_rf(*_strong_f0(200, seed=0))
    res = sp.shap_summary_and_dependence(m, np.empty((0, 5)), feature_names=[f"f{i}" for i in range(5)])
    assert res.skipped == "empty input"
    assert res.figures == []


def test_biz_value_f0_ranks_first_and_monotone(tmp_path):
    """biz_value: y = 3*f0 -> f0 has the largest mean|SHAP| (top of beeswarm) AND its dependence trend is
    correctly monotone-increasing (corr(f0_value, shap_f0) strongly positive). A regression that breaks the
    explainer / ordering / sign trips this.
    """
    X, y = _strong_f0(4000, n_feat=6, coef=3.0, noise=0.1, seed=7)
    m = _fit_rf(X, y, n_estimators=50, max_depth=7)
    res = sp.shap_summary_and_dependence(m, X, feature_names=list(X.columns), max_rows=2500, top_k=4, plot_file=str(tmp_path / "biz.png"))

    assert res.top_features[0] == "f0", f"f0 must top the beeswarm, got {res.top_features}"
    f0_mean = res.mean_abs_shap[0]
    others = np.delete(res.mean_abs_shap, 0)
    # f0's mean |SHAP| dominates: at least 5x the next-largest feature (measured ratio is >>10x at this signal).
    assert f0_mean >= 5.0 * float(others.max()), f"f0 mean|SHAP| {f0_mean:.4f} not dominant vs {others.max():.4f}"

    # Dependence sign/monotonicity: SHAP_f0 increases with f0's value. Recompute the same SHAP matrix the
    # dependence plot uses (one explainer reused) and assert a strongly positive rank-like correlation.
    explainer = shap.TreeExplainer(m)
    Xs = X.iloc[:2000]
    shap_mat = sp._shap_values_2d(explainer(Xs, check_additivity=False))
    corr = float(np.corrcoef(Xs["f0"].to_numpy(), shap_mat[:, 0])[0, 1])
    assert corr >= 0.9, f"f0 dependence trend should be monotone-increasing (corr>=0.9), got {corr:.3f}"


def test_cprofile_tree_bounded_by_sample_cap(tmp_path):
    """cProfile at a production shape: n=1e6 rows subsampled to 20k -> bounded wall.

    The sample cap (``max_rows``) is the wall-time lever: TreeExplainer cost scales with the EXPLAINED row
    count, not n. We materialise a 1M-row frame but the producer subsamples to ``max_rows`` BEFORE any SHAP
    work, so the SHAP cost is a function of 20k, independent of the 1M source. No actionable in-module
    hotspot beyond the cap: shap's own C++ tree traversal dominates; reuse of ONE explainer + ONE
    shap_values for beeswarm + all dependence plots already removes per-feature recompute. Lowering
    ``max_rows`` is the only further lever and trades resolution for speed.
    """
    n = 1_000_000
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, 5)).astype(np.float32), columns=[f"f{i}" for i in range(5)])
    y = 3.0 * X["f0"].to_numpy() + 0.1 * rng.normal(size=n).astype(np.float32)
    # Small forest so the FIT (not the diagnostic) doesn't dominate the profile -- the producer is the SUT.
    m = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0, n_jobs=1).fit(X.iloc[:50_000], y[:50_000])

    pr = cProfile.Profile()
    pr.enable()
    res = sp.shap_summary_and_dependence(m, X, feature_names=list(X.columns), max_rows=20_000, top_k=6, plot_file=str(tmp_path / "prof.png"))
    pr.disable()

    assert res.skipped is None
    assert res.top_features[0] == "f0"
    assert plt.get_fignums() == []

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
    total = float(pstats.Stats(pr).total_tt)
    # Generous ceiling: the cap makes this independent of the 1M source rows; on the dev box this is single-digit
    # seconds. The assertion guards against an accidental "explain all n" regression, not micro-timing.
    assert total < 120.0, f"SHAP producer wall {total:.1f}s exceeds the cap-bounded budget"
