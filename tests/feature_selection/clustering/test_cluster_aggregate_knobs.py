"""Cluster-discovery knob coverage for ``run_cluster_aggregate_step`` / ``_discover_clusters``.

Closes the knob holes flagged in the FS-coverage audit (coverage_asymmetry_wrappers-14): the seven
cluster-discovery parameters that shape WHICH clusters get found were untested -- ``min_member_relevance``,
``max_cluster_size``, ``homogeneity_tau``, ``max_candidates``, ``mi_eps``, ``edge_significance``,
``is_polars_input`` -- plus the param_axes-17 ask to parametrize over the full 9-method roster
(``CLUSTER_AGGREGATE_METHODS``) instead of a hand-picked subset, so a new aggregator auto-enrolls.

Fixture: ONE latent factor ``z`` reflected in 6 graded-relevance members (loadings 1.4..0.55) + an
independent signal + 4 pure-noise columns (``make_latent_reflections``). On this frame the discoverer
finds exactly ONE cluster of the 6 reflections; the noise and the independent signal stay out. The knobs
are then driven directly against the discovery / orchestration functions (the same harness the existing
``test_gate_rejects_when_no_mi_gain`` uses), so the assertions are behavioural -- measured cluster
membership / recipe content / appended-count, never source inspection.

Calibration (measured 2026-06-10, n=4000, nb=8, seed=42): member relevances x0=0.281, x1=0.254,
x2=0.215, x3=0.190, x4=0.148, x5=0.103; homogeneous cluster PC1 var-ratio 0.671; a distinct_sd=0.9
heterogeneous cluster var-ratio 0.491 (so tau=0.99 rejects, tau=0.0 appends). All floors here are exact
structural facts (membership / lengths / counts), not noisy magnitudes, so they need no margin.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tests.feature_selection.conftest import fast_subset, is_fast_mode  # noqa: F401
from tests.feature_selection._biz_val_synth import make_latent_reflections

from mlframe.feature_selection.filters._cluster_aggregate import (
    CLUSTER_AGGREGATE_METHODS,
    _discover_clusters,
    run_cluster_aggregate_step,
)
from mlframe.feature_selection.filters.discretization import discretize_array
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

warnings.filterwarnings("ignore")


# Graded loadings -> graded per-member relevance (x0 strongest .. x5 weakest); 4 pure-noise cols + the
# independent signal sit outside the single latent cluster. n=4000 keeps the MI/corr estimates stable.
_LOADINGS = (1.4, 1.2, 1.0, 0.85, 0.7, 0.55)
_NOISE_SD = (0.7,) * 6
_N_NOISE = 4
_NB = 8


def _build(n: int = 4000, seed: int = 42, nb: int = _NB, **mlr):
    """Binned-matrix + raw-frame harness for ``run_cluster_aggregate_step``.

    Returns ``(X, y, info, data, cols, nbins, target_indices, nb)`` where ``data`` is the int-binned
    feature+target matrix the discoverer consumes, ``X`` the raw pandas frame the recipe replays from,
    and ``cols`` the column names with the target appended last.
    """
    X_arr, y, info = make_latent_reflections(
        n=n,
        loadings=_LOADINGS,
        noise_sd=_NOISE_SD,
        n_noise=_N_NOISE,
        indep_weight=0.4,
        seed=seed,
        **mlr,
    )
    p = X_arr.shape[1]
    names = [f"x{i}" for i in range(p)]
    X = pd.DataFrame(X_arr, columns=names)
    binned = [discretize_array(arr=X[c].to_numpy(), n_bins=nb, method="quantile", dtype=np.int32) for c in names]
    binned.append(y.astype(np.int32))
    data = np.column_stack(binned).astype(np.int32)
    nbins = np.array([nb] * p + [2], dtype=np.int64)
    return X, y, info, data, [*names, "y"], nbins, (p,), nb


def _discover(data, cols, nbins, X, target_indices, **over):
    kw = dict(
        data=data,
        cols=cols,
        nbins=nbins,
        X=X,
        target_indices=target_indices,
        feature_names_in_=[c for c in cols if c != "y"],
        categorical_idx=(),
        cached_MIs={},
        min_member_relevance=0.0,
        corr_threshold=0.5,
        min_cluster_size=3,
        max_cluster_size=12,
        homogeneity_tau=0.6,
        max_candidates=200,
        mi_eps=1e-6,
        edge_significance=3.0,
        dtype=np.int32,
    )
    kw.update(over)
    return _discover_clusters(**kw)


def _run(data, cols, nbins, X, target_indices, nb, methods=("mean_z", "pca_pc1", "mean_inv_var"), **over):
    """Drive ``run_cluster_aggregate_step``; return ``(n_added, summary, engineered_recipes)``."""
    kw = dict(
        data=data.copy(),
        cols=list(cols),
        nbins=nbins.copy(),
        X=X,
        target_indices=target_indices,
        feature_names_in_=[c for c in cols if c != "y"],
        categorical_idx=(),
        cached_MIs={},
        engineered_recipes={},
        quantization_nbins=nb,
        quantization_method="quantile",
        quantization_dtype=np.int32,
        methods=methods,
        mi_prevalence=1.0,
        corr_threshold=0.5,
        min_cluster_size=3,
        verbose=0,
    )
    kw.update(over)
    recipes = kw["engineered_recipes"]
    out = run_cluster_aggregate_step(**kw)
    n_added = out[4]
    summary = out[7]
    return n_added, summary, recipes


def _member_names(clusters, cols):
    return [sorted(cols[m] for m in c["members"]) for c in clusters]


# ---------------------------------------------------------------------------
# Sanity: the fixture yields exactly one latent cluster of the six reflections
# ---------------------------------------------------------------------------


def test_fixture_discovers_single_latent_cluster():
    """The 6 graded reflections collapse to ONE cluster; the independent signal + 4 noise cols stay out.
    This pins the baseline every knob test perturbs from."""
    X, _y, _info, data, cols, nbins, ti, _nb = _build()
    clusters = _discover(data, cols, nbins, X, ti)
    assert len(clusters) == 1, f"one latent factor must give one cluster; got {_member_names(clusters, cols)}"
    members = set(cols[m] for m in clusters[0]["members"])
    assert members == {"x0", "x1", "x2", "x3", "x4", "x5"}, members
    # The independent signal (x6) and the four noise cols (x7..x10) must NOT be pulled in.
    assert not (members & {"x6", "x7", "x8", "x9", "x10"})


# ---------------------------------------------------------------------------
# (a) / param_axes-17: every method in the 9-roster yields a finite aggregate recipe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", fast_subset(sorted(CLUSTER_AGGREGATE_METHODS), 3), ids=str)
def test_each_method_yields_finite_aggregate_recipe(method):
    """Parametrized over ``sorted(CLUSTER_AGGREGATE_METHODS)`` (per-method id) so a future aggregator
    auto-enrolls. Each method, used as the SOLE method with the gate disabled (``mi_prevalence=0`` so the
    accept/reject decision can't mask a NaN recipe), must append exactly one aggregate whose replayed
    column is entirely finite (no NaN/inf) and carries the requested method tag."""
    X, _y, _info, data, cols, nbins, ti, nb = _build()
    n_added, summary, recipes = _run(data, cols, nbins, X, ti, nb, methods=(method,), mi_prevalence=0.0)
    assert n_added == 1, f"method {method} should append one aggregate on a clean cluster"
    assert len(recipes) == 1
    recipe = next(iter(recipes.values()))
    assert recipe.kind == "cluster_aggregate"
    assert recipe.extra["method"] == method
    col = np.asarray(apply_recipe(recipe, X), dtype=np.float64)
    assert col.shape[0] == X.shape[0]
    assert np.all(np.isfinite(col)), f"method {method} produced a non-finite aggregate column"
    assert summary[0]["method"] == method


# ---------------------------------------------------------------------------
# (b) max_cluster_size truncation: rep + top-|corr| members, length <= cap
# ---------------------------------------------------------------------------


def test_max_cluster_size_truncates_to_representative_plus_top_corr():
    """``max_cluster_size=3`` truncates the 6-member cluster to the representative (highest-relevance
    member, x0) + the top-|corr|-to-rep members (NOT the top-relevance ones -- the source rule at
    _cluster_aggregate.py:360-364). The recipe ``src_names`` therefore has length 3, includes the rep,
    and is a subset of the full reflection set."""
    X, _y, _info, data, cols, nbins, ti, nb = _build()
    full = _discover(data, cols, nbins, X, ti)
    assert len(full[0]["members"]) == 6  # untruncated baseline

    clusters = _discover(data, cols, nbins, X, ti, max_cluster_size=3)
    assert len(clusters) == 1
    members = [cols[m] for m in clusters[0]["members"]]
    assert len(members) <= 3, members
    assert cols[clusters[0]["rep"]] == "x0", "representative is the highest-relevance member"
    assert "x0" in members, "truncation must retain the representative"
    assert set(members) <= {"x0", "x1", "x2", "x3", "x4", "x5"}

    # The recipe ``src_names`` mirrors the truncated membership.
    n_added, _summary, recipes = _run(data, cols, nbins, X, ti, nb, methods=("mean_z",), max_cluster_size=3)
    assert n_added == 1
    recipe = next(iter(recipes.values()))
    assert len(recipe.src_names) <= 3
    assert "x0" in recipe.src_names
    assert set(recipe.src_names) <= {"x0", "x1", "x2", "x3", "x4", "x5"}


# ---------------------------------------------------------------------------
# (c) min_member_relevance: pool filter drops members below the floor
# ---------------------------------------------------------------------------


def test_min_member_relevance_drops_weakest_member():
    """The weakest reflection x5 has marginal relevance ~0.103 (measured). A ``min_member_relevance``
    floor set above it (0.12) but below the next member x4 (~0.148) removes x5 from the discovered
    cluster -- and from the built recipe's ``src_names`` -- while keeping x0..x4."""
    X, _y, _info, data, cols, nbins, ti, nb = _build()

    clusters = _discover(data, cols, nbins, X, ti, min_member_relevance=0.12)
    assert len(clusters) == 1
    members = set(cols[m] for m in clusters[0]["members"])
    assert "x5" not in members, f"x5 (rel~0.103) must be filtered by floor 0.12; got {sorted(members)}"
    assert {"x0", "x1", "x2", "x3", "x4"} <= members

    n_added, _summary, recipes = _run(data, cols, nbins, X, ti, nb, methods=("mean_z",), min_member_relevance=0.12)
    assert n_added == 1
    recipe = next(iter(recipes.values()))
    assert "x5" not in recipe.src_names, recipe.src_names
    assert "x0" in recipe.src_names


# ---------------------------------------------------------------------------
# (d) homogeneity_tau gate DIRECTION: strict rejects heterogeneous cluster, permissive appends
# ---------------------------------------------------------------------------


def test_homogeneity_tau_gate_direction_on_heterogeneous_cluster():
    """``distinct_sd=0.9`` injects a per-reflection distinct signal -> the cluster is multi-factor, its
    PC1 variance-ratio drops to ~0.49 (measured). The unidimensionality gate is then DIRECTIONAL: a
    strict ``homogeneity_tau=0.99`` rejects it (no aggregate appended), while a permissive ``tau=0.0``
    accepts and appends. Pins both directions so the gate sense can't silently invert."""
    Xh, _yh, _infoh, datah, colsh, nbinsh, tih, nbh = _build(distinct_sd=0.9, seed=11)

    strict = _discover(datah, colsh, nbinsh, Xh, tih, homogeneity_tau=0.99)
    permissive = _discover(datah, colsh, nbinsh, Xh, tih, homogeneity_tau=0.0)
    assert len(strict) == 0, "extreme tau must reject a heterogeneous (multi-factor) cluster"
    assert len(permissive) >= 1, "permissive tau must accept the heterogeneous cluster"

    n_strict, _, rec_strict = _run(datah, colsh, nbinsh, Xh, tih, nbh, methods=("mean_z", "pca_pc1"), homogeneity_tau=0.99)
    n_perm, _, rec_perm = _run(datah, colsh, nbinsh, Xh, tih, nbh, methods=("mean_z", "pca_pc1"), homogeneity_tau=0.0)
    assert n_strict == 0 and not rec_strict, "strict tau -> no aggregate appended"
    assert n_perm >= 1 and rec_perm, "permissive tau -> aggregate appended"


# ---------------------------------------------------------------------------
# (e) max_candidates: pool starvation degrades gracefully (<=1 cluster, no crash)
# ---------------------------------------------------------------------------


def test_max_candidates_pool_starvation_no_crash():
    """``max_candidates=2`` caps the candidate pool at 2 members, below ``min_cluster_size=3`` -> the
    discoverer returns at most one cluster (here zero) WITHOUT raising, and the orchestration step
    appends nothing. Guards the pool-cap edge against an index/empty-pool crash."""
    X, _y, _info, data, cols, nbins, ti, nb = _build()

    clusters = _discover(data, cols, nbins, X, ti, max_candidates=2)
    assert len(clusters) <= 1, f"a 2-member pool cannot form a >=3-member cluster; got {_member_names(clusters, cols)}"

    n_added, _summary, recipes = _run(data, cols, nbins, X, ti, nb, max_candidates=2)
    assert n_added == 0 and not recipes


# ---------------------------------------------------------------------------
# (f) edge_significance / mi_eps sweep: strict prunes all edges -> zero clusters; permissive recovers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "knob",
    [
        pytest.param("edge_significance", id="edge_significance"),
        pytest.param("mi_eps", id="mi_eps"),
    ],
)
def test_edge_threshold_sweep_prunes_then_recovers(knob):
    """The finite-sample edge floor is ``max(mi_eps, edge_significance*(na-1)(nb-1)/(2n))``. Driving
    EITHER term to a huge value (1e9) lifts the floor above every real reflection-pair MI, so no edges
    survive and zero clusters form; the permissive defaults recover the single latent cluster. Pins the
    monotone pruning direction of both edge-test knobs."""
    X, _y, _info, data, cols, nbins, ti, _nb = _build()

    strict = _discover(data, cols, nbins, X, ti, **{knob: 1e9})
    permissive = _discover(data, cols, nbins, X, ti, edge_significance=3.0, mi_eps=1e-6)
    assert len(strict) == 0, f"{knob}=1e9 must prune all edges -> zero clusters"
    assert len(permissive) == 1, "permissive edge thresholds recover the latent cluster"
    assert set(cols[m] for m in permissive[0]["members"]) == {"x0", "x1", "x2", "x3", "x4", "x5"}


# ---------------------------------------------------------------------------
# (g) is_polars_input parity: pl.DataFrame gives the same recipe as the pandas run
# ---------------------------------------------------------------------------


def test_is_polars_input_parity_with_pandas():
    """``is_polars_input=True`` on the same data as a ``pl.DataFrame`` must build a recipe whose name,
    method and ``src_names`` equal the pandas run, with weights allclose. ``pca_pc1`` is forced as the
    sole method so the recipe deterministically carries a linear weight vector to compare."""
    pl = pytest.importorskip("polars")
    X, _y, _info, data, cols, nbins, ti, nb = _build()
    Xpl = pl.from_pandas(X)

    n_pd, _sum_pd, rec_pd = _run(data, cols, nbins, X, ti, nb, methods=("pca_pc1",))
    n_pl, _sum_pl, rec_pl = _run(data, cols, nbins, Xpl, ti, nb, methods=("pca_pc1",), is_polars_input=True)

    assert n_pd == 1 and n_pl == 1
    r_pd = next(iter(rec_pd.values()))
    r_pl = next(iter(rec_pl.values()))
    assert r_pd.name == r_pl.name
    assert r_pd.extra["method"] == r_pl.extra["method"] == "pca_pc1"
    assert r_pd.src_names == r_pl.src_names
    w_pd = np.asarray(r_pd.extra["weights"], dtype=np.float64)
    w_pl = np.asarray(r_pl.extra["weights"], dtype=np.float64)
    assert np.allclose(w_pd, w_pl), f"polars vs pandas weights diverge: {w_pd} vs {w_pl}"
    # Replayed aggregate columns must match value-for-value too (parity is end-to-end, not just metadata).
    col_pd = np.asarray(apply_recipe(r_pd, X), dtype=np.float64)
    col_pl = np.asarray(apply_recipe(r_pl, X), dtype=np.float64)
    assert np.array_equal(col_pd, col_pl)
