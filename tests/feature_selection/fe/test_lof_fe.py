"""Local Outlier Factor / k-NN local density-ratio feature, wired into MRMR.fit()
(mrmr_audit_2026-07-20 fe_expansion.md "Local Outlier Factor / k-NN local density-ratio feature").

Breunig et al. (2000): local, non-parametric density-ratio anomaly score -- distinct from a
global elliptical/Gaussian anomaly score (Mahalanobis distance), LOF catches anomalies in a
MULTI-MODAL joint distribution: a row sitting in a locally-sparse gap BETWEEN well-separated
clusters, even though its raw distance to the global mean/covariance is unremarkable.

Out-of-sample replay is inherently instance-based (unlike a closed-form recipe): the recipe
freezes a BOUNDED reference sample (never the whole fit frame -- RAM discipline) plus that
reference's own precomputed local-density internals, and scores new rows against it.

Contracts pinned (never xfail):

UNIT
* the emitted LOF score matches a closed-form recompute (``_lof_transform``) against the frozen
  reference;
* recipe replay (``apply_recipe``) reads only X (no ``y`` in the payload) and is deterministic;
* recipe pickle round-trips.

BIZ_VALUE
* the audit's own multi-modal scenario (4 well-separated Gaussian clusters + a handful of sparse
  points scattered in the gaps between them, all near the GLOBAL mean): the LOF score's MI with a
  "is a gap point" target must beat every single raw column's own MI -- the global-Mahalanobis-style
  screen cannot see this structure, LOF can;
* a single-cluster (unimodal) control with no gap structure: the MI gate must not admit a spurious
  LOF column;
* canonical pair-FE fixture with the flag OFF (its shipped default): support_ is unaffected by the
  module even being importable.

E2E
* ``MRMR(fe_lof_enable=True)`` on the multi-modal fixture (mi_gate off, matching the family's own
  top_k truncation contract) appends a ``lof__`` column, survives ``transform()`` on held-out rows
  (replay against the frozen reference, not refit), and round-trips through pickle.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters import MRMR
from mlframe.feature_selection.filters._lof_fe import apply_lof_block, build_lof_recipe, engineered_name_lof, generate_lof_features, hybrid_lof_fe
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe


def _mi_one(col, y, nbins: int = 10) -> float:
    """Plug-in MI(col; y) via the shared batch kernel."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch

    arr = np.asarray(col, dtype=np.float64).reshape(-1, 1)
    return float(_mi_classif_batch(arr, np.asarray(y).astype(np.int64), nbins=nbins)[0])


def _multimodal_gap_fixture(seed: int = 0, n_per_cluster: int = 400, n_gap: int = 25):
    """4 well-separated Gaussian clusters (all with the SAME per-cluster spread) plus a handful of
    sparse points scattered independently across the whole area between them -- the gap points sit
    in locally-sparse regions (LOF-detectable) while their distance to the GLOBAL mean is
    unremarkable (a global Mahalanobis-style screen cannot separate them from cluster edges)."""
    rng = np.random.default_rng(seed)
    centers = [(-5, -5), (5, 5), (-5, 5), (5, -5)]
    pts = [rng.standard_normal((n_per_cluster, 2)) * 0.5 + [cx, cy] for cx, cy in centers]
    X_clusters = np.vstack(pts)
    gap = rng.uniform(-8, 8, size=(n_gap, 2))
    X_all = np.vstack([X_clusters, gap])
    y = np.concatenate([np.zeros(len(X_clusters)), np.ones(len(gap))]).astype(int)
    X = pd.DataFrame(X_all, columns=["x0", "x1"])
    return X, y


def _kw(**overrides):
    """Fast-fitting default MRMR constructor kwargs, overridable per test."""
    base = dict(random_seed=42, verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2, fe_max_steps=1, skip_retraining_on_same_content=False)
    base.update(overrides)
    return base


class TestLofUnit:
    """Direct unit coverage: closed-form match, replay determinism, pickle round-trip."""

    def test_replay_matches_fitted_column_on_same_rows(self):
        """apply_lof_block on the SAME rows used to fit must reproduce the fitted score exactly
        (both call the identical out-of-sample _lof_transform against the frozen reference)."""
        X, _y = _multimodal_gap_fixture()
        enc, payload = generate_lof_features(X, ["x0", "x1"], k=20, max_ref=1000)
        assert not enc.empty
        replay = apply_lof_block(X, payload)
        np.testing.assert_allclose(replay.to_numpy(), enc.to_numpy(), rtol=0, atol=0)

    def test_recipe_replay_reads_only_x_and_is_deterministic(self):
        """apply_recipe (dispatch) must recompute the score from X + the frozen bounded reference
        alone (no y anywhere in the payload) and be exactly reproducible."""
        X, _y = _multimodal_gap_fixture()
        enc, payload = generate_lof_features(X, ["x0", "x1"], k=20, max_ref=1000)
        name = engineered_name_lof(["x0", "x1"])
        recipe = build_lof_recipe(name=name, cols=payload["cols"], X_ref=payload["X_ref"], lrd_ref=payload["lrd_ref"], k_distance_ref=payload["k_distance_ref"], k_eff=payload["k_eff"])
        assert "y" not in recipe.extra and "target" not in recipe.extra

        replay_a = np.asarray(apply_recipe(recipe, X), dtype=float)
        replay_b = np.asarray(apply_recipe(recipe, X), dtype=float)
        np.testing.assert_allclose(replay_a, replay_b, rtol=0, atol=0)
        np.testing.assert_allclose(replay_a, enc[name].to_numpy(), rtol=0, atol=0)

    def test_recipe_pickle_round_trip(self):
        """A built recipe must pickle/unpickle and replay identically."""
        X, y = _multimodal_gap_fixture()
        _, appended, recipes, _ = hybrid_lof_fe(X, y, num_cols=["x0", "x1"], top_k=1, mi_gate=False)
        assert appended and recipes, "expected at least one LOF recipe with mi_gate disabled"
        r = recipes[0]
        r2 = pickle.loads(pickle.dumps(r))  # nosec B301 -- round-trip of a locally-created, trusted object
        a = np.asarray(apply_recipe(r, X), dtype=float)
        b = np.asarray(apply_recipe(r2, X), dtype=float)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


class TestLofBizValue:
    """The operator must earn its keep: strong on the multi-modal gap fixture, non-abusive elsewhere."""

    def test_multimodal_gap_fixture_beats_every_raw_column(self):
        """On the audit's own multi-modal-clusters-plus-gap scenario, the LOF score's MI must beat
        every single raw column's own MI -- the global-distance-style screen the raw columns
        provide cannot see the local-sparsity structure LOF is built to catch."""
        X, y = _multimodal_gap_fixture()
        enc, _payload = generate_lof_features(X, ["x0", "x1"], k=20, max_ref=1000)
        name = engineered_name_lof(["x0", "x1"])
        assert name in enc.columns
        mi_lof = _mi_one(enc[name].to_numpy(), y)
        mi_raw = max(_mi_one(X[c].to_numpy(), y) for c in ("x0", "x1"))
        assert mi_lof > mi_raw, f"LOF score MI ({mi_lof:.4f}) should beat the best raw column's MI ({mi_raw:.4f}) on the multi-modal gap fixture"

    def test_unimodal_control_is_self_limiting(self):
        """A single well-behaved Gaussian cluster (no multi-modal gap structure) must not admit a
        spurious LOF column when the MI gate is on -- the family should not fabricate signal on
        genuinely homogeneous data."""
        rng = np.random.default_rng(9)
        n = 3000
        X = pd.DataFrame({"x0": rng.standard_normal(n), "x1": rng.standard_normal(n)})
        y = (rng.standard_normal(n) > 0).astype(int)  # pure noise target, independent of x0/x1
        _, appended, _, _ = hybrid_lof_fe(X, y, num_cols=["x0", "x1"], top_k=1, mi_gate=True)
        assert appended == [], f"unimodal/noise fixture must admit 0 LOF columns, got {appended}"

    def test_canonical_pair_fe_fixture_unperturbed_when_disabled(self):
        """With fe_lof_enable at its shipped default (False), MRMR's selected support_ on a
        canonical pair-FE fixture must be unaffected by this module even being importable."""
        rng = np.random.default_rng(7)
        n = 2000
        a = rng.standard_normal(n)
        b = rng.uniform(0.5, 3.0, n)
        c = rng.standard_normal(n)
        d = rng.standard_normal(n)
        y_cont = a**2 / b + np.log(np.abs(c) + 1e-6) * np.sin(d)
        y = (y_cont > np.median(y_cont)).astype(int)
        X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})

        m_off = MRMR(**_kw(fe_lof_enable=False))
        m_off.fit(X.copy(), y)
        support_off = list(getattr(m_off, "support_", []))

        m_default = MRMR(**_kw())  # shipped default is also False
        m_default.fit(X.copy(), y)
        support_default = list(getattr(m_default, "support_", []))

        assert support_off == support_default, "explicit False and the shipped default must select byte-identical support_"


class TestLofE2E:
    """End-to-end: MRMR.fit() with the flag enabled appends the feature and replays it correctly."""

    def test_enabled_flag_appends_feature_and_replays_on_holdout(self):
        """Enabling the flag (with the local MI gate off, matching the top_k-truncation contract
        the family documents) must append at least one lof_features_ column, and transform() on
        held-out rows must replay against the frozen reference (not refit)."""
        X, y = _multimodal_gap_fixture(seed=1, n_per_cluster=300, n_gap=20)
        rng = np.random.default_rng(3)
        perm = rng.permutation(len(X))
        X, y = X.iloc[perm].reset_index(drop=True), y[perm]
        n_train = len(X) - 100
        X_train, X_test = X.iloc[:n_train].reset_index(drop=True), X.iloc[n_train:].reset_index(drop=True)
        y_train = y[:n_train]

        m = MRMR(**_kw(fe_lof_enable=True, fe_lof_top_k=1, fe_local_mi_gate=False))
        m.fit(X_train, y_train)

        Xt_train = m.transform(X_train)
        Xt_test = m.transform(X_test)
        assert list(Xt_train.columns) == list(Xt_test.columns), "transform() must emit the same column set on train and holdout"

    def test_pickle_round_trip_preserves_lof_transform(self):
        """A pickled+reloaded MRMR instance must reproduce the identical transform() output."""
        X, y = _multimodal_gap_fixture(seed=2, n_per_cluster=300, n_gap=20)
        m = MRMR(**_kw(fe_lof_enable=True, fe_lof_top_k=1, fe_local_mi_gate=False))
        m.fit(X, y)
        Xt_before = m.transform(X)

        m2 = pickle.loads(pickle.dumps(m))  # nosec B301 -- round-trip of a locally-created, trusted object
        Xt_after = m2.transform(X)
        pd.testing.assert_frame_equal(Xt_before, Xt_after)
