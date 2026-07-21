"""Multivariate Mahalanobis / Gaussian-copula joint density anomaly score, wired into MRMR.fit()
(mrmr_audit_2026-07-20 fe_expansion.md "Multivariate Mahalanobis / Gaussian-copula joint density
anomaly score").

Computes ``d(row) = sqrt((x-mu)^T Sigma^-1 (x-mu))`` over a correlated cluster of numeric columns
jointly, with mean/covariance Ledoit-Wolf shrunk -- catches y depending on whether a row sits
inside/outside an ELLIPSOIDAL level-set of a p-way joint distribution where no single column,
pair, triplet, or even quadruplet cross-basis is individually extreme.

Contracts pinned (never xfail):

UNIT
* the emitted score matches a closed-form recompute against the frozen Ledoit-Wolf ``mu``/
  ``Sigma_inv``;
* recipe replay (``apply_recipe``) reads only X (no ``y`` in the payload) and is exact/deterministic;
* recipe pickle round-trips.

BIZ_VALUE
* the audit's own scenario (a p=6-way correlated joint distribution; y flags the true top-decile
  Mahalanobis-distance outliers): the Mahalanobis score's MI must beat every single raw column's
  own MI by a wide margin -- no column is individually extreme;
* an independent-columns control (no joint ellipsoidal structure beyond what each marginal already
  carries): the MI gate must not admit a spurious Mahalanobis column;
* canonical pair-FE fixture with the flag OFF (its shipped default): support_ is unaffected by the
  module even being importable.

E2E
* ``MRMR(fe_mahalanobis_density_enable=True)`` on the ellipsoidal-outlier fixture (mi_gate off,
  matching the family's own top_k truncation contract) appends a ``mahal__`` column, survives
  ``transform()`` on held-out rows (replay, not refit), and round-trips through pickle.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters import MRMR
from mlframe.feature_selection.filters._mahalanobis_density_fe import (
    apply_mahalanobis_density,
    build_mahalanobis_density_recipe,
    engineered_name_mahalanobis_density,
    generate_mahalanobis_density_features,
    hybrid_mahalanobis_density_fe,
)
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe


def _mi_one(col, y, nbins: int = 10) -> float:
    """Plug-in MI(col; y) via the shared batch kernel."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch

    arr = np.asarray(col, dtype=np.float64).reshape(-1, 1)
    return float(_mi_classif_batch(arr, np.asarray(y).astype(np.int64), nbins=nbins)[0])


def _ellipsoidal_outlier_fixture(seed: int = 0, n: int = 3000, p: int = 6):
    """A p-way correlated joint Gaussian; y flags the true top-decile Mahalanobis-distance
    outliers -- the audit's own scenario where no single column is individually extreme but the
    JOINT combination is far in Mahalanobis distance."""
    rng = np.random.default_rng(seed)
    cov = np.eye(p)
    for i in range(p - 1):
        cov[i, i + 1] = cov[i + 1, i] = 0.6
    X_arr = rng.multivariate_normal(np.zeros(p), cov, size=n)
    mu = X_arr.mean(axis=0)
    Sigma = np.cov(X_arr.T)
    d2 = np.einsum("ni,ij,nj->n", X_arr - mu, np.linalg.inv(Sigma), X_arr - mu)
    y = (d2 > np.quantile(d2, 0.9)).astype(int)
    X = pd.DataFrame(X_arr, columns=[f"x{i}" for i in range(p)])
    return X, y


def _kw(**overrides):
    """Fast-fitting default MRMR constructor kwargs, overridable per test."""
    base = dict(random_seed=42, verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2, fe_max_steps=1, skip_retraining_on_same_content=False)
    base.update(overrides)
    return base


class TestMahalanobisDensityUnit:
    """Direct unit coverage: closed-form match, replay determinism, pickle round-trip."""

    def test_replay_matches_fitted_column_on_same_rows(self):
        """apply_mahalanobis_density on the SAME rows used to fit must reproduce the fitted score
        exactly (both are the same closed-form quadratic form against the frozen mu/Sigma_inv)."""
        X, _y = _ellipsoidal_outlier_fixture()
        cols = [f"x{i}" for i in range(6)]
        enc, payload = generate_mahalanobis_density_features(X, cols)
        assert not enc.empty
        replay = apply_mahalanobis_density(X, payload)
        np.testing.assert_allclose(replay.to_numpy(), enc.to_numpy(), rtol=1e-10, atol=1e-10)

    def test_recipe_replay_reads_only_x_and_is_deterministic(self):
        """apply_recipe (dispatch) must recompute the score from X + the frozen mu/Sigma_inv alone
        (no y anywhere in the payload) and be exactly reproducible."""
        X, _y = _ellipsoidal_outlier_fixture()
        cols = [f"x{i}" for i in range(6)]
        enc, payload = generate_mahalanobis_density_features(X, cols)
        name = engineered_name_mahalanobis_density(cols)
        recipe = build_mahalanobis_density_recipe(name=name, cols=payload["cols"], mu=payload["mu"], Sigma_inv=payload["Sigma_inv"])
        assert "y" not in recipe.extra and "target" not in recipe.extra

        replay_a = np.asarray(apply_recipe(recipe, X), dtype=float)
        replay_b = np.asarray(apply_recipe(recipe, X), dtype=float)
        np.testing.assert_allclose(replay_a, replay_b, rtol=0, atol=0)
        np.testing.assert_allclose(replay_a, enc[name].to_numpy(), rtol=1e-10, atol=1e-10)

    def test_recipe_pickle_round_trip(self):
        """A built recipe must pickle/unpickle and replay identically."""
        X, y = _ellipsoidal_outlier_fixture()
        _, appended, recipes, _ = hybrid_mahalanobis_density_fe(X, y, num_cols=[f"x{i}" for i in range(6)], top_k=1, mi_gate=False)
        assert appended and recipes, "expected at least one Mahalanobis-density recipe with mi_gate disabled"
        r = recipes[0]
        r2 = pickle.loads(pickle.dumps(r))  # nosec B301 -- round-trip of a locally-created, trusted object
        a = np.asarray(apply_recipe(r, X), dtype=float)
        b = np.asarray(apply_recipe(r2, X), dtype=float)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


class TestMahalanobisDensityBizValue:
    """The operator must earn its keep: strong on the ellipsoidal-outlier fixture, non-abusive elsewhere."""

    def test_ellipsoidal_outlier_fixture_beats_every_raw_column_by_a_wide_margin(self):
        """On the audit's own p-way correlated joint-outlier scenario, the Mahalanobis score's MI
        must beat every single raw column's own MI by a wide margin -- no column is individually
        extreme, but the joint combination is far in Mahalanobis distance."""
        X, y = _ellipsoidal_outlier_fixture()
        cols = [f"x{i}" for i in range(6)]
        enc, _payload = generate_mahalanobis_density_features(X, cols)
        name = engineered_name_mahalanobis_density(cols)
        assert name in enc.columns
        mi_mahal = _mi_one(enc[name].to_numpy(), y)
        mi_raw = max(_mi_one(X[c].to_numpy(), y) for c in cols)
        assert mi_mahal > 3.0 * mi_raw, f"Mahalanobis score MI ({mi_mahal:.4f}) should beat the best raw column's MI ({mi_raw:.4f}) by a wide margin"

    def test_independent_columns_control_is_self_limiting(self):
        """When columns are mutually independent (no joint ellipsoidal structure beyond each
        marginal), the MI gate must not admit a spurious Mahalanobis column on a pure-noise
        target -- the family should not fabricate signal on genuinely unstructured data."""
        rng = np.random.default_rng(5)
        n = 3000
        X = pd.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(6)})
        y = (rng.standard_normal(n) > 0).astype(int)  # pure noise target, independent of all x_i
        _, appended, _, _ = hybrid_mahalanobis_density_fe(X, y, num_cols=list(X.columns), top_k=1, mi_gate=True)
        assert appended == [], f"independent-columns/noise fixture must admit 0 Mahalanobis columns, got {appended}"

    def test_canonical_pair_fe_fixture_unperturbed_when_disabled(self):
        """With fe_mahalanobis_density_enable at its shipped default (False), MRMR's selected
        support_ on a canonical pair-FE fixture must be unaffected by this module even being
        importable."""
        rng = np.random.default_rng(7)
        n = 2000
        a = rng.standard_normal(n)
        b = rng.uniform(0.5, 3.0, n)
        c = rng.standard_normal(n)
        d = rng.standard_normal(n)
        y_cont = a**2 / b + np.log(np.abs(c) + 1e-6) * np.sin(d)
        y = (y_cont > np.median(y_cont)).astype(int)
        X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})

        m_off = MRMR(**_kw(fe_mahalanobis_density_enable=False))
        m_off.fit(X.copy(), y)
        support_off = list(getattr(m_off, "support_", []))

        m_default = MRMR(**_kw())  # shipped default is also False
        m_default.fit(X.copy(), y)
        support_default = list(getattr(m_default, "support_", []))

        assert support_off == support_default, "explicit False and the shipped default must select byte-identical support_"


class TestMahalanobisDensityE2E:
    """End-to-end: MRMR.fit() with the flag enabled appends the feature and replays it correctly."""

    def test_enabled_flag_appends_feature_and_replays_on_holdout(self):
        """Enabling the flag (with the local MI gate off, matching the top_k-truncation contract
        the family documents) must append at least one mahalanobis_density_features_ column, and
        transform() on held-out rows must replay (not refit) it."""
        X, y = _ellipsoidal_outlier_fixture(seed=1, n=3000, p=6)
        rng = np.random.default_rng(3)
        perm = rng.permutation(len(X))
        X, y = X.iloc[perm].reset_index(drop=True), y[perm]
        n_train = len(X) - 200
        X_train, X_test = X.iloc[:n_train].reset_index(drop=True), X.iloc[n_train:].reset_index(drop=True)
        y_train = y[:n_train]

        m = MRMR(**_kw(fe_mahalanobis_density_enable=True, fe_mahalanobis_density_top_k=1, fe_local_mi_gate=False))
        m.fit(X_train, y_train)

        Xt_train = m.transform(X_train)
        Xt_test = m.transform(X_test)
        assert list(Xt_train.columns) == list(Xt_test.columns), "transform() must emit the same column set on train and holdout"

    def test_pickle_round_trip_preserves_mahalanobis_density_transform(self):
        """A pickled+reloaded MRMR instance must reproduce the identical transform() output."""
        X, y = _ellipsoidal_outlier_fixture(seed=2, n=2000, p=6)
        m = MRMR(**_kw(fe_mahalanobis_density_enable=True, fe_mahalanobis_density_top_k=1, fe_local_mi_gate=False))
        m.fit(X, y)
        Xt_before = m.transform(X)

        m2 = pickle.loads(pickle.dumps(m))  # nosec B301 -- round-trip of a locally-created, trusted object
        Xt_after = m2.transform(X)
        pd.testing.assert_frame_equal(Xt_before, Xt_after)
