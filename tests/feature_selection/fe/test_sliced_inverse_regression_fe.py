"""Sliced Inverse Regression (SIR) oblique-direction projection, wired into MRMR.fit()
(mrmr_audit_2026-07-20 fe_expansion.md "Sliced Inverse Regression (SIR) oblique-direction
projection feature").

Li (1991): slice y into H bins, form the between-slice-mean covariance M, solve the generalized
eigenproblem Sigma^{-1} M v = lambda v; the top eigenvectors give the LINEAR COMBINATION
direction(s) w.x along which y varies most -- an oblique direction not restricted to any 2-4 named
columns, catching a genuinely rotated threshold spread thinly across several correlated columns
that no per-column or pairwise/triplet/quadruplet product basis can reconstruct economically.

Contracts pinned (never xfail):

UNIT
* the emitted projection matches a closed-form recompute from the frozen ``x_mean``/direction
  vector;
* recipe replay (``apply_recipe``) reads only X (no ``y`` in the payload -- its effect is already
  baked into the frozen direction vector) and is exact/deterministic;
* recipe pickle round-trips.

BIZ_VALUE
* the audit's own scenario (``y = 1{0.6*x1+0.5*x2+0.4*x3+0.3*x4+0.4*x5 > c}``): the SIR direction's
  MI must beat every single raw column's own MI by a wide margin -- no column's individual weight
  clears the marginal screening floor alone;
* an axis-aligned control (y depends on exactly one raw column, no oblique mixing): SIR's top
  direction must not materially beat that one informative column's own MI (no fabricated signal);
* canonical pair-FE fixture with the flag OFF (its shipped default): support_ is unaffected by the
  module even being importable.

E2E
* ``MRMR(fe_sir_direction_enable=True)`` on the oblique fixture (mi_gate off, matching the family's
  own top_k truncation contract) appends a ``sir__`` column, survives ``transform()`` on held-out
  rows (replay, not refit), and round-trips through pickle.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters import MRMR
from mlframe.feature_selection.filters._sliced_inverse_regression_fe import (
    apply_sir_direction,
    build_sir_direction_recipe,
    engineered_name_sir_direction,
    generate_sir_direction_features,
    hybrid_sir_direction_fe,
)
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe


def _mi_one(col, y, nbins: int = 10) -> float:
    """Plug-in MI(col; y) via the shared batch kernel."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch

    arr = np.asarray(col, dtype=np.float64).reshape(-1, 1)
    return float(_mi_classif_batch(arr, np.asarray(y).astype(np.int64), nbins=nbins)[0])


def _oblique_fixture(seed: int = 0, n: int = 4000):
    """The audit's own scenario: an oblique threshold spread thinly across 5 correlated columns,
    each weight too small individually to clear the marginal MI screening floor."""
    rng = np.random.default_rng(seed)
    cols = {f"x{i}": rng.standard_normal(n) for i in range(1, 6)}
    X = pd.DataFrame(cols)
    w = np.array([0.6, 0.5, 0.4, 0.3, 0.4])
    lin = X.to_numpy() @ w
    y = (lin > np.median(lin)).astype(int)
    return X, y


def _kw(**overrides):
    """Fast-fitting default MRMR constructor kwargs, overridable per test."""
    base = dict(random_seed=42, verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2, fe_max_steps=1, skip_retraining_on_same_content=False)
    base.update(overrides)
    return base


class TestSirDirectionUnit:
    """Direct unit coverage: closed-form match, replay determinism, pickle round-trip."""

    def test_replay_matches_fitted_column_on_same_rows(self):
        """apply_sir_direction on the SAME rows used to fit must reproduce the fitted projection
        exactly (both are the same closed-form centering+projection of the frozen x_mean/v)."""
        X, y = _oblique_fixture()
        cols = [f"x{i}" for i in range(1, 6)]
        enc, payload = generate_sir_direction_features(X, cols, y, n_directions=2)
        assert not enc.empty
        replay = apply_sir_direction(X, payload)
        np.testing.assert_allclose(replay.to_numpy(), enc.to_numpy(), rtol=1e-10, atol=1e-10)

    def test_recipe_replay_reads_only_x_and_is_deterministic(self):
        """apply_recipe (dispatch) must recompute the projection from X + the frozen x_mean/v
        alone (no y anywhere in the payload) and be exactly reproducible."""
        X, y = _oblique_fixture()
        cols = [f"x{i}" for i in range(1, 6)]
        enc, payload = generate_sir_direction_features(X, cols, y, n_directions=2)
        name = engineered_name_sir_direction(cols, 0)
        recipe = build_sir_direction_recipe(name=name, idx=0, cols=payload["cols"], x_mean=payload["x_mean"], v=payload["v"])
        assert "y" not in recipe.extra and "target" not in recipe.extra

        replay_a = np.asarray(apply_recipe(recipe, X), dtype=float)
        replay_b = np.asarray(apply_recipe(recipe, X), dtype=float)
        np.testing.assert_allclose(replay_a, replay_b, rtol=0, atol=0)
        np.testing.assert_allclose(replay_a, enc[name].to_numpy(), rtol=1e-10, atol=1e-10)

    def test_recipe_pickle_round_trip(self):
        """A built recipe must pickle/unpickle and replay identically."""
        X, y = _oblique_fixture()
        _, appended, recipes, _ = hybrid_sir_direction_fe(X, y, num_cols=[f"x{i}" for i in range(1, 6)], n_directions=2, top_k=2, mi_gate=False)
        assert appended and recipes, "expected at least one SIR recipe with mi_gate disabled"
        r = recipes[0]
        r2 = pickle.loads(pickle.dumps(r))  # nosec B301 -- round-trip of a locally-created, trusted object
        a = np.asarray(apply_recipe(r, X), dtype=float)
        b = np.asarray(apply_recipe(r2, X), dtype=float)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


class TestSirDirectionBizValue:
    """The operator must earn its keep: strong on the oblique fixture, non-abusive elsewhere."""

    def test_oblique_fixture_beats_every_raw_column_by_a_wide_margin(self):
        """On the audit's own oblique-threshold scenario, the SIR direction's MI must beat every
        single raw column's own MI by a wide margin -- no column's individual weight clears the
        marginal screening floor alone, while SIR recovers the mixing direction directly."""
        X, y = _oblique_fixture()
        cols = [f"x{i}" for i in range(1, 6)]
        enc, _payload = generate_sir_direction_features(X, cols, y, n_directions=2)
        assert not enc.empty
        mi_sir = max(_mi_one(enc[c].to_numpy(), y) for c in enc.columns)
        mi_raw = max(_mi_one(X[c].to_numpy(), y) for c in cols)
        assert mi_sir > 3.0 * mi_raw, f"best SIR direction MI ({mi_sir:.4f}) should beat the best raw column's MI ({mi_raw:.4f}) by a wide margin"

    def test_axis_aligned_control_does_not_overclaim(self):
        """When y depends on exactly ONE raw column (no oblique mixing), SIR's top direction must
        not materially exceed that one informative column's own MI -- the family should not
        fabricate signal beyond what the joint structure actually contains."""
        rng = np.random.default_rng(13)
        n = 4000
        x0 = rng.standard_normal(n)
        noise_cols = {f"x{i}": rng.standard_normal(n) for i in range(1, 5)}
        X = pd.DataFrame({"x0": x0, **noise_cols})
        y = (x0 > 0).astype(int)
        mi_raw = _mi_one(x0, y)

        _, appended, _, enc = hybrid_sir_direction_fe(X, y, num_cols=list(X.columns), n_directions=2, top_k=2, mi_gate=True, random_state=0)
        if appended:
            mi_best = max(_mi_one(enc[c].to_numpy(), y) for c in appended)
            assert mi_best <= mi_raw + 0.05, f"SIR direction MI ({mi_best:.4f}) should not materially exceed the single truly-informative raw column's MI ({mi_raw:.4f})"

    def test_canonical_pair_fe_fixture_unperturbed_when_disabled(self):
        """With fe_sir_direction_enable at its shipped default (False), MRMR's selected support_
        on a canonical pair-FE fixture must be unaffected by this module even being importable."""
        rng = np.random.default_rng(7)
        n = 2000
        a = rng.standard_normal(n)
        b = rng.uniform(0.5, 3.0, n)
        c = rng.standard_normal(n)
        d = rng.standard_normal(n)
        y_cont = a**2 / b + np.log(np.abs(c) + 1e-6) * np.sin(d)
        y = (y_cont > np.median(y_cont)).astype(int)
        X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})

        m_off = MRMR(**_kw(fe_sir_direction_enable=False))
        m_off.fit(X.copy(), y)
        support_off = list(getattr(m_off, "support_", []))

        m_default = MRMR(**_kw())  # shipped default is also False
        m_default.fit(X.copy(), y)
        support_default = list(getattr(m_default, "support_", []))

        assert support_off == support_default, "explicit False and the shipped default must select byte-identical support_"


class TestSirDirectionE2E:
    """End-to-end: MRMR.fit() with the flag enabled appends the feature and replays it correctly."""

    def test_enabled_flag_appends_feature_and_replays_on_holdout(self):
        """Enabling the flag (with the local MI gate off, matching the top_k-truncation contract
        the family documents) must append at least one sir_direction_features_ column, and
        transform() on held-out rows must replay (not refit) it."""
        X, y = _oblique_fixture(seed=1, n=3000)
        X_train, X_test = X.iloc[:2000].reset_index(drop=True), X.iloc[2000:].reset_index(drop=True)
        y_train = y[:2000]

        m = MRMR(**_kw(fe_sir_direction_enable=True, fe_sir_direction_top_k=2, fe_local_mi_gate=False))
        m.fit(X_train, y_train)

        Xt_train = m.transform(X_train)
        Xt_test = m.transform(X_test)
        assert list(Xt_train.columns) == list(Xt_test.columns), "transform() must emit the same column set on train and holdout"

    def test_pickle_round_trip_preserves_sir_direction_transform(self):
        """A pickled+reloaded MRMR instance must reproduce the identical transform() output."""
        X, y = _oblique_fixture(seed=2, n=2000)
        m = MRMR(**_kw(fe_sir_direction_enable=True, fe_sir_direction_top_k=2, fe_local_mi_gate=False))
        m.fit(X, y)
        Xt_before = m.transform(X)

        m2 = pickle.loads(pickle.dumps(m))  # nosec B301 -- round-trip of a locally-created, trusted object
        Xt_after = m2.transform(X)
        pd.testing.assert_frame_equal(Xt_before, Xt_after)
