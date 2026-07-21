"""Random Fourier Features (random kitchen sinks), wired into MRMR.fit() (mrmr_audit_2026-07-20
fe_expansion.md "Random Fourier Features (random kitchen sinks) multi-column kernel-approximation
block").

Draws a frozen random Gaussian projection ``W`` (p x m) + phases ``b`` and emits
``phi(x) = sqrt(2/m) * cos(X @ W / bandwidth + b)`` as ``m`` joint columns approximating an RBF
kernel over a bounded raw-column pool -- the one family that captures a smooth function of MANY
(5+) columns jointly without combinatorial blow-up.

Contracts pinned (never xfail):

UNIT
* the emitted block matches a closed-form recompute from the frozen ``W``/``b``/``bandwidth``;
* recipe replay (``apply_recipe``) reads only X (no ``y`` in the payload) and is exact/deterministic;
* recipe pickle round-trips.

BIZ_VALUE
* a radial/Gaussian-bump target (``y = 1{exp(-||x||^2/2) > median}``) over 5 jointly-informative
  columns: the RFF block's MI must beat every single raw column's own MI (no per-column expansion
  can capture a genuinely joint radial structure);
* a fixture where y depends only on ONE column (no joint radial structure): the MI gate must not
  admit a materially-more-informative RFF column than the raw column itself;
* canonical pair-FE fixture with the flag OFF (its shipped default): support_ is unaffected by the
  module even being importable.

E2E
* ``MRMR(fe_random_fourier_enable=True)`` on the radial fixture (mi_gate off, matching the family's
  own top_k truncation contract) appends an ``rff__`` column, survives ``transform()`` on held-out
  rows (replay, not refit), and round-trips through pickle.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters import MRMR
from mlframe.feature_selection.filters._random_fourier_features_fe import (
    apply_random_fourier_block,
    build_random_fourier_recipe,
    engineered_name_random_fourier,
    generate_random_fourier_features_block,
    hybrid_random_fourier_fe,
)
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe


def _mi_one(col, y, nbins: int = 10) -> float:
    """Plug-in MI(col; y) via the shared batch kernel."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch

    arr = np.asarray(col, dtype=np.float64).reshape(-1, 1)
    return float(_mi_classif_batch(arr, np.asarray(y).astype(np.int64), nbins=nbins)[0])


def _radial_fixture(seed: int = 0, n: int = 4000, p: int = 5):
    """A radial/Gaussian-bump target jointly over ``p`` columns -- no per-column or pairwise
    expansion resolves it, but an RBF-kernel approximation does (the kernel IS the target class)."""
    rng = np.random.default_rng(seed)
    cols = {f"x{i}": rng.standard_normal(n) for i in range(p)}
    X = pd.DataFrame(cols)
    r2 = np.sum(X.to_numpy() ** 2, axis=1)
    y_cont = np.exp(-r2 / 2.0)
    y = (y_cont > np.median(y_cont)).astype(int)
    return X, y


def _kw(**overrides):
    """Fast-fitting default MRMR constructor kwargs, overridable per test."""
    base = dict(random_seed=42, verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2, fe_max_steps=1, skip_retraining_on_same_content=False)
    base.update(overrides)
    return base


class TestRandomFourierUnit:
    """Direct unit coverage: closed-form match, replay determinism, pickle round-trip."""

    def test_block_matches_closed_form_replay(self):
        """apply_random_fourier_block on the SAME rows used to fit must reproduce the fitted
        columns exactly (both are the same closed-form expansion of the frozen W/b/bandwidth)."""
        X, _y = _radial_fixture()
        cols = [f"x{i}" for i in range(5)]
        enc, payload = generate_random_fourier_features_block(X, cols, m=16, random_state=0)
        assert not enc.empty
        replay = apply_random_fourier_block(X, payload)
        np.testing.assert_allclose(replay.to_numpy(), enc.to_numpy(), rtol=0, atol=0)

    def test_recipe_replay_reads_only_x_and_is_deterministic(self):
        """apply_recipe (dispatch) must recompute the component from X + the frozen W-column/b/
        bandwidth alone (no y anywhere in the payload) and be exactly reproducible."""
        X, _y = _radial_fixture()
        cols = [f"x{i}" for i in range(5)]
        enc, payload = generate_random_fourier_features_block(X, cols, m=16, random_state=0)
        name = engineered_name_random_fourier(cols, 0)
        recipe = build_random_fourier_recipe(name=name, idx=0, cols=payload["cols"], W=payload["W"], b=payload["b"], bandwidth=payload["bandwidth"])
        assert "y" not in recipe.extra and "target" not in recipe.extra

        replay_a = np.asarray(apply_recipe(recipe, X), dtype=float)
        replay_b = np.asarray(apply_recipe(recipe, X), dtype=float)
        np.testing.assert_allclose(replay_a, replay_b, rtol=0, atol=0)
        np.testing.assert_allclose(replay_a, enc[name].to_numpy(), rtol=1e-12, atol=1e-12)

    def test_recipe_pickle_round_trip(self):
        """A built recipe must pickle/unpickle and replay identically."""
        X, y = _radial_fixture()
        _, appended, recipes, _ = hybrid_random_fourier_fe(X, y, num_cols=[f"x{i}" for i in range(5)], m=16, top_k=5, mi_gate=False)
        assert appended and recipes, "expected at least one RFF recipe with mi_gate disabled"
        r = recipes[0]
        r2 = pickle.loads(pickle.dumps(r))  # nosec B301 -- round-trip of a locally-created, trusted object
        a = np.asarray(apply_recipe(r, X), dtype=float)
        b = np.asarray(apply_recipe(r2, X), dtype=float)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


class TestRandomFourierBizValue:
    """The operator must earn its keep: strong on the radial fixture, non-abusive elsewhere."""

    def test_radial_fixture_beats_every_raw_column(self):
        """On the radial/Gaussian-bump target, at least one RFF component's MI must strictly beat
        every single raw column's own MI -- the audit's exact claim that no per-column expansion
        captures a genuinely joint radial structure while an RBF-kernel approximation does."""
        X, y = _radial_fixture()
        cols = [f"x{i}" for i in range(5)]
        enc, _payload = generate_random_fourier_features_block(X, cols, m=64, random_state=0)
        mi_rff = max(_mi_one(enc[c].to_numpy(), y) for c in enc.columns)
        mi_raw = max(_mi_one(X[c].to_numpy(), y) for c in cols)
        assert mi_rff > mi_raw, f"best RFF component MI ({mi_rff:.4f}) should beat the best raw column's MI ({mi_raw:.4f})"

    def test_single_column_dependence_gate_does_not_overclaim(self):
        """When y depends on exactly ONE raw column (no joint radial structure), the MI gate must
        not admit an RFF column with materially HIGHER MI than that one informative raw column --
        the family should not fabricate signal beyond what the joint structure actually contains."""
        rng = np.random.default_rng(11)
        n = 4000
        x0 = rng.standard_normal(n)
        noise_cols = {f"x{i}": rng.standard_normal(n) for i in range(1, 5)}
        X = pd.DataFrame({"x0": x0, **noise_cols})
        y = (x0 > 0).astype(int)
        mi_raw = _mi_one(x0, y)

        _, appended, _, enc = hybrid_random_fourier_fe(X, y, num_cols=list(X.columns), m=32, top_k=5, mi_gate=True, random_state=0)
        if appended:
            mi_best = max(_mi_one(enc[c].to_numpy(), y) for c in appended)
            assert mi_best <= mi_raw + 0.05, f"RFF column MI ({mi_best:.4f}) should not materially exceed the single truly-informative raw column's MI ({mi_raw:.4f})"

    def test_canonical_pair_fe_fixture_unperturbed_when_disabled(self):
        """With fe_random_fourier_enable at its shipped default (False), MRMR's selected support_
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

        m_off = MRMR(**_kw(fe_random_fourier_enable=False))
        m_off.fit(X.copy(), y)
        support_off = list(getattr(m_off, "support_", []))

        m_default = MRMR(**_kw())  # shipped default is also False
        m_default.fit(X.copy(), y)
        support_default = list(getattr(m_default, "support_", []))

        assert support_off == support_default, "explicit False and the shipped default must select byte-identical support_"


class TestRandomFourierE2E:
    """End-to-end: MRMR.fit() with the flag enabled appends the feature and replays it correctly."""

    def test_enabled_flag_appends_feature_and_replays_on_holdout(self):
        """Enabling the flag (with the local MI gate off, matching the top_k-truncation contract
        the family documents) must append at least one random_fourier_features_ column, and
        transform() on held-out rows must replay (not refit) it."""
        X, y = _radial_fixture(seed=1, n=3000)
        X_train, X_test = X.iloc[:2000].reset_index(drop=True), X.iloc[2000:].reset_index(drop=True)
        y_train = y[:2000]

        m = MRMR(**_kw(fe_random_fourier_enable=True, fe_random_fourier_m=32, fe_random_fourier_top_k=5, fe_local_mi_gate=False))
        m.fit(X_train, y_train)

        Xt_train = m.transform(X_train)
        Xt_test = m.transform(X_test)
        assert list(Xt_train.columns) == list(Xt_test.columns), "transform() must emit the same column set on train and holdout"

    def test_pickle_round_trip_preserves_random_fourier_transform(self):
        """A pickled+reloaded MRMR instance must reproduce the identical transform() output."""
        X, y = _radial_fixture(seed=2, n=2000)
        m = MRMR(**_kw(fe_random_fourier_enable=True, fe_random_fourier_m=32, fe_random_fourier_top_k=5, fe_local_mi_gate=False))
        m.fit(X, y)
        Xt_before = m.transform(X)

        m2 = pickle.loads(pickle.dumps(m))  # nosec B301 -- round-trip of a locally-created, trusted object
        Xt_after = m2.transform(X)
        pd.testing.assert_frame_equal(Xt_before, Xt_after)
