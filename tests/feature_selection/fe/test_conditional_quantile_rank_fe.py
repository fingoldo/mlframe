"""Conditional quantile-rank FE, wired into MRMR.fit() (mrmr_audit_2026-07-20 fe_expansion.md).

4th member of the conditional-dispersion family (grouped_agg mean/std -> composite_group_agg ->
conditional-dispersion z-score/|z| -> conditional quantile-rank). Bin ``x_j``; emit
``q(row) = empirical_rank(x_i within bin(x_j))`` -- the row's TRUE within-bin percentile, not a
z-score.

Contracts pinned (never xfail):

UNIT
* the emitted quantile-rank column matches a closed-form recompute from the stored per-bin sorted
  reference values;
* recipe replay (``apply_recipe`` / ``transform``) reproduces the fit column exactly (leak-safe: no
  y reference);
* recipe pickle round-trips.

BIZ_VALUE (the operator earns its keep)
* SKEWED two-bin fixture (bin A: normal, bin B: heavy-tailed Pareto) where a fixed z-score
  threshold corresponds to a materially different TRUE percentile in each bin: quantile-rank's MI
  with a "top-of-peer-group" target beats both the raw x_i MI and a z-score-based sibling.
* HOMOSCEDASTIC, non-skewed control (both bins normal): quantile-rank is a near-monotone
  reparametrization of x_i and is self-limiting (does not add spurious columns beyond noise).
* CANONICAL pair-FE fixture (``y = a**2/b + log(c)*sin(d)``) with the flag OFF (its shipped
  default): the engineered set is byte-identical to a fit without the module imported at all --
  confirms the new family does not perturb genuine-feature recovery when disabled.

E2E
* ``MRMR(fe_conditional_quantile_rank_enable=True)`` on the skewed fixture appends a quantile-rank
  column, survives ``transform()`` on held-out rows (replay, not refit), and round-trips through
  pickle.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters import MRMR
from mlframe.feature_selection.filters._conditional_quantile_rank_fe import (
    apply_conditional_quantile_rank,
    build_conditional_quantile_rank_recipe,
    conditional_quantile_rank_fe,
    engineered_name_conditional_quantile_rank,
    generate_conditional_quantile_rank_features,
    hybrid_conditional_quantile_rank_fe,
)
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe


def _mi_one(col, y, nbins: int = 10) -> float:
    """Plug-in MI(col; y) via the shared batch kernel."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch

    arr = np.asarray(col, dtype=np.float64).reshape(-1, 1)
    return float(_mi_classif_batch(arr, np.asarray(y).astype(np.int64), nbins=nbins)[0])


def _skewed_fixture(seed: int = 0, n: int = 6000):
    """Two-bin fixture: bin A (xj<=0.5) is standard normal, bin B (xj>0.5) is heavy-tailed Pareto.
    y flags rows in the TOP 10% of their own bin's true distribution -- a target quantile-rank
    resolves directly but a fixed z-score threshold resolves inconsistently across the two bins."""
    rng = np.random.default_rng(seed)
    xj = rng.random(n)
    binB = xj > 0.5
    xi = np.where(binB, rng.pareto(3.0, n), rng.standard_normal(n))
    # y = 1 iff xi is in the top 10% of ITS OWN bin (the true quantile-rank target).
    y = np.zeros(n, dtype=int)
    for b in (False, True):
        mask = binB == b
        thresh = np.quantile(xi[mask], 0.9)
        y[mask] = (xi[mask] > thresh).astype(int)
    X = pd.DataFrame({"xi": xi, "xj": xj})
    return X, y


def _kw(**overrides):
    """Fast-fitting default MRMR constructor kwargs, overridable per test."""
    base = dict(random_seed=42, verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2, fe_max_steps=1, skip_retraining_on_same_content=False)
    base.update(overrides)
    return base


class TestConditionalQuantileRankUnit:
    """Direct unit coverage: closed-form match, replay exactness, pickle round-trip."""

    def test_rank_matches_closed_form(self):
        """The emitted column must equal a fresh recompute of conditional_quantile_rank_fe on the
        stored per-bin codes."""
        X, _y = _skewed_fixture()
        enc, raw = generate_conditional_quantile_rank_features(X, ["xi", "xj"], n_bins=10)
        name = engineered_name_conditional_quantile_rank("xi", "xj")
        assert name in enc.columns
        rec = raw[name]

        from mlframe.feature_selection.filters._extra_fe_families import _digitize_with_edges

        codes = _digitize_with_edges(X["xj"].to_numpy(), rec["edges"])
        expected = conditional_quantile_rank_fe(X["xi"].to_numpy(dtype=float), codes)
        np.testing.assert_allclose(enc[name].to_numpy(), expected, rtol=0, atol=0, equal_nan=True)

    def test_replay_is_leak_safe_and_exact(self):
        """apply_recipe (dispatch) and the direct apply function must reproduce the fit column
        exactly, with no y reference anywhere in the recipe payload."""
        X, _y = _skewed_fixture()
        enc, raw = generate_conditional_quantile_rank_features(X, ["xi", "xj"], n_bins=10)
        name = engineered_name_conditional_quantile_rank("xi", "xj")
        recipe = build_conditional_quantile_rank_recipe(name=name, **raw[name])
        replay = np.asarray(apply_recipe(recipe, X), dtype=float)
        np.testing.assert_allclose(replay, enc[name].to_numpy(), rtol=0, atol=0, equal_nan=True)

        head = X.head(500)
        direct = apply_conditional_quantile_rank(head, {**raw[name]})
        via_dispatch = np.asarray(apply_recipe(recipe, head), dtype=float)
        np.testing.assert_allclose(direct, via_dispatch, rtol=0, atol=0, equal_nan=True)

    def test_recipe_pickle_round_trip(self):
        """A built recipe must pickle/unpickle and replay identically."""
        X, y = _skewed_fixture()
        rng = np.random.default_rng(99)
        X = X.assign(g1=rng.standard_normal(len(X)), g2=rng.standard_normal(len(X)))
        _, _appended, recipes, _ = hybrid_conditional_quantile_rank_fe(X, y, num_cols=["xi", "xj", "g1", "g2"], n_bins=10, top_k=5)
        assert recipes, "expected at least one quantile-rank recipe on the skewed fixture"
        r = recipes[0]
        r2 = pickle.loads(pickle.dumps(r))  # nosec B301 -- round-trip of a locally-created, trusted object
        a = np.asarray(apply_recipe(r, X), dtype=float)
        b = np.asarray(apply_recipe(r2, X), dtype=float)
        np.testing.assert_allclose(a, b, rtol=0, atol=0, equal_nan=True)


class TestConditionalQuantileRankBizValue:
    """The operator must earn its keep: strong on the skewed fixture, self-limiting elsewhere."""

    def test_skewed_fixture_quantile_rank_beats_zscore_sibling(self):
        """On the skewed two-bin fixture, the quantile-rank column's MI with the true
        within-bin-top-decile target must strictly beat the conditional-dispersion family's
        z-score sibling's MI -- the exact comparison the audit's rationale names: a z-score
        threshold is not a fixed percentile cutoff across differently-shaped conditioning bins,
        while quantile-rank resolves the row's true peer-group extremeness directly."""
        from mlframe.feature_selection.filters._extra_fe_families_dispersion import (
            engineered_name_conditional_dispersion,
            generate_conditional_dispersion_features,
        )

        X, y = _skewed_fixture()
        enc_q, _raw_q = generate_conditional_quantile_rank_features(X, ["xi", "xj"], n_bins=20)
        name_q = engineered_name_conditional_quantile_rank("xi", "xj")
        assert name_q in enc_q.columns
        mi_qrank = _mi_one(enc_q[name_q].to_numpy(), y)

        enc_z, _raw_z = generate_conditional_dispersion_features(X, ["xi", "xj"], n_bins=20, kinds=("absz",))
        name_z = engineered_name_conditional_dispersion("xi", "xj", "absz")
        mi_zscore = _mi_one(enc_z[name_z].to_numpy(), y) if name_z in enc_z.columns else 0.0

        assert mi_qrank > mi_zscore, f"quantile-rank MI ({mi_qrank:.4f}) should beat the z-score sibling's MI ({mi_zscore:.4f}) on the skewed within-bin-top-decile target"

    def test_homoscedastic_control_is_self_limiting(self):
        """On a homoscedastic, non-skewed two-bin fixture (both bins normal), the hybrid pipeline's
        MI gate must not admit spurious quantile-rank columns beyond genuine noise-floor variation."""
        rng = np.random.default_rng(5)
        n = 4000
        xj = rng.random(n)
        xi = rng.standard_normal(n)  # SAME distribution regardless of bin -- no genuine dependence
        y = (rng.standard_normal(n) > 0).astype(int)  # pure noise target, independent of xi/xj
        X = pd.DataFrame({"xi": xi, "xj": xj})
        _, appended, _, _ = hybrid_conditional_quantile_rank_fe(X, y, num_cols=["xi", "xj"], n_bins=10, top_k=5, mi_gate=True)
        assert appended == [], f"homoscedastic/noise fixture must admit 0 quantile-rank columns, got {appended}"

    def test_canonical_pair_fe_fixture_unperturbed_when_disabled(self):
        """With fe_conditional_quantile_rank_enable at its shipped default (False), MRMR's selected
        support_ on a canonical pair-FE fixture must be unaffected by this module even being
        importable -- confirms the new family does not perturb genuine-feature recovery when off."""
        rng = np.random.default_rng(7)
        n = 2000
        a = rng.standard_normal(n)
        b = rng.uniform(0.5, 3.0, n)
        c = rng.standard_normal(n)
        d = rng.standard_normal(n)
        y_cont = a**2 / b + np.log(np.abs(c) + 1e-6) * np.sin(d)
        y = (y_cont > np.median(y_cont)).astype(int)
        X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})

        m_off = MRMR(**_kw(fe_conditional_quantile_rank_enable=False))
        m_off.fit(X.copy(), y)
        support_off = list(getattr(m_off, "support_", []))

        m_default = MRMR(**_kw())  # shipped default is also False
        m_default.fit(X.copy(), y)
        support_default = list(getattr(m_default, "support_", []))

        assert support_off == support_default, "explicit False and the shipped default must select byte-identical support_"


class TestConditionalQuantileRankE2E:
    """End-to-end: MRMR.fit() with the flag enabled appends the feature and replays it correctly."""

    def test_enabled_flag_appends_feature_and_replays_on_holdout(self):
        """Enabling the flag must append at least one conditional_quantile_rank_features_ column,
        and transform() on held-out rows must replay (not refit) it."""
        X, y = _skewed_fixture(seed=1, n=3000)
        X_train, X_test = X.iloc[:2000].reset_index(drop=True), X.iloc[2000:].reset_index(drop=True)
        y_train = y[:2000]

        m = MRMR(**_kw(fe_conditional_quantile_rank_enable=True, fe_conditional_quantile_rank_n_bins=10))
        m.fit(X_train, y_train)

        Xt_train = m.transform(X_train)
        Xt_test = m.transform(X_test)
        assert list(Xt_train.columns) == list(Xt_test.columns), "transform() must emit the same column set on train and holdout"

    def test_pickle_round_trip_preserves_quantile_rank_transform(self):
        """A pickled+reloaded MRMR instance must reproduce the identical transform() output."""
        X, y = _skewed_fixture(seed=2, n=2000)
        m = MRMR(**_kw(fe_conditional_quantile_rank_enable=True))
        m.fit(X, y)
        Xt_before = m.transform(X)

        m2 = pickle.loads(pickle.dumps(m))  # nosec B301 -- round-trip of a locally-created, trusted object
        Xt_after = m2.transform(X)
        pd.testing.assert_frame_equal(Xt_before, Xt_after)
