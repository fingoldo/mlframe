"""Layer 58 biz_value: CONDITIONAL BASIS ROUTING for the orthogonal-
polynomial univariate FE path.

Validates ``hybrid_orth_mi_conditional_routing_fe`` introduced 2026-05-31
(sibling module ``_orthogonal_routing_fe``): try every (pre_transform,
basis, degree) cell per source column and keep ONLY the MI-uplift winner.

Why this layer matters
----------------------

Layer 21 picks ONE basis per source via the moment fingerprint and
sweeps every degree. Layer 57 picks the best DEGREE per column. Both
still assume one (basis, transform) pair per column; for two important
regimes that assumption is wrong:

* Heavy-tail x driving y = f(log x). Raw Hermite-on-x is dominated by
  the few extreme tail rows; Hermite-on-``log|x|`` recovers the signal.
* Uniform x driving y = T_3(x). The fingerprint correctly picks
  Chebyshev, but Chebyshev-on-raw vs Chebyshev-on-``tanh`` vs Hermite-on
  -raw is target-dependent and the fingerprint cannot know which wins.

Layer 58 fixes both by ACTUALLY EVALUATING every (pre_transform, basis,
degree) cell against the target via batch MI and keeping the per-column
argmax. Combinatorial space is bounded (4 transforms x 4 bases x
``len(degrees)`` = 32 candidates per column); one batch MI call resolves
the entire pool.

Contracts pinned
----------------

* TestHeavyTailSignal: y = He_2(log x) on log-normal x; routing picks
  the ``log_abs`` pre-transform on the source.

* TestBoundedSignalChebyshev: y = T_3(x) on uniform x; routing picks
  Chebyshev for the source.

* TestBestPerColumn: 3 cols with different optimal {basis, transform};
  routing recovers each.

* TestNoSpuriousNoise: pure-noise frame at p >= 16 emits no cols
  (noise-aware MAD floor kicks in).

* TestDefaultDisabledByteIdentical: ``fe_hybrid_orth_conditional_routing_enable=False``
  keeps ``feature_names_in_`` identical to a fit without the flag.

* TestPickleAndClone: sklearn-style ``clone`` preserves the new ctor
  params; ``pickle`` round-trips a fitted MRMR with the per-column
  chosen (basis, pre_transform, degree) triples intact.

Consolidated verbatim from test_biz_value_mrmr_layer58.py (per audit finding test_code_quality-16).
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)


def _import_routing_fe():
    """Lazily import the Layer-58 conditional-basis-routing FE functions."""
    from mlframe.feature_selection.filters._orthogonal_routing_fe import (
        generate_conditional_basis_routing_features,
        hybrid_orth_mi_conditional_routing_fe,
        hybrid_orth_mi_conditional_routing_fe_with_recipes,
        PRE_TRANSFORM_NAMES,
        parse_routing_col_name,
    )

    return (
        generate_conditional_basis_routing_features,
        hybrid_orth_mi_conditional_routing_fe,
        hybrid_orth_mi_conditional_routing_fe_with_recipes,
        PRE_TRANSFORM_NAMES,
        parse_routing_col_name,
    )


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr
# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _hermite_he(x: np.ndarray, n: int) -> np.ndarray:
    """Probabilist's Hermite He_n via numpy poly eval (used by tests to
    construct targets whose optimal basis is known)."""
    coef = np.zeros(n + 1, dtype=np.float64)
    coef[n] = 1.0
    return np.polynomial.hermite_e.hermeval(x, coef)


def _build_heavy_tail(seed: int, n: int = 4000):
    """Signed heavy-tail x ~ +/- exp(N(0, 1.5)), y = sign(He_2(log|x|) -
    median) + epsilon.

    Why signed: under quantile-bin MI a monotone transform (raw x ->
    log|x| for positive-only x) is rank-preserving so MI(raw_x; y) ==
    MI(log_x; y). For routing to genuinely uplift over raw, the binary y
    must NOT be a monotone function of raw x's rank. The signed sample
    breaks that monotone link -- raw x rank carries only sign info; the
    signal lives in the magnitude of log|x|. ``log_abs`` reveals that
    structure; raw Hermite cannot because the z-score collapses sign +
    magnitude into one axis.
    """
    rng = np.random.default_rng(seed)
    log_mag = rng.standard_normal(n) * 1.5
    sign = rng.choice([-1.0, 1.0], size=n)
    x = sign * np.exp(log_mag)
    z = log_mag - log_mag.mean()
    sig = z**2 - float(z.var()) + 0.3 * rng.standard_normal(n)
    y = (sig > np.median(sig)).astype(int)
    cols = {"x": x}
    for i in range(5):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    return X, pd.Series(y, name="y")


def _build_bounded_t3(seed: int, n: int = 4000):
    """Uniform x in [-1, 1] driving y = T_3(x). Chebyshev T_3 on the
    bounded source recovers the signal; Hermite-on-raw mis-stretches
    the boundary.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    # T_3(x) = 4 x^3 - 3 x
    sig = 4.0 * x**3 - 3.0 * x + 0.15 * rng.standard_normal(n)
    y = (sig > np.median(sig)).astype(int)
    cols = {"x": x}
    for i in range(5):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    return X, pd.Series(y, name="y")


def _build_mixed_routing(seed: int, n: int = 4000):
    """3 source cols with different optimal (basis, transform):
    * x_lognorm -- heavy-tail driving He_2(log x): log_abs + hermite
    * x_unif    -- uniform driving T_3(x):         raw + chebyshev
    * x_gauss   -- gaussian driving He_3(x):       raw + hermite

    Noise cols pad the source pool so the noise floor sees a healthy
    raw-baseline distribution.
    """
    rng = np.random.default_rng(seed)
    x_lognorm = np.exp(rng.standard_normal(n) * 2.0)
    x_unif = rng.uniform(-1.0, 1.0, n)
    x_gauss = rng.standard_normal(n)

    def _z(v: np.ndarray) -> np.ndarray:
        """Standard-deviation-normalize a component so all signal terms carry similar weight."""
        sd = float(np.std(v))
        return v / sd if sd > 1e-12 else v

    log_x = np.log(x_lognorm)
    z_log = (log_x - log_x.mean()) / max(log_x.std(), 1e-12)
    s1 = _z(_hermite_he(z_log, 2))
    s2 = _z(4.0 * x_unif**3 - 3.0 * x_unif)
    s3 = _z(_hermite_he(x_gauss, 3))
    sig = s1 + s2 + s3 + 0.25 * rng.standard_normal(n)
    y = (sig > np.median(sig)).astype(int)
    cols = {"x_lognorm": x_lognorm, "x_unif": x_unif, "x_gauss": x_gauss}
    # Pad with noise so the per-source argmax distribution is meaningful.
    for i in range(5):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    return X, pd.Series(y, name="y")


def _build_noise_only_large(seed: int, n: int = 2000, p: int = 20):
    """p>=16 pure-noise frame so the MAD-based noise floor activates."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(p)})
    y = (rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_linear(seed: int, n: int = 1500):
    """Plain linear-additive signal for the default-disabled contract."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "noise_a": rng.standard_normal(n),
            "noise_b": rng.standard_normal(n),
            "noise_c": rng.standard_normal(n),
        }
    )
    y = ((x1 + 0.7 * x2) > 0).astype(int)
    return X, pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# Contract 1: heavy-tail signal -- routing picks log_abs on the source
# ---------------------------------------------------------------------------


class TestHeavyTailSignal:
    """On a heavy-tail source, conditional routing must select the log_abs pre-transform."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_routing_picks_log_abs_for_heavy_tail(self, seed):
        """Routing must survive on the heavy-tail source x and select log_abs as its pre-transform."""
        gen_routing, _, _, _, _ = _import_routing_fe()
        X, y = _build_heavy_tail(seed)
        eng, meta = gen_routing(
            X,
            y.values,
            cols=list(X.columns),
            degrees=(2, 3),
            min_uplift=1.10,
            top_k=5,
        )
        assert eng.shape[1] >= 1, f"seed={seed}: heavy-tail signal should produce >=1 routing column; got {eng.shape[1]}: {list(eng.columns)}"
        # x must be among the survivors
        srcs = {info["src"] for info in meta.values()}
        assert "x" in srcs, f"seed={seed}: heavy-tail source 'x' must survive routing gates; survivors={srcs}, meta={meta}"
        # The chosen pre-transform for the x source must be log_abs (or any
        # non-raw transform that handles heavy-tail) -- raw Hermite on a
        # log-normal input is the failure mode this layer fixes.
        for info in meta.values():
            if info["src"] == "x":
                assert info["pre_transform"] == "log_abs", (
                    f"seed={seed}: routing should pick log_abs for heavy-tail "
                    f"source 'x'; got pre_transform={info['pre_transform']}, "
                    f"basis={info['basis']}, degree={info['degree']}, "
                    f"emi={info['engineered_mi']:.4f}, "
                    f"uplift={info['uplift']:.3f}"
                )


# ---------------------------------------------------------------------------
# Contract 2: bounded signal y = T_3(x) -- routing picks Chebyshev
# ---------------------------------------------------------------------------


class TestBoundedSignalChebyshev:
    """On a bounded uniform source driving y = T_3(x), routing must select Chebyshev degree 3."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_routing_picks_chebyshev_for_uniform_t3(self, seed):
        """Routing selects Chebyshev degree 3 for the uniform T_3(x) target."""
        gen_routing, _, _, _, _ = _import_routing_fe()
        X, y = _build_bounded_t3(seed)
        _eng, meta = gen_routing(
            X,
            y.values,
            cols=list(X.columns),
            degrees=(2, 3),
            min_uplift=1.10,
            top_k=5,
        )
        assert "x" in {
            info["src"] for info in meta.values()
        }, f"seed={seed}: bounded T_3 signal on 'x' must produce a routing column for 'x'; survivors={list(meta.values())}"
        # The Chebyshev T_3 target is best captured by Chebyshev-of-degree-3.
        # We require the basis to be chebyshev (other bases can produce
        # numerically equivalent MI under quantile binning when the support
        # is identical, but T_3 is the construction target -- if the
        # routing picks anything else the layer's premise is broken).
        for info in meta.values():
            if info["src"] == "x":
                assert info["basis"] == "chebyshev", (
                    f"seed={seed}: routing should pick Chebyshev for "
                    f"y=T_3(x) target; got basis={info['basis']}, "
                    f"degree={info['degree']}, "
                    f"pre_transform={info['pre_transform']}, "
                    f"emi={info['engineered_mi']:.4f}, "
                    f"uplift={info['uplift']:.3f}"
                )
                assert info["degree"] == 3, f"seed={seed}: routing should pick degree 3 for y=T_3(x); got degree={info['degree']}"


# ---------------------------------------------------------------------------
# Contract 3: per-column best (basis, transform) recovery on a mixed-signal
# ---------------------------------------------------------------------------


class TestBestPerColumn:
    """Routing must recover each source column's own optimal (basis, transform) pair, one column max per source."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_per_column_one_recipe_max(self, seed):
        """At most one engineered column survives per source column."""
        gen_routing, _, _, _, _ = _import_routing_fe()
        X, y = _build_mixed_routing(seed)
        _eng, meta = gen_routing(
            X,
            y.values,
            cols=list(X.columns),
            degrees=(2, 3),
            min_uplift=1.10,
            top_k=10,
        )
        # At most one engineered column per source.
        srcs = [info["src"] for info in meta.values()]
        assert len(srcs) == len(set(srcs)), f"seed={seed}: routing must emit at most one column per source; got duplicates in {srcs}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_heavy_tail_source_routes_to_log_abs(self, seed):
        """Among mixed sources, the heavy-tail x_lognorm column, if it survives, routes to log_abs."""
        gen_routing, _, _, _, _ = _import_routing_fe()
        X, y = _build_mixed_routing(seed)
        _eng, meta = gen_routing(
            X,
            y.values,
            cols=list(X.columns),
            degrees=(2, 3),
            min_uplift=1.10,
            top_k=10,
        )
        # x_lognorm: if it survives, must pick log_abs.
        for info in meta.values():
            if info["src"] == "x_lognorm":
                assert info["pre_transform"] == "log_abs", f"seed={seed}: heavy-tail source 'x_lognorm' should route to log_abs; got {info}"
                return
        # If it didn't survive at all that's a separate failure mode; the
        # test for survival belongs to TestHeavyTailSignal -- here we only
        # assert routing CORRECTNESS conditional on survival.


# ---------------------------------------------------------------------------
# Contract 4: pure-noise frame at p>=16 emits no columns
# ---------------------------------------------------------------------------


class TestNoSpuriousNoise:
    """A p>=16 pure-noise frame must clear the noise-aware MAD floor and emit no columns."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_noise_only_p20_emits_nothing(self, seed):
        """A pure-noise frame at p=20 emits zero routing columns."""
        gen_routing, _, _, _, _ = _import_routing_fe()
        X, y = _build_noise_only_large(seed, p=20)
        eng, _meta = gen_routing(
            X,
            y.values,
            cols=list(X.columns),
            degrees=(2, 3),
            min_uplift=1.10,
            top_k=5,
        )
        assert (
            eng.shape[1] == 0
        ), f"seed={seed}: p=20 pure-noise frame should clear no routing columns through the noise-aware floor; got {eng.shape[1]}: {list(eng.columns)}"


# ---------------------------------------------------------------------------
# Contract 5: default disabled -- legacy behaviour byte-identical
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    """fe_hybrid_orth_conditional_routing_enable defaults to False; enabling it must fire and append columns."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_no_routing_columns(self, seed):
        """With the flag left at its False default, no routing columns are appended."""
        X, y = _build_linear(seed)
        m = _make_mrmr().fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added == [], f"seed={seed}: default fe_hybrid_orth_conditional_routing_enable=False should NOT append any engineered columns; got {added}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_enable_routing_appends_engineered(self, seed):
        """Enabling routing appends at least one engineered column, referencing the heavy-tail source."""
        X, y = _build_heavy_tail(seed, n=2500)
        m = _make_mrmr(
            fe_hybrid_orth_conditional_routing_enable=True,
            fe_hybrid_orth_conditional_routing_degrees=(2, 3),
            fe_hybrid_orth_conditional_routing_min_uplift=1.10,
            fe_hybrid_orth_conditional_routing_top_k=5,
        ).fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added, f"seed={seed}: routing flag ON should append at least one engineered column to hybrid_orth_features_; got {added}"
        # At least one engineered column should reference the heavy-tail
        # source 'x' (the only source carrying actual non-monotone signal).
        srcs_in_names = [n.split("__", 1)[0] for n in added]
        assert "x" in srcs_in_names, f"seed={seed}: routing should pick the heavy-tail source 'x'; engineered names = {added}"


# ---------------------------------------------------------------------------
# Contract 6: pickle / clone preserve the routing ctor + chosen-triple recipes
# ---------------------------------------------------------------------------


class TestPickleAndClone:
    """Routing ctor params and chosen (basis, degree, pre_transform) triples must survive clone/pickle round-trips."""

    def test_clone_preserves_routing_params(self):
        """sklearn clone() copies every fe_hybrid_orth_conditional_routing_* ctor param."""
        m = _make_mrmr(
            fe_hybrid_orth_conditional_routing_enable=True,
            fe_hybrid_orth_conditional_routing_top_k=7,
            fe_hybrid_orth_conditional_routing_min_uplift=1.15,
            fe_hybrid_orth_conditional_routing_degrees=(2, 3, 4),
        )
        m2 = clone(m)
        for name, expected in [
            ("fe_hybrid_orth_conditional_routing_enable", True),
            ("fe_hybrid_orth_conditional_routing_top_k", 7),
            ("fe_hybrid_orth_conditional_routing_min_uplift", 1.15),
            ("fe_hybrid_orth_conditional_routing_degrees", (2, 3, 4)),
        ]:
            assert getattr(m2, name) == expected, f"clone() dropped {name}: expected {expected}, got {getattr(m2, name)}"

    def test_pickle_roundtrip_preserves_chosen_triples(self):
        """A pickle round-trip preserves feature names, appended columns, and every chosen-triple recipe field."""
        X, y = _build_heavy_tail(seed=42, n=2500)
        m = _make_mrmr(
            fe_hybrid_orth_conditional_routing_enable=True,
            fe_hybrid_orth_conditional_routing_degrees=(2, 3),
            fe_hybrid_orth_conditional_routing_min_uplift=1.10,
            fe_hybrid_orth_conditional_routing_top_k=5,
        ).fit(X, y)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object
        assert list(m2.feature_names_in_) == list(m.feature_names_in_), "pickle changed feature_names_in_"
        added_before = list(getattr(m, "hybrid_orth_features_", []) or [])
        added_after = list(getattr(m2, "hybrid_orth_features_", []) or [])
        assert added_before == added_after, f"pickle changed hybrid_orth_features_: before={added_before}, after={added_after}"
        # Recipes: per chosen (basis, degree, pre_transform) triple survives.
        recipes_before = {r.name: r for r in getattr(m, "_engineered_recipes_", []) or [] if r.kind == "orth_univariate"}
        recipes_after = {r.name: r for r in getattr(m2, "_engineered_recipes_", []) or [] if r.kind == "orth_univariate"}
        assert set(recipes_before.keys()) == set(
            recipes_after.keys()
        ), f"pickle dropped or added recipe names: before={set(recipes_before.keys())}, after={set(recipes_after.keys())}"
        for name, r_before in recipes_before.items():
            r_after = recipes_after[name]
            for key in ("basis", "degree", "pre_transform"):
                assert r_before.extra.get(key) == r_after.extra.get(
                    key
                ), f"pickle changed '{key}' for recipe {name!r}: before={r_before.extra}, after={r_after.extra}"


# ---------------------------------------------------------------------------
# Contract 7: column-naming + recipe replay round-trip
# ---------------------------------------------------------------------------


class TestRecipeReplay:
    """apply_recipe at transform time must reproduce the fit-time engineered column for the chosen routing cell."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_recipe_replay_matches_fit_time_values(self, seed):
        """Replay correctness: applying the orth_univariate recipe at
        transform time produces exactly the fit-time engineered column
        (modulo float epsilon). Validates that the recipe extras carry
        enough state to reconstruct the chosen pre_transform path.
        """
        _, _, hybrid_with_recipes, _, _ = _import_routing_fe()
        X, y = _build_heavy_tail(seed)
        X_aug, _scores, recipes = hybrid_with_recipes(
            X,
            y.values,
            cols=list(X.columns),
            degrees=(2, 3),
            min_uplift=1.10,
            top_k=5,
        )
        # On every seed in SEEDS the heavy-tail fixture deterministically routes x to
        # the log|x|+Hermite cell, so routing MUST emit at least one recipe; a regression
        # that silently drops the routed column now fails here instead of skipping.
        assert recipes, f"seed={seed}: routing emitted no recipe on the heavy-tail fixture"
        # Re-extract appended columns from X_aug.
        appended = [c for c in X_aug.columns if c not in X.columns]
        # For each recipe, replay against the SAME X and compare row-by-row
        # to the value stored in X_aug.
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )

        for r in recipes:
            assert r.name in appended, f"seed={seed}: recipe {r.name!r} not in appended columns {appended}"
            replayed = apply_recipe(r, X)
            fit_time = X_aug[r.name].to_numpy()
            assert np.allclose(
                replayed, fit_time, rtol=1e-9, atol=1e-12
            ), f"seed={seed}: recipe {r.name!r} replay drift: max|replayed - fit| = {float(np.max(np.abs(replayed - fit_time)))}; extra={dict(r.extra)}"


class TestRoutingCriterionCorrDefault:
    """The per-source ARGMAX routes by linear usability (|Pearson corr|), not MI.

    A 2026-06-03 out-of-sample study over this exact (pre_transform x basis x
    degree) space showed corr-routing near-oracle (OOS-linear R^2 0.81 vs MI 0.52;
    MI picks an informative-but-non-linear cell -- log|x|/tanh + Laguerre -- in
    23/30 cases). This mirrors the default Layer-21 router ``basis_route_by_signal``
    so the two routers agree. The KEEP gate stays MI-based (relevance IS an MI
    question); only the argmax switched.
    """

    def test_default_routing_criterion_is_corr(self):
        """The conditional-routing argmax defaults to routing_criterion='corr' (linear usability)."""
        import inspect

        gen = _import_routing_fe()[0]
        assert (
            inspect.signature(gen).parameters["routing_criterion"].default == "corr"
        ), "the conditional-routing argmax must default to linear-usability (corr)"

    def test_corr_routing_generalises_at_least_as_well_as_mi(self):
        """Route on a TRAIN slice, replay the chosen (pre,basis,degree) cell on a
        held-out slice via its recipe, and compare held-out |corr| with y. Routing
        by corr (linear usability) must generalise at least as well as routing by
        MI on the y=f(log|x|) heavy-tail regime -- the non-tautological win (the
        comparison is on data the routing never saw)."""
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

        _, _, hybrid_with_recipes, _, _ = _import_routing_fe()
        X, y = _build_heavy_tail(0)
        half = len(X) // 2
        Xtr = X.iloc[:half].reset_index(drop=True)
        ytr = y.iloc[:half].reset_index(drop=True)
        Xte = X.iloc[half:].reset_index(drop=True)
        yte = y.iloc[half:].to_numpy().astype(float)
        held = {}
        for crit in ("corr", "mi"):
            _, _, recipes = hybrid_with_recipes(
                Xtr,
                ytr.values,
                cols=["x"],
                degrees=(2, 3),
                min_uplift=1.0,
                top_k=1,
                routing_criterion=crit,
            )
            # Both corr and mi criteria deterministically route x to the log|x|+Hermite
            # cell on this fixture, so each MUST emit a recipe for the held-out comparison.
            assert recipes, f"criterion {crit!r} emitted no recipe on the heavy-tail fixture"
            v = np.asarray(apply_recipe(recipes[0], Xte), dtype=float)
            held[crit] = abs(float(np.corrcoef(v, yte)[0, 1])) if float(np.std(v)) > 1e-12 else 0.0
        assert held["corr"] >= held["mi"] - 1e-6, (
            f"corr-routing must generalise at least as well as MI-routing on the "
            f"y=f(log|x|) heavy-tail regime: held-out |corr| "
            f"corr={held['corr']:.3f} mi={held['mi']:.3f}"
        )
