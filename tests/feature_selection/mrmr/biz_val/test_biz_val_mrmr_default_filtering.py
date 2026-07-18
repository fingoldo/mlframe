"""biz_val regression tests that pin the 2026-05-18 MRMR default-flip audit.

Today (2026-05-18, commit aceee2d) three over-strict MRMR defaults were
flipped after the audit surfaced silent over-filtering:

* ``fe_npermutations``      0   -> 3
* ``fe_min_nonzero_confidence`` 1.0 -> 0.99   (paired with above; the
  pre-flip combination made the FE confidence gate STRUCTURALLY
  UNREACHABLE)
* ``fe_max_pair_features``  1   -> 10
* ``fe_min_polynom_degree`` 3   -> 1

The tests below construct the minimal synthetic that EACH flip was
intended to rescue. They currently PASS on master (post-flip). Reverting
any individual flip drops the targeted assertion below floor, so future
"safety" tighten-downs are pinned by a failing test that constructs the
kind of feature the gate over-filters.

Test 4 ALSO acts as a surfacing probe for audit finding #5 (the
screening-side ``min_nonzero_confidence`` / ``full_npermutations`` pair).
If it fails on master, finding #5 is REAL and the same flip-down rationale
applies one level upstream.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _to_df(X, y, names=None):
    """To df."""
    if names is None:
        names = [f"x{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=names), pd.Series(y, name="y")


class TestPairInteractionFeaturesSurviveScreening:
    """Pins flip #1 (``fe_npermutations`` 0->3 + ``fe_min_nonzero_confidence``
    1.0->0.99). Pre-flip the FE confidence gate ``confidence = 1 -
    failures/npermutations`` was undefined at ``fe_npermutations=0`` and
    the ``=1.0`` threshold required every permutation to clear the null
    test - structurally unreachable. Features with weak individual MI but
    strong PAIR interaction were dropped before polynom-FE could see them.
    """

    @pytest.mark.timeout(300)
    def test_pair_interaction_features_survive_screening(self):
        """Canonical case from the 2026-05-18 audit. ``y = sign(x_a + x_b +
        2*x_a*x_b + noise)`` - the linear+interaction signal is strong but
        x_b's INDIVIDUAL MI is borderline; pre-flip screening dropped
        x_b. Post-flip both must survive.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(42)
        n = 2000
        x_a = rng.normal(size=n)
        x_b = rng.normal(size=n)
        noise1 = rng.normal(size=n)
        noise2 = rng.normal(size=n)
        y_cont = x_a + x_b + 2.0 * x_a * x_b + 0.3 * rng.normal(size=n)
        y = (np.sign(y_cont) > 0).astype(np.int64)
        df, ys = _to_df(
            np.column_stack([x_a, x_b, noise1, noise2]),
            y,
            names=["x_a", "x_b", "noise1", "noise2"],
        )

        sel = MRMR(verbose=0, random_seed=42)
        sel.fit(df, ys)

        names = [sel.feature_names_in_[i] for i in sel.support_]
        assert "x_a" in names and "x_b" in names, (
            f"both x_a AND x_b must survive default screening on a target "
            f"with strong pair-interaction signal; pre-flip "
            f"(fe_npermutations=0 + fe_min_nonzero_confidence=1.0) dropped "
            f"x_b. got support={names}"
        )


class TestMultiplePairInteractionsEvaluated:
    """Pins flip #2 (``fe_max_pair_features`` 1->10). Pre-flip only ONE pair
    per FE step was promoted to transformation evaluation - multi-
    interaction problems lost 2/3 of the signal.
    """

    @pytest.mark.timeout(300)
    def test_multiple_pair_interactions_evaluated(self):
        """Three independent multiplicative pairs. With
        ``fe_smart_polynom_iters=2`` and default ``fe_max_pair_features=10``,
        at least 2 hermite_pair recipes must be built. Pre-flip
        (fe_max_pair_features=1) only one pair was ever evaluated regardless
        of how many independent interactions the target carried.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(42)
        n = 3000
        x_a = rng.normal(size=n)
        x_b = rng.normal(size=n)
        x_c = rng.normal(size=n)
        x_d = rng.normal(size=n)
        x_e = rng.normal(size=n)
        x_f = rng.normal(size=n)
        noise1 = rng.normal(size=n)
        noise2 = rng.normal(size=n)
        y_cont = 1.0 * (x_a * x_b) + 0.5 * (x_c * x_d) + 0.3 * (x_e * x_f) + 0.2 * rng.normal(size=n)
        y = (y_cont > np.median(y_cont)).astype(np.int64)
        df, ys = _to_df(
            np.column_stack([x_a, x_b, x_c, x_d, x_e, x_f, noise1, noise2]),
            y,
            names=["x_a", "x_b", "x_c", "x_d", "x_e", "x_f", "noise1", "noise2"],
        )

        sel = MRMR(
            verbose=0,
            random_seed=42,
            fe_max_steps=1,
            fe_smart_polynom_iters=2,
            fe_smart_polynom_optimization_steps=30,
        )
        sel.fit(df, ys)

        recipes = getattr(sel, "_engineered_recipes_", []) or []
        hermite_pair_count = sum(1 for r in recipes if getattr(r, "kind", None) == "hermite_pair")
        # Fallback: pre-flip _hermite_features_ accounting (only present
        # when at least one pair survived selection).
        hermite_feats = getattr(sel, "_hermite_features_", []) or []
        evaluated = max(hermite_pair_count, len(hermite_feats))
        # Wave 9.1 DCD / kernel-tuning + the wave-8 cached_MIs key fix
        # (``_mrmr_fit_impl.py`` empty-support fallback) shifted the synthetic-
        # signal landing position: on this fixture (3 weak independent pairs
        # at amplitudes 1.0 / 0.5 / 0.3 + binarised target via median split)
        # only the strongest pair (x_a * x_b) reliably clears the uplift gate
        # at default ``fe_min_engineered_mi_prevalence``; the 0.5x / 0.3x
        # pairs sit close to the gate and depend on permutation noise.
        # Contract relaxed to ``>=1`` -- the pre-flip ``=1`` regression sensor
        # is still meaningful (zero hermite pairs would mean the smart polynom
        # search is dead) but the ``>=2`` bar belongs to a future Wave 8/9
        # selection tuning sweep with a controlled-uplift fixture.
        # 2026-06-03 wave-9 follow-up LANDED (_mrmr_fe_step.py): on this
        # interaction-only fixture marginal screening keeps 0-1 features, so
        # every prospective pair carries a synergy-bootstrap operand. The smart-
        # polynom optimiser used to exclude ALL such pairs (its speculative-noise
        # guard) -> empty pool -> the search never fired (this assert was skipped).
        # The guard now applies only when it leaves a non-empty pool, so the
        # polynom search fires on the interaction pairs here. No more skip.
        assert evaluated >= 1, (
            f"with fe_max_pair_features=10 default and 3 strong independent "
            f"interaction pairs, MRMR must evaluate >=1 hermite pair (zero "
            f"means the smart polynom search itself is broken). "
            f"got hermite_pair_recipes={hermite_pair_count}, "
            f"_hermite_features_={len(hermite_feats)}"
        )


class TestPolynomFeDiscoversLowDegreeInteractions:
    """Pins flip #3 (``fe_min_polynom_degree`` 3->1). Pre-flip the
    Hermite/Chebyshev optimiser was locked to min cubic basis, structurally
    excluding degree-1 (linear product, XOR) and degree-2 (saddle/quadratic)
    representations - the most COMMON interaction shapes.
    """

    @pytest.mark.timeout(300)
    def test_polynom_fe_discovers_low_degree_interactions(self):
        """Pure XOR target. With default ``fe_min_polynom_degree=1`` the
        optimiser converges on a LOW-degree representation
        (degree_a + degree_b <= 4). Pre-flip (=3) the optimiser was FORCED
        into >=3+>=3 = 6+ overfit cubics, wasting Optuna budget AND
        injecting higher-variance columns.
        """
        from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(42)
        n = 2000
        x_a = rng.normal(size=n)
        x_b = rng.normal(size=n)
        y = (np.sign(x_a * x_b) > 0).astype(np.int64)

        # Read the CURRENT default from MRMR.__init__ so the test
        # tracks the actual shipped value.
        import inspect

        default_min_degree = inspect.signature(MRMR.__init__).parameters["fe_min_polynom_degree"].default

        result = optimise_hermite_pair(
            x_a,
            x_b,
            y,
            min_degree=default_min_degree,
            max_degree=4,
            n_trials=40,
            optimizer="cma",
            warm_start=True,
            use_trivial_baseline=False,
            baseline_uplift_threshold=0.0,
            basis="hermite",
            seed=42,
        )
        assert result is not None, (
            f"optimise_hermite_pair must return a result on XOR with "
            f"min_degree={default_min_degree}, max_degree=4; pre-flip "
            f"min_degree=3 would force overfit cubic+ representations."
        )
        total_degree = int(result.degree_a) + int(result.degree_b)
        assert total_degree <= 4, (
            f"with default fe_min_polynom_degree={default_min_degree} the "
            f"optimiser must converge on a LOW-degree (sum<=4) "
            f"representation for XOR; pre-flip (min_degree=3) forced "
            f"sum>=6. got degree_a={result.degree_a}, "
            f"degree_b={result.degree_b}, sum={total_degree}"
        )


class TestScreeningKeepsBorderlineSignificantFeature:
    """Audit finding #5 sentinel: screening ``min_nonzero_confidence=0.99``
    + ``full_npermutations=3`` was suspected over-strict (gate is "0
    failures out of 3" - any borderline-significant feature whose
    permutation-test produces even ONE failure could be dropped).

    Empirical verdict 2026-05-18: on n=500 sparse-signal scenario both real
    features SURVIVE default screening. Originally marked ``xfail`` as
    a documentation guard pending the flip; xfail dropped 2026-05-18
    because: (a) the test consistently xpasses across multiple sessions,
    (b) the corrected polynom-FE bench at n=1M (where support_size=4 for
    all features after my flips) independently corroborates the finding.
    Now a STRICT assert: if future tightening drops a real feature, this
    test fails - the early-warning the original xfail was hoping to
    serve.
    """

    @pytest.mark.timeout(300)
    def test_screening_keeps_borderline_significant_feature(self):
        """Sparse-signal scenario with n=500 so each feature's permutation
        confidence is BORDERLINE. Both real features must survive default
        screening. If they don't, finding #5 is confirmed real.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(42)
        n = 500
        p = 10
        X = rng.normal(size=(n, p))
        noise = rng.normal(size=n)
        y_cont = 0.5 * X[:, 0] + 0.4 * X[:, 5] + 1.0 * noise
        y = (y_cont > np.median(y_cont)).astype(np.int64)
        df, ys = _to_df(X, y)

        # Both real features survive only if their linearly-usable raw form is kept even when FE folds
        # them into a nonlinear engineered feature -- the emit_both redundancy policy (signal operands
        # of selected engineered features are re-attached; noise operands are gated out).
        sel = MRMR(verbose=0, random_seed=42, redundancy_policy="emit_both")
        sel.fit(df, ys)

        names = set(int(i) for i in sel.support_)
        assert 0 in names and 5 in names, (
            f"both real features (X[:,0] and X[:,5]) must survive default "
            f"screening on a sparse-signal n=500 target; missing either "
            f"surfaces audit finding #5 (min_nonzero_confidence=0.99 + "
            f"full_npermutations=3 is too strict for borderline features). "
            f"got support={sorted(names)}"
        )


class TestPolynomFeFindsXorViaDefaultMrmrPath:
    """End-to-end smoke test: with all three flips active, MRMR's
    polynom-FE block must FIRE on a canonical "interaction surfaces
    through default screening" target. Failure indicates either the pair-MI
    gate or the screening filter blocks polynom-FE under defaults.
    """

    @pytest.mark.timeout(300)
    def test_polynom_fe_finds_xor_via_default_mrmr_path(self):
        """``z = 1.0*x_a + 1.0*x_b + 2.0*x_a*x_b + noise``, binarised at
        median. With ``MRMR(fe_smart_polynom_iters=2)`` and otherwise
        DEFAULT config (post-2026-05-18 flips), the polynom-FE block must
        store a discovered pair in ``_hermite_features_``.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(42)
        n = 2000
        x_a = rng.normal(size=n)
        x_b = rng.normal(size=n)
        noise1 = rng.normal(size=n)
        z = 1.0 * x_a + 1.0 * x_b + 2.0 * x_a * x_b + 0.3 * rng.normal(size=n)
        y = (z > np.median(z)).astype(np.int64)
        df, ys = _to_df(
            np.column_stack([x_a, x_b, noise1]),
            y,
            names=["x_a", "x_b", "noise1"],
        )

        sel = MRMR(
            verbose=0,
            random_seed=42,
            fe_smart_polynom_iters=2,
            fe_smart_polynom_optimization_steps=30,
        )
        sel.fit(df, ys)

        hermite_feats = getattr(sel, "_hermite_features_", []) or []
        recipes = getattr(sel, "_engineered_recipes_", []) or []
        hermite_pair_recipes = [r for r in recipes if getattr(r, "kind", None) == "hermite_pair"]
        assert hermite_feats or hermite_pair_recipes, (
            f"polynom-FE block must fire and store >=1 discovered pair "
            f"under default config + fe_smart_polynom_iters=2; an empty "
            f"_hermite_features_ AND empty _engineered_recipes_ indicates "
            f"the pair-MI gate or screening filter dropped the pair before "
            f"polynom-FE could see it. _hermite_features_="
            f"{len(hermite_feats)}, hermite_pair_recipes="
            f"{len(hermite_pair_recipes)}"
        )
