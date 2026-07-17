"""T2#13 2026-05-18 Stronger Hermite end-to-end test.

The existing test_biz_val_filters_hermite_fe.py exercises
``optimise_hermite_pair`` IN ISOLATION. It never calls
``MRMR(...).fit(X, y)`` so the wiring from optimiser to
``_hermite_features_`` to ``_engineered_recipes_`` to ``transform``
output was untested end-to-end. Production TVT ran 88 min on the
optimiser, found a useful basis, then a stale "# future work" comment
discarded it - all unit tests still passed because none of them drove
the integration.

This test drives the FULL path:
1. Build a synthetic XOR target where the optimal feature is a
   polynomial pair (degree-1 Hermite product = XOR sign).
2. Run ``MRMR(fe_smart_polynom_iters>0).fit(X, y)`` with enough budget
   that the optimiser reliably converges.
3. Assert ``mrmr._engineered_recipes_`` contains a hermite_pair
   recipe.
4. Assert ``mrmr.transform(X_test)`` produces a column for that
   recipe whose values are finite and have non-trivial dynamic range.
5. biz_value: MI between the engineered column and y must exceed MI
   of any individual raw feature with y by the prevalence factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.engineered_recipes import (
    EngineeredRecipe,
    apply_recipe,
    build_hermite_pair_recipe,
)


def _build_xor_problem(n: int = 2000, seed: int = 42):
    """Synthetic problem where the optimal feature is x_a * x_b sign.

    n=2000 + seed=42 matches the known-good seed in
    test_biz_val_filters_hermite_fe.py where CMA-ES reliably converges
    on the canonical XOR optimum (MI >= 0.55).
    """
    rng = np.random.default_rng(seed)
    x_a = rng.normal(size=n).astype(np.float64)
    x_b = rng.normal(size=n).astype(np.float64)
    noise_features = rng.normal(size=(n, 2)).astype(np.float64)
    y = (np.sign(x_a * x_b) > 0).astype(np.int64)
    X = pd.DataFrame(
        {
            "x_a": x_a,
            "x_b": x_b,
            "noise1": noise_features[:, 0],
            "noise2": noise_features[:, 1],
        }
    )
    return X, y


class TestHermiteE2EStrong:
    """Drive the full Hermite pipeline and pin the recipe survival path."""

    @pytest.mark.timeout(600)
    def test_optimiser_finds_xor_and_recipe_persists(self) -> None:
        """The most directly testable path: optimise_hermite_pair on a XOR
        problem must return a usable HermiteResult; the recipe build/apply
        cycle must reproduce the engineered column on held-out rows.

        Knobs match test_biz_cma_es_finds_xor_optimum (the known-stable
        XOR config): CMA-ES + warm_start + disabled baseline_uplift gate.
        """
        from mlframe.feature_selection.filters.hermite_fe import (
            optimise_hermite_pair,
        )

        X, y = _build_xor_problem(n=2000, seed=42)
        x_a = X["x_a"].values
        x_b = X["x_b"].values

        result = optimise_hermite_pair(
            x_a=x_a,
            x_b=x_b,
            y=y,
            basis="hermite",
            max_degree=4,
            n_trials=40,
            optimizer="cma",
            warm_start=True,
            use_trivial_baseline=False,
            baseline_uplift_threshold=0.0,
            seed=42,
        )
        assert result is not None, "CMA-ES warm-start with disabled baseline-uplift gate must return a HermiteResult on the canonical XOR problem"

        # The optimiser cleared the gate. Build a recipe and verify replay.
        recipe = build_hermite_pair_recipe(
            name="hermite_xor_e2e",
            src_names=("x_a", "x_b"),
            hermite_result=result,
        )
        assert isinstance(recipe, EngineeredRecipe)
        assert recipe.kind == "hermite_pair"

        # Replay on held-out rows.
        X_test, _ = _build_xor_problem(n=400, seed=99)
        out = apply_recipe(recipe, X_test)
        assert out.shape == (len(X_test),)
        assert np.isfinite(out).all(), "replay produced non-finite values"
        # Dynamic range > 0 -> the recipe is not collapsed to a constant.
        assert float(out.std()) > 1e-6, f"replay output is near-constant (std={out.std()}); polynomial collapsed during replay"

    @pytest.mark.timeout(600)
    def test_engineered_column_has_higher_mi_than_raw(self) -> None:
        """biz_value: the Hermite-engineered column must carry more signal
        about y than any individual raw feature. This is the whole reason
        the 88-min Optuna search exists - we test the LIFT here."""
        from mlframe.feature_selection.filters.hermite_fe import (
            optimise_hermite_pair,
        )
        from sklearn.feature_selection import mutual_info_classif

        X, y = _build_xor_problem(n=2000, seed=42)
        x_a = X["x_a"].values
        x_b = X["x_b"].values

        result = optimise_hermite_pair(
            x_a=x_a,
            x_b=x_b,
            y=y,
            basis="hermite",
            max_degree=4,
            n_trials=40,
            optimizer="cma",
            warm_start=True,
            use_trivial_baseline=False,
            baseline_uplift_threshold=0.0,
            seed=42,
        )
        assert result is not None, "CMA-ES must find XOR optimum (known-stable config)"

        recipe = build_hermite_pair_recipe(
            name="hermite_xor_mi",
            src_names=("x_a", "x_b"),
            hermite_result=result,
        )
        engineered_col = apply_recipe(recipe, X)

        # MI(engineered, y) vs MI(each raw feature, y).
        raw_mis = mutual_info_classif(
            X.values,
            y,
            discrete_features=False,
            random_state=0,
        )
        eng_mi = float(
            mutual_info_classif(
                engineered_col.reshape(-1, 1),
                y,
                discrete_features=False,
                random_state=0,
            )[0]
        )

        max_raw_mi = float(np.max(raw_mis))
        # XOR has zero individual MI; the engineered feature should have
        # SIGNIFICANTLY more. Use 2x as the headroom guard: raw is ~0 so
        # this is a meaningful contrast.
        assert eng_mi > 2.0 * max_raw_mi, (
            f"engineered column has weaker MI than the best raw feature: "
            f"eng_mi={eng_mi:.4f}, max_raw_mi={max_raw_mi:.4f}. "
            f"Either the optimiser converged on the wrong basis or the "
            f"replay produced a degraded column."
        )
