"""Unit + biz_value tests for ``CompositeTargetEstimator.monotone_constraints``.

The wrapper forwards a per-feature monotonicity constraint vector to the inner
GBDT at fit, enforced on the T (residual) target. For additive-residual cores
(``diff`` / ``linear_residual``) the inverse adds a base-only term back, so a
feature constrained monotone in T is also monotone in y at fixed base.

Coverage:
- A +1 constraint on a feature whose true T-relationship is non-monotone makes
  the inner produce predictions monotone-increasing in that feature, where the
  UNCONSTRAINED inner violates monotonicity (biz_value: the constraint wins).
- The constraint vector length is validated against the POST-drop feature count:
  for a grouped transform the group_column is dropped before the inner sees X,
  so the constraint must match (n_columns - 1), and a vector sized to the FULL
  column count raises.
- A length mismatch (no dropped columns) raises ValueError.
- An inner that does not expose ``monotone_constraints`` raises TypeError.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

lgb = pytest.importorskip("lightgbm")

from mlframe.training.composite.estimator import CompositeTargetEstimator


def _inner():
    # Small, deterministic tree ensemble; enough depth to chase a non-monotone
    # bump when left UNCONSTRAINED so the constraint has something to fix.
    return lgb.LGBMRegressor(
        n_estimators=200,
        num_leaves=31,
        min_child_samples=5,
        learning_rate=0.1,
        random_state=0,
        verbose=-1,
    )


def _make_nonmonotone_in_T(n: int = 4000, seed: int = 0):
    """Synthetic where T = y - base has a NON-monotone bump in feature ``f``.

    ``f`` in [0, 1]; the true T-signal rises then dips then rises (a wiggle an
    unconstrained tree will happily reproduce). A +1 monotone constraint on
    ``f`` forces the fitted surface to be non-decreasing in ``f``.
    """
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.0, 1.0, size=n)
    base = rng.normal(10.0, 2.0, size=n)
    # Non-monotone T component in f: ramp up, dip in the middle, ramp up again.
    t_signal = 3.0 * f - 12.0 * np.exp(-((f - 0.5) ** 2) / 0.01) + rng.normal(0.0, 0.05, size=n)
    # diff transform: T = y - base  ->  y = t_signal + base.
    y = t_signal + base
    X = pd.DataFrame({"base": base, "f": f})
    return X, y, f


def _max_monotone_violation(f_grid_pred: np.ndarray) -> float:
    """Largest downward step in predictions ordered by ascending feature."""
    d = np.diff(f_grid_pred)
    return float(-d.min()) if d.size and d.min() < 0 else 0.0


class TestMonotoneConstraintEnforced:
    def test_constraint_makes_T_predictions_monotone(self) -> None:
        X, y, _ = _make_nonmonotone_in_T()
        # Grid over f at a fixed base to read the marginal T-surface in f.
        grid = pd.DataFrame({"base": np.full(200, 10.0), "f": np.linspace(0.0, 1.0, 200)})

        # Unconstrained: the inner follows the dip -> a real downward step.
        est_free = CompositeTargetEstimator(
            base_estimator=_inner(),
            transform_name="diff",
            base_column="base",
        ).fit(X, y)
        t_free = est_free.estimator_.predict(grid[["base", "f"]])
        viol_free = _max_monotone_violation(t_free)

        # Constrained +1 on f (0 on base): the inner T-surface is non-decreasing in f.
        est_mono = CompositeTargetEstimator(
            base_estimator=_inner(),
            transform_name="diff",
            base_column="base",
            monotone_constraints=[0, 1],
        ).fit(X, y)
        t_mono = est_mono.estimator_.predict(grid[["base", "f"]])
        viol_mono = _max_monotone_violation(t_mono)

        assert viol_free > 0.5, f"the unconstrained inner should violate monotonicity in T on this synthetic; max downward step was only {viol_free:.4f}"
        assert viol_mono <= 1e-9, f"the +1-constrained inner must be non-decreasing in T; max downward step was {viol_mono:.6f}"

    def test_constraint_carries_through_to_y_on_additive_core(self) -> None:
        # diff is additive (y = T + base); at fixed base, y is monotone in f iff T is.
        X, y, _ = _make_nonmonotone_in_T()
        grid = pd.DataFrame({"base": np.full(200, 10.0), "f": np.linspace(0.0, 1.0, 200)})
        est = CompositeTargetEstimator(
            base_estimator=_inner(),
            transform_name="diff",
            base_column="base",
            monotone_constraints=[0, 1],
        ).fit(X, y)
        y_hat = est.predict(grid)
        assert _max_monotone_violation(y_hat) <= 1e-9


class TestConstraintLengthValidation:
    def test_grouped_transform_validates_against_post_drop_count(self) -> None:
        # linear_residual_grouped drops the group_column before the inner fits,
        # so the inner trains on (n_cols - 1) features. A constraint vector sized
        # to the post-drop count fits; one sized to the FULL column count raises.
        rng = np.random.default_rng(1)
        n = 1500
        g = rng.integers(0, 3, size=n)
        base = rng.normal(5.0, 1.0, size=n)
        f = rng.uniform(0.0, 1.0, size=n)
        y = base * (1.0 + 0.2 * g) + 2.0 * f + rng.normal(0.0, 0.1, size=n)
        X = pd.DataFrame({"base": base, "f": f, "grp": g.astype(str)})
        # X has 3 columns; the inner trains on 2 (grp dropped). post-drop = 2.
        ok = CompositeTargetEstimator(
            base_estimator=_inner(),
            transform_name="linear_residual_grouped",
            base_column="base",
            group_column="grp",
            monotone_constraints=[0, 1],
        )
        ok.fit(X, y)  # length 2 == post-drop count -> succeeds.
        assert ok.estimator_.get_params()["monotone_constraints"] == [0, 1]

        # Length 3 (= FULL column count, includes the dropped group_column) raises.
        bad = CompositeTargetEstimator(
            base_estimator=_inner(),
            transform_name="linear_residual_grouped",
            base_column="base",
            group_column="grp",
            monotone_constraints=[0, 0, 1],
        )
        with pytest.raises(ValueError, match="post-drop feature count"):
            bad.fit(X, y)

    def test_plain_length_mismatch_raises(self) -> None:
        X, y, _ = _make_nonmonotone_in_T(n=500)
        est = CompositeTargetEstimator(
            base_estimator=_inner(),
            transform_name="diff",
            base_column="base",
            monotone_constraints=[1],  # X has 2 columns; vector is length 1.
        )
        with pytest.raises(ValueError, match="length 1 but the inner estimator trains on 2"):
            est.fit(X, y)

    def test_bad_constraint_values_raise(self) -> None:
        X, y, _ = _make_nonmonotone_in_T(n=500)
        est = CompositeTargetEstimator(
            base_estimator=_inner(),
            transform_name="diff",
            base_column="base",
            monotone_constraints=[0, 2],  # 2 is not a valid constraint code.
        )
        with pytest.raises(ValueError, match="must each be"):
            est.fit(X, y)

    def test_inner_without_support_raises(self) -> None:
        from sklearn.linear_model import LinearRegression

        X, y, _ = _make_nonmonotone_in_T(n=500)
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="diff",
            base_column="base",
            monotone_constraints=[0, 1],
        )
        with pytest.raises(TypeError, match="does not accept"):
            est.fit(X, y)


def test_default_none_unchanged() -> None:
    # No constraint configured -> the inner is fit with its own default params.
    X, y, _ = _make_nonmonotone_in_T(n=500)
    est = CompositeTargetEstimator(
        base_estimator=_inner(),
        transform_name="diff",
        base_column="base",
    ).fit(X, y)
    # LightGBM does not list monotone_constraints in its default get_params (it
    # is a booster **kwargs extra), so when the wrapper never forwards one the
    # key is simply absent -- the inner is fit with its own defaults untouched.
    assert "monotone_constraints" not in est.estimator_.get_params()
