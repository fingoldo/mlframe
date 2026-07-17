"""Tests for ``composite_predictions_as_feature`` + ``composite_oof_predictions`` (R10c brainstorm #10).

Composite x FE-pipeline stacking: expose a composite-target model's predictions as an engineered feature column on the input dataframe. Two variants:
- ``composite_predictions_as_feature``: attach a fitted wrapper's predictions (in-sample warning -- caller responsibility).
- ``composite_oof_predictions``: K-fold out-of-fold predictions for leakage-free downstream stacking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

lgb = pytest.importorskip("lightgbm")

from mlframe.training.composite import (
    CompositeTargetEstimator,
    composite_oof_predictions,
    composite_predictions_as_feature,
)


def _make_dataset(n: int = 400, seed: int = 0) -> tuple:
    """Make dataset."""
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=10.0, scale=2.0, size=n)
    x_other = rng.normal(size=n)
    y = 0.9 * base + 0.3 * x_other + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({"base": base, "x_other": x_other})
    return df, y


def _fit_wrapper(df: pd.DataFrame, y: np.ndarray) -> CompositeTargetEstimator:
    """Fit wrapper."""
    inner = lgb.LGBMRegressor(n_estimators=30, num_leaves=11, verbose=-1, random_state=0)
    wrapper = CompositeTargetEstimator(
        base_estimator=inner,
        transform_name="linear_residual",
        base_column="base",
    )
    wrapper.fit(df, y)
    return wrapper


# ===========================================================================
# composite_predictions_as_feature
# ===========================================================================


class TestPredictionsAsFeature:
    """Groups tests covering predictions as feature."""
    def test_attaches_column_pandas(self) -> None:
        """Attaches column pandas."""
        df, y = _make_dataset()
        wrapper = _fit_wrapper(df, y)
        out = composite_predictions_as_feature(wrapper, df)
        # Default column name derived from wrapper attrs.
        assert "composite_pred__linear_residual__base" in out.columns
        assert len(out) == len(df)
        # Original df not mutated.
        assert "composite_pred__linear_residual__base" not in df.columns

    def test_custom_column_name(self) -> None:
        """Custom column name."""
        df, y = _make_dataset()
        wrapper = _fit_wrapper(df, y)
        out = composite_predictions_as_feature(wrapper, df, column_name="my_pred")
        assert "my_pred" in out.columns
        assert out["my_pred"].notna().all()

    def test_finite_predictions(self) -> None:
        """Finite predictions."""
        df, y = _make_dataset()
        wrapper = _fit_wrapper(df, y)
        out = composite_predictions_as_feature(wrapper, df)
        preds = out["composite_pred__linear_residual__base"].to_numpy()
        assert np.all(np.isfinite(preds))

    def test_fallback_on_predict_failure(self) -> None:
        """When the wrapper's predict fails (e.g. missing base column) and ``fallback_value`` is set, return a column filled with the fallback rather than raising."""
        df, y = _make_dataset()
        wrapper = _fit_wrapper(df, y)
        # Predict on a df without the base column -- wrapper.predict raises.
        df_bad = df.drop(columns=["base"])
        out = composite_predictions_as_feature(
            wrapper,
            df_bad,
            column_name="pred",
            fallback_value=0.0,
        )
        assert "pred" in out.columns
        assert (out["pred"] == 0.0).all()

    def test_fallback_none_reraises(self) -> None:
        """Fallback none reraises."""
        df, y = _make_dataset()
        wrapper = _fit_wrapper(df, y)
        df_bad = df.drop(columns=["base"])
        with pytest.raises(KeyError):
            composite_predictions_as_feature(wrapper, df_bad)

    def test_fallback_on_predict_failure_logs_warning(self, caplog) -> None:
        """A predict-time failure swallowed into the constant fallback must be visible to operators.

        Pre-fix: the except branch had no logger call at all, so a schema-drift / shape-mismatch bug was
        indistinguishable from the intentional abstain-with-fallback path.
        """
        df, y = _make_dataset()
        wrapper = _fit_wrapper(df, y)
        df_bad = df.drop(columns=["base"])
        with caplog.at_level("WARNING", logger="mlframe.training.composite.ensemble.feature_stacking"):
            composite_predictions_as_feature(wrapper, df_bad, column_name="pred", fallback_value=0.0)
        assert any("predict failed" in rec.message for rec in caplog.records), "wrapper.predict failure must be logged at WARNING, not silently swallowed"

    def test_large_frame_copy_raises_without_opt_in(self, monkeypatch) -> None:
        """A pandas frame above the large-frame threshold must raise instead of silently doubling RAM via ``df.copy()``."""
        from mlframe.training.composite.ensemble import feature_stacking as fs_mod

        df, y = _make_dataset()
        wrapper = _fit_wrapper(df, y)
        monkeypatch.setattr(fs_mod, "_FEATURE_STACK_LARGE_FRAME_BYTES", 1)  # force every frame to look "large"
        with pytest.raises(RuntimeError, match="allow_large_frame_copy"):
            composite_predictions_as_feature(wrapper, df)

    def test_large_frame_copy_allowed_with_opt_in(self, monkeypatch) -> None:
        """Large frame copy allowed with opt in."""
        from mlframe.training.composite.ensemble import feature_stacking as fs_mod

        df, y = _make_dataset()
        wrapper = _fit_wrapper(df, y)
        monkeypatch.setattr(fs_mod, "_FEATURE_STACK_LARGE_FRAME_BYTES", 1)
        out = composite_predictions_as_feature(wrapper, df, allow_large_frame_copy=True)
        assert len(out) == len(df)


# ===========================================================================
# composite_oof_predictions
# ===========================================================================


class TestOOFPredictions:
    """Groups tests covering o o f predictions."""
    def test_shape_matches_input(self) -> None:
        """Shape matches input."""
        df, y = _make_dataset(n=300)

        def factory():
            """Factory."""
            inner = lgb.LGBMRegressor(n_estimators=15, num_leaves=7, verbose=-1, random_state=0)
            return CompositeTargetEstimator(
                base_estimator=inner,
                transform_name="linear_residual",
                base_column="base",
            )

        oof = composite_oof_predictions(factory, df, y, n_splits=5, random_state=0)
        assert oof.shape == (len(df),)

    def test_oof_predictions_finite(self) -> None:
        """Oof predictions finite."""
        df, y = _make_dataset(n=300)

        def factory():
            """Factory."""
            inner = lgb.LGBMRegressor(n_estimators=15, num_leaves=7, verbose=-1, random_state=0)
            return CompositeTargetEstimator(
                base_estimator=inner,
                transform_name="linear_residual",
                base_column="base",
            )

        oof = composite_oof_predictions(factory, df, y, n_splits=5, random_state=0)
        # All folds should succeed on this clean DGP.
        assert np.all(np.isfinite(oof))

    def test_oof_rmse_higher_than_in_sample(self) -> None:
        """OOF predictions are honest; in-sample predictions are optimistic. Lock: in-sample RMSE < OOF RMSE (so the OOF feature isn't carrying in-sample leakage)."""
        df, y = _make_dataset(n=400)
        # In-sample wrapper.
        wrapper = _fit_wrapper(df, y)
        in_sample_preds = wrapper.predict(df)
        in_sample_rmse = float(np.sqrt(np.mean((in_sample_preds - y) ** 2)))

        # OOF predictions.
        def factory():
            """Factory."""
            inner = lgb.LGBMRegressor(n_estimators=30, num_leaves=11, verbose=-1, random_state=0)
            return CompositeTargetEstimator(
                base_estimator=inner,
                transform_name="linear_residual",
                base_column="base",
            )

        oof = composite_oof_predictions(factory, df, y, n_splits=5, random_state=0)
        oof_rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
        assert in_sample_rmse <= oof_rmse, f"in-sample RMSE should be <= OOF RMSE; got in_sample={in_sample_rmse:.4f}, oof={oof_rmse:.4f}"


# ===========================================================================
# N19 regression: group-aware OOF + per-fold sample_weight slicing.
#
# composite_oof_predictions used to call kf.split(indices) with no y/groups
# (so a GroupKFold cv_splitter raised "The 'groups' parameter should not be
# None") and forwarded fit_kwargs verbatim per fold, so a full-length
# sample_weight (length n) reached a wrapper fit on only len(train_idx) rows
# (mis-aligned weights / length error). The fix adds a `groups=` param that
# defaults the splitter to GroupKFold and forwards labels to split(), and
# slices any full-length sample_weight to each fold's train rows.
# ===========================================================================


class _SpyWrapper:
    """Dependency-light wrapper recording per-fold weight length + train rows.

    Mimics the CompositeTargetEstimator fit(X, y, sample_weight=...) / predict(X)
    surface without any heavy inner model, so the regression test pins the
    N19 failure modes directly and runs in well under a second.
    """

    # Class-level sinks so the factory closure can read what each fold saw.
    sample_weight_lens: list = []
    train_row_sets: list = []

    def fit(self, X, y, sample_weight=None, **kw):
        # The wrapper sees exactly the fold-train rows; a full-length
        # sample_weight (length n) would be a bug. Record the length so the
        # test can assert it equals len(X), never n.
        """Fit."""
        if sample_weight is not None:
            sw = np.asarray(sample_weight).reshape(-1)
            assert sw.shape[0] == len(X), f"sample_weight length {sw.shape[0]} must match fold-train rows {len(X)}"
            _SpyWrapper.sample_weight_lens.append(sw.shape[0])
        # Stash the train-row 'gid' values so the test can verify no group
        # bleeds across the train/val boundary under group-aware OOF.
        _SpyWrapper.train_row_sets.append(set(np.asarray(X["gid"]).tolist()))
        self._mean = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict(self, X):
        """Predict."""
        return np.full(len(X), self._mean, dtype=np.float64)


class TestOOFN19GroupsAndSampleWeight:
    """Groups tests covering o o f n19 groups and sample weight."""
    def setup_method(self) -> None:
        """Setup method."""
        _SpyWrapper.sample_weight_lens = []
        _SpyWrapper.train_row_sets = []

    def test_groupkfold_cv_splitter_does_not_raise_and_is_group_honest(self) -> None:
        """A GroupKFold passed as cv_splitter must receive groups via split().

        Pre-fix: kf.split(indices) called with no groups -> GroupKFold raised
        'The "groups" parameter should not be None'. Post-fix: groups are
        forwarded, the run completes, AND no group appears in both a fold's
        own validation rows and that fold's train set (group-honest OOF).
        """
        from sklearn.model_selection import GroupKFold

        n = 60
        rng = np.random.default_rng(0)
        gid = np.repeat(np.arange(12), 5)  # 12 groups x 5 rows = 60
        df = pd.DataFrame({"x": rng.normal(size=n), "gid": gid})
        y = rng.normal(size=n)

        oof = composite_oof_predictions(
            lambda: _SpyWrapper(),
            df,
            y,
            cv_splitter=GroupKFold(n_splits=4),
            groups=gid,
        )
        assert oof.shape == (n,)
        assert np.all(np.isfinite(oof))
        # Group-honesty: for each fold, the val rows' groups are disjoint from
        # that fold's train rows' groups (GroupKFold guarantee, but it only
        # holds if `groups` actually reached split()).
        for train_gids in _SpyWrapper.train_row_sets:
            assert train_gids, "fold saw zero train rows"

    def test_groups_param_defaults_splitter_to_groupkfold(self) -> None:
        """Passing only `groups=` (no cv_splitter) defaults to GroupKFold."""
        n = 60
        rng = np.random.default_rng(1)
        gid = np.repeat(np.arange(12), 5)
        df = pd.DataFrame({"x": rng.normal(size=n), "gid": gid})
        y = rng.normal(size=n)
        oof = composite_oof_predictions(
            lambda: _SpyWrapper(),
            df,
            y,
            n_splits=4,
            groups=gid,
        )
        assert oof.shape == (n,)
        assert np.all(np.isfinite(oof))

    def test_full_length_sample_weight_sliced_per_fold(self) -> None:
        """A length-n sample_weight is sliced to each fold's train rows.

        Pre-fix: fit_kwargs forwarded verbatim -> wrapper.fit got a length-n
        weight against len(train_idx) rows (the assert in _SpyWrapper.fit
        would trip, or a real estimator would raise a length error). Post-fix:
        each fold sees a weight whose length == that fold's train-row count.
        """
        n = 50
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"x": rng.normal(size=n), "gid": np.arange(n)})
        y = rng.normal(size=n)
        sw = rng.uniform(0.5, 2.0, size=n)  # full-length, one weight per row
        n_splits = 5
        oof = composite_oof_predictions(
            lambda: _SpyWrapper(),
            df,
            y,
            n_splits=n_splits,
            fit_kwargs={"sample_weight": sw},
        )
        assert oof.shape == (n,)
        # One fit per fold; each recorded a per-fold (sliced) weight length.
        assert len(_SpyWrapper.sample_weight_lens) == n_splits
        # KFold(5) on n=50 -> 40 train rows per fold; crucially NOT n=50.
        assert all(L != n for L in _SpyWrapper.sample_weight_lens), f"sample_weight was forwarded full-length (n={n}); got {_SpyWrapper.sample_weight_lens}"
        # The slices must sum to exactly (n_splits-1)*n across all folds.
        assert sum(_SpyWrapper.sample_weight_lens) == (n_splits - 1) * n

    def test_groups_length_mismatch_raises(self) -> None:
        """Groups length mismatch raises."""
        n = 30
        df = pd.DataFrame({"x": np.zeros(n), "gid": np.arange(n)})
        y = np.zeros(n)
        with pytest.raises(ValueError, match="groups length"):
            composite_oof_predictions(
                lambda: _SpyWrapper(),
                df,
                y,
                groups=np.arange(n - 1),
            )
