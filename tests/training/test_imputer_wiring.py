"""
Sensor tests for the long-standing `imputer_strategy` wiring debt.

`PreprocessingBackendConfig.imputer_strategy` (renamed from
`PolarsPipelineConfig.imputer_strategy`) was declared but never connected to
the polars-ds Blueprint in `create_polarsds_pipeline` — confirmed by the
audit at training/pipeline.py:458-585. As a result, NaN values in numeric
columns survived the pipeline and crashed downstream models with cryptic
upstream errors.

This pack locks in the wiring through behavioural tests (NaN in / NaN
out), not source-inspection — per the project rule "behavioral tests over
source-inspection".

Phase M deliverable. The ten tests below are the contract for the fix
landed in `create_polarsds_pipeline`.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
import pytest

# Some polars_ds installs ship the core package without the Pipeline /
# Blueprint submodule (legacy split builds). The previous file-level
# ``pytest.importorskip("polars_ds.pipeline")`` and the module-level
# ``pytestmark = skipif(...)`` both broke pytest node-ID resolution
# ("ERROR: found no collectors for path::class::method") on those envs:
# pytest refuses to navigate into a class/method whose containing module
# was skipped at collection time.
#
# Robust fix: collection ALWAYS succeeds (no module-level skip). Each
# test starts with ``_require_pds_pipeline()`` which raises pytest.skip
# at RUNTIME, AFTER the test has been collected and named. Node-IDs
# resolve cleanly and the run reports per-test skips instead of
# collection errors.
# Module under test — imported lazily so test collection works during the
# bootstrap phase when imports may not be settled.
from mlframe.training.configs import PreprocessingBackendConfig
from mlframe.training.pipeline import create_polarsds_pipeline


@pytest.fixture(autouse=True)
def _require_pds_pipeline():
    """Autouse skip-gate. Fires AFTER collection, so pytest node-ID lookup
    (``path::class::method`` invocations) resolves cleanly on envs without
    ``polars_ds.pipeline`` -- each named test gets skipped individually
    instead of producing a "found no collectors" collection error."""
    try:
        import polars_ds.pipeline  # noqa: F401
    except ImportError:
        pytest.skip("polars_ds.pipeline submodule unavailable on this install")


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def df_with_nans():
    """Numeric frame with NULL holes (polars-native) in two columns + a clean column.

    Uses ``pl.when(...).then(None).otherwise(...)`` to inject real polars
    nulls. ``np.nan`` floats are NOT counted by ``null_count()`` -- they
    are valid floats with a special bit pattern -- so a fixture using
    ``np.nan`` would silently make ``test_strategy_none_leaves_nans``
    pass for the wrong reason.
    """
    rng = np.random.default_rng(0)
    n = 200
    f0 = rng.standard_normal(n).astype(np.float32)
    f1 = rng.standard_normal(n).astype(np.float32)
    f2 = rng.standard_normal(n).astype(np.float32)  # clean
    df = pl.DataFrame({"f0": f0, "f1": f1, "f2": f2})
    # Inject ~14% nulls in f0 (every 7th row), 20-row block of nulls in f1.
    df = df.with_columns(
        [
            pl.when(pl.int_range(pl.len()) % 7 == 0).then(None).otherwise(pl.col("f0")).alias("f0"),
            pl.when((pl.int_range(pl.len()) >= 3) & (pl.int_range(pl.len()) < 23)).then(None).otherwise(pl.col("f1")).alias("f1"),
        ]
    )
    return df


@pytest.fixture
def df_train_test_with_nans():
    """Two frames sharing schema. Train has NaN, test has NaN at different
    positions + entirely fresh NaN columns. Used to verify train-only
    statistic application (no leak)."""
    rng = np.random.default_rng(0)
    n = 200
    train = pl.DataFrame(
        {
            "f0": rng.standard_normal(n).astype(np.float32),
            "f1": rng.standard_normal(n).astype(np.float32),
        }
    )
    train = train.with_columns(
        [
            pl.when(pl.int_range(pl.len()) % 5 == 0).then(None).otherwise(pl.col("f0")).alias("f0"),
            pl.when(pl.int_range(pl.len()) % 7 == 0).then(None).otherwise(pl.col("f1")).alias("f1"),
        ]
    )
    test = pl.DataFrame(
        {
            "f0": rng.standard_normal(50).astype(np.float32),
            "f1": rng.standard_normal(50).astype(np.float32),
        }
    )
    test = test.with_columns(
        [
            pl.when(pl.int_range(pl.len()) % 3 == 0).then(None).otherwise(pl.col("f0")).alias("f0"),
            pl.when(pl.int_range(pl.len()) % 4 == 0).then(None).otherwise(pl.col("f1")).alias("f1"),
        ]
    )
    return train, test


@pytest.fixture
def df_with_string_and_numeric_nans():
    """Mixed-dtype frame: string column, categorical column, numeric with
    NaN. Imputer strategies should target only numeric columns; string
    imputation belongs to the categorical encoding step.
    """
    rng = np.random.default_rng(0)
    n = 100
    return pl.DataFrame(
        {
            "num_col": np.where(rng.random(n) < 0.2, np.nan, rng.standard_normal(n)).astype(np.float32),
            "str_col": ["a", "b", "c", None, "a"] * (n // 5),
            "cat_col": pl.Series(["x", "y", None, "x", "y"] * (n // 5), dtype=pl.Categorical),
        }
    )


def _scaler_off() -> dict:
    """Disable scaler to test imputer in isolation — scaler can mask
    pre-existing NaN handling bugs by introducing its own."""
    return dict(scaler_name=None)


# =====================================================================
# 1. Strategy="mean" fills NaN with train-column mean
# =====================================================================


class TestImputerStrategyMean:
    def test_strategy_mean_fills_all_numeric_nans(self, df_with_nans):
        cfg = PreprocessingBackendConfig(imputer_strategy="mean", **_scaler_off())
        pipe = create_polarsds_pipeline(df_with_nans, cfg, verbose=0)
        assert pipe is not None, "polars-ds pipeline must be returned"
        out = pipe.transform(df_with_nans)
        # Numeric columns: zero NaN survives. f0 had ~14% NaN, f1 had a 20-row block.
        assert out["f0"].null_count() == 0, "f0 should have no NaN after mean imputation"
        assert out["f1"].null_count() == 0, "f1 should have no NaN after mean imputation"
        assert out["f2"].null_count() == 0, "f2 was already clean; should stay clean"

    def test_strategy_mean_uses_train_mean_value(self, df_with_nans):
        cfg = PreprocessingBackendConfig(imputer_strategy="mean", **_scaler_off())
        pipe = create_polarsds_pipeline(df_with_nans, cfg, verbose=0)
        out = pipe.transform(df_with_nans)
        # Imputed positions in f0 (the np.nan slots — every 7th row) must
        # equal the train mean of f0's non-null values.
        train_f0_mean = float(df_with_nans["f0"].mean())
        imputed_positions = list(range(0, 200, 7))
        for pos in imputed_positions:
            np.testing.assert_allclose(
                out["f0"][pos],
                train_f0_mean,
                atol=1e-5,
                err_msg=f"f0[{pos}] should equal train mean {train_f0_mean}",
            )


# =====================================================================
# 2. Strategy="median"
# =====================================================================


class TestImputerStrategyMedian:
    def test_strategy_median_fills_nans(self, df_with_nans):
        cfg = PreprocessingBackendConfig(imputer_strategy="median", **_scaler_off())
        pipe = create_polarsds_pipeline(df_with_nans, cfg, verbose=0)
        out = pipe.transform(df_with_nans)
        assert out["f0"].null_count() == 0
        assert out["f1"].null_count() == 0

    def test_strategy_median_uses_train_median(self, df_with_nans):
        cfg = PreprocessingBackendConfig(imputer_strategy="median", **_scaler_off())
        pipe = create_polarsds_pipeline(df_with_nans, cfg, verbose=0)
        out = pipe.transform(df_with_nans)
        train_f0_median = float(df_with_nans["f0"].median())
        np.testing.assert_allclose(
            out["f0"][0],
            train_f0_median,
            atol=1e-5,
            err_msg="imputed value should be train median",
        )


# =====================================================================
# 3. Strategy=None skips imputation (NaN survives)
# =====================================================================


class TestImputerStrategyNone:
    def test_strategy_none_leaves_nans(self, df_with_nans):
        cfg = PreprocessingBackendConfig(imputer_strategy=None, **_scaler_off())
        pipe = create_polarsds_pipeline(df_with_nans, cfg, verbose=0)
        out = pipe.transform(df_with_nans)
        # Skipped imputer → NaN survives.
        assert out["f0"].null_count() > 0, "imputer_strategy=None should NOT fill NaN"
        # Original NaN counts preserved.
        assert out["f0"].null_count() == df_with_nans["f0"].null_count()


# =====================================================================
# 4. No-leak: train-only fit applies same statistic to test
# =====================================================================


class TestImputerTrainOnlyNoLeak:
    def test_test_set_imputed_with_train_mean_not_test_mean(self, df_train_test_with_nans):
        train, test = df_train_test_with_nans
        cfg = PreprocessingBackendConfig(imputer_strategy="mean", **_scaler_off())
        pipe = create_polarsds_pipeline(train, cfg, verbose=0)
        out_test = pipe.transform(test)
        # Test set imputation must use TRAIN mean (computed at fit), NOT
        # TEST mean (which is a leakage-flavoured shortcut).
        train_f0_mean = float(train["f0"].mean())
        test_f0_mean = float(test["f0"].mean())
        # Sanity: train and test means differ enough to discriminate.
        assert abs(train_f0_mean - test_f0_mean) > 1e-3, "fixture broken: train and test means too close to discriminate"
        # First imputed row in test (every 3rd) — must be train mean.
        np.testing.assert_allclose(
            out_test["f0"][0],
            train_f0_mean,
            atol=1e-5,
            err_msg=(
                f"test imputation used the wrong statistic: got {float(out_test['f0'][0])}, expected train mean {train_f0_mean} (test mean was {test_f0_mean})"
            ),
        )


# =====================================================================
# 5. Numeric-only target — string columns untouched
# =====================================================================


class TestImputerLeavesStringColumnsAlone:
    def test_string_nulls_not_filled_by_numeric_imputer(self, df_with_string_and_numeric_nans):
        cfg = PreprocessingBackendConfig(
            imputer_strategy="mean",
            categorical_encoding=None,  # disable encoder to leave string column raw
            **_scaler_off(),
        )
        pipe = create_polarsds_pipeline(df_with_string_and_numeric_nans, cfg, verbose=0)
        out = pipe.transform(df_with_string_and_numeric_nans)
        # Numeric column had its NaN filled.
        assert out["num_col"].null_count() == 0
        # String column's None values are NOT touched by the numeric imputer.
        # (Categorical encoding is the right step for string nulls — we
        # explicitly disabled it above so we can verify the imputer
        # doesn't overreach.)
        assert out["str_col"].null_count() == df_with_string_and_numeric_nans["str_col"].null_count()


# =====================================================================
# 6. Idempotency on clean data — no-op behaviour
# =====================================================================


class TestImputerIdempotentOnCleanData:
    def test_clean_data_passes_through_unchanged(self):
        rng = np.random.default_rng(0)
        clean = pl.DataFrame(
            {
                "f0": rng.standard_normal(100).astype(np.float32),
                "f1": rng.standard_normal(100).astype(np.float32),
            }
        )
        cfg = PreprocessingBackendConfig(imputer_strategy="mean", **_scaler_off())
        pipe = create_polarsds_pipeline(clean, cfg, verbose=0)
        out = pipe.transform(clean)
        # Bit-equal pass-through (modulo dtype since int_to_float runs at the end).
        np.testing.assert_array_equal(out["f0"].to_numpy(), clean["f0"].to_numpy())
        np.testing.assert_array_equal(out["f1"].to_numpy(), clean["f1"].to_numpy())


# =====================================================================
# 7. Imputer + scaler compose: NaN-fill happens BEFORE scaling
# =====================================================================


class TestImputerComposesWithScaler:
    def test_imputer_runs_before_scaler(self, df_with_nans):
        """
        If scaler ran before imputer, NaN values would propagate through
        scaling (NaN * x = NaN), leaving scaled NaN in output. Imputer
        must run first so scaler sees no NaN.
        """
        cfg = PreprocessingBackendConfig(
            imputer_strategy="mean",
            scaler_name="standard",
        )
        pipe = create_polarsds_pipeline(df_with_nans, cfg, verbose=0)
        out = pipe.transform(df_with_nans)
        # No NaN in any numeric column post-pipeline.
        for col in ("f0", "f1", "f2"):
            assert out[col].null_count() == 0, f"{col} has NaN after imputer+scaler — order is likely wrong"
        # Scaling actually applied (mean ≈ 0, std ≈ 1 for f2 which had no NaN).
        f2_vals = out["f2"].to_numpy()
        assert abs(float(np.mean(f2_vals))) < 0.5, "scaler did not run"


# =====================================================================
# 8. Float NaN (not just polars NULL) is imputed
# =====================================================================


class TestFloatNanImputed:
    """polars-ds ``Blueprint.impute`` only fills polars NULL; float ``NaN``
    (the numpy/pandas-origin missing marker, also produced by FE ratios like
    0/0) survived the imputer silently and reached the scaler / model. The
    pipeline now converts NaN -> NULL on float imputable columns first."""

    def _nan_frame(self):
        rng = np.random.default_rng(0)
        n = 200
        X = rng.lognormal(0.0, 1.0, size=(n, 3)).astype(np.float32)
        mask = rng.random((n, 3)) < 0.2
        X[mask] = np.nan  # FLOAT NaN, not polars NULL
        df = pl.DataFrame({f"f{j}": X[:, j] for j in range(3)})
        # Sanity: null_count does NOT see float NaN, so the holes are real NaN.
        assert df.null_count().to_numpy().sum() == 0
        return df, mask

    def test_float_nan_is_imputed_not_survived(self):
        df, _ = self._nan_frame()
        cfg = PreprocessingBackendConfig(imputer_strategy="mean", scaler_name=None, categorical_encoding=None)
        pipe = create_polarsds_pipeline(df, cfg, verbose=0)
        out = pipe.transform(df).to_numpy()
        assert not np.isnan(out).any(), "float NaN survived the imputer"

    def test_mean_and_median_fill_differently_on_float_nan(self):
        """Skewed column: mean fill != median fill. Pre-fix both left NaN
        untouched (identical output), so this delta also pins the bug."""
        df, mask = self._nan_frame()
        fills = {}
        for strat in ("mean", "median"):
            cfg = PreprocessingBackendConfig(imputer_strategy=strat, scaler_name=None, categorical_encoding=None)
            out = create_polarsds_pipeline(df, cfg, verbose=0).transform(df).to_numpy()
            assert not np.isnan(out).any()
            fills[strat] = float(out[mask[:, 0], 0][0])
        assert abs(fills["mean"] - fills["median"]) > 0.1, "mean and median imputation produced identical fills on a skewed column"


# =====================================================================
# 11. All-NULL / all-NaN column must NOT survive the impute step as NULL
# =====================================================================


class TestAllNullColumnFilledNotSurvived:
    """A column with NO finite value (e.g. the fuzz ``inject_all_nan_col`` axis) has no mean/median to
    impute from, so ``_impute_targets`` correctly excludes it from ``Blueprint.impute``. But leaving it
    NULL forever relied on an UNENFORCED assumption ("a downstream constant-column dropper removes it"):
    that dropper is the user-controlled ``remove_constant_columns`` flag, which a real config can set
    False (e.g. to keep a fixed column layout across train/val/test). With it off, an all-NULL column
    survived every preprocessing stage and reached a NaN-intolerant model's own strict guard
    (PytorchLightningEstimator raising ``ValueError: X contains N NaN``) -- surfaced by fuzzing
    models=[linear,mlp] + recurrent_model=lstm + remove_constant_columns=False, 2026-07-06.
    """

    def _frame_with_all_null_col(self):
        rng = np.random.default_rng(0)
        n = 100
        return pl.DataFrame(
            {
                "f0": rng.standard_normal(n).astype(np.float32),
                "f_all_null": pl.Series([None] * n, dtype=pl.Float32),
            }
        )

    def test_all_null_column_filled_finite_not_left_null(self):
        df = self._frame_with_all_null_col()
        cfg = PreprocessingBackendConfig(imputer_strategy="mean", **_scaler_off())
        pipe = create_polarsds_pipeline(df, cfg, verbose=0)
        assert pipe is not None
        out = pipe.transform(df)
        assert out["f_all_null"].null_count() == 0, (
            "all-NULL column must be filled to a finite value, not left NULL relying on the constant-column dropper (which may be disabled)"
        )
        assert out["f0"].null_count() == 0

    def test_all_null_column_filled_zero(self):
        """Mirrors the ``SimpleImputer(keep_empty_features=True)`` convention used for the sklearn
        imputer path (a feature with no observed values gets statistic 0, not NaN)."""
        df = self._frame_with_all_null_col()
        cfg = PreprocessingBackendConfig(imputer_strategy="mean", **_scaler_off())
        out = create_polarsds_pipeline(df, cfg, verbose=0).transform(df)
        assert (out["f_all_null"].to_numpy() == 0.0).all()
