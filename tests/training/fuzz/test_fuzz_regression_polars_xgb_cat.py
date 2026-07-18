"""Regression sensors for fuzz-suite polars / XGBoost categorical crashes.

Promoted from ``test_fuzz_suite.py`` n1000 combos:

* XGBoost ``cat_container.h: Found a category not in the training set`` -- a val
  row carried a categorical level absent from the train DMatrix category universe
  (c0011).
* polars-ds ``InvalidOperationError: expected List data type ... got Float32`` --
  ``imputer_strategy='mode'`` routed through polars-ds's ``mode().list.first()``
  which breaks on polars versions where ``Series.mode()`` is non-List (c0074/...).
* polars-ds one-hot / ordinal rejecting ``pl.Enum`` columns
  ("not string/categorical types") on ``input_type='polars_enum'`` combos.
"""

from __future__ import annotations

import numpy as np
import pytest

pl = pytest.importorskip("polars")


class TestXgbEvalCategoricalAlignment:
    """``xgb_shim._align_eval_categoricals`` makes the val frame's categorical
    dtype match train's so XGBoost's enable_categorical eval doesn't reject a
    val-only category (fuzz c0011)."""

    def test_polars_val_enum_widened_to_train(self):
        """Polars val enum widened to train."""
        from mlframe.training.xgb_shim import _align_eval_categoricals

        tr_enum = pl.Enum(["alpha"])
        va_enum = pl.Enum(["alpha", "beta"])
        tr = pl.DataFrame({"c": pl.Series(["alpha"] * 5).cast(tr_enum)})
        va = pl.DataFrame({"c": pl.Series(["alpha", "beta", "alpha", "beta", "alpha"]).cast(va_enum)})
        out = _align_eval_categoricals(tr, va)
        # val 'c' now uses the TRAIN enum; 'beta' (OOV) becomes null, not a crash.
        assert out.schema["c"] == tr_enum
        assert out["c"].null_count() == 2

    def test_xgb_fit_with_train_absent_val_category(self):
        """Xgb fit with train absent val category."""
        pytest.importorskip("xgboost")
        from mlframe.training.xgb_shim import XGBClassifierWithDMatrixReuse

        enum = pl.Enum(["alpha", "beta"])
        n = 80
        tr = pl.DataFrame({"c": pl.Series(["alpha"] * n).cast(enum), "x": np.random.randn(n).astype("float32")})
        # val carries 'beta', absent from train rows -- shared enum must keep it safe.
        va = pl.DataFrame({"c": pl.Series(["alpha"] * 20 + ["beta"] * 10).cast(enum), "x": np.random.randn(30).astype("float32")})
        ytr = np.random.randint(0, 2, n)
        yva = np.random.randint(0, 2, 30)
        m = XGBClassifierWithDMatrixReuse(
            n_estimators=5,
            enable_categorical=True,
            tree_method="hist",
        )
        m.fit(tr, ytr, eval_set=[(va, yva)])
        assert m.predict(va).shape[0] == 30


class TestPolarsdsModeImpute:
    """``create_polarsds_pipeline`` must impute ``mode`` natively (polars-ds's
    ``mode().list.first()`` breaks on current polars) (fuzz c0074/...)."""

    def test_mode_impute_pipeline_builds_and_fills(self):
        """Mode impute pipeline builds and fills."""
        from mlframe.training.configs import PreprocessingBackendConfig
        from mlframe.training.pipeline import create_polarsds_pipeline

        pytest.importorskip("polars_ds")
        n = 200
        rng = np.random.default_rng(0)
        v = rng.standard_normal(n).astype("float32")
        v[::8] = np.nan
        df = pl.DataFrame({"num_0": pl.Series(v), "cat_0": pl.Series([["A", "B", "C"][i % 3] for i in range(n)]).cast(pl.Enum(["A", "B", "C"]))})
        cfg = PreprocessingBackendConfig(
            prefer_polarsds=True,
            scaler_name=None,
            imputer_strategy="mode",
            categorical_encoding=None,
            skip_categorical_encoding=True,
        )
        pipe = create_polarsds_pipeline(df, cfg, verbose=0)
        assert pipe is not None
        out = pipe.transform(df)
        assert out["num_0"].null_count() == 0


class TestPolarsdsEnumEncode:
    """``create_polarsds_pipeline`` casts ``pl.Enum`` to Categorical before the
    polars-ds encoder (which rejects Enum) so polars_enum combos encode."""

    @pytest.mark.parametrize("encoding", ["onehot", "ordinal"])
    def test_enum_columns_encode(self, encoding):
        """Enum columns encode."""
        from mlframe.training.configs import PreprocessingBackendConfig
        from mlframe.training.pipeline import create_polarsds_pipeline

        pytest.importorskip("polars_ds")
        n = 150
        df = pl.DataFrame(
            {
                "num_0": pl.Series(np.random.default_rng(1).standard_normal(n).astype("float32")),
                "cat_0": pl.Series([["A", "B", "C"][i % 3] for i in range(n)]).cast(pl.Enum(["A", "B", "C"])),
            }
        )
        cfg = PreprocessingBackendConfig(
            prefer_polarsds=True,
            scaler_name=None,
            imputer_strategy=None,
            categorical_encoding=encoding,
            skip_categorical_encoding=False,
        )
        pipe = create_polarsds_pipeline(df, cfg, verbose=0)
        assert pipe is not None
        out = pipe.transform(df)
        if encoding == "onehot":
            assert any(c.startswith("cat_0_") for c in out.columns)
        else:
            assert out.schema["cat_0"].is_numeric()


class TestRankerCategoricalRobustness:
    """LTR native rankers: predict-time category alignment + CatBoost None-fill
    (fuzz c0030 / c0141)."""

    def _ranker_frames(self, train_cats, val_cats):
        """Ranker frames."""
        import pandas as pd

        rng = np.random.default_rng(0)
        ntr, nva = len(train_cats), len(val_cats)
        Xtr = pd.DataFrame({"x": rng.standard_normal(ntr).astype("float32"), "c": train_cats})
        Xva = pd.DataFrame({"x": rng.standard_normal(nva).astype("float32"), "c": val_cats})
        ytr = rng.integers(0, 3, ntr)
        gtr = np.array([ntr // 2, ntr - ntr // 2])
        return Xtr, ytr, gtr, Xva

    def test_lgb_ranker_predict_on_object_cat_frame(self):
        """Lgb ranker predict on object cat frame."""
        pytest.importorskip("lightgbm")
        from mlframe.training.ranking.ranking import _fit_lgb_ranker, predict_ranker_scores

        Xtr, ytr, _gtr, Xva = self._ranker_frames(
            [["A", "B", "C"][i % 3] for i in range(60)],
            [["A", "B"][i % 2] for i in range(40)],
        )
        gids_tr = np.repeat(np.arange(2), [30, 30])
        gids_va = np.repeat(np.arange(2), [20, 20])
        fitted = _fit_lgb_ranker(
            Xtr,
            ytr,
            gids_tr,
            Xva,
            ytr[:40],
            gids_va,
            obj_kwargs={"objective": "lambdarank"},
            model_kwargs={"n_estimators": 5},
            cat_features=["c"],
            early_stopping_rounds=None,
            verbose=False,
        )
        # Predict on a RAW object-dtype frame (categories differ) must not raise.
        scores = predict_ranker_scores(fitted, Xva)
        assert scores.shape[0] == 40

    def test_xgb_ranker_predict_on_object_cat_frame(self):
        """Xgb ranker predict on object cat frame."""
        pytest.importorskip("xgboost")
        from mlframe.training.ranking.ranking import _fit_xgb_ranker, predict_ranker_scores

        Xtr, ytr, gids_tr, Xva = self._ranker_frames(
            [["A", "B", "C"][i % 3] for i in range(60)],
            [["A", "B"][i % 2] for i in range(40)],
        )
        gids_tr = np.repeat(np.arange(2), [30, 30])
        gids_va = np.repeat(np.arange(2), [20, 20])
        fitted = _fit_xgb_ranker(
            Xtr,
            ytr,
            gids_tr,
            Xva,
            ytr[:40],
            gids_va,
            obj_kwargs={"objective": "rank:ndcg"},
            model_kwargs={"n_estimators": 5},
            cat_features=["c"],
            early_stopping_rounds=None,
            verbose=False,
        )
        scores = predict_ranker_scores(fitted, Xva)
        assert scores.shape[0] == 40

    def test_cb_ranker_fits_with_none_in_object_cat(self):
        """Cb ranker fits with none in object cat."""
        pytest.importorskip("catboost")
        from mlframe.training.ranking.ranking import _fit_cb_ranker, predict_ranker_scores

        Xtr, ytr, gids_tr, Xva = self._ranker_frames(
            [["A", "B", None][i % 3] for i in range(60)],
            [["A", None][i % 2] for i in range(40)],
        )
        gids_tr = np.repeat(np.arange(2), [30, 30])
        # Pre-fix: CatBoost Pool raised "must be real number, not NoneType".
        fitted = _fit_cb_ranker(
            Xtr,
            ytr,
            gids_tr,
            None,
            None,
            None,
            obj_kwargs={"loss_function": "YetiRank"},
            model_kwargs={"iterations": 5},
            cat_features=["c"],
            early_stopping_rounds=None,
            verbose=False,
        )
        scores = predict_ranker_scores(fitted, Xva)
        assert scores.shape[0] == 40
