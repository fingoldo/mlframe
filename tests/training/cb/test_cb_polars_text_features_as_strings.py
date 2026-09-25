"""CatBoost never receives a polars Categorical column as a text feature.

CatBoost 1.2.10 corrupts memory on that input: building a Pool from such a frame crashed 6 of 6 fresh processes within 60
constructions, while the same column as ``String`` survived 6 of 6. mlframe's own CatBoost fallback tests died with an
access violation inside ``Pool.__init__`` in about one run in four.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import polars as pl
import pytest

from mlframe.training.cb._cb_polars_text import cb_text_features_as_strings, model_text_feature_names, text_columns_as_strings


def _frame(n: int = 50, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "num": rng.standard_normal(n).astype(np.float32),
        "true_cat": pl.Series(rng.choice(["r", "g", "b"], size=n)).cast(pl.Categorical),
        "skills_text": pl.Series(np.array([f"s_{i:04d}" for i in range(20)])[rng.integers(0, 20, size=n)]).cast(pl.Categorical),
    })


class TestCast:
    """Only text features that are categorical change; everything else is returned as the very same object."""

    def test_categorical_text_feature_becomes_string(self):
        out = text_columns_as_strings(_frame(), ["skills_text"])
        assert out.schema["skills_text"] == pl.String
        assert out.schema["true_cat"] == pl.Categorical, "cat features stay categorical: that is what CatBoost wants for them"

    def test_enum_text_feature_becomes_string(self):
        df = _frame().with_columns(pl.col("skills_text").cast(pl.String).cast(pl.Enum([f"s_{i:04d}" for i in range(20)])))
        assert text_columns_as_strings(df, ["skills_text"]).schema["skills_text"] == pl.String

    @pytest.mark.parametrize("text", [None, [], ["num"], ["absent"]])
    def test_nothing_to_cast_returns_the_same_frame(self, text):
        df = _frame()
        assert text_columns_as_strings(df, text) is df

    def test_pandas_is_left_alone(self):
        pdf = _frame().to_pandas()
        assert text_columns_as_strings(pdf, ["skills_text"]) is pdf


def test_the_fit_helper_covers_train_and_every_eval_frame():
    """The Pool path, the val Pool and the plain fit fallback all read these two places."""
    tr, va = _frame(seed=1), _frame(seed=2)
    fit_params = {"text_features": ["skills_text"], "eval_set": (va, np.zeros(va.height))}
    out = cb_text_features_as_strings("CatBoostClassifier", tr, fit_params)
    assert out.schema["skills_text"] == pl.String
    assert fit_params["eval_set"][0].schema["skills_text"] == pl.String


def test_non_catboost_models_are_not_touched():
    tr = _frame()
    fit_params = {"text_features": ["skills_text"]}
    assert cb_text_features_as_strings("LGBMClassifier", tr, fit_params) is tr


def test_predict_side_reads_the_models_own_text_features():
    """At predict time the model, not fit_params, says which columns are text."""

    class _Fitted:
        feature_names_ = ["num", "true_cat", "skills_text"]

        def get_text_feature_indices(self):
            return [2]

    assert model_text_feature_names(_Fitted()) == ["skills_text"]
    assert model_text_feature_names(object()) == []


def test_catboost_survives_the_input_that_used_to_crash_it(tmp_path):
    """Run in a subprocess: before the fix this killed the interpreter in 6 of 6 processes; a test must not take pytest
    down with it."""
    pytest.importorskip("catboost")
    script = tmp_path / "repro.py"
    script.write_text(textwrap.dedent('''
        import numpy as np, polars as pl, catboost
        from mlframe.training.cb._cb_polars_text import cb_text_features_as_strings
        for i in range(60):
            rng = np.random.default_rng(i)
            n = 500
            df = pl.DataFrame({
                "num": rng.standard_normal(n).astype(np.float32),
                "true_cat": pl.Series(rng.choice(["r", "g", "b"], size=n)).cast(pl.Categorical),
                "skills_text": pl.Series(np.array([f"s_{j:04d}" for j in range(200)])[rng.integers(0, 200, size=n)]).cast(pl.Categorical),
            })
            fit_params = {"cat_features": ["true_cat"], "text_features": ["skills_text"]}
            safe = cb_text_features_as_strings("CatBoostClassifier", df, fit_params)
            catboost.Pool(safe, label=np.arange(n) % 2, cat_features=["true_cat"], text_features=["skills_text"])
        print("survived")
    '''))
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and "survived" in proc.stdout, f"exit {proc.returncode}: {proc.stderr[-400:]}"


def test_real_catboost_fits_and_predicts_on_a_polars_frame_with_a_categorical_text_feature(tmp_path):
    """End to end through mlframe's trainer and predict fallback, with real CatBoost, in a subprocess."""
    pytest.importorskip("catboost")
    script = tmp_path / "e2e.py"
    script.write_text(textwrap.dedent('''
        import numpy as np, polars as pl, catboost
        from mlframe.training.trainer import _train_model_with_fallback
        from mlframe.training.cb._cb_pool import _predict_with_fallback

        def frame(seed, n):
            rng = np.random.default_rng(seed)
            return pl.DataFrame({
                "num": rng.standard_normal(n).astype(np.float32),
                "true_cat": pl.Series(rng.choice(["r", "g", "b"], size=n)).cast(pl.Categorical),
                "skills_text": pl.Series(np.array([f"s_{j:04d}" for j in range(200)])[rng.integers(0, 200, size=n)]).cast(pl.Categorical),
            })

        for rep in range(5):
            tr, va = frame(rep, 500), frame(100 + rep, 100)
            model = catboost.CatBoostClassifier(iterations=5, verbose=0, thread_count=1)
            _train_model_with_fallback(
                model=model, model_obj=model, model_type_name="CatBoostClassifier", train_df=tr,
                train_target=np.arange(tr.height) % 2,
                fit_params={"cat_features": ["true_cat"], "text_features": ["skills_text"], "eval_set": (va, np.arange(va.height) % 2)},
                verbose=False,
            )
            proba = np.asarray(_predict_with_fallback(model, va, method="predict_proba"))
            assert proba.shape == (va.height, 2) and np.isfinite(proba).all()
        print("survived")
    '''))
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0 and "survived" in proc.stdout, f"exit {proc.returncode}: {proc.stderr[-600:]}"
