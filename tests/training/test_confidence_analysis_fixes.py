"""Regression guards for three pre-existing confidence_analysis bugs
exposed by the fuzz-suite sweep on 2026-04-26 (Session 7 batch 5):

* c0056 (cb+hgb+xgb pl_nullable n=5000): CB Pool crashed on ``text_0``
  AND ``emb_0`` columns because confidence_analysis caller (HGB/XGB)
  doesn't pass text_features / embedding_features in fit_params, and
  the polars-side auto-detect didn't cover ``pl.Utf8`` / ``pl.List``
  / ``pl.Array`` / ``pl.Struct``.
* c0081 (hgb+lgb pandas n=300): degenerate confidence_targets — all
  identical → CB raises "All train targets are equal".
* c0088 (cb+hgb+xgb pandas n=600): CB CPU fallback hung past the
  pytest timeout because ``confidence_model_kwargs`` has no
  ``iterations`` cap → defaults to 1000 boosting rounds.

Each test exercises ``run_confidence_analysis`` directly with the
minimal trigger condition. Faster than running the full fuzz combo.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import polars as pl

from mlframe.training.trainer import run_confidence_analysis


# ---------------------------------------------------------------------------
# Bug 1: pl.Utf8 / pl.List / pl.Array / pl.Struct columns leak into CB Pool
# ---------------------------------------------------------------------------


def test_confidence_analysis_drops_polars_utf8_columns(caplog):
    """Polars Utf8 column without explicit text_features must be
    auto-dropped from the confidence Pool, not crash CB."""
    rng = np.random.default_rng(0)
    n = 200
    test_df = pl.DataFrame(
        {
            "x_num": pl.Series(rng.standard_normal(n).astype(np.float32)),
            "text_0": pl.Series([f"sentence_{i % 50}" for i in range(n)]),  # pl.Utf8
        }
    )
    test_target = rng.integers(0, 2, size=n).astype(np.int64)
    test_probs = rng.uniform(size=(n, 2))
    test_probs /= test_probs.sum(axis=1, keepdims=True)

    # Pass text_features=None to simulate the HGB/XGB call site where
    # fit_params doesn't carry text_features.
    result = run_confidence_analysis(
        test_df=test_df,
        test_target=test_target,
        test_probs=test_probs,
        cat_features=None,
        text_features=None,
        embedding_features=None,
        use_shap=False,
        verbose=False,
    )
    # No crash → the auto-detect dropped text_0; result is a fitted CB.
    assert result is not None


def test_confidence_analysis_drops_polars_list_columns():
    """Polars List(Float32) embedding column must be auto-dropped."""
    rng = np.random.default_rng(0)
    n = 200
    embeddings = [list(rng.standard_normal(8).astype(np.float32)) for _ in range(n)]
    test_df = pl.DataFrame(
        {
            "x_num": pl.Series(rng.standard_normal(n).astype(np.float32)),
            "emb_0": pl.Series(embeddings, dtype=pl.List(pl.Float32)),
        }
    )
    test_target = rng.integers(0, 2, size=n).astype(np.int64)
    test_probs = rng.uniform(size=(n, 2))
    test_probs /= test_probs.sum(axis=1, keepdims=True)

    # caller passes embedding_features=None (HGB/XGB site)
    result = run_confidence_analysis(
        test_df=test_df,
        test_target=test_target,
        test_probs=test_probs,
        cat_features=None,
        text_features=None,
        embedding_features=None,
        use_shap=False,
        verbose=False,
    )
    assert result is not None


def test_confidence_analysis_drops_polars_array_columns():
    """Polars Array(Float32, k) embedding column must be auto-dropped."""
    rng = np.random.default_rng(0)
    n = 200
    arr_data = rng.standard_normal((n, 4)).astype(np.float32)
    test_df = pl.DataFrame(
        {
            "x_num": pl.Series(rng.standard_normal(n).astype(np.float32)),
            "emb_arr": pl.Series(
                arr_data.tolist(),
                dtype=pl.Array(pl.Float32, 4),
            ),
        }
    )
    test_target = rng.integers(0, 2, size=n).astype(np.int64)
    test_probs = rng.uniform(size=(n, 2))
    test_probs /= test_probs.sum(axis=1, keepdims=True)

    result = run_confidence_analysis(
        test_df=test_df,
        test_target=test_target,
        test_probs=test_probs,
        cat_features=None,
        text_features=None,
        embedding_features=None,
        use_shap=False,
        verbose=False,
    )
    assert result is not None


def test_confidence_analysis_pandas_object_dtype_still_dropped():
    """Pre-existing pandas-side handler still works (regression guard
    for the earlier 2026-04-28 fix). Object-dtype text column must
    drop."""
    rng = np.random.default_rng(0)
    n = 200
    test_df = pd.DataFrame(
        {
            "x_num": rng.standard_normal(n).astype(np.float32),
            "text_obj": [f"s_{i % 30}" for i in range(n)],
        }
    )
    test_df["text_obj"] = test_df["text_obj"].astype(object)
    test_target = rng.integers(0, 2, size=n).astype(np.int64)
    test_probs = rng.uniform(size=(n, 2))
    test_probs /= test_probs.sum(axis=1, keepdims=True)

    result = run_confidence_analysis(
        test_df=test_df,
        test_target=test_target,
        test_probs=test_probs,
        cat_features=None,
        text_features=None,
        embedding_features=None,
        use_shap=False,
        verbose=False,
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Bug 2: degenerate confidence_targets (all equal)
# ---------------------------------------------------------------------------


def test_confidence_analysis_skips_when_targets_all_equal(caplog):
    """When all confidence_targets are identical, skip with WARN
    (CB rejects the fit otherwise: "All train targets are equal")."""
    rng = np.random.default_rng(0)
    n = 100
    test_df = pd.DataFrame(
        {
            "x_num": rng.standard_normal(n).astype(np.float32),
            "x_int": rng.integers(0, 5, size=n).astype(np.int32),
        }
    )
    test_target = np.zeros(n, dtype=np.int64)  # all class 0
    # All test_probs identical [1.0, 0.0] → confidence_targets = [1.0, 1.0, ...]
    test_probs = np.tile([[1.0, 0.0]], (n, 1))

    with caplog.at_level(logging.WARNING):
        result = run_confidence_analysis(
            test_df=test_df,
            test_target=test_target,
            test_probs=test_probs,
            use_shap=False,
            verbose=False,
        )
    assert result is None  # Skipped, no fit attempted
    assert any("confidence_targets are equal" in rec.message for rec in caplog.records), "expected the degenerate-target skip warning to fire"


def test_confidence_analysis_runs_when_targets_have_two_values():
    """The skip threshold is n_unique < 2; with exactly 2 unique
    values the fit proceeds normally."""
    rng = np.random.default_rng(0)
    n = 100
    test_df = pd.DataFrame(
        {
            "x_num": rng.standard_normal(n).astype(np.float32),
            "x_int": rng.integers(0, 5, size=n).astype(np.int32),
        }
    )
    test_target = rng.integers(0, 2, size=n).astype(np.int64)
    # Half-and-half: probs[:, true_class] is either 0.9 or 0.5 → 2 unique values
    test_probs = np.zeros((n, 2))
    high_conf_rows = rng.uniform(size=n) < 0.5
    test_probs[high_conf_rows, test_target[high_conf_rows]] = 0.9
    test_probs[~high_conf_rows, test_target[~high_conf_rows]] = 0.5
    test_probs[high_conf_rows, 1 - test_target[high_conf_rows]] = 0.1
    test_probs[~high_conf_rows, 1 - test_target[~high_conf_rows]] = 0.5

    result = run_confidence_analysis(
        test_df=test_df,
        test_target=test_target,
        test_probs=test_probs,
        use_shap=False,
        verbose=False,
    )
    assert result is not None  # Two unique values → fit proceeds


# ---------------------------------------------------------------------------
# Bug 3: bounded iterations prevent CPU hang
# ---------------------------------------------------------------------------


def test_confidence_analysis_default_iterations_cap():
    """The confidence model gets a sensible default iterations cap
    (200) and early_stopping_rounds (30) so its CPU fallback can't
    spin forever. Verified by checking the fitted model's params."""
    rng = np.random.default_rng(0)
    n = 200
    test_df = pd.DataFrame(
        {
            "x_num": rng.standard_normal(n).astype(np.float32),
            "x_int": rng.integers(0, 5, size=n).astype(np.int32),
        }
    )
    test_target = rng.integers(0, 2, size=n).astype(np.int64)
    test_probs = rng.uniform(size=(n, 2))
    test_probs /= test_probs.sum(axis=1, keepdims=True)

    # No confidence_model_kwargs → defaults must be applied.
    result = run_confidence_analysis(
        test_df=test_df,
        test_target=test_target,
        test_probs=test_probs,
        confidence_model_kwargs={},  # explicitly empty so we can inspect
        use_shap=False,
        verbose=False,
    )
    assert result is not None
    # The fitted CatBoostRegressor's iterations attribute reflects the cap.
    assert result.get_params().get("iterations") == 200
    assert result.get_params().get("early_stopping_rounds") == 30


def test_confidence_analysis_caller_can_override_iterations_cap():
    """If the caller explicitly passes iterations / early_stopping_rounds,
    the explicit value wins over the new defaults."""
    rng = np.random.default_rng(0)
    n = 200
    test_df = pd.DataFrame(
        {
            "x_num": rng.standard_normal(n).astype(np.float32),
            "x_int": rng.integers(0, 5, size=n).astype(np.int32),
        }
    )
    test_target = rng.integers(0, 2, size=n).astype(np.int64)
    test_probs = rng.uniform(size=(n, 2))
    test_probs /= test_probs.sum(axis=1, keepdims=True)

    result = run_confidence_analysis(
        test_df=test_df,
        test_target=test_target,
        test_probs=test_probs,
        confidence_model_kwargs={"iterations": 50, "early_stopping_rounds": 10},
        use_shap=False,
        verbose=False,
    )
    assert result is not None
    assert result.get_params().get("iterations") == 50
    assert result.get_params().get("early_stopping_rounds") == 10


def test_confidence_analysis_completes_within_time_budget():
    """End-to-end timing guard: with the default iteration cap the
    confidence fit on 200 rows / 5 features must complete well under
    30 seconds. If this regresses, suspect the iteration default."""
    rng = np.random.default_rng(0)
    n = 200
    test_df = pd.DataFrame({f"x_{i}": rng.standard_normal(n).astype(np.float32) for i in range(5)})
    test_target = rng.integers(0, 2, size=n).astype(np.int64)
    test_probs = rng.uniform(size=(n, 2))
    test_probs /= test_probs.sum(axis=1, keepdims=True)

    t0 = time.perf_counter()
    result = run_confidence_analysis(
        test_df=test_df,
        test_target=test_target,
        test_probs=test_probs,
        use_shap=False,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    assert result is not None
    assert elapsed < 30.0, f"confidence_analysis took {elapsed:.1f}s on 200x5 — iteration cap default may have been removed."


def test_confidence_analysis_accepts_iterations_synonym():
    """Regression (fuzz c0002, 2026-05-31): a caller passing the CatBoost
    iteration-budget under a synonym spelling (``n_estimators`` /
    ``num_boost_round`` / ``num_trees``) must not collide with the
    function's internal ``iterations`` default.

    Pre-fix ``run_confidence_analysis`` did
    ``confidence_model_kwargs.setdefault("iterations", 200)`` unconditionally;
    when the caller already supplied ``n_estimators`` CatBoost raised
    ``only one of the parameters iterations, n_estimators, num_boost_round,
    num_trees should be initialized`` and the whole confidence pass crashed.
    The guard now skips the default when ANY synonym is already present.
    """
    rng = np.random.default_rng(0)
    n = 200
    test_df = pd.DataFrame(
        {
            "x_num": rng.standard_normal(n).astype(np.float32),
            "x_int": rng.integers(0, 5, size=n).astype(np.int32),
        }
    )
    test_target = rng.integers(0, 2, size=n).astype(np.int64)
    test_probs = rng.uniform(size=(n, 2))
    test_probs /= test_probs.sum(axis=1, keepdims=True)

    for synonym in ("n_estimators", "num_boost_round", "num_trees"):
        result = run_confidence_analysis(
            test_df=test_df,
            test_target=test_target,
            test_probs=test_probs,
            confidence_model_kwargs={synonym: 25, "max_depth": 4},
            use_shap=False,
            verbose=False,
        )
        assert result is not None, f"confidence fit crashed on synonym {synonym!r}"
        # The caller-supplied budget wins uncontested; no `iterations` key
        # was injected to shadow it.
        params = result.get_params()
        assert params.get(synonym) == 25 or params.get("iterations") == 25, f"caller {synonym}=25 was not honoured; params={params}"
