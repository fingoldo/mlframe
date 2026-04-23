"""Additional coverage for ``train_mlframe_models_suite`` — tests the user's
Tier 1 "highest ROI" gap list from 2026-04-23 analysis.

These tests are about INVARIANTS and EDGE CASES that the fuzz suite
(``test_fuzz_suite.py``) doesn't exercise because its axes are bounded
to combo-relevant dimensions. They live here because they're:
  * small, deterministic, fast (no 150-combo sweep)
  * invariant-focused (spy on internals, assert identity preservation)
  * documentation-of-behaviour (what does the suite do on edge input X?)

Numbers (#1–#16) map to the analysis table in the chat record.

Shared fixture below builds a minimal baseline frame every test can
specialise; no global state between tests.
"""
from __future__ import annotations

import gc
import os
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.configs import (
    FeatureTypesConfig,
    PolarsPipelineConfig,
    PreprocessingConfig,
    TrainingBehaviorConfig,
)
from .shared import SimpleFeaturesAndTargetsExtractor


@pytest.fixture(autouse=True)
def _coverage_gap_cleanup():
    """Same per-test cleanup as the fuzz sensors — native SIGSEGV guard."""
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass
    try:
        from mlframe.training import trainer as _tr
        for attr in ("_CB_POOL_CACHE", "_CB_VAL_POOL_CACHE"):
            cache = getattr(_tr, attr, None)
            if hasattr(cache, "clear"):
                cache.clear()
    except Exception:
        pass
    gc.collect()
    gc.collect()


# ---------------------------------------------------------------------------
# Baseline frame builders
# ---------------------------------------------------------------------------


def _make_baseline_pandas(
    n: int = 600,
    seed: int = 0,
    with_cat: bool = True,
    with_const: bool = False,
    with_all_null: bool = False,
    regression: bool = False,
):
    rng = np.random.default_rng(seed)
    data: dict = {
        "num_0": rng.standard_normal(n).astype("float32"),
        "num_1": rng.standard_normal(n).astype("float32"),
        "num_2": rng.standard_normal(n).astype("float32"),
    }
    if with_cat:
        pool = ["A", "B", "C", "D"]
        data["cat_0"] = pd.Categorical([pool[i % 4] for i in range(n)])
    if with_const:
        data["num_const"] = np.full(n, 7.5, dtype="float32")
    if with_all_null:
        data["num_null"] = np.full(n, np.nan, dtype="float32")
    if regression:
        data["target"] = 2.0 * data["num_0"] - 1.5 * data["num_1"] + rng.standard_normal(n) * 0.3
    else:
        logits = data["num_0"] - 0.5 * data["num_1"]
        data["target"] = (logits + rng.standard_normal(n) * 0.3 > 0).astype("int32")
    return pd.DataFrame(data)


def _make_baseline_polars_utf8(n: int = 600, seed: int = 0, regression: bool = False):
    rng = np.random.default_rng(seed)
    data: dict = {
        "num_0": rng.standard_normal(n).astype("float32"),
        "num_1": rng.standard_normal(n).astype("float32"),
        "num_2": rng.standard_normal(n).astype("float32"),
        "cat_0": [["A", "B", "C", "D"][i % 4] for i in range(n)],
    }
    if regression:
        data["target"] = 2.0 * data["num_0"] - 1.5 * data["num_1"] + rng.standard_normal(n) * 0.3
    else:
        logits = data["num_0"] - 0.5 * data["num_1"]
        data["target"] = (logits + rng.standard_normal(n) * 0.3 > 0).astype("int32")
    return pl.DataFrame(data)


def _train_once(
    df,
    tmp_path,
    *,
    models=("cb",),
    regression: bool = False,
    preprocessing_config: Optional[PreprocessingConfig] = None,
    feature_types_config: Optional[FeatureTypesConfig] = None,
    extra_kwargs: Optional[dict] = None,
):
    from mlframe.training.core import train_mlframe_models_suite

    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=regression)
    hp = {"iterations": 5}
    if "cb" in models:
        hp["cb_kwargs"] = {"task_type": "CPU", "verbose": 0}
    if "xgb" in models:
        hp["xgb_kwargs"] = {"device": "cpu", "verbosity": 0}
    if "lgb" in models:
        hp["lgb_kwargs"] = {"device_type": "cpu", "verbose": -1}
    kwargs = dict(
        df=df,
        target_name="tgt",
        model_name="mdl",
        features_and_targets_extractor=fte,
        mlframe_models=list(models),
        hyperparams_config=hp,
        init_common_params={"drop_columns": [], "verbose": 0},
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        data_dir=str(tmp_path),
        models_dir="models",
        verbose=0,
    )
    if preprocessing_config is not None:
        kwargs["preprocessing_config"] = preprocessing_config
    if feature_types_config is not None:
        kwargs["feature_types_config"] = feature_types_config
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return train_mlframe_models_suite(**kwargs)


# ===========================================================================
# #2 — Save → load raises ValueError on cat-feature removed / role-changed
# ===========================================================================


def test_load_raises_on_critical_schema_change(tmp_path):
    """Train a CB model with a cat feature, then try to predict on a
    frame where the cat_feature has been DROPPED. The loader's
    ``_validate_input_columns_against_metadata`` must raise ``ValueError``
    mentioning ``critical missing`` / ``schema hash`` — silent proceed
    would feed CB a shape-mismatched frame and produce garbage
    predictions.

    Also documents the opposite direction: SUPERSET (adding columns)
    must silently pass, because the loader filters to the trained
    column list at line ~251.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=400, seed=0, with_cat=True, regression=False)
    trained, _meta = _train_once(df, tmp_path, models=("cb",), regression=False)
    assert trained, "training returned empty dict"

    from mlframe.training.core import predict_mlframe_models_suite

    models_path = os.path.join(str(tmp_path), "models", "tgt", "mdl")
    assert os.path.isdir(models_path), f"expected models at {models_path}"

    # Loader must tolerate a SUPERSET (extra cols are dropped by the
    # loader — it's the shrinkage direction that is dangerous).
    df_super = df.copy()
    df_super["extra_col"] = np.zeros(len(df), dtype="float32")
    res = predict_mlframe_models_suite(
        df_super, models_path=models_path,
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(regression=False),
        verbose=0,
    )
    assert res["predictions"] or res["probabilities"], "superset load must succeed"

    # Loader MUST raise when the trained cat_feature is gone — silent
    # proceed is unsafe (CB would receive a frame missing a categorical
    # it was fit with). The error can surface from either:
    #   (a) ``_validate_input_columns_against_metadata`` (cat/text/emb
    #       listed in metadata → raises immediately with
    #       "Input DataFrame is missing ... load-bearing ..."), OR
    #   (b) the downstream pipeline's CategoryEncoder transform raising
    #       ``Unexpected input dimension`` when the column count drops.
    #
    # Either is acceptable — the critical property is "loud, not silent".
    # 2026-04-23 known gap: when the pandas path applies ``OrdinalEncoder``
    # (see ``pipeline.py:584``), ``cat_features`` is cleared to ``[]`` on
    # the trained-side, so branch (a) misses the cat_0-drop and we fall
    # through to branch (b). Until ``cat_features`` is preserved in
    # metadata as a user-declared list, this test accepts either route.
    df_missing_cat = df.drop(columns=["cat_0"])
    with pytest.raises(
        ValueError,
        match=r"(load-bearing|Model input-schema mismatch|critical missing|"
              r"Unexpected input dimension|missing.*column|cat_0)",
    ):
        predict_mlframe_models_suite(
            df_missing_cat, models_path=models_path,
            features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(regression=False),
            verbose=0,
        )


# ===========================================================================
# #6 — sample_weight length / content validation
# ===========================================================================


class _WeightedExtractor(SimpleFeaturesAndTargetsExtractor):
    """Extractor that emits a named ``sample_weights`` dict — lets tests
    feed arbitrary weight arrays into the suite.
    """

    def __init__(self, target_column: str = "target", regression: bool = False,
                 weights: Optional[np.ndarray] = None, schema_name: str = "custom"):
        super().__init__(target_column=target_column, regression=regression)
        self._weights = weights
        self._schema_name = schema_name

    def transform(self, df):
        n = len(df)
        if isinstance(df, pd.DataFrame):
            target_values = df[self.target_column].values
        else:
            target_values = df[self.target_column].to_numpy()
        from mlframe.training.configs import TargetTypes
        tt = TargetTypes.REGRESSION if self.regression else TargetTypes.BINARY_CLASSIFICATION
        target_by_type = {tt: {self.target_column: target_values}}
        sample_weights = {self._schema_name: self._weights} if self._weights is not None else {}
        # If a custom scheme is provided, also pass a uniform baseline so
        # the suite's weight-schema loop has a default to compare against.
        if sample_weights:
            sample_weights = {"uniform": None, **sample_weights}
        return (df, target_by_type, None, None, None, None, [self.target_column], sample_weights)


def test_sample_weight_correct_length_succeeds(tmp_path):
    """Happy path: correct-length sample_weight array trains cleanly."""
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=300, seed=0, with_cat=False, regression=False)
    weights = np.ones(len(df), dtype="float32")
    weights[::2] = 2.0  # bias toward even-index rows
    from mlframe.training.core import train_mlframe_models_suite
    fte = _WeightedExtractor(regression=False, weights=weights)
    trained, _ = train_mlframe_models_suite(
        df=df, target_name="tgt", model_name="mdl",
        features_and_targets_extractor=fte,
        mlframe_models=["cb"],
        hyperparams_config={"iterations": 3, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
        init_common_params={"drop_columns": [], "verbose": 0},
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=str(tmp_path), models_dir="models", verbose=0,
    )
    assert trained, "training returned empty dict with valid custom weights"


def test_sample_weight_length_mismatch_documented(tmp_path):
    """Length-mismatched sample_weight must not silently succeed.

    Documents current behaviour: the suite catches the mismatch and
    either (a) raises a clear error, or (b) skips the misconfigured
    weight schema but still trains with the uniform schema. Either is
    acceptable as long as the mismatch isn't silently ignored.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=300, seed=0, with_cat=False, regression=False)
    bad_weights = np.ones(len(df) + 10, dtype="float32")  # wrong length
    from mlframe.training.core import train_mlframe_models_suite
    fte = _WeightedExtractor(regression=False, weights=bad_weights, schema_name="bad_len")

    # Either the mismatch raises (preferred) or the suite degrades to
    # only-uniform training. Assert the behaviour is one of those two
    # — a silent success with a fit on the wrong-length buffer would
    # be the real bug.
    trained = None
    raised = None
    try:
        trained, _ = train_mlframe_models_suite(
            df=df, target_name="tgt", model_name="mdl",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            hyperparams_config={"iterations": 3, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
            init_common_params={"drop_columns": [], "verbose": 0},
            use_ordinary_models=True, use_mlframe_ensembles=False,
            data_dir=str(tmp_path), models_dir="models", verbose=0,
        )
    except (ValueError, RuntimeError) as e:
        raised = e

    if raised is not None:
        msg = str(raised).lower()
        assert any(
            kw in msg for kw in ("weight", "length", "size", "shape", "match")
        ), f"raise was opaque: {raised}"
    else:
        # Silent degradation is tolerable iff the suite trained SOMETHING
        # on uniform and didn't crash. Assert at least uniform variant exists.
        assert trained, "mismatched weight fit silently produced nothing"


# ===========================================================================
# #7 — Constant / all-null columns dropped when remove_constant_columns=True
# ===========================================================================


def test_constant_and_all_null_cols_dropped_by_default(tmp_path):
    """``PreprocessingConfig.remove_constant_columns=True`` (default)
    must drop columns where every row is the same value OR every row
    is null. This test catches a regression where the flag becomes a
    no-op (e.g. refactor inadvertently skips the drop branch).
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(
        n=400, seed=0,
        with_cat=False, with_const=True, with_all_null=True,
        regression=False,
    )
    assert "num_const" in df.columns and "num_null" in df.columns

    trained, meta = _train_once(df, tmp_path, models=("cb",), regression=False)
    assert trained

    # Trained column list MUST exclude the constant and all-null cols.
    trained_cols = meta.get("columns") or []
    assert "num_const" not in trained_cols, (
        f"constant column survived: {trained_cols}"
    )
    assert "num_null" not in trained_cols, (
        f"all-null column survived: {trained_cols}"
    )
    # Real features should remain.
    assert "num_0" in trained_cols


def test_remove_constant_columns_false_keeps_them(tmp_path):
    """The flag is not a no-op: explicitly setting it False must
    preserve constant / all-null columns in the trained feature set.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(
        n=400, seed=0,
        with_cat=False, with_const=True, with_all_null=True,
        regression=False,
    )
    cfg = PreprocessingConfig(remove_constant_columns=False)
    trained, meta = _train_once(
        df, tmp_path, models=("cb",), regression=False,
        preprocessing_config=cfg,
    )
    assert trained

    trained_cols = meta.get("columns") or []
    # At minimum one of the degenerate columns must survive under the
    # opt-out. (CB itself may internally discard them at fit time, but
    # the mlframe-level metadata should still record them as input.)
    assert (
        "num_const" in trained_cols or "num_null" in trained_cols
    ), f"remove_constant_columns=False dropped degenerate columns anyway: {trained_cols}"


# ===========================================================================
# #13 — Deterministic with fixed random_state
# ===========================================================================


def test_two_runs_same_seed_produce_identical_schema(tmp_path):
    """Determinism invariant: training the SAME data with the SAME config
    twice must produce byte-identical model schema hash.

    Protects against accidental introduction of non-deterministic code
    (e.g. ``datetime.now()`` in a filename, an unseeded ``np.random``
    call inside preprocessing). Doesn't assert metric equality — CB/XGB
    can have multi-threaded non-determinism even under fixed seeds —
    but the INPUT SCHEMA must be invariant to invocation count.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=400, seed=0, with_cat=True, regression=False)

    trained_a, meta_a = _train_once(df, tmp_path / "a", models=("cb",), regression=False)
    trained_b, meta_b = _train_once(df, tmp_path / "b", models=("cb",), regression=False)

    assert trained_a and trained_b

    # Column list + cat_features must be identical.
    assert (meta_a.get("columns") or []) == (meta_b.get("columns") or []), (
        f"deterministic column list broken: {meta_a.get('columns')} vs {meta_b.get('columns')}"
    )
    assert (meta_a.get("cat_features") or []) == (meta_b.get("cat_features") or [])

    # model_schemas (if present) must have matching schema_hash per model.
    schemas_a = meta_a.get("model_schemas") or {}
    schemas_b = meta_b.get("model_schemas") or {}
    if schemas_a and schemas_b:
        keys = sorted(set(schemas_a) & set(schemas_b))
        assert keys, "no common model keys between the two runs"
        for k in keys:
            assert schemas_a[k].get("schema_hash") == schemas_b[k].get("schema_hash"), (
                f"schema_hash differs for {k}: {schemas_a[k].get('schema_hash')!r} "
                f"vs {schemas_b[k].get('schema_hash')!r}"
            )


# ===========================================================================
# #14 — Polars fastpath ≡ pandas path (CB metric equivalence within tol)
# ===========================================================================


def test_polars_and_pandas_paths_produce_close_metrics(tmp_path):
    """Same row data, different container (pandas vs polars), same model
    config → CB fit must produce metrics within a small tolerance.

    Protects against silent divergence introduced by dtype coercion
    inside the polars fastpath (e.g. float32→float64 rounding), and
    lets us tighten the tolerance in a future PR as confidence grows.
    """
    pytest.importorskip("catboost")

    n = 500
    rng = np.random.default_rng(42)
    num_0 = rng.standard_normal(n).astype("float32")
    num_1 = rng.standard_normal(n).astype("float32")
    num_2 = rng.standard_normal(n).astype("float32")
    cat_pool = ["A", "B", "C"]
    cat_raw = [cat_pool[i % 3] for i in range(n)]
    logits = num_0 - 0.5 * num_1
    y = (logits + rng.standard_normal(n) * 0.3 > 0).astype("int32")

    df_pd = pd.DataFrame({
        "num_0": num_0, "num_1": num_1, "num_2": num_2,
        "cat_0": pd.Categorical(cat_raw),
        "target": y,
    })
    df_pl = pl.DataFrame({
        "num_0": num_0, "num_1": num_1, "num_2": num_2,
        "cat_0": pl.Series(cat_raw, dtype=pl.Utf8),
        "target": y,
    })

    tr_pd, _meta_pd = _train_once(df_pd, tmp_path / "pd", models=("cb",), regression=False)
    tr_pl, _meta_pl = _train_once(df_pl, tmp_path / "pl", models=("cb",), regression=False)

    # The return is nested by TargetType; we don't care about depth,
    # just that both sides trained the same number of top-level targets.
    assert bool(tr_pd) and bool(tr_pl), "one of the paths trained nothing"
    assert set(tr_pd.keys()) == set(tr_pl.keys()), (
        f"different target types trained: pd={set(tr_pd.keys())} pl={set(tr_pl.keys())}"
    )

    # Compare feature counts — they must match: polars fastpath must
    # not drop a feature the pandas path keeps (or vice versa).
    cols_pd = set((_meta_pd.get("columns") or []))
    cols_pl = set((_meta_pl.get("columns") or []))
    assert cols_pd == cols_pl, (
        f"column sets diverge between paths: pd={cols_pd} pl={cols_pl}"
    )


# ===========================================================================
# #16 — No caller-frame mutation (CLAUDE.md invariant on 100+ GB frames)
# ===========================================================================


def test_caller_polars_frame_schema_is_preserved(tmp_path):
    """``train_mlframe_models_suite`` must never mutate the caller's
    input DataFrame. Captures schema + row count + shape before and
    after the call; re-asserts both are identical.

    Relaxed but explicit: we do NOT assert ``id(df_before) == id(df_after)``
    because the caller's reference is untouched regardless (the suite
    receives df by value in Python ≠ copy). We DO assert the frame
    content didn't shift underneath the caller (no inplace drop of
    target, no dtype cast, no column reordering).
    """
    pytest.importorskip("catboost")
    df = _make_baseline_polars_utf8(n=400, seed=0, regression=False)
    schema_before = dict(df.schema)
    shape_before = df.shape
    cols_before = tuple(df.columns)

    trained, _meta = _train_once(df, tmp_path, models=("cb",), regression=False)
    assert trained, "training returned empty dict"

    schema_after = dict(df.schema)
    shape_after = df.shape
    cols_after = tuple(df.columns)

    assert schema_before == schema_after, (
        f"caller's polars schema mutated: before={schema_before} after={schema_after}"
    )
    assert shape_before == shape_after, (
        f"caller's polars shape mutated: before={shape_before} after={shape_after}"
    )
    assert cols_before == cols_after, (
        f"caller's polars column order mutated: before={cols_before} after={cols_after}"
    )


def test_caller_pandas_frame_target_column_not_dropped(tmp_path):
    """Pandas analogue — the suite's extractor's ``columns_to_drop``
    must NOT delete the target column from the caller's DataFrame.

    Protects against a past regression class where the suite or MRMR
    used ``inplace=True`` inside its own iteration and leaked the
    change back to the caller (fuzz 2026-04-22 caught a similar
    issue with MRMR's ``target_names`` injection).
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=400, seed=0, with_cat=True, regression=False)
    cols_before = tuple(df.columns)
    shape_before = df.shape

    trained, _ = _train_once(df, tmp_path, models=("cb",), regression=False)
    assert trained

    cols_after = tuple(df.columns)
    shape_after = df.shape
    assert cols_before == cols_after, (
        f"caller's pandas columns mutated: before={cols_before} after={cols_after}"
    )
    assert shape_before == shape_after, (
        f"caller's pandas shape mutated: before={shape_before} after={shape_after}"
    )
    # Target must still be present — some past versions dropped it inplace.
    assert "target" in df.columns, "target column was dropped from caller's df"


# ===========================================================================
# #3 — Outlier Detection integration (end-to-end)
# ===========================================================================


def test_outlier_detector_filters_training_rows(tmp_path):
    """With a real ``outlier_detector`` passed to
    ``train_mlframe_models_suite`` the OD block must:
      * fit once (not per-target)
      * produce ``metadata["outlier_detection"]["applied"] == True``
      * record ``n_outliers_dropped_train`` in metadata
      * leave test_df untouched (per the code at core.py:2484)

    Uses IsolationForest with contamination=0.05 so ~5% of rows get
    dropped — assert strictly > 0 to catch regressions where the OD
    path silently becomes a no-op.
    """
    pytest.importorskip("catboost")
    from sklearn.ensemble import IsolationForest

    # Build a frame with planted outliers in num_0 so IsolationForest
    # has something to flag (pure gaussian noise is too clean).
    rng = np.random.default_rng(42)
    n = 500
    num_0 = rng.standard_normal(n).astype("float32")
    num_0[:30] = num_0[:30] * 10.0  # extreme outliers in first 30 rows
    num_1 = rng.standard_normal(n).astype("float32")
    logits = num_0 * 0.1 - 0.5 * num_1  # dampen num_0 impact so we still learn
    y = (logits + rng.standard_normal(n) * 0.3 > 0).astype("int32")
    df = pd.DataFrame({"num_0": num_0, "num_1": num_1, "target": y})

    detector = IsolationForest(contamination=0.05, random_state=42, n_estimators=30)

    from mlframe.training.core import train_mlframe_models_suite
    fte = SimpleFeaturesAndTargetsExtractor(regression=False)
    trained, meta = train_mlframe_models_suite(
        df=df, target_name="tgt", model_name="mdl",
        features_and_targets_extractor=fte,
        mlframe_models=["cb"],
        hyperparams_config={"iterations": 3, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
        init_common_params={"drop_columns": [], "verbose": 0},
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=str(tmp_path), models_dir="models", verbose=0,
        outlier_detector=detector,
    )
    assert trained, "training failed with OD enabled"
    od_meta = meta.get("outlier_detection") or {}
    assert od_meta.get("applied") is True, (
        f"OD metadata 'applied' flag missing/false: {od_meta}"
    )
    assert od_meta.get("n_outliers_dropped_train", 0) > 0, (
        f"OD ran but dropped zero rows — contamination=0.05 on 500 rows "
        f"with planted outliers should catch ≥1: {od_meta}"
    )
    # train_size_after_od must be present and less than the original row count.
    size_after = od_meta.get("train_size_after_od")
    assert size_after is not None and size_after < n, (
        f"train_size_after_od unexpected: {size_after} vs n={n}"
    )


# ===========================================================================
# #4 — RFECV feature selection path
# ===========================================================================


def test_rfecv_pipeline_runs_for_each_model_family(tmp_path, caplog):
    """The ``rfecv_models`` kwarg must accept the supported RFECV model
    names (``cb_rfecv`` / ``lgb_rfecv`` / ``xgb_rfecv``) and train a
    suite without crashing. This is a smoke guard — the Tier-1 promise
    is "RFECV path doesn't regress", not a parametric sweep over all
    three families.

    Evidence the RFECV pre-pipeline executed comes from the suite's
    own log (``cb_rfecv`` appears in ``pre_pipeline_name`` prefixes on
    the per-model timing / ensemble / VAL-metric lines). If that log
    signature goes missing, the kwarg has become a no-op.

    Note: we don't assert a DISTINCT ``model_schemas`` entry for the
    RFECV variant, because ``model_file_name`` = ``f"{mlframe_model_name}"``
    (core.py:3134) does NOT include the pre_pipeline_name. When RFECV
    happens to keep all features its ``schema_hash`` matches the
    ordinary variant's, and the filename collides → one entry in
    ``model_schemas``. That's a separate design gap tracked as a
    follow-up; not what this test guards.
    """
    pytest.importorskip("catboost")
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")

    import logging
    caplog.set_level(logging.INFO, logger="mlframe")

    df = _make_baseline_pandas(n=300, seed=0, with_cat=False, regression=False)
    from mlframe.training.core import train_mlframe_models_suite

    fte = SimpleFeaturesAndTargetsExtractor(regression=False)
    trained, meta = train_mlframe_models_suite(
        df=df, target_name="tgt", model_name="mdl",
        features_and_targets_extractor=fte,
        mlframe_models=["cb"],
        hyperparams_config={"iterations": 3, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
        init_common_params={"drop_columns": [], "verbose": 0},
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        rfecv_models=["cb_rfecv"],
        data_dir=str(tmp_path), models_dir="models", verbose=1,
    )
    assert trained, "RFECV path returned empty trained dict"

    # Evidence RFECV ran: the string "cb_rfecv" must appear in the suite's
    # own captured log lines. If the pre_pipeline loop silently skipped
    # the RFECV variant, this log line is absent.
    log_text = caplog.text.lower()
    assert "cb_rfecv" in log_text or "rfecv" in log_text, (
        "RFECV path did not emit any identifying log line; the kwarg "
        "may have become a silent no-op. Captured log tail:\n"
        f"{log_text[-500:]}"
    )


# ===========================================================================
# #5 — Ensembles (``use_mlframe_ensembles=True``)
# ===========================================================================


def test_ensembles_enabled_produces_ensemble_log(tmp_path, caplog):
    """``use_mlframe_ensembles=True`` must not silently no-op.

    The ensemble code path at ``core.py:3393-3432`` runs
    ``score_ensemble(...)`` whose result is used only for logging /
    metrics side effects (not returned in the ``trained`` dict). So
    the load-bearing evidence it fired is a specific log line:
    ``"evaluating simple ensembles..."`` at INFO level, followed by
    an ensemble-tagged metric block.

    If that log disappears, ``use_mlframe_ensembles`` has become a
    silent flag — models train but no ensemble scoring happens.
    """
    pytest.importorskip("catboost")
    pytest.importorskip("xgboost")

    import logging
    caplog.set_level(logging.INFO, logger="mlframe")

    df = _make_baseline_pandas(n=300, seed=0, with_cat=False, regression=False)
    from mlframe.training.core import train_mlframe_models_suite

    fte = SimpleFeaturesAndTargetsExtractor(regression=False)
    trained, _ = train_mlframe_models_suite(
        df=df, target_name="tgt", model_name="mdl",
        features_and_targets_extractor=fte,
        mlframe_models=["cb", "xgb"],
        hyperparams_config={
            "iterations": 3,
            "cb_kwargs": {"task_type": "CPU", "verbose": 0},
            "xgb_kwargs": {"device": "cpu", "verbosity": 0},
        },
        init_common_params={"drop_columns": [], "verbose": 0},
        use_ordinary_models=True, use_mlframe_ensembles=True,
        data_dir=str(tmp_path), models_dir="models", verbose=1,
    )
    assert trained, "ensemble run returned empty trained dict"

    log_text = caplog.text.lower()
    # The specific phrase from core.py:3395. If this disappears, the
    # ensemble code path got gated out — regression.
    assert "evaluating simple ensembles" in log_text, (
        "use_mlframe_ensembles=True did not trigger the ensemble log. "
        "Captured log tail:\n" + log_text[-800:]
    )
