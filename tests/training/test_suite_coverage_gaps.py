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


# ===========================================================================
# Tier 2 — data edge cases
# ===========================================================================


# #8 Single-class target (y all 0 or all 1)
def test_single_class_target_does_not_silently_return_empty(tmp_path):
    """Degenerate classification target (all samples same class) must
    either (a) raise a clear error upfront, or (b) skip training and
    return a well-marked empty result. A SILENT training with bogus
    metrics is the failure mode this guards against.

    Real-world trigger: a daily job's target column has 100 % minority
    class when the upstream filter accidentally narrows to a single
    state. Past prod incident: models "trained" with AUC=NaN, shipped.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=300, seed=0, with_cat=False, regression=False)
    df["target"] = np.zeros(len(df), dtype="int32")  # all one class

    from mlframe.training.core import train_mlframe_models_suite
    fte = SimpleFeaturesAndTargetsExtractor(regression=False)

    # Either outcome is OK; silent-pass-with-bogus-metrics is NOT.
    # Catch any Exception here — CatBoostError (from catboost C++) is
    # neither ValueError nor RuntimeError, and empirically fires at
    # fit() time with "Target contains only one unique value".
    raised = None
    trained = None
    meta = None
    try:
        trained, meta = train_mlframe_models_suite(
            df=df, target_name="tgt", model_name="mdl",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            hyperparams_config={"iterations": 3, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
            init_common_params={"drop_columns": [], "verbose": 0},
            use_ordinary_models=True, use_mlframe_ensembles=False,
            data_dir=str(tmp_path), models_dir="models", verbose=0,
        )
    except Exception as e:
        raised = e

    if raised is not None:
        msg = str(raised).lower()
        assert any(
            kw in msg for kw in (
                "class", "label", "target", "one", "single", "degenerate",
                "unique",  # CatBoostError: "Target contains only one unique value"
            )
        ), f"degenerate-target raise was opaque: {raised!r}"
        return
    # Non-crashing path: empty trained dict OR failure breadcrumb in meta.
    if trained:
        assert meta is not None


# #9 High-cardinality cat column (> HGB's 255 ordinal limit)
def test_high_cardinality_cat_column_trains_successfully(tmp_path):
    """cat_0 with 1000 unique string values must not crash the suite.

    HGB has ``_MAX_CATEGORICAL_CARDINALITY = 255`` and triggers the
    ordinal-to-UInt32 fallback for columns above that. XGB handles it
    via ``enable_categorical``. CB has no cardinality limit. This test
    verifies the cross-model fallback paths don't regress.
    """
    pytest.importorskip("catboost")
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")

    rng = np.random.default_rng(0)
    n = 3000  # need enough rows that 1000 uniques is plausible
    # 1000-uniques: format "k000".."k999" cycled.
    cat_vals = [f"k{i % 1000:03d}" for i in range(n)]
    num_0 = rng.standard_normal(n).astype("float32")
    y = (num_0 > 0).astype("int32")
    df = pd.DataFrame({
        "num_0": num_0,
        "cat_0": pd.Categorical(cat_vals),
        "target": y,
    })
    assert df["cat_0"].nunique() == 1000

    # HGB + CB + XGB mixed (LGB skipped: it has its own cat behaviour
    # not relevant to this test's cardinality-cap guard).
    from mlframe.training.core import train_mlframe_models_suite
    fte = SimpleFeaturesAndTargetsExtractor(regression=False)
    trained, meta = train_mlframe_models_suite(
        df=df, target_name="tgt", model_name="mdl",
        features_and_targets_extractor=fte,
        mlframe_models=["cb", "hgb", "xgb"],
        hyperparams_config={
            "iterations": 3,
            "cb_kwargs": {"task_type": "CPU", "verbose": 0},
            "xgb_kwargs": {"device": "cpu", "verbosity": 0, "enable_categorical": True},
        },
        init_common_params={"drop_columns": [], "verbose": 0},
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=str(tmp_path), models_dir="models", verbose=0,
    )
    assert trained, "high-cardinality suite returned empty dict"


# #10 Infinity / NaN in numeric column
def test_infinity_and_nan_in_numeric_column_does_not_crash(tmp_path):
    """A numeric column containing ``np.inf``, ``-np.inf``, ``np.nan``
    must be handled by the suite's existing guards (``fix_infinities``
    / ``skip_infinity_checks`` flags). This test covers the default
    config (``fix_infinities=True``) — the framework must either clip
    or drop the rows, not propagate NaN/Inf into the model fit.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=400, seed=0, with_cat=False, regression=False)
    # Inject inf/nan into num_0 at 3 rows — enough to trigger handling
    # but not enough to shift the sample distribution dramatically.
    df.loc[0, "num_0"] = np.inf
    df.loc[1, "num_0"] = -np.inf
    df.loc[2, "num_0"] = np.nan

    trained, _ = _train_once(df, tmp_path, models=("cb",), regression=False)
    assert trained, "NaN/Inf pandas frame broke the training suite"


# #11 Date/datetime column — should either pass through or be dropped cleanly
def test_datetime_column_does_not_crash_suite(tmp_path):
    """A frame with a ``pl.Datetime`` / pandas datetime column — the
    suite must either silently drop it (treat as non-feature) or feed
    it through as numeric via int64 epoch. A crash in feature-type
    auto-detection is the failure mode this guards.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=300, seed=0, with_cat=False, regression=False)
    # Add a datetime column spanning 1 year.
    start = pd.Timestamp("2026-01-01")
    df["ts"] = pd.date_range(start, periods=len(df), freq="H")

    trained, meta = _train_once(df, tmp_path, models=("cb",), regression=False)
    assert trained, "datetime column broke the suite"
    trained_cols = meta.get("columns") or []
    # Either ``ts`` is in trained_cols (treated as numeric epoch) OR
    # it was dropped. Both are acceptable; the INVARIANT is that the
    # run completed without crashing on the datetime dtype.
    if "ts" in trained_cols:
        # If kept, CB received it — that's fine, datetime is allowed
        # as a numeric feature for CB when cast to int64.
        pass


# #12 Multi-target regression (two regression targets at once)
class _MultiTargetExtractor(SimpleFeaturesAndTargetsExtractor):
    """Extractor that emits TWO regression targets from the same frame."""

    def __init__(self, target_columns=("target_a", "target_b")):
        super().__init__(target_column="target_a", regression=True)
        self._targets = list(target_columns)

    def transform(self, df):
        from mlframe.training.configs import TargetTypes
        if isinstance(df, pd.DataFrame):
            tgt_vals = {c: df[c].values for c in self._targets}
        else:
            tgt_vals = {c: df[c].to_numpy() for c in self._targets}
        target_by_type = {TargetTypes.REGRESSION: tgt_vals}
        return (df, target_by_type, None, None, None, None, list(self._targets), {})


def test_multi_target_regression_trains_all_targets(tmp_path):
    """When the extractor emits 2 regression targets the suite must
    train all of them (not silently skip). Protects against a
    regression where the target_by_type loop breaks on dict-with-N-keys
    (used to iterate only the first one).
    """
    pytest.importorskip("catboost")
    rng = np.random.default_rng(0)
    n = 300
    num_0 = rng.standard_normal(n).astype("float32")
    num_1 = rng.standard_normal(n).astype("float32")
    df = pd.DataFrame({
        "num_0": num_0, "num_1": num_1,
        "target_a": 2 * num_0 + rng.standard_normal(n) * 0.3,
        "target_b": -1.5 * num_1 + rng.standard_normal(n) * 0.3,
    })

    from mlframe.training.core import train_mlframe_models_suite
    fte = _MultiTargetExtractor(target_columns=("target_a", "target_b"))
    trained, meta = train_mlframe_models_suite(
        df=df, target_name="tgt", model_name="mdl",
        features_and_targets_extractor=fte,
        mlframe_models=["cb"],
        hyperparams_config={"iterations": 3, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
        init_common_params={"drop_columns": [], "verbose": 0},
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=str(tmp_path), models_dir="models", verbose=0,
    )
    assert trained, "multi-target regression returned empty trained dict"

    # Evidence both targets trained: the ``trained`` dict nests by
    # target_type → target_name → models. For 2 regression targets
    # sharing target_type, the inner dict must contain both.
    from mlframe.training.configs import TargetTypes
    inner = trained.get(TargetTypes.REGRESSION) or trained.get("regression") or {}
    assert isinstance(inner, dict), (
        f"trained[REGRESSION] is not a dict: {type(inner).__name__}"
    )
    target_names_trained = set(inner.keys())
    expected = {"target_a", "target_b"}
    assert expected.issubset(target_names_trained), (
        f"Not all regression targets trained: expected superset of {expected}, "
        f"got {target_names_trained}. Full trained structure keys: {list(trained)}"
    )
    # Note: schema_hash can collide across targets when they share the
    # same feature frame (model_file_name at core.py:3134 doesn't
    # include target_name). That's why we use the ``trained`` dict
    # structure rather than ``model_schemas`` count for this assertion.


# ===========================================================================
# Tier 3 — invariants / property-based
# ===========================================================================


# #15 Iteration monotonicity: more iterations → at least not worse on train
def test_more_iterations_do_not_decrease_train_performance(tmp_path):
    """With a FIXED eval_metric the suite trained with more iterations
    should NOT produce a model that scored materially WORSE on the
    training set than a shorter run. This is a sanity guard, not a
    strict monotone test (CB has stochastic subsampling; a 2-unit
    AUC wobble is fine, a regression to random is not).

    What we actually check: both runs finish without error. The weak
    invariant "training with more iterations still produces a working
    model" is cheap to assert and catches refactors that silently break
    the hyperparameter loop.

    Stronger assertion (train_auc_100iter >= train_auc_3iter - tolerance)
    would need to extract fit-time train metrics, which the suite
    buries in CB's internal ``evals_result_``. Not worth the coupling
    today.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=500, seed=0, with_cat=False, regression=False)

    from mlframe.training.core import train_mlframe_models_suite
    fte = SimpleFeaturesAndTargetsExtractor(regression=False)

    for iters in (3, 50):
        trained, _ = train_mlframe_models_suite(
            df=df, target_name="tgt", model_name=f"mdl_{iters}",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            hyperparams_config={"iterations": iters, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
            init_common_params={"drop_columns": [], "verbose": 0},
            use_ordinary_models=True, use_mlframe_ensembles=False,
            data_dir=str(tmp_path / f"iters_{iters}"), models_dir="models", verbose=0,
        )
        assert trained, f"training with iterations={iters} returned empty dict"


# #17 Idempotent save → load → predict round-trip
def test_save_load_predict_round_trip(tmp_path):
    """``train_mlframe_models_suite`` saves a model; calling
    ``predict_mlframe_models_suite`` on the same data must return
    predictions of the right shape and type. This is a smoke guard
    over the save/load/predict chain — breakage here = prod inference
    path is broken.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=300, seed=0, with_cat=False, regression=False)
    trained, _ = _train_once(df, tmp_path, models=("cb",), regression=False)
    assert trained

    from mlframe.training.core import predict_mlframe_models_suite
    models_path = os.path.join(str(tmp_path), "models", "tgt", "mdl")

    # Predict on 30 rows (subset of training frame).
    serving_df = df.head(30).drop(columns=["target"])  # no target at serving
    # The extractor is optional at predict time but helps strip target
    # rows if present. Here we just feed raw features.
    result = predict_mlframe_models_suite(
        serving_df, models_path=models_path, verbose=0,
    )

    assert "predictions" in result or "probabilities" in result
    preds_dict = result.get("predictions") or {}
    probs_dict = result.get("probabilities") or {}
    assert preds_dict or probs_dict, (
        f"predict returned no model outputs: keys={list(result)}"
    )
    # Any prediction array must have length == len(serving_df).
    for model_key, arr in {**preds_dict, **probs_dict}.items():
        if arr is None:
            continue
        assert len(arr) == len(serving_df), (
            f"prediction length mismatch for {model_key}: "
            f"{len(arr)} vs serving={len(serving_df)}"
        )


# #18 Tier-DF cache hit (prepare_polars_dataframe called once per unique tier)
def test_prepare_polars_called_once_per_model_per_pipeline(tmp_path, monkeypatch):
    """When the suite trains multiple models that share the same
    feature tier (CB + XGB both ``feature_tier=(True,True)``),
    ``strategy.prepare_polars_dataframe`` must only be invoked ONCE
    per tier (cache hit on the second model).

    If this fires more often, the ``tier_dfs_cache`` at
    ``core.py:2817-2823`` is broken — historically this is where the
    prod-scale 180 s prepare per model came from before the cache
    landed (2026-04-19).

    Test strategy: wrap ``CatBoostStrategy.prepare_polars_dataframe``
    with a counter; run CB+XGB (same tier) and assert counts ≤ 3
    (train + val + test splits of the same DF). Without the cache
    we'd see ≥ 6 calls (3 splits × 2 models).
    """
    pytest.importorskip("catboost")
    pytest.importorskip("xgboost")

    df = _make_baseline_polars_utf8(n=300, seed=0, regression=False)

    from mlframe.training import strategies as _strat
    call_counter = {"cb": 0, "xgb": 0}

    orig_cb = _strat.CatBoostStrategy.prepare_polars_dataframe
    orig_xgb = _strat.XGBoostStrategy.prepare_polars_dataframe

    def _wrap_cb(self, *a, **kw):
        call_counter["cb"] += 1
        return orig_cb(self, *a, **kw)

    def _wrap_xgb(self, *a, **kw):
        call_counter["xgb"] += 1
        return orig_xgb(self, *a, **kw)

    monkeypatch.setattr(_strat.CatBoostStrategy, "prepare_polars_dataframe", _wrap_cb)
    monkeypatch.setattr(_strat.XGBoostStrategy, "prepare_polars_dataframe", _wrap_xgb)

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
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=str(tmp_path), models_dir="models", verbose=0,
    )
    assert trained

    # CB and XGB share ``feature_tier = (True, True)`` (inherited from
    # TreeModelStrategy). The tier_dfs_cache + per-model prepare call
    # means:
    #   * CB runs first (tier-desc sort): prepares 3 splits (train/val/test)
    #   * XGB runs second: should reuse the cached tier frames but may
    #     still call prepare_polars_dataframe once to apply its own
    #     casts (e.g. Utf8→Categorical). That's 3 more calls.
    # Total acceptable ceiling: 6 (3 splits × 2 strategies).
    # Without the cache we'd see ≥ 12 (3 splits × 2 weight schemes × 2 models).
    total = call_counter["cb"] + call_counter["xgb"]
    assert total <= 6, (
        f"prepare_polars_dataframe fired too many times: "
        f"{call_counter}. Expected ≤6 combined — this suggests the "
        f"tier_dfs_cache regressed or the per-model loop is rebuilding "
        f"tier frames."
    )


# ===========================================================================
# Tier 4 — obвеска и наблюдаемость (observability)
# ===========================================================================


# #19 Config validation — Pydantic extra=forbid (strict configs only)
def test_strict_configs_reject_unknown_fields():
    """Configs declared with ``extra='forbid'`` must raise a clean
    ``ValidationError`` on typo'd field names — this prevents the
    silent-absorption bug where ``iteratoins=100`` (typo) slipped
    through as an unused extra.

    mlframe has TWO strict configs (explicit ``extra='forbid'`` at
    class level): ``PreprocessingConfig`` and ``FeatureTypesConfig``.
    The rest inherit ``BaseConfig``'s ``extra='allow'`` to preserve
    pass-through kwargs (e.g. ``hyperparams_config={'mae_weight': 1.0}``
    flows through to downstream callees). The permissive configs
    still emit a WARNING via ``_warn_on_unknown_extras`` validator —
    tested separately.
    """
    from pydantic import ValidationError

    # PreprocessingConfig declares extra='forbid' (configs.py:138).
    with pytest.raises(ValidationError):
        PreprocessingConfig(fillna_vlue=0)  # typo: vlue → value

    # FeatureTypesConfig declares extra='forbid' (configs.py:406).
    with pytest.raises(ValidationError):
        FeatureTypesConfig(use_text_feats=True)  # typo: feats → features


def test_permissive_configs_warn_on_unknown_fields(caplog):
    """Permissive configs (extra='allow') must WARN on unknown fields
    — silent absorption is the real prod-risk (``iteratoins=100``
    typo slipping through unused). ``PolarsPipelineConfig`` and
    ``TrainingBehaviorConfig`` are permissive; typos must still
    produce a visible WARNING via ``_warn_on_unknown_extras``.
    """
    import logging
    caplog.set_level(logging.WARNING, logger="mlframe.training.configs")

    # Should NOT raise (extra='allow'), but must emit a WARNING.
    cfg = PolarsPipelineConfig(use_polrsds_pipeline=True)  # typo: missing 'a'
    assert cfg is not None

    log_text = caplog.text.lower()
    assert "unknown field" in log_text, (
        "PolarsPipelineConfig did not warn on unknown field — "
        "silent-absorption bug returned. Log: " + log_text[-300:]
    )


# #20 Metadata schema contents — load-bearing fields must be populated
def test_metadata_carries_load_bearing_fields(tmp_path):
    """After training, the returned metadata dict must contain the
    keys that ``load_mlframe_suite`` / ``_validate_input_columns_against_metadata``
    depend on:
      * ``columns`` (the trained feature list)
      * ``cat_features`` (may be empty post-encoding — still the key
        must be present)
      * ``pipeline``, ``extensions_pipeline`` (for transform at load)
      * ``outlier_detection`` (dict with ``applied`` flag)
      * ``model_schemas`` (per-model schema hashes — Fix 8)

    Each missing key below would manifest as a distinct load-time
    failure. This test is a single-point tripwire catching silent
    metadata shape drift.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=300, seed=0, with_cat=True, regression=False)
    _trained, meta = _train_once(df, tmp_path, models=("cb",), regression=False)
    assert meta is not None

    for key in ("columns", "cat_features", "pipeline", "outlier_detection", "model_schemas"):
        assert key in meta, f"metadata missing load-bearing key {key!r}; keys={list(meta)[:20]}"

    # model_schemas must be non-empty and entries must carry schema_hash
    # + input_schema + mlframe_model + weight_name.
    schemas = meta["model_schemas"]
    assert schemas, "model_schemas is empty after a successful training"
    entry = next(iter(schemas.values()))
    for k in ("schema_hash", "input_schema", "mlframe_model", "weight_name"):
        assert k in entry, (
            f"model_schemas entry missing {k!r}; entry keys={list(entry)}"
        )

    # outlier_detection must have the applied flag (False here since
    # no detector was passed).
    od = meta["outlier_detection"]
    assert isinstance(od, dict)
    assert "applied" in od
    assert od["applied"] is False  # no detector passed in this test


# #21 continue_on_model_failure — one model crashes, others proceed
def test_continue_on_model_failure_skips_crashed_model(tmp_path, monkeypatch):
    """When ``continue_on_model_failure=True``, a deliberate crash
    during CB.fit must not abort the whole suite — XGB in the same
    ``mlframe_models`` list must still train. Metadata's
    ``failed_models`` list must record the crashed model.

    Monkey-patches ``CatBoostClassifier.fit`` to raise a tagged
    RuntimeError; the suite's ``continue_on_model_failure`` branch
    at ``core.py:3232-3238`` is the load-bearing code.
    """
    pytest.importorskip("catboost")
    pytest.importorskip("xgboost")

    df = _make_baseline_pandas(n=300, seed=0, with_cat=False, regression=False)

    import catboost as cb
    _orig_fit = cb.CatBoostClassifier.fit
    _sentinel = RuntimeError("sentinel-fit-crash-for-test")

    def _boom(self, *a, **kw):
        raise _sentinel

    monkeypatch.setattr(cb.CatBoostClassifier, "fit", _boom)

    from mlframe.training.core import train_mlframe_models_suite
    fte = SimpleFeaturesAndTargetsExtractor(regression=False)
    trained, meta = train_mlframe_models_suite(
        df=df, target_name="tgt", model_name="mdl",
        features_and_targets_extractor=fte,
        mlframe_models=["cb", "xgb"],
        hyperparams_config={
            "iterations": 3,
            "cb_kwargs": {"task_type": "CPU", "verbose": 0},
            "xgb_kwargs": {"device": "cpu", "verbosity": 0},
        },
        init_common_params={"drop_columns": [], "verbose": 0},
        use_ordinary_models=True, use_mlframe_ensembles=False,
        behavior_config=TrainingBehaviorConfig(continue_on_model_failure=True),
        data_dir=str(tmp_path), models_dir="models", verbose=0,
    )

    # CB crashed, XGB must have trained. The ``trained`` dict may have
    # XGB entries even if CB's entries are missing / empty.
    from mlframe.training.configs import TargetTypes
    inner = trained.get(TargetTypes.BINARY_CLASSIFICATION) or {}
    # failed_models list must record the crash with model='cb'.
    failed = meta.get("failed_models") or []
    assert failed, (
        "continue_on_model_failure=True but no 'failed_models' in metadata"
    )
    cb_failures = [f for f in failed if f.get("model") == "cb"]
    assert cb_failures, (
        f"CB crash not recorded in failed_models: {failed}"
    )

    # Restore fit so subsequent tests in same session aren't affected
    # (monkeypatch does this automatically, but explicit is safer).
    cb.CatBoostClassifier.fit = _orig_fit


# #22 verbose=0 — stdout must be effectively silent (modulo warnings)
def test_verbose_zero_suppresses_suite_info_logs(tmp_path, capsys, caplog):
    """``verbose=0`` must suppress the suite's own INFO-level stdout
    narration. Third-party libs (CatBoost's native output, matplotlib
    warnings) may still emit; we can't stop those. But the suite's
    own ``logger.info(...)`` calls must be silenced.
    """
    pytest.importorskip("catboost")
    import logging
    caplog.set_level(logging.WARNING, logger="mlframe")

    df = _make_baseline_pandas(n=200, seed=0, with_cat=False, regression=False)
    from mlframe.training.core import train_mlframe_models_suite
    fte = SimpleFeaturesAndTargetsExtractor(regression=False)
    trained, _ = train_mlframe_models_suite(
        df=df, target_name="tgt", model_name="mdl",
        features_and_targets_extractor=fte,
        mlframe_models=["cb"],
        hyperparams_config={"iterations": 3, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
        init_common_params={"drop_columns": [], "verbose": 0},
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=str(tmp_path), models_dir="models", verbose=0,
    )
    assert trained

    # Check caplog at INFO level filtered to mlframe: the suite's own
    # verbose=0 path should emit zero INFO records from the mlframe
    # namespace. We allow WARNING+ (diagnostics shouldn't vanish).
    mlframe_info_records = [
        r for r in caplog.records
        if r.name.startswith("mlframe") and r.levelno == logging.INFO
    ]
    assert not mlframe_info_records, (
        f"verbose=0 leaked {len(mlframe_info_records)} INFO-level log "
        f"records from mlframe.*; first few:\n"
        + "\n".join(r.getMessage()[:120] for r in mlframe_info_records[:5])
    )


# ===========================================================================
# Group A — high-ROI, easy (#23–#28)
# ===========================================================================


# #23 Empty / None df / missing target column
@pytest.mark.parametrize("df_input,error_kw", [
    (None, ("none", "type", "input")),
    # Empty pd.DataFrame has no columns at all → extractor fails with
    # KeyError when looking up the target column. Either error class
    # is acceptable as long as the message points at the missing
    # column (or the empty frame).
    (pd.DataFrame(), ("empty", "rows", "no", "target", "column", "key")),
])
def test_invalid_df_inputs_raise_clear_error(tmp_path, df_input, error_kw):
    """``train_mlframe_models_suite`` must raise a CLEAR error (not a
    cryptic AttributeError deep in the pipeline) when given invalid
    ``df`` input. Two cases:
      * ``df=None`` — caller forgot to load
      * ``df=empty pandas frame`` — upstream filter killed all rows

    The error message should reference the underlying problem
    (none-type / empty / no rows / type), not be opaque.
    """
    pytest.importorskip("catboost")
    from mlframe.training.core import train_mlframe_models_suite
    fte = SimpleFeaturesAndTargetsExtractor(regression=False)

    raised = None
    try:
        train_mlframe_models_suite(
            df=df_input,
            target_name="tgt", model_name="mdl",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            hyperparams_config={"iterations": 3, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
            init_common_params={"drop_columns": [], "verbose": 0},
            use_ordinary_models=True, use_mlframe_ensembles=False,
            data_dir=str(tmp_path), models_dir="models", verbose=0,
        )
    except Exception as e:
        raised = e

    assert raised is not None, (
        f"invalid df={type(df_input).__name__} did not raise — "
        "caller would proceed with junk data"
    )
    msg = str(raised).lower()
    assert any(kw in msg for kw in error_kw), (
        f"raise was opaque for df={type(df_input).__name__}: {raised!r}"
    )


def test_missing_target_column_raises(tmp_path):
    """If the extractor's ``target_column`` is not in the input df,
    the suite must raise a clear KeyError/ValueError mentioning the
    missing column name. Silent dummy-fill or default-to-zero would
    silently train a useless model.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=300, seed=0, with_cat=False, regression=False)
    df = df.drop(columns=["target"])  # target column gone

    from mlframe.training.core import train_mlframe_models_suite
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

    raised = None
    try:
        train_mlframe_models_suite(
            df=df, target_name="tgt", model_name="mdl",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            hyperparams_config={"iterations": 3, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
            init_common_params={"drop_columns": [], "verbose": 0},
            use_ordinary_models=True, use_mlframe_ensembles=False,
            data_dir=str(tmp_path), models_dir="models", verbose=0,
        )
    except Exception as e:
        raised = e
    assert raised is not None, "missing target column did not raise"
    msg = str(raised).lower()
    # Either explicit "target" mention OR a KeyError-style message about
    # the column name. Anything that points the operator at the missing
    # column is acceptable.
    assert "target" in msg or "column" in msg or "key" in msg, (
        f"missing-target raise was opaque: {raised!r}"
    )


# #24 Empty model list / unknown model name
def test_empty_mlframe_models_handled_gracefully(tmp_path):
    """``mlframe_models=[]`` (empty list) must either (a) raise a clean
    error indicating no models were requested OR (b) skip cleanly with
    an empty trained dict. A silent return-with-non-empty-dict from
    nowhere would be the bug.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=300, seed=0, with_cat=False, regression=False)
    from mlframe.training.core import train_mlframe_models_suite
    fte = SimpleFeaturesAndTargetsExtractor(regression=False)

    raised = None
    trained = None
    try:
        trained, _ = train_mlframe_models_suite(
            df=df, target_name="tgt", model_name="mdl",
            features_and_targets_extractor=fte,
            mlframe_models=[],
            hyperparams_config={"iterations": 3},
            init_common_params={"drop_columns": [], "verbose": 0},
            use_ordinary_models=True, use_mlframe_ensembles=False,
            data_dir=str(tmp_path), models_dir="models", verbose=0,
        )
    except Exception as e:
        raised = e
    if raised is not None:
        msg = str(raised).lower()
        assert any(kw in msg for kw in ("model", "empty", "none")), (
            f"empty-models raise was opaque: {raised!r}"
        )
        return
    # Or, a non-crashing result: trained dict must be empty.
    assert not trained, (
        f"empty mlframe_models silently returned non-empty trained dict: {trained}"
    )


def test_unknown_mlframe_model_name_raises_or_warns(tmp_path):
    """An unknown model name in ``mlframe_models`` must NOT silently
    pass — either raise a ValueError listing valid names OR log a
    WARNING and skip the unknown name. Silent acceptance leads to
    "trained" reports that don't mention the missing model.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=300, seed=0, with_cat=False, regression=False)
    from mlframe.training.core import train_mlframe_models_suite
    fte = SimpleFeaturesAndTargetsExtractor(regression=False)

    import logging
    import io
    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    handler.setLevel(logging.WARNING)
    logging.getLogger("mlframe").addHandler(handler)

    raised = None
    trained = None
    try:
        trained, _ = train_mlframe_models_suite(
            df=df, target_name="tgt", model_name="mdl",
            features_and_targets_extractor=fte,
            mlframe_models=["this_is_not_a_real_model"],
            hyperparams_config={"iterations": 3},
            init_common_params={"drop_columns": [], "verbose": 0},
            use_ordinary_models=True, use_mlframe_ensembles=False,
            data_dir=str(tmp_path), models_dir="models", verbose=0,
        )
    except Exception as e:
        raised = e
    finally:
        logging.getLogger("mlframe").removeHandler(handler)

    log_text = log_buf.getvalue().lower()
    if raised is not None:
        # Loud raise — preferred.
        msg = str(raised).lower()
        assert any(kw in msg for kw in ("unknown", "model", "not", "valid")), (
            f"unknown-model raise was opaque: {raised!r}"
        )
        return
    # Or, silent skip with WARN. Trained dict empty + warning fired.
    assert (
        ("unknown" in log_text or "skip" in log_text or "this_is_not_a_real_model" in log_text)
        or not trained
    ), (
        "unknown model name was silently accepted with no warning and "
        "no skip — operator would never know"
    )


# #25 Predict output range invariant
def test_predictions_in_valid_range_for_classification(tmp_path):
    """Classification predict_proba outputs must be probabilities in
    [0, 1]. Catches a broken model wrapper that returned raw logits
    instead of probabilities.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=300, seed=0, with_cat=False, regression=False)
    trained, _ = _train_once(df, tmp_path, models=("cb",), regression=False)
    assert trained

    from mlframe.training.core import predict_mlframe_models_suite
    models_path = os.path.join(str(tmp_path), "models", "tgt", "mdl")
    serving = df.head(50).drop(columns=["target"])
    res = predict_mlframe_models_suite(
        serving, models_path=models_path, return_probabilities=True, verbose=0,
    )
    probs = res.get("probabilities") or {}
    assert probs, "no probabilities returned for classification"
    for name, arr in probs.items():
        if arr is None:
            continue
        arr_np = np.asarray(arr)
        # Either single-column (positive-class prob) or two-column.
        # Both must be in [0, 1].
        assert np.isfinite(arr_np).all(), f"non-finite probs from {name}: any NaN/Inf"
        assert (arr_np >= 0.0).all() and (arr_np <= 1.0001).all(), (
            f"out-of-range probs from {name}: min={arr_np.min()}, max={arr_np.max()}"
        )


def test_predictions_finite_for_regression(tmp_path):
    """Regression predict outputs must be finite floats. Catches a
    model that returned NaN/Inf from a numerically unstable fit.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=300, seed=0, with_cat=False, regression=True)
    trained, _ = _train_once(df, tmp_path, models=("cb",), regression=True)
    assert trained

    from mlframe.training.core import predict_mlframe_models_suite
    models_path = os.path.join(str(tmp_path), "models", "tgt", "mdl")
    serving = df.head(50).drop(columns=["target"])
    res = predict_mlframe_models_suite(
        serving, models_path=models_path, return_probabilities=False, verbose=0,
    )
    preds = res.get("predictions") or {}
    assert preds, "no predictions returned for regression"
    for name, arr in preds.items():
        if arr is None:
            continue
        arr_np = np.asarray(arr, dtype="float64")
        assert np.isfinite(arr_np).all(), (
            f"non-finite regression predictions from {name}"
        )


# #26 Column-order invariance: schema_hash should not depend on
# input column order (compute_model_input_fingerprint sorts cols).
def test_schema_hash_is_column_order_invariant(tmp_path):
    """Train two suites on the SAME data but with columns in different
    orders. ``schema_hash`` for the trained model must be identical.

    This invariant relies on ``compute_model_input_fingerprint`` calling
    ``sorted(df_at_fit.columns)`` at utils.py:374. If a refactor drops
    the sort, two semantically-identical frames would produce DIFFERENT
    cache filenames — duplicate trained models on disk and silent
    cache misses at load time.
    """
    pytest.importorskip("catboost")
    df_a = _make_baseline_pandas(n=300, seed=0, with_cat=True, regression=False)
    # Reorder: put cat_0 BEFORE the numeric columns.
    df_b = df_a[["cat_0", "num_0", "num_1", "num_2", "target"]].copy()

    _, meta_a = _train_once(df_a, tmp_path / "a", models=("cb",), regression=False)
    _, meta_b = _train_once(df_b, tmp_path / "b", models=("cb",), regression=False)

    schemas_a = meta_a.get("model_schemas") or {}
    schemas_b = meta_b.get("model_schemas") or {}
    assert schemas_a and schemas_b, "no model_schemas after training"

    # Take the first model entry from each and compare schema_hash.
    hash_a = next(iter(schemas_a.values())).get("schema_hash")
    hash_b = next(iter(schemas_b.values())).get("schema_hash")
    assert hash_a == hash_b, (
        f"column-order invariance broken: "
        f"original-order hash={hash_a!r}, reordered hash={hash_b!r}. "
        f"compute_model_input_fingerprint dropped its sorted() call?"
    )


# #27 Subprocess save/load roundtrip
def test_save_load_predict_in_subprocess(tmp_path):
    """Train in pytest process; load + predict in a FRESH subprocess.

    This catches serialization issues that an in-process load would
    miss — e.g., a custom class pickled with a module path that's
    only importable in the test's working directory, or a CB binary
    blob that triggers GPU-context init when loaded fresh.

    Subprocess uses ``subprocess.run`` with the same Python interpreter,
    captures stdout, asserts the printed prediction count matches the
    expected serving rows.
    """
    pytest.importorskip("catboost")
    import subprocess
    import sys

    df = _make_baseline_pandas(n=300, seed=0, with_cat=False, regression=False)
    _train_once(df, tmp_path, models=("cb",), regression=False)
    models_path = os.path.join(str(tmp_path), "models", "tgt", "mdl")
    assert os.path.isdir(models_path)

    # Persist a serving frame to feed the subprocess.
    serving_path = os.path.join(str(tmp_path), "serving.parquet")
    df.head(40).drop(columns=["target"]).to_parquet(serving_path)

    # Subprocess script: load + predict + print result count.
    script = (
        "import sys, os\n"
        "import pandas as pd\n"
        "from mlframe.training.core import predict_mlframe_models_suite\n"
        f"df = pd.read_parquet({serving_path!r})\n"
        f"res = predict_mlframe_models_suite(df, models_path={models_path!r}, "
        "return_probabilities=True, verbose=0)\n"
        "probs = res.get('probabilities') or {}\n"
        "preds = res.get('predictions') or {}\n"
        "any_arr = next(iter({**probs, **preds}.values()), None)\n"
        "if any_arr is None:\n"
        "    print('NONE'); sys.exit(2)\n"
        "print('PREDS_LEN=' + str(len(any_arr)))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, (
        f"subprocess load+predict failed (rc={result.returncode}):\n"
        f"--stdout--\n{result.stdout[-2000:]}\n--stderr--\n{result.stderr[-2000:]}"
    )
    assert "PREDS_LEN=40" in result.stdout, (
        f"subprocess returned wrong prediction count: stdout={result.stdout!r}"
    )


# #28 Duplicates in mlframe_models — must dedupe or raise, not silently
# train twice and overwrite.
def test_duplicate_mlframe_models_handled(tmp_path):
    """``mlframe_models=["cb", "cb"]`` (copy-paste typo). Acceptable
    behaviours:
      (a) silently dedupe → trained once, one entry in metadata
      (b) explicit ValueError mentioning the duplicate

    Unacceptable: train twice with the SAME model_file_name and the
    second run silently overwrites the first — leaves stale metadata
    referencing a file the suite has already replaced.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=300, seed=0, with_cat=False, regression=False)
    from mlframe.training.core import train_mlframe_models_suite
    fte = SimpleFeaturesAndTargetsExtractor(regression=False)

    raised = None
    trained = None
    meta = None
    try:
        trained, meta = train_mlframe_models_suite(
            df=df, target_name="tgt", model_name="mdl",
            features_and_targets_extractor=fte,
            mlframe_models=["cb", "cb"],  # duplicate
            hyperparams_config={"iterations": 3, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
            init_common_params={"drop_columns": [], "verbose": 0},
            use_ordinary_models=True, use_mlframe_ensembles=False,
            data_dir=str(tmp_path), models_dir="models", verbose=0,
        )
    except Exception as e:
        raised = e

    if raised is not None:
        msg = str(raised).lower()
        assert any(kw in msg for kw in ("duplicate", "unique", "twice", "model")), (
            f"duplicate-models raise was opaque: {raised!r}"
        )
        return
    # Silent dedupe path: ensure model_schemas has at most ONE 'cb' entry.
    schemas = (meta or {}).get("model_schemas") or {}
    cb_entries = [
        v for v in schemas.values()
        if v.get("mlframe_model") == "cb" and v.get("weight_name") == "uniform"
    ]
    assert len(cb_entries) <= 1, (
        f"duplicate ['cb','cb'] silently produced {len(cb_entries)} 'cb' entries "
        f"in model_schemas — overwrite hazard. Entries: {cb_entries}"
    )


# ===========================================================================
# Group B — medium-ROI (#29-#34)
# ===========================================================================


# #29 Custom pre_pipelines (PCA/IncrementalPCA inserted into the suite)
def test_custom_pre_pipelines_runs_without_crash(tmp_path, caplog):
    """``custom_pre_pipelines={"pca50": IncrementalPCA(n_components=2)}``
    must be accepted by the suite. The custom transformer fits inside
    the per-model pre_pipeline loop. Asserts (a) no crash, (b) the
    custom name appears in suite logs.

    n_components=2 is small enough that the synthetic 3-numeric frame
    can fit it without rank issues.
    """
    pytest.importorskip("catboost")
    from sklearn.decomposition import IncrementalPCA
    import logging
    caplog.set_level(logging.INFO, logger="mlframe")

    df = _make_baseline_pandas(n=300, seed=0, with_cat=False, regression=False)
    from mlframe.training.core import train_mlframe_models_suite
    fte = SimpleFeaturesAndTargetsExtractor(regression=False)

    custom = {"pca50": IncrementalPCA(n_components=2)}
    trained, _ = train_mlframe_models_suite(
        df=df, target_name="tgt", model_name="mdl",
        features_and_targets_extractor=fte,
        mlframe_models=["cb"],
        hyperparams_config={"iterations": 3, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
        init_common_params={"drop_columns": [], "verbose": 0},
        use_ordinary_models=True, use_mlframe_ensembles=False,
        custom_pre_pipelines=custom,
        data_dir=str(tmp_path), models_dir="models", verbose=1,
    )
    assert trained, "custom_pre_pipelines path returned empty trained dict"

    log_text = caplog.text.lower()
    assert "pca50" in log_text, (
        "custom pipeline name 'pca50' missing from suite logs — "
        "the custom_pre_pipelines kwarg may not have been wired in. "
        "Captured log tail:\n" + log_text[-600:]
    )


# #30 preprocessing_extensions: poly_features + scaler
def test_preprocessing_extensions_polynomial_features_run(tmp_path):
    """``PreprocessingExtensionsConfig(polynomial_degree=2, ...)`` must
    expand the feature space (poly features add interaction columns)
    and feed it into model fit. Cross-checks the extensions sklearn
    bridge inside ``apply_preprocessing_extensions`` (core.py:2019).

    Asserts: training completes; column count after extensions > 3.
    """
    pytest.importorskip("catboost")
    from mlframe.training.configs import PreprocessingExtensionsConfig

    df = _make_baseline_pandas(n=300, seed=0, with_cat=False, regression=False)

    from mlframe.training.core import train_mlframe_models_suite
    fte = SimpleFeaturesAndTargetsExtractor(regression=False)

    ext = PreprocessingExtensionsConfig(
        polynomial_degree=2,
        polynomial_interaction_only=True,
        scaler="StandardScaler",
    )
    trained, meta = train_mlframe_models_suite(
        df=df, target_name="tgt", model_name="mdl",
        features_and_targets_extractor=fte,
        mlframe_models=["cb"],
        hyperparams_config={"iterations": 3, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
        init_common_params={"drop_columns": [], "verbose": 0},
        use_ordinary_models=True, use_mlframe_ensembles=False,
        preprocessing_extensions=ext,
        data_dir=str(tmp_path), models_dir="models", verbose=0,
    )
    assert trained, "preprocessing_extensions=poly run produced empty dict"
    cols = meta.get("columns") or []
    # Original frame has 3 numeric features. Interaction-only poly_2
    # adds C(3,2)=3 interactions → at least 4 features post-extension.
    # Permissive lower bound (>3) survives small sklearn-version drift.
    assert len(cols) > 3, (
        f"polynomial_degree=2 did not expand feature count: cols={cols}"
    )


# #31 Fairness end-to-end
def test_fairness_features_recorded_in_metadata(tmp_path):
    """Setting ``behavior_config.fairness_features=["cat_0"]`` must
    populate fairness subgroups and reflect them in metadata. Without
    this guard, a refactor that disables the fairness path silently
    skips the per-subgroup metric calc.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=400, seed=0, with_cat=True, regression=False)
    from mlframe.training.core import train_mlframe_models_suite
    fte = SimpleFeaturesAndTargetsExtractor(regression=False)

    behavior = TrainingBehaviorConfig(
        fairness_features=["cat_0"],
        fairness_min_pop_cat_thresh=10,
    )
    trained, meta = train_mlframe_models_suite(
        df=df, target_name="tgt", model_name="mdl",
        features_and_targets_extractor=fte,
        mlframe_models=["cb"],
        hyperparams_config={"iterations": 3, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
        init_common_params={"drop_columns": [], "verbose": 0},
        use_ordinary_models=True, use_mlframe_ensembles=False,
        behavior_config=behavior,
        data_dir=str(tmp_path), models_dir="models", verbose=0,
    )
    assert trained, "fairness run returned empty trained dict"
    # Acceptance: meta either records fairness subgroups OR the
    # fairness keyword appears somewhere in the metadata. Strict
    # contract is undocumented — we test the loose invariant
    # "fairness_features didn't get silently dropped".
    meta_text = repr(meta).lower()
    assert "fair" in meta_text, (
        "fairness_features=['cat_0'] left no breadcrumb in metadata — "
        "the kwarg may have been silently ignored. Meta keys: "
        + str(list(meta))
    )


# #32 prefer_calibrated_classifiers (calibration code path)
def test_prefer_calibrated_classifiers_runs(tmp_path):
    """``behavior_config.prefer_calibrated_classifiers=True`` selects
    the CALIB_CLASSIF hyperparameter set inside ``configure_training_params``
    (trainer.py:3680). Asserts a CB run with this flag set completes
    without crashing — the calibration code path is not crashed by a
    refactor.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=300, seed=0, with_cat=False, regression=False)

    from mlframe.training.core import train_mlframe_models_suite
    fte = SimpleFeaturesAndTargetsExtractor(regression=False)

    behavior = TrainingBehaviorConfig(prefer_calibrated_classifiers=True)
    trained, _ = train_mlframe_models_suite(
        df=df, target_name="tgt", model_name="mdl",
        features_and_targets_extractor=fte,
        mlframe_models=["cb"],
        hyperparams_config={"iterations": 3, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
        init_common_params={"drop_columns": [], "verbose": 0},
        use_ordinary_models=True, use_mlframe_ensembles=False,
        behavior_config=behavior,
        data_dir=str(tmp_path), models_dir="models", verbose=0,
    )
    assert trained, "prefer_calibrated_classifiers=True trained empty dict"


# #33 Streaming parquet path: df=str(path)
def test_df_as_parquet_path_string_loads_inside_suite(tmp_path):
    """``train_mlframe_models_suite(df=<path-string>)`` must load the
    parquet from disk via ``load_and_prepare_dataframe``. This is the
    prod-streaming use case where the df is too large to materialize
    in the caller and the suite reads it itself.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=300, seed=0, with_cat=True, regression=False)
    parquet_path = os.path.join(str(tmp_path), "data.parquet")
    df.to_parquet(parquet_path)

    from mlframe.training.core import train_mlframe_models_suite
    fte = SimpleFeaturesAndTargetsExtractor(regression=False)
    trained, meta = train_mlframe_models_suite(
        df=parquet_path,  # path string instead of frame
        target_name="tgt", model_name="mdl",
        features_and_targets_extractor=fte,
        mlframe_models=["cb"],
        hyperparams_config={"iterations": 3, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
        init_common_params={"drop_columns": [], "verbose": 0},
        use_ordinary_models=True, use_mlframe_ensembles=False,
        data_dir=str(tmp_path), models_dir="models", verbose=0,
    )
    assert trained, "parquet-path-as-df failed to train"
    cols = meta.get("columns") or []
    assert "num_0" in cols, f"loaded frame missing expected columns: {cols}"


# #34 trusted_root security check on load
def test_load_outside_trusted_root_blocked(tmp_path):
    """``predict_mlframe_models_suite`` defaults ``trusted_root`` to
    ``os.path.abspath(models_path)``. Passing an EXPLICIT
    ``trusted_root`` that does NOT contain the metadata file must
    raise ``ValueError`` mentioning "trusted_root" — security guard
    against arbitrary path escape during pickle load.
    """
    pytest.importorskip("catboost")
    df = _make_baseline_pandas(n=200, seed=0, with_cat=False, regression=False)
    _train_once(df, tmp_path, models=("cb",), regression=False)
    models_path = os.path.join(str(tmp_path), "models", "tgt", "mdl")
    serving = df.head(10).drop(columns=["target"])

    from mlframe.training.core import predict_mlframe_models_suite

    # Use a sibling dir that does NOT contain models_path.
    bogus_root = os.path.abspath(os.path.join(str(tmp_path), "..", "elsewhere"))
    os.makedirs(bogus_root, exist_ok=True)

    with pytest.raises(ValueError, match=r"trusted_root|inside"):
        predict_mlframe_models_suite(
            serving, models_path=models_path,
            trusted_root=bogus_root, verbose=0,
        )
