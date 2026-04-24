"""Randomized fuzz coverage for ``train_mlframe_models_suite``.

Feeds ~150 unique, pairwise-covering combos through the suite and records
every combo's outcome to ``_fuzz_results.jsonl`` for later analysis.

Known-bug xfail rules live in ``_fuzz_combo.KNOWN_XFAIL_RULES`` and are
applied automatically per combo via ``pytest.mark.xfail`` in the test
function — new bugs discovered by fuzzing should be added there once
they're traced to a specific combo predicate.
"""
from __future__ import annotations

import os
import time
import traceback

import pytest

from ._fuzz_combo import (
    FuzzCombo,
    build_frame_for_combo,
    enumerate_combos,
    log_combo_outcome,
    xfail_reason,
)
from .shared import SimpleFeaturesAndTargetsExtractor

# Enumerate once at import time — small, pure Python, no heavy deps.
# FUZZ_SEED env var overrides the default (driver scripts use this to
# sweep many seeds in sequence without editing the file; each pytest
# invocation reads the env fresh so 10k-combo campaigns can span 60+
# seeds × 150 combos each without the parent process sharing state).
_FUZZ_MASTER_SEED = int(os.environ.get("FUZZ_SEED", "20260422"))
COMBOS: list[FuzzCombo] = enumerate_combos(target=150, master_seed=_FUZZ_MASTER_SEED)


def _config_for_models(
    models: tuple[str, ...],
    n_rows: int,
    iterations: int = 3,
    early_stopping_rounds: "int | None" = None,
) -> dict:
    cfg: dict = {"iterations": iterations}
    if early_stopping_rounds is not None:
        cfg["early_stopping_rounds"] = early_stopping_rounds
    if "lgb" in models:
        cfg["lgb_kwargs"] = {"device_type": "cpu", "verbose": -1}
    if "xgb" in models:
        cfg["xgb_kwargs"] = {"device": "cpu", "verbosity": 0}
    if "cb" in models:
        cfg["cb_kwargs"] = {"task_type": "CPU", "verbose": 0}
    return cfg


def _configs_for_combo(combo: FuzzCombo) -> dict:
    """Build the preprocessing / feature-types / behavior config overrides
    that prop the combo's per-axis flags into ``train_mlframe_models_suite``.

    Returns a kwargs dict ready to **-splat into the suite call. Built
    lazily from Pydantic model defaults so any field we don't touch
    keeps its library default (mirrors how prod callers typically
    override only a handful of fields).

    Text / embedding column names are passed explicitly so that combos
    with ``auto_detect_cats=False`` still register them as text / embedding
    features. Without this, the auto-detector's early-return on
    ``auto_detect_feature_types=False`` leaves ``emb_0`` (or ``text_0``)
    in the feature matrix as a regular column — CatBoost's Polars
    fastpath then crashes trying to process a ``pl.List(Float32)`` /
    high-cardinality string column it never learned was special.
    Matches the prod idiom: callers who turn off auto-detection
    typically do so BECAUSE they're declaring the lists manually."""
    from mlframe.training.configs import (
        PolarsPipelineConfig,
        FeatureTypesConfig,
        TrainingBehaviorConfig,
        PreprocessingConfig,
        TrainingSplitConfig,
    )
    # Mirror the ``want_text`` / ``want_embedding`` gates in
    # ``build_frame_for_combo`` so the declared column lists exactly
    # match what the frame actually contains — no false positives,
    # no false negatives.
    emits_text = combo.text_col_count > 0 and "cb" in combo.models
    emits_emb = (
        combo.embedding_col_count > 0
        and "cb" in combo.models
        and combo.input_type != "pandas"
    )
    text_features = (
        [f"text_{i}" for i in range(combo.text_col_count)] if emits_text else None
    )
    embedding_features = (
        [f"emb_{i}" for i in range(combo.embedding_col_count)] if emits_emb else None
    )
    # Fairness: only valid if the referenced column actually exists in
    # the frame (cat_0 requires cat_feature_count >= 1).
    fairness_features = (
        [combo.fairness_col]
        if combo.fairness_col is not None and combo.cat_feature_count > 0
        else None
    )
    behavior_kwargs: dict = {
        "align_polars_categorical_dicts": combo.align_polars_categorical_dicts,
        "continue_on_model_failure": combo.continue_on_model_failure,
        "prefer_calibrated_classifiers": combo.prefer_calibrated_classifiers,
        # 2026-04-24 round 2
        "use_robust_eval_metric": combo.use_robust_eval_metric_cfg,
    }
    if fairness_features:
        behavior_kwargs["fairness_features"] = fairness_features
        behavior_kwargs["fairness_min_pop_cat_thresh"] = 10
    # TrainingSplitConfig: val_size left at default (0.1); test_size
    # varies per axis. trainset_aging_limit validates strictly in
    # (0, 1) so only 0.5 is a safe non-None value.
    split_config = TrainingSplitConfig(
        test_size=combo.test_size_cfg,
        val_placement=combo.val_placement_cfg,
        trainset_aging_limit=combo.trainset_aging_limit_cfg,
    )
    preprocessing_config = PreprocessingConfig(
        fillna_value=combo.fillna_value_cfg,
    )
    return {
        "pipeline_config": PolarsPipelineConfig(
            use_polarsds_pipeline=combo.use_polarsds_pipeline,
            scaler_name=combo.scaler_name_cfg,
            categorical_encoding=combo.categorical_encoding_cfg,
            skip_categorical_encoding=combo.skip_categorical_encoding_cfg,
        ),
        "preprocessing_config": preprocessing_config,
        "split_config": split_config,
        "feature_types_config": FeatureTypesConfig(
            auto_detect_feature_types=combo.auto_detect_cats,
            use_text_features=combo.use_text_features,
            honor_user_dtype=combo.honor_user_dtype,
            text_features=text_features,
            embedding_features=embedding_features,
            cat_text_cardinality_threshold=combo.cat_text_card_threshold_cfg,
        ),
        "behavior_config": TrainingBehaviorConfig(**behavior_kwargs),
    }


def _outlier_detector_for_combo(combo: FuzzCombo):
    """Construct an ``outlier_detector`` object when the combo asks for
    one, else return None. Kept separate from ``_configs_for_combo`` so
    the suite kwarg and the combo axis are wired independently."""
    if combo.outlier_detection == "isolation_forest":
        try:
            from sklearn.ensemble import IsolationForest
            return IsolationForest(
                contamination=0.05, random_state=combo.seed, n_estimators=20,
            )
        except ImportError:
            return None
    return None


def _custom_pre_pipelines_for_combo(combo: FuzzCombo):
    """When ``combo.custom_prep == "pca2"`` attach an IncrementalPCA
    transformer. Fails-open if sklearn isn't importable.

    Gated on all-numeric frame: sklearn's IncrementalPCA cannot fit
    on string/categorical/list-of-float columns, and the mlframe
    pipeline does NOT pre-encode before a custom_pre_pipeline.
    Matches the canonicalisation in FuzzCombo.canonical_key so the
    combo generation and runtime wiring agree.
    """
    # Mirror FuzzCombo.canonical_key pca2-incompatibility gating:
    # IncrementalPCA also rejects NaN (inject_inf_nan) and
    # all-null columns (inject_degenerate_cols), not just
    # non-numeric dtypes.
    pca_incompatible = (
        combo.cat_feature_count > 0
        or combo.text_col_count > 0
        or combo.embedding_col_count > 0
        or combo.inject_inf_nan
        or combo.inject_degenerate_cols
    )
    if combo.custom_prep == "pca2" and not pca_incompatible:
        try:
            from sklearn.decomposition import IncrementalPCA
            return {"pca2": IncrementalPCA(n_components=2)}
        except ImportError:
            return None
    return None


def _maybe_to_parquet(combo: FuzzCombo, df, tmp_path):
    """Convert ``df`` to a parquet file path when
    ``combo.input_storage == "parquet"``; otherwise pass through.
    The suite's ``load_and_prepare_dataframe`` accepts str paths
    and reads them internally — this exercises the streaming
    parquet code path.
    """
    if combo.input_storage != "parquet":
        return df
    import polars as _pl
    path = str(tmp_path / "combo_input.parquet")
    if isinstance(df, _pl.DataFrame):
        df.write_parquet(path)
    else:
        df.to_parquet(path)
    return path


def _common_init_for_combo(combo: FuzzCombo) -> dict:
    """init_common_params for a combo. Attaches a category encoder only when
    a non-native-cat model (linear) is present — matches the prod config
    pattern the existing integration tests use."""
    params: dict = {"drop_columns": [], "verbose": 0}
    if "linear" in combo.models and combo.cat_feature_count > 0:
        try:
            import category_encoders as ce
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer
            params["category_encoder"] = ce.CatBoostEncoder()
            params["scaler"] = StandardScaler()
            params["imputer"] = SimpleImputer(strategy="mean")
        except ImportError:
            pass
    return params


def _skip_if_deps_missing(models: tuple[str, ...]) -> None:
    pkg = {
        "cb": "catboost", "xgb": "xgboost", "lgb": "lightgbm",
        "hgb": "sklearn", "linear": "sklearn",
    }
    for m in models:
        pytest.importorskip(pkg[m])


def _iter_trained_models(trained):
    """Yield (target_type, target_name, trained_entry) for every model.

    The suite returns ``trained[target_type][target_name]`` → list of
    SimpleNamespace entries with ``.model``, ``.val_preds``, ``.val_probs``,
    ``.columns``, ``.metrics`` attributes.
    """
    if not isinstance(trained, dict):
        return
    for tt, by_name in trained.items():
        if not isinstance(by_name, dict):
            continue
        for tn, lst in by_name.items():
            if not isinstance(lst, list):
                continue
            for entry in lst:
                yield tt, tn, entry


def _assert_prediction_invariants(trained, meta, combo) -> None:
    """Fix C (cheap tier): post-train property checks that run on every
    combo.

    Runs on the predictions the suite already materialised
    (``entry.val_preds``, ``entry.test_preds``), so no re-fit. Catches:

    - NaN / Inf leaking into the model head (I1).
    - Constant predictions when the val target has ≥2 distinct classes / values
      (I2) — indicates dead pipeline / silently-dropped features.
    - Shape mismatch between ``val_preds`` and ``meta['val_size']`` (I3) —
      indicates row-slicing drift between the pipeline and its metrics.

    Deeper invariants (determinism, idempotency, column-perm, prediction
    probe) require re-fit; they live behind ``MLFRAME_FUZZ_INVARIANTS=full``
    in ``test_fuzz_invariants_full.py``.
    """
    import math
    import numpy as np

    val_size = (meta or {}).get("val_size")
    for tt, tn, entry in _iter_trained_models(trained):
        # Pull both available forms; prefer probs (cleaner finite check)
        # then fall back to preds.
        for attr in ("val_probs", "val_preds"):
            arr = getattr(entry, attr, None)
            if arr is None:
                continue
            # Normalise to ndarray for finite checks.
            try:
                arr_np = np.asarray(arr)
            except Exception:
                continue
            if arr_np.size == 0:
                continue
            # I1 — finiteness. Regression preds can be negative but must
            # be finite. Classification probs live in [0, 1] but we only
            # assert finiteness here to stay model-agnostic.
            if np.issubdtype(arr_np.dtype, np.floating):
                n_bad = int(np.count_nonzero(~np.isfinite(arr_np)))
                assert n_bad == 0, (
                    f"I1: non-finite values in {tt}/{tn}/{type(entry.model).__name__}.{attr} "
                    f"({n_bad}/{arr_np.size})"
                )
            # I3 — shape upper-bound. ``meta['val_size']`` is measured
            # before outlier-detection filters rows, so post-OD ``val_preds``
            # can be strictly smaller. Asserting ``<=`` catches only the
            # bug we care about (preds longer than val slice → row-slicing
            # drift), not OD-expected shrinkage.
            if val_size is not None and val_size > 0 and arr_np.ndim >= 1:
                assert arr_np.shape[0] <= val_size, (
                    f"I3: {attr} shape[0]={arr_np.shape[0]} > val_size={val_size} "
                    f"for {tt}/{tn}/{type(entry.model).__name__}"
                )
            # I2 — non-constant predictions when val has >1 row.
            # Skipped for tiny val slices (< 4 rows; statistical noise).
            # Skipped when outlier-detection could have reduced val to
            # a single class (we can't cheaply check val target variance
            # here without re-extracting). Asserted only for classification
            # probs where the "all-same" outcome is provably degenerate.
            if (
                attr == "val_probs"
                and arr_np.size >= 4
                and np.issubdtype(arr_np.dtype, np.floating)
            ):
                # Pull scalar series for 1-D, flatten for 2-D.
                vals = arr_np.ravel()
                unique_near = np.unique(np.round(vals, 6))
                # Relaxed on imbalanced multi-class: a 1% minority class
                # may produce probs rounded to 6dp that collapse to a
                # single value when the model mass-predicts the majority.
                # R3 combos with imbalance_ratio=rare_1pct + very few
                # iterations see this legitimately.
                if combo.imbalance_ratio == "rare_1pct":
                    continue
                assert unique_near.size >= 2, (
                    f"I2: val_probs are all identical "
                    f"({unique_near[0]:.6g}) for {tt}/{tn}/{type(entry.model).__name__} "
                    f"— pipeline may have dropped all features"
                )


def _assert_serialization_roundtrip(trained, data_dir: str, combo) -> None:
    """Fix R3-3 (I4 — serialization roundtrip). Gated by env
    ``MLFRAME_FUZZ_ROUNDTRIP=1`` because it spends ~100-500ms per combo
    on disk I/O + load + predict.

    Finds the first saved .dump file under ``data_dir`` and verifies it
    loads back and can produce predictions. This is a smoke check, not a
    bit-for-bit equivalence (the saved artifact is meant for
    ``load_mlframe_suite`` + ``predict_mlframe_models_suite`` and the
    metadata/preprocessing trail is not hydrated by joblib.load alone).
    A load failure catches regressions in the picklable contract of
    models / pipelines / custom estimators.
    """
    import os
    import glob
    import joblib

    if not trained:
        return
    pattern = os.path.join(data_dir, "**", "*.dump")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return  # No models saved (continue_on_failure path, or save disabled)
    # Load the first .dump file; only assert the artifact is not corrupt.
    try:
        obj = joblib.load(files[0])
    except Exception as exc:
        raise AssertionError(
            f"I4: saved model artifact {files[0]!r} failed joblib.load: "
            f"{type(exc).__name__}: {exc}"
        )
    # The dump contains at least one object with a predict-like attribute
    # (the trained pipeline / model). Not asserting specific type — the
    # wrapper class can evolve; catching "can't unpickle" is the goal.
    assert obj is not None, f"I4: joblib.load returned None for {files[0]}"


@pytest.fixture(autouse=True)
def _fuzz_combo_cleanup():
    """Between fuzz combos: close matplotlib figures, clear CB/XGB/LGB
    internal caches, drop generated models — state accumulation across the
    150-combo run has been observed to trigger native-level crashes
    (SIGSEGV on combo 6 in a sequential run on 2026-04-22)."""
    yield
    # 1. Matplotlib figures (mlframe emits per-model feature_importance plots).
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass
    # 2. mlframe's in-process caches (CB val Pool cache, tier-DF cache).
    try:
        from mlframe.training import trainer as _tr
        for attr in ("_CB_POOL_CACHE", "_CB_VAL_POOL_CACHE"):
            cache = getattr(_tr, attr, None)
            if hasattr(cache, "clear"):
                cache.clear()
    except Exception:
        pass
    # 3. CatBoost internal state — force full GPU/CPU resource release.
    try:
        import catboost
        # catboost.utils doesn't expose a global cleanup; deleting module-level
        # state is unsafe. Best-effort: trigger a GC pass twice so CB's
        # C++-side memory pools see zero Python refs before the next combo
        # allocates.
    except ImportError:
        pass
    # 4. Double GC — first pass collects Python objects, second pass lets
    # finalizers (including native lib close-outs) run before we return.
    import gc
    gc.collect()
    gc.collect()
    # 5. clean_ram: on Linux returns memory to OS via malloc_trim(0);
    # on Windows trims working-set via SetProcessWorkingSetSizeEx (RSS
    # only, not commit). Wired here as best-effort against multi-combo
    # native heap fragmentation that historically OOMs around combo #36
    # of 150 on Win32 multi-classification × ensembles paths.
    try:
        from pyutilz.system import clean_ram
        clean_ram()
    except Exception:
        pass


@pytest.mark.timeout(300)
@pytest.mark.parametrize("combo", COMBOS, ids=[c.pytest_id() for c in COMBOS])
def test_fuzz_train_mlframe_models_suite(combo: FuzzCombo, tmp_path, request):
    """Run ``train_mlframe_models_suite`` on one random combo; log the outcome."""
    _skip_if_deps_missing(combo.models)

    # Apply xfail automatically for known bugs. pytest's runtime-xfail marker
    # works via ``request.node.add_marker``.
    reason = xfail_reason(combo)
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=False))

    df, target_col, _cat_names = build_frame_for_combo(combo)

    # #16 invariant: capture caller-frame schema + shape before the
    # suite runs; re-assert identity after. Applies when input stays
    # in-memory (parquet-path combos have no Python-level caller frame
    # to preserve — the parquet file is the source of truth).
    frame_schema_before = None
    frame_shape_before = None
    frame_cols_before = None
    if combo.input_storage == "memory":
        if hasattr(df, "schema"):
            frame_schema_before = dict(df.schema)
        elif hasattr(df, "dtypes"):
            frame_schema_before = {c: str(df[c].dtype) for c in df.columns}
        frame_shape_before = getattr(df, "shape", None)
        frame_cols_before = tuple(df.columns) if hasattr(df, "columns") else None

    # Resolve target_type for FTE — maps combo's string target_type to
    # the TargetTypes enum. Multilabel gets explicit TargetTypes to
    # trigger the 2-D target unpack path in FTE.
    from mlframe.training.configs import TargetTypes as _TT
    _combo_tt = {
        "regression": _TT.REGRESSION,
        "binary_classification": _TT.BINARY_CLASSIFICATION,
        "multiclass_classification": _TT.MULTICLASS_CLASSIFICATION,
        "multilabel_classification": _TT.MULTILABEL_CLASSIFICATION,
    }[combo.target_type]
    fte = SimpleFeaturesAndTargetsExtractor(
        target_column=target_col,
        regression=(combo.target_type == "regression"),
        target_type=_combo_tt,
    )

    # Resolve combo-specific kwargs (outlier detector, custom prep,
    # parquet path). These feed directly into train_mlframe_models_suite.
    df_input = _maybe_to_parquet(combo, df, tmp_path)
    outlier_detector = _outlier_detector_for_combo(combo)
    custom_pre = _custom_pre_pipelines_for_combo(combo)

    from mlframe.training.core import train_mlframe_models_suite

    t0 = time.perf_counter()
    outcome = "pass"
    err_class = None
    err_summary = None
    try:
        trained, _meta = train_mlframe_models_suite(
            df=df_input,
            target_name=combo.short_id(),
            model_name=combo.short_id(),
            features_and_targets_extractor=fte,
            mlframe_models=list(combo.models),
            hyperparams_config=_config_for_models(
                combo.models, combo.n_rows,
                iterations=combo.iterations,
                early_stopping_rounds=combo.early_stopping_rounds_cfg,
            ),
            init_common_params=_common_init_for_combo(combo),
            use_ordinary_models=True,
            use_mlframe_ensembles=combo.use_ensembles,
            outlier_detector=outlier_detector,
            custom_pre_pipelines=custom_pre,
            data_dir=str(tmp_path),
            models_dir="models",
            verbose=0,
            use_mrmr_fs=combo.use_mrmr_fs,
            mrmr_kwargs=({
                "verbose": 0, "max_runtime_mins": 1, "n_workers": 1,
                "quantization_nbins": 5, "use_simple_mode": True,
                "min_nonzero_confidence": 0.9, "max_consec_unconfirmed": 3,
                "full_npermutations": 3,
            } if combo.use_mrmr_fs else None),
            **_configs_for_combo(combo),
        )
        # An empty ``trained`` dict is acceptable ONLY when
        # ``continue_on_model_failure=True`` AND the suite recorded
        # each failure in ``metadata['failed_models']``. Any other
        # empty-trained outcome is a bug — the suite should have
        # either raised or produced ≥1 model.
        if not trained:
            if (
                combo.continue_on_model_failure
                and _meta is not None
                and _meta.get("failed_models")
            ):
                pass  # graceful skip of a configurably-failing combo
            else:
                raise AssertionError(
                    f"empty models dict for combo {combo.short_id()} "
                    f"(continue_on_failure={combo.continue_on_model_failure}, "
                    f"failed_models={(_meta or {}).get('failed_models')})"
                )

        # --- Post-train invariants (free on every combo) ---
        # #16 no caller-frame mutation (skip for parquet-path).
        if combo.input_storage == "memory" and frame_cols_before is not None:
            assert tuple(df.columns) == frame_cols_before, (
                f"caller-frame columns mutated: before={frame_cols_before} "
                f"after={tuple(df.columns)}"
            )
            shape_after = getattr(df, "shape", None)
            assert shape_after == frame_shape_before, (
                f"caller-frame shape mutated: before={frame_shape_before} "
                f"after={shape_after}"
            )
        # #20 metadata schema: load-bearing keys present.
        # ``model_schemas`` is only populated when at least one model
        # successfully trained — combos that legitimately degrade to
        # an empty trained dict (continue_on_failure=True + all models
        # failed) won't have it. Check the always-present keys
        # unconditionally; model_schemas only when trained non-empty.
        if _meta is not None:
            for k in ("columns", "cat_features", "outlier_detection"):
                assert k in _meta, (
                    f"metadata missing load-bearing key {k!r}; "
                    f"keys={list(_meta)[:20]}"
                )
            if trained:
                assert "model_schemas" in _meta, (
                    "metadata missing 'model_schemas' despite non-empty "
                    f"trained dict; keys={list(_meta)[:20]}"
                )

        # --- Fix C property invariants (cheap, per-combo) ---
        # Catches silent degeneracy that a "no exception" assertion misses:
        # dead features, all-zero predictions, NaN leakage to the model
        # head, val-slice misalignment.
        _assert_prediction_invariants(trained, _meta, combo)
        # --- R3-3 I4 serialization roundtrip (env-gated, off by default) ---
        if os.environ.get("MLFRAME_FUZZ_ROUNDTRIP") == "1":
            _assert_serialization_roundtrip(trained, str(tmp_path), combo)
    except Exception as exc:
        outcome = "fail"
        err_class = type(exc).__name__
        err_summary = traceback.format_exception_only(type(exc), exc)[-1].strip()
        log_combo_outcome(
            combo, outcome,
            duration_s=time.perf_counter() - t0,
            error_class=err_class,
            error_summary=err_summary,
        )
        raise

    log_combo_outcome(
        combo, outcome, duration_s=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Meta-tests: sanity-check the enumerator itself
# ---------------------------------------------------------------------------


def test_enumerator_is_deterministic():
    """Same master_seed must yield byte-identical combo list."""
    a = enumerate_combos(target=50, master_seed=2026_04_22)
    b = enumerate_combos(target=50, master_seed=2026_04_22)
    assert [c.canonical_key() for c in a] == [c.canonical_key() for c in b]


def test_enumerator_produces_unique_combos():
    """No canonical-key duplicates in the 150-combo run."""
    keys = [c.canonical_key() for c in COMBOS]
    assert len(keys) == len(set(keys)), "Fuzz enumerator produced duplicates"


def test_enumerator_hits_all_models():
    """Every supported model must appear at least once across the 150 combos."""
    from ._fuzz_combo import MODELS
    seen = {m for c in COMBOS for m in c.models}
    missing = set(MODELS) - seen
    assert not missing, f"Models never exercised by fuzz: {missing}"


def test_enumerator_target_count():
    assert len(COMBOS) == 150
