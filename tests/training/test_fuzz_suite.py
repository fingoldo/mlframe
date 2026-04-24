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


def _config_for_models(models: tuple[str, ...], n_rows: int, iterations: int = 3) -> dict:
    cfg: dict = {"iterations": iterations}
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
    }
    if fairness_features:
        behavior_kwargs["fairness_features"] = fairness_features
        behavior_kwargs["fairness_min_pop_cat_thresh"] = 10
    return {
        "pipeline_config": PolarsPipelineConfig(
            use_polarsds_pipeline=combo.use_polarsds_pipeline,
        ),
        "feature_types_config": FeatureTypesConfig(
            auto_detect_feature_types=combo.auto_detect_cats,
            use_text_features=combo.use_text_features,
            honor_user_dtype=combo.honor_user_dtype,
            text_features=text_features,
            embedding_features=embedding_features,
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
    transformer. Fails-open if sklearn isn't importable."""
    if combo.custom_prep == "pca2":
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


@pytest.mark.timeout(120)
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

    fte = SimpleFeaturesAndTargetsExtractor(
        target_column=target_col,
        regression=(combo.target_type == "regression"),
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
                combo.models, combo.n_rows, iterations=combo.iterations,
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
        assert trained, f"empty models dict for combo {combo.short_id()}"

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
        if _meta is not None:
            for k in ("columns", "cat_features", "outlier_detection", "model_schemas"):
                assert k in _meta, (
                    f"metadata missing load-bearing key {k!r}; "
                    f"keys={list(_meta)[:20]}"
                )
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
