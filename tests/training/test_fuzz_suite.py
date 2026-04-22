"""Randomized fuzz coverage for ``train_mlframe_models_suite``.

Feeds ~150 unique, pairwise-covering combos through the suite and records
every combo's outcome to ``_fuzz_results.jsonl`` for later analysis.

Known-bug xfail rules live in ``_fuzz_combo.KNOWN_XFAIL_RULES`` and are
applied automatically per combo via ``pytest.mark.xfail`` in the test
function — new bugs discovered by fuzzing should be added there once
they're traced to a specific combo predicate.
"""
from __future__ import annotations

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
COMBOS: list[FuzzCombo] = enumerate_combos(target=150, master_seed=2026_04_22)


def _config_for_models(models: tuple[str, ...], n_rows: int) -> dict:
    cfg: dict = {"iterations": 3 if n_rows <= 300 else 5}
    if "lgb" in models:
        cfg["lgb_kwargs"] = {"device_type": "cpu", "verbose": -1}
    if "xgb" in models:
        cfg["xgb_kwargs"] = {"device": "cpu", "verbosity": 0}
    if "cb" in models:
        cfg["cb_kwargs"] = {"task_type": "CPU", "verbose": 0}
    return cfg


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
    # 3. Python-level GC so native libraries release their allocations
    # before the next combo imports a fresh model.
    import gc
    gc.collect()


@pytest.mark.timeout(60)
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

    fte = SimpleFeaturesAndTargetsExtractor(
        target_column=target_col,
        regression=(combo.target_type == "regression"),
    )

    from mlframe.training.core import train_mlframe_models_suite

    t0 = time.perf_counter()
    outcome = "pass"
    err_class = None
    err_summary = None
    try:
        trained, _meta = train_mlframe_models_suite(
            df=df,
            target_name=combo.short_id(),
            model_name=combo.short_id(),
            features_and_targets_extractor=fte,
            mlframe_models=list(combo.models),
            hyperparams_config=_config_for_models(combo.models, combo.n_rows),
            init_common_params=_common_init_for_combo(combo),
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
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
        )
        assert trained, f"empty models dict for combo {combo.short_id()}"
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
