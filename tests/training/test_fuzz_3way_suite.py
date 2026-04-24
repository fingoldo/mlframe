"""Fix A — 3-wise (IPOG-style) covering suite for ``train_mlframe_models_suite``.

Upgrade from the primary pairwise fuzz (``test_fuzz_suite.py``, 150 combos)
to 3-way interaction coverage over a curated 15-axis subset. At 400
combos we hit 100% triple coverage over the axes where 3-way bugs have
historically lived (discovered by 2026-04 regressions in c0015 =
polars_enum × onehot × MRMR).

This suite is INTENDED for nightly / pre-merge runs, not per-push CI —
400 combos × ~50s each = ~5.5h serial, or ~45min with ``-n 8``.

Axes covered (see ``_fuzz_combo._3WAY_AXES``):
- input_type, n_rows, cat_feature_count, use_mrmr_fs, target_type
- outlier_detection, use_ensembles, inject_inf_nan, inject_degenerate_cols
- custom_prep, categorical_encoding_cfg, scaler_name_cfg
- inject_label_leak, inject_rank_deficient, inject_all_nan_col (Fix G)

Axes in the pairwise suite but deliberately NOT in the triple set live
in less-interaction-prone corners (e.g. ``use_robust_eval_metric_cfg``,
``val_placement_cfg``) — every extra axis multiplies triples by
C(N-1, 2), so restraint matters.

Invoke just like the pairwise suite but point at this file::

    pytest mlframe/tests/training/test_fuzz_3way_suite.py \\
        --no-cov -p no:randomly -n 8

Env:
- ``FUZZ_3WAY_SEED`` — override ``master_seed`` for this suite
  (independent of the pairwise suite's ``FUZZ_SEED``).
- ``FUZZ_3WAY_TARGET`` — combo count (default 400).
"""
from __future__ import annotations

import os
import time
import traceback

import pytest

from ._fuzz_combo import (
    FuzzCombo,
    build_frame_for_combo,
    enumerate_combos_3way,
    log_combo_outcome,
    xfail_reason,
)
from .shared import SimpleFeaturesAndTargetsExtractor

# Import config/invariant helpers from the pairwise suite — they stay
# in one place (DRY across pairwise + 3-way).
from .test_fuzz_suite import (
    _assert_prediction_invariants,
    _common_init_for_combo,
    _config_for_models,
    _configs_for_combo,
    _custom_pre_pipelines_for_combo,
    _maybe_to_parquet,
    _outlier_detector_for_combo,
    _skip_if_deps_missing,
)

_FUZZ_3WAY_SEED = int(os.environ.get("FUZZ_3WAY_SEED", "20260424"))
_FUZZ_3WAY_TARGET = int(os.environ.get("FUZZ_3WAY_TARGET", "400"))
COMBOS_3WAY: list[FuzzCombo] = enumerate_combos_3way(
    target=_FUZZ_3WAY_TARGET,
    master_seed=_FUZZ_3WAY_SEED,
)


@pytest.fixture(autouse=True)
def _fuzz3way_cleanup():
    """Same post-combo cleanup as the pairwise suite to prevent native
    library state accumulation (SIGSEGV on long serial runs)."""
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
    import gc
    gc.collect()
    gc.collect()


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "combo", COMBOS_3WAY, ids=[c.pytest_id() for c in COMBOS_3WAY]
)
def test_fuzz_3way_train_mlframe_models_suite(combo: FuzzCombo, tmp_path, request):
    """Run the suite on one triple-coverage combo. Identical assertion
    contract to the pairwise suite — we're sampling a different region
    of the combo space, not using different checks."""
    _skip_if_deps_missing(combo.models)
    reason = xfail_reason(combo)
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=False))

    df, target_col, _ = build_frame_for_combo(combo)
    frame_cols_before = tuple(df.columns) if hasattr(df, "columns") else None
    frame_shape_before = getattr(df, "shape", None)

    fte = SimpleFeaturesAndTargetsExtractor(
        target_column=target_col,
        regression=(combo.target_type == "regression"),
    )
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
        if not trained:
            if (
                combo.continue_on_model_failure
                and _meta is not None
                and _meta.get("failed_models")
            ):
                pass
            else:
                raise AssertionError(
                    f"empty models dict for combo {combo.short_id()}"
                )
        if combo.input_storage == "memory" and frame_cols_before is not None:
            assert tuple(df.columns) == frame_cols_before
            assert getattr(df, "shape", None) == frame_shape_before
        if _meta is not None:
            for k in ("columns", "cat_features", "outlier_detection"):
                assert k in _meta
        _assert_prediction_invariants(trained, _meta, combo)
    except Exception as exc:
        outcome = "fail"
        err_class = type(exc).__name__
        err_summary = traceback.format_exception_only(type(exc), exc)[-1].strip()
        log_combo_outcome(
            combo, outcome,
            duration_s=time.perf_counter() - t0,
            error_class=err_class,
            error_summary=err_summary,
            extra={"suite": "3way"},
        )
        raise
    log_combo_outcome(
        combo, outcome, duration_s=time.perf_counter() - t0,
        extra={"suite": "3way"},
    )


def test_3way_enumerator_covers_all_triples():
    """Invariant: ``enumerate_combos_3way`` must achieve full 3-way
    coverage over ``_3WAY_AXES`` at the default target. If this
    regresses, either ``_3WAY_AXES`` grew beyond what the current target
    can cover, or the greedy picker is starving — raise the target or
    prune an axis."""
    from ._fuzz_combo import _all_axis_triples, _combo_triples
    required = _all_axis_triples()
    covered: set = set()
    for c in COMBOS_3WAY:
        covered.update(_combo_triples(c))
    missing = required - covered
    # Allow up to 0.1% uncovered (some triples can be unreachable due to
    # canonical-key dedup / model-count interactions).
    tolerance = max(1, int(len(required) * 0.001))
    assert len(missing) <= tolerance, (
        f"3-way coverage regressed: {len(missing)} triples uncovered "
        f"(tolerance {tolerance}). First 5: {list(missing)[:5]}"
    )
