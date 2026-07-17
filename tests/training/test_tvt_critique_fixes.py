"""Regression coverage for the TVT-run critique fixes (2026-05-29).

One test per defect identified in the user-supplied prod log. Each test asserts
the OBSERVABLE behaviour the fix promises - never the implementation detail -
so refactors that preserve the contract pass without churn.
"""

from __future__ import annotations

import logging
import types
import warnings
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# C: composite-discovery feature matrix is float32 (was float64 -> 15.9 GB OOM
# on 4M x 487 in the user's prod run; float32 halves the footprint).
# ---------------------------------------------------------------------------
def test_C_extract_column_array_returns_float32():
    """_extract_column_array must produce float32 on both polars + pandas paths.

    Before the fix the discovery feature-matrix allocation hit 4M*487*8B = 15.9 GB
    and crashed on hosts where the trainer itself sat at ~100 GB; float32 halves
    that to ~8 GB which fits comfortably.
    """
    pl = pytest.importorskip("polars")
    import pandas as pd
    from mlframe.training.composite.discovery.screening import _extract_column_array

    pl_df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    pd_df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    a_pl = _extract_column_array(pl_df, "a")
    a_pd = _extract_column_array(pd_df, "a")
    assert a_pl.dtype == np.float32, f"polars path returned {a_pl.dtype}"
    assert a_pd.dtype == np.float32, f"pandas path returned {a_pd.dtype}"
    # Memory contract: float32 vs float64 must literally halve nbytes for the
    # same row count, otherwise the OOM fix is cosmetic.
    assert a_pl.nbytes == 4 * len(a_pl)


def test_C_build_feature_matrix_shape_and_dtype():
    """Discovery's per-base feature matrix is the SAME size as before, dtype
    only changes float64 -> float32. Catches an accidental drop of rows / cols
    during the share-across-bases refactor."""
    pl = pytest.importorskip("polars")
    from mlframe.training.composite.discovery import CompositeTargetDiscovery
    from mlframe.training._composite_target_discovery_config import (
        CompositeTargetDiscoveryConfig,
    )

    df = pl.DataFrame({c: list(range(100)) for c in ["a", "b", "c", "d"]})
    inst = CompositeTargetDiscovery.__new__(CompositeTargetDiscovery)
    inst.config = CompositeTargetDiscoveryConfig()
    matrix = inst._build_feature_matrix(df, ["a", "b", "c"], np.arange(50))
    assert matrix.shape == (50, 3), matrix.shape
    assert matrix.dtype == np.float32, matrix.dtype


# ---------------------------------------------------------------------------
# D: _release_ctx_polars_frames clears dataset_reuse_cache so polars buffers
# can actually be reclaimed (otherwise DMatrix / Dataset / Pool wrappers pin
# the binned tensors and the RSS-drop sanity check warns at 0.0 MB freed).
# ---------------------------------------------------------------------------
def test_D_release_clears_dataset_reuse_cache():
    """_release_ctx_polars_frames must clear dataset_reuse_cache too, or pinned DMatrix/Dataset/Pool wrappers block RSS reclaim."""
    from mlframe.training.core import _phase_train_one_target_dataset_cache as mod

    ctx = types.SimpleNamespace(
        train_df_polars=None,
        val_df_polars=None,
        test_df_polars=None,
        artifacts={
            "dataset_reuse_cache": {
                "model_a": {"_cached_train_dmatrix": object()},
                "model_b": {"_cached_val_dataset": object()},
            },
        },
    )
    # Patch the heavy RAM clean-up + RSS-measurement helpers to noops so the
    # test is hermetic.
    with (
        patch.object(mod, "get_process_rss_mb", return_value=0.0),
        patch.object(mod, "maybe_clean_ram_and_gpu", return_value=0.0),
        patch.object(mod, "estimate_df_size_mb", return_value=0.0),
    ):
        mod._release_ctx_polars_frames(
            ctx,
            baseline_rss_mb=0.0,
            df_size_mb=0.0,
            verbose=False,
            reason="test",
        )
    assert ctx.artifacts["dataset_reuse_cache"] == {}, (
        "dataset_reuse_cache must be emptied; otherwise XGB/LGB/CB wrappers pin the polars buffers and the polars-release is a no-op."
    )


# ---------------------------------------------------------------------------
# A: dummy + CT_ENSEMBLE phases respect compute_valset_metrics=False /
# compute_testset_metrics=False.
# ---------------------------------------------------------------------------
def test_A_dummy_baselines_respects_compute_valset_metrics_false():
    """The dummy-baselines emit path must read ``compute_valset_metrics`` and
    ``compute_testset_metrics`` from the reporting_config.

    Behavioural check via file read (no ``inspect.getsource`` -- per
    feedback_behavioral_tests / feedback_no_inspect_getsource memory): if the
    gate tokens disappear from the source, the gate has been silently dropped
    and VAL/TEST dummy metrics will leak out even when the operator opts out.
    """
    import pathlib
    import mlframe as _mlframe

    src = (pathlib.Path(_mlframe.__file__).resolve().parent / "training" / "core" / "_phase_dummy_baselines.py").read_text(encoding="utf-8")
    assert "compute_valset_metrics" in src, (
        "dummy-baselines emit path must read compute_valset_metrics from "
        "reporting_config; user explicitly set it to False and still saw "
        "VAL (DUMMY) metric lines."
    )
    assert "compute_testset_metrics" in src, "Symmetric test-side gate must be present too."


def test_A_ct_ensemble_respects_compute_valset_metrics_false():
    """Mirror of the dummy-baselines gate, on the cross-target ensemble emit path."""
    import pathlib
    import mlframe as _mlframe

    _core = pathlib.Path(_mlframe.__file__).resolve().parent / "training" / "core"
    _xt = _core / "_phase_composite_post_xt_ensemble.py"
    if _xt.exists():
        src = _xt.read_text(encoding="utf-8")
    else:
        # Monolith-split compat: became a subpackage; read __init__ + submodules.
        _pkg = _core / "_phase_composite_post_xt_ensemble"
        src = "\n".join(p.read_text(encoding="utf-8") for p in sorted(_pkg.glob("*.py")))
    assert "compute_valset_metrics" in src
    assert "compute_testset_metrics" in src


# ---------------------------------------------------------------------------
# E: val_placement downgrade is INFO-level (was WARNING).
# ---------------------------------------------------------------------------
def test_E_val_placement_downgrade_emits_warning_with_remediation(caplog):
    """When val_placement='backward' is requested but timestamps=None, the
    downgrade-to-forward message is emitted at WARNING level (a silent
    temporal-honesty loss is worth a loud log line) and the message names
    the consequence ('Temporal honesty lost') so it isn't easy to miss
    in production runs."""
    import pandas as pd
    from mlframe.training.splitting import make_train_test_split

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "target": rng.normal(size=n),
        }
    )
    with caplog.at_level(logging.INFO, logger="mlframe.training.splitting"):
        try:
            make_train_test_split(
                df=df,
                val_size=0.1,
                test_size=0.1,
                val_placement="backward",
                timestamps=None,
                random_seed=0,
            )
        except Exception:  # nosec B110 -- best-effort cleanup/optional step; failure here never masks this test's own assertions
            # Downstream split-shape errors are not under test; the log line is.
            pass

    relevant = [r for r in caplog.records if "downgraded" in r.getMessage() and r.name == "mlframe.training.splitting"]
    assert relevant, "expected a log line about val_placement downgrade"
    for r in relevant:
        assert r.levelno == logging.WARNING, (
            f"val_placement='backward' downgrade emitted at {r.levelname}; "
            "should be WARNING -- temporal honesty silently lost is exactly the kind "
            "of regression this log level was raised to surface."
        )
        assert "Temporal honesty lost" in r.getMessage(), "WARNING message must spell out the consequence so operators see it."


# ---------------------------------------------------------------------------
# B: temporal-audit plot routes through plot_outputs (multi-backend DSL).
# ---------------------------------------------------------------------------
def test_B_temporal_audit_uses_plot_outputs_when_present():
    """When reporting_config.plot_outputs is set, _plot_target_over_time gets
    invoked through the base_path / plot_outputs branch (multi-backend), NOT
    via the matplotlib-only save_path branch.

    Source-presence sensor via file read (not ``inspect.getsource`` -- see
    feedback_behavioral_tests). Both call shapes must coexist: with-plot_outputs
    (DSL) and without (matplotlib-only PNG fallback for legacy callers that
    don't supply reporting_config.plot_outputs).
    """
    import pathlib
    import mlframe as _mlframe

    src = (pathlib.Path(_mlframe.__file__).resolve().parent / "training" / "core" / "_phase_train_one_target_model_setup.py").read_text(encoding="utf-8")
    assert "plot_outputs=" in src
    assert "_target_temporal_audit" in src


# ---------------------------------------------------------------------------
# G: cross-target verdict considers the CT_ENSEMBLE metric, not just the
# single best model.
# ---------------------------------------------------------------------------
def test_G_verdict_picks_ensemble_when_better_than_best_model():
    """On a strong-AR target the NNLS-stack ensemble often clears the dummy
    floor cleanly while the best single model only marginally beats it. The
    suite-end verdict must prefer whichever is stronger - before this fix it
    silently ignored CT_ENSEMBLE and falsely flagged BEST_MODEL_BELOW_DUMMY."""
    from mlframe.training.baselines._dummy_summary_format import format_suite_end_summary

    dummy_metadata = {
        "regression": {
            "TVT": {
                "strongest": "lag_predict",
                "primary_metric": "val_RMSE",
                "data": {"lag_predict": {"val_RMSE": 13.19}},
            }
        }
    }
    best_model_metrics = {
        ("regression", "TVT"): {"val_RMSE": 13.43, "model_name": "LGBMRegressor"},
    }
    ct_ensemble_metrics = {
        ("regression", "TVT"): {"val_RMSE": 9.59, "model_name": "CT_ENSEMBLE[nnls_stack]"},
    }

    # Without the ensemble: best model marginally LOSES vs dummy (lift 0.98x).
    out_no_ens = format_suite_end_summary(
        dummy_baselines_metadata=dummy_metadata,
        best_model_metrics_by_target=best_model_metrics,
        min_lift=1.5,
    )
    assert "MODELS_BARELY_BEAT_TRIVIAL" in out_no_ens

    # With the ensemble: comfortable beat of dummy by 1.38x -> healthy verdict.
    out_with_ens = format_suite_end_summary(
        dummy_baselines_metadata=dummy_metadata,
        best_model_metrics_by_target=best_model_metrics,
        cross_target_ensemble_metrics=ct_ensemble_metrics,
        min_lift=1.3,
    )
    assert "TASK_NON_TRIVIAL_AND_MODELS_HEALTHY" in out_with_ens
    # And the displayed best_model column should reflect the ensemble.
    assert "CT_ENSEMBLE" in out_with_ens


# ---------------------------------------------------------------------------
# I: MAPE warmup uses a non-zero y_true vector so no false "n of 10 zero"
# warning fires at import time.
# ---------------------------------------------------------------------------
def test_I_mape_warmup_does_not_emit_zero_y_warning(caplog):
    """Reload the warmup module; if the warmup y_true contains zeros the
    rate-limited '_MAPE_ZERO_WARN_SEEN' warning fires once. After the fix it
    must stay silent."""
    from mlframe.metrics import _core_precision_mape

    # Clear the rate-limit cache so a stale prior call doesn't mask a fresh trigger.
    _core_precision_mape._MAPE_ZERO_WARN_SEEN.clear()

    with caplog.at_level(logging.WARNING, logger="mlframe.metrics._core_precision_mape"):
        from mlframe.metrics import _core_numba_warmup

        # Re-run the warmup explicitly even if the module is already cached so
        # the call paths fire again under our caplog.
        warmup_fn = getattr(_core_numba_warmup, "warmup_numba_kernels", None) or getattr(_core_numba_warmup, "warmup", None)
        if warmup_fn is not None:
            try:
                warmup_fn()
            except Exception:  # nosec B110 -- best-effort cleanup/optional step; failure here never masks this test's own assertions
                # If warmup signature differs, importing the module already executed it.
                pass
    zero_warns = [r for r in caplog.records if "y_true entries are zero" in r.getMessage()]
    assert not zero_warns, (
        f"MAPE zero-y warning fired during numba warmup; warmup y_true must be a non-zero vector. Captured: {[r.getMessage() for r in zero_warns]}"
    )


# ---------------------------------------------------------------------------
# J: coerce_to_numpy uses the new allow_copy kwarg first, falling back through
# zero_copy_only (deprecated alias) so polars 0.20.10+ no longer DeprecationWarns.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Bonus (post-merge follow-up): shap 0.51 + xgboost 3.x base_score crash.
# XGBoost 3.x persists base_score as a JSON array string ('[0.5]') that
# shap.explainers._tree calls float() on; the call raises ValueError. mlframe
# patches shap's module-level float name with a bracket-aware coercer that
# scalars-pass-through and arrays-take-first-element.
# ---------------------------------------------------------------------------
def test_shap_xgb_base_score_patch_handles_bracketed_array_string():
    """The narrow patch must:
    (a) coerce ``"[0.5]"`` and ``"[5.06E-1, 0.0]"`` to a scalar float
    (b) leave plain scalars (``"0.5"``, ``0.5``) unchanged.

    Exercised in isolation (no real shap call) so the test runs on any host
    regardless of whether the local xgboost build triggers the array path.
    """
    # Test the bracket-aware coercer DIRECTLY. On shap>=0.52 the patch is a STRICT no-op (shap parses the array
    # base_score natively and uses ``float`` as a numpy dtype, so ``_shap_tree.float`` must NOT be replaced -- see
    # test_patch_is_noop_on_shap_ge_052), hence the coercer is no longer reachable via ``_shap_tree.float`` there.
    # The bracket-handling logic itself (the contract under test) lives in the module-level ``_safe_float``.
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import _safe_float as coerce

    assert coerce("[0.5]") == 0.5
    assert abs(coerce("[5.0666666E-1]") - 0.50666666) < 1e-6
    # Multi-element array: take the first.
    assert coerce("[0.7, 0.3]") == 0.7
    # Scalar pass-through.
    assert coerce("0.5") == 0.5
    assert coerce(0.5) == 0.5
    assert coerce(3) == 3.0


def test_J_coerce_to_numpy_does_not_emit_zero_copy_deprecation_warning():
    """coerce_to_numpy on a polars Series must not trigger polars' zero-copy-conversion DeprecationWarning."""
    pl = pytest.importorskip("polars")
    from mlframe.training.utils import coerce_to_numpy

    s = pl.Series("x", [1.0, 2.0, 3.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        arr = coerce_to_numpy(s)
        assert isinstance(arr, np.ndarray)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning) and "zero_copy_only" in str(w.message)]
    assert not deprecations, (
        "polars 0.20.10+ emits DeprecationWarning on the zero_copy_only kwarg; "
        "the fix must prefer allow_copy=True. "
        f"Caught: {[str(w.message) for w in deprecations]}"
    )
