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
                "model_b": {"_cached_val_datasets": {("sig",): object()}},
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
    assert (
        ctx.artifacts["dataset_reuse_cache"] == {}
    ), "dataset_reuse_cache must be emptied; otherwise XGB/LGB/CB wrappers pin the polars buffers and the polars-release is a no-op."


# ---------------------------------------------------------------------------
# A: dummy + CT_ENSEMBLE phases respect compute_valset_metrics=False /
# compute_testset_metrics=False.
# ---------------------------------------------------------------------------
def _dummy_baselines_kwargs(reporting_config, metadata):
    """Keyword set for a real ``run_dummy_baselines`` call on a small regression problem."""
    import pandas as pd

    from mlframe.training.configs import DummyBaselinesConfig

    rng = np.random.default_rng(0)
    n = 120
    frame = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = rng.normal(size=n)
    return dict(
        target_type="regression",
        cur_target_name="y",
        target_name="y",
        model_name="m",
        current_train_target=y[:80],
        current_val_target=y[80:100],
        current_test_target=y[100:],
        filtered_train_df=frame.iloc[:80],
        filtered_val_df=frame.iloc[80:100],
        test_df_pd=frame.iloc[100:],
        filtered_train_idx=np.arange(80),
        filtered_val_idx=np.arange(80, 100),
        test_idx=np.arange(100, 120),
        timestamps=None,
        cat_features=[],
        dummy_baselines_config=DummyBaselinesConfig(enabled=True),
        quantile_regression_config=None,
        reporting_config=reporting_config,
        _dropped_high_card_data={},
        train_od_idx=None,
        val_od_idx=None,
        plot_file="",
        metadata=metadata,
        target_by_type={},
        _split_preds_probs=lambda raw, target_type: (raw, None),
    )


@pytest.mark.parametrize(
    "compute_val, compute_test, expected_titles",
    [(False, False, []), (True, False, ["VAL (DUMMY) "]), (False, True, ["TEST (DUMMY) "]), (True, True, ["VAL (DUMMY) ", "TEST (DUMMY) "])],
)
def test_A_dummy_baselines_respects_compute_valset_metrics_false(monkeypatch, compute_val, compute_test, expected_titles):
    """The dummy-baselines emit path reports a split only when the matching ``compute_*set_metrics`` flag is on."""
    from mlframe.training.core import _phase_dummy_baselines as mod

    titles: list = []
    monkeypatch.setattr(mod, "report_model_perf", lambda **kw: titles.append(kw["report_title"]))
    reporting = types.SimpleNamespace(compute_valset_metrics=compute_val, compute_testset_metrics=compute_test, plot_outputs=None, plot_dpi=None)
    metadata: dict = {}
    mod.run_dummy_baselines(**_dummy_baselines_kwargs(reporting, metadata))
    assert metadata.get("dummy_baselines_status") != "failed", metadata.get("dummy_baselines_status_detail")
    assert "regression" in metadata["dummy_baselines"]
    assert titles == expected_titles


@pytest.mark.parametrize(
    "compute_val, compute_test, expected",
    [(False, False, []), (True, False, ["val"]), (False, True, ["test"]), (True, True, ["val", "test"])],
)
def test_A_ct_ensemble_respects_compute_valset_metrics_false(compute_val, compute_test, expected):
    """The cross-target ensemble report covers a split only when the matching ``compute_*set_metrics`` flag is on."""
    from mlframe.training.core._phase_composite_post_xt_ensemble._xt_ensemble_helpers import _ct_ensemble_split_plan

    reporting = types.SimpleNamespace(compute_valset_metrics=compute_val, compute_testset_metrics=compute_test)
    val_idx, val_df, test_idx, test_df = np.arange(3), object(), np.arange(5), object()
    plan = _ct_ensemble_split_plan(reporting, val_idx, val_df, test_idx, test_df)
    assert [entry[0] for entry in plan] == expected
    by_split = {entry[0]: entry for entry in plan}
    if "val" in by_split:
        assert by_split["val"][1:] == ("VAL (CT_ENSEMBLE) ", val_idx, val_df)
    if "test" in by_split:
        assert by_split["test"][1:] == ("TEST (CT_ENSEMBLE) ", test_idx, test_df)
    assert [e[0] for e in _ct_ensemble_split_plan(None, val_idx, val_df, test_idx, test_df)] == ["val", "test"]


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
        train, val, test = make_train_test_split(
            df=df,
            val_size=0.1,
            test_size=0.1,
            val_placement="backward",
            timestamps=None,
            random_seed=0,
        )[:3]
    assert len(train) + len(val) + len(test) == n

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
@pytest.mark.parametrize("plot_outputs", ["plotly[html]+matplotlib[png]", None])
def test_B_temporal_audit_uses_plot_outputs_when_present(monkeypatch, plot_outputs):
    """With ``reporting_config.plot_outputs`` set the temporal-audit chart goes through the multi-backend DSL; without it, the matplotlib PNG fallback."""
    from mlframe.training.core import _phase_train_one_target_model_setup as mod

    calls: list = []
    monkeypatch.setattr(mod, "_plot_target_over_time", lambda audit, **kw: calls.append((audit, kw)))
    reporting = types.SimpleNamespace(plot_outputs=plot_outputs)
    mod._save_temporal_audit_plot("AUDIT", types.SimpleNamespace(target_temporal_audit_save_plot=True), reporting, "out/run")
    if plot_outputs:
        assert calls == [("AUDIT", {"plot_outputs": plot_outputs, "base_path": "out/run_target_temporal_audit"})]
    else:
        assert calls == [("AUDIT", {"save_path": "out/run_target_temporal_audit.png"})]

    calls.clear()
    mod._save_temporal_audit_plot("AUDIT", types.SimpleNamespace(target_temporal_audit_save_plot=False), reporting, "out/run")
    mod._save_temporal_audit_plot("AUDIT", types.SimpleNamespace(target_temporal_audit_save_plot=True), reporting, "")
    assert calls == []


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
    # lift = 13.19/13.43 = 0.98: the model really is worse than the baseline, which is what this half of the
    # test set up. The verdict now says so; it used to read MODELS_BARELY_BEAT_TRIVIAL, which claims a win.
    assert "BEST_MODEL_BELOW_DUMMY" in out_no_ens

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
    """The numba warmup's MAPE call uses a non-zero y_true, so the rate-limited zero-y_true warning stays silent; a zero vector does trigger it."""
    from mlframe.metrics import _core_numba_warmup, _core_precision_mape
    from mlframe.metrics.core import maximum_absolute_percentage_error

    _core_precision_mape._MAPE_ZERO_WARN_SEEN.clear()
    with caplog.at_level(logging.WARNING):
        _core_numba_warmup._prewarm_numba_cach_lazy_first_real_call()
    assert not [r for r in caplog.records if "warmup failed" in r.getMessage()], [r.getMessage() for r in caplog.records]
    zero_warns = [r for r in caplog.records if "y_true entries are zero" in r.getMessage()]
    assert not zero_warns, f"MAPE zero-y warning fired during numba warmup: {[r.getMessage() for r in zero_warns]}"

    caplog.clear()
    _core_precision_mape._MAPE_ZERO_WARN_SEEN.clear()
    with caplog.at_level(logging.WARNING):
        maximum_absolute_percentage_error(np.array([0.0, 0.0, 1.0, 2.0]), np.array([0.1, 0.2, 1.1, 2.1]))
    assert [r for r in caplog.records if "y_true entries are zero" in r.getMessage()], "positive control: a zero y_true must trigger the warning"


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
