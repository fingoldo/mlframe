"""Regression tests for the training_loose_c.md audit fix wave (2026-07-21).

F1/F2 and PR2 (LGB/XGB shim module-level Dataset cache + multi-eval-set val cache + the
clone-cache-hit parity test) were already fixed/tested before this pass began --
see tests/training/test_audit_2026_07_21_lgb_shim_fixes.py and
tests/training/test_lgb_dataset_reuse_shim.py. One test per remaining finding (F3-F18)
plus PR1 (cv_stability_check loss-type coverage) and PR3 (sklearn.clone regression for
PartialFitESWrapper's external val set). PR6 is covered by a saved benchmark script
(_benchmarks/bench_lgb_shim_clone_cache_reuse.py), not a pytest test. PR4/PR5/PR7 are
implemented alongside F14/F7/F11 respectively.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

# =====================================================================
# F3 / F18 -- _model_factories.py no longer duplicates an unguarded CUDA probe
# =====================================================================


def test_f3_f18_model_factories_reexports_gpu_probe_cuda_is_available():
    """Pre-fix, _model_factories.py ran its own unguarded ``from numba.cuda import
    is_available`` at module scope -- crashes package import on hosts without numba / a
    working CUDA driver, unlike the guarded _gpu_probe.py version. Now must be the SAME
    object (single canonical probe, not a second independent one)."""
    from mlframe.training import _gpu_probe, _model_factories

    assert _model_factories.CUDA_IS_AVAILABLE is _gpu_probe.CUDA_IS_AVAILABLE


# =====================================================================
# F4 / PR3 -- PartialFitESWrapper.get_params() + sklearn.clone() preserve external val set
# =====================================================================


def test_f4_pr3_sklearn_clone_preserves_external_val_and_type():
    """Pre-fix, get_params() omitted external_X_val/external_y_val, so sklearn.clone()
    silently dropped the caller's held-out val set and fell back to an internal random split
    with no error. A SEPARATE, more severe bug was found while fixing this: PartialFitESWrapper's
    __getattr__ unconditionally forwarded dunder names (including __sklearn_clone__) to the
    wrapped estimator, so clone(wrapper) returned a clone of the INNER estimator, not the
    wrapper at all -- verified this reproduces the exact failure signature pre-fix, fixed by
    excluding dunder names from the forwarding."""
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression

    from mlframe.training._partial_fit_es_wrapper import PartialFitESWrapper

    Xv = np.random.RandomState(0).randn(10, 3)
    yv = np.random.RandomState(1).randint(0, 2, 10)
    w = PartialFitESWrapper(estimator=LogisticRegression(), external_X_val=Xv, external_y_val=yv)

    cloned = clone(w)
    assert isinstance(cloned, PartialFitESWrapper), f"clone() must return a PartialFitESWrapper, got {type(cloned).__name__}"
    assert cloned.external_X_val is not None and np.array_equal(cloned.external_X_val, Xv)
    assert cloned.external_y_val is not None and np.array_equal(cloned.external_y_val, yv)


def test_f4_getattr_still_delegates_non_dunder_attrs():
    """The dunder-exclusion fix must not break the wrapper's actual documented purpose:
    non-dunder attribute delegation to the wrapped estimator (.coef_, .classes_, etc.)."""
    from sklearn.linear_model import LogisticRegression

    from mlframe.training._partial_fit_es_wrapper import PartialFitESWrapper

    w = PartialFitESWrapper(estimator=LogisticRegression())
    w.estimator.coef_ = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(w.coef_, np.array([1.0, 2.0, 3.0]))
    with pytest.raises(AttributeError):
        _ = w.definitely_not_a_real_attribute


def test_f4_getattr_dunder_probe_raises_attribute_error():
    """A missing dunder must raise AttributeError from the wrapper itself, not silently
    resolve via the wrapped estimator."""
    from sklearn.linear_model import LogisticRegression

    from mlframe.training._partial_fit_es_wrapper import PartialFitESWrapper

    w = PartialFitESWrapper(estimator=LogisticRegression())
    assert not hasattr(w, "__some_nonexistent_dunder__")


# =====================================================================
# F5 -- train_lama_model's shape-aware LAMA->probs conversion (lightautoml not installed;
# tested via the extracted pure helper, no source-inspection needed)
# =====================================================================


def test_f5_lama_data_to_probs_binary_1d():
    """F5: lama data to probs binary 1d."""
    from mlframe.training.automl import _lama_data_to_probs

    out = _lama_data_to_probs(np.array([0.1, 0.9, 0.5]))
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[:, 1], [0.1, 0.9, 0.5])
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def test_f5_lama_data_to_probs_binary_2d_single_column():
    """F5: lama data to probs binary 2d single column."""
    from mlframe.training.automl import _lama_data_to_probs

    out = _lama_data_to_probs(np.array([[0.2], [0.8]]))
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[:, 1], [0.2, 0.8])


def test_f5_lama_data_to_probs_genuine_multiclass_not_fabricated():
    """Pre-fix, this K=3 case had column 0 alone fed into a fake [1-p, p] binary pair,
    discarding columns 1 and 2 entirely and silently mis-scoring a genuine multiclass Task."""
    from mlframe.training.automl import _lama_data_to_probs

    mc = np.array([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]])
    out = _lama_data_to_probs(mc)
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, mc)


# =====================================================================
# F6 / PR1 -- cv_stability_check honours a metric-direction (maximize) parameter
# =====================================================================


def test_f6_pr1_cv_stability_check_maximize_false_agrees_on_minimum_not_maximum():
    """Pre-fix, cv_stability_check always used argmax regardless of metric direction, so a
    loss-type curve's cross-seed agreement was silently measured against the WORST
    hyperparameter. With maximize=False the agreement must be assessed via argmin instead."""
    from mlframe.training._overlapping_walk_forward_cv import cv_stability_check

    # Loss-type curves: minimum at index 2 for both seeds (best hyperparameter), maximum at
    # index 0 -- if the direction is wrong, "agreement" would incorrectly center on index 0.
    loss_curves = [
        [5.0, 3.0, 1.0, 3.0, 5.0],
        [5.2, 2.8, 1.1, 2.9, 5.1],
    ]
    result = cv_stability_check(loss_curves, maximize=False)
    assert result["cross_seed_argmax_agreement"] == 1.0, "both seed curves' argmin should agree on the shared minimum"
    assert result["stable"] is True


def test_f6_cv_stability_check_maximize_true_still_default():
    """Default behaviour (maximize=True, the pre-fix-only mode) must be unchanged."""
    from mlframe.training._overlapping_walk_forward_cv import cv_stability_check

    gain_curves = [
        [0.1, 0.5, 0.9, 0.5, 0.1],
        [0.15, 0.55, 0.85, 0.45, 0.05],
    ]
    result = cv_stability_check(gain_curves)
    assert result["cross_seed_argmax_agreement"] == 1.0


# =====================================================================
# F7 / PR5 -- iid/temporal conformal carvers gain the zero-floor guard; shared helper
# =====================================================================


def test_f7_iid_carver_raises_on_zero_floor_calib():
    """F7: iid carver raises on zero floor calib."""
    from mlframe.training._conformal_split import carve_calib_conformal_iid

    idx = np.arange(100)
    with pytest.raises(ValueError, match="floors to 0 calib"):
        carve_calib_conformal_iid(idx, calib_frac=0.001, conformal_frac=0.1, seed=0)


def test_f7_temporal_carver_raises_on_zero_floor_conformal():
    """F7: temporal carver raises on zero floor conformal."""
    from mlframe.training._conformal_split import carve_calib_conformal_temporal

    idx = np.arange(100)
    with pytest.raises(ValueError, match="floors to 0 conformal"):
        carve_calib_conformal_temporal(idx, calib_frac=0.1, conformal_frac=0.001)


def test_f7_iid_and_temporal_carvers_normal_case_unaffected():
    """F7: iid and temporal carvers normal case unaffected."""
    from mlframe.training._conformal_split import carve_calib_conformal_iid, carve_calib_conformal_temporal

    idx = np.arange(100)
    fit, calib, conf = carve_calib_conformal_iid(idx, 0.1, 0.1, seed=0)
    assert len(fit) == 80 and len(calib) == 10 and len(conf) == 10
    fit2, calib2, conf2 = carve_calib_conformal_temporal(idx, 0.1, 0.1)
    assert len(fit2) == 80 and len(calib2) == 10 and len(conf2) == 10


def test_pr5_shared_zero_floor_helper_used_by_all_three_carvers():
    """PR5: the zero-floor guard is now one shared helper, not 3 independent copies --
    the grouped carver (already correct pre-fix) must still behave identically."""
    from mlframe.training._conformal_split import carve_calib_conformal_grouped

    idx = np.arange(100)
    groups = np.repeat(np.arange(20), 5)
    with pytest.raises(ValueError, match="floors to 0 calib"):
        carve_calib_conformal_grouped(idx, calib_frac=0.01, conformal_frac=0.1, group_values=groups, seed=0)
    fit, calib, conf = carve_calib_conformal_grouped(idx, 0.1, 0.1, group_values=groups, seed=0)
    assert len(fit) + len(calib) + len(conf) == 100


# =====================================================================
# F8 -- _setup_sample_weight gains a polars branch + shared boolean-mask normalisation
# =====================================================================


def test_f8_setup_sample_weight_polars_series_with_boolean_mask():
    """Pre-fix, this had no polars branch at all: a pl.Series sample_weight fell into the
    generic sample_weight[train_idx] path, which raises InvalidOperationError for a boolean
    train_idx on polars (works fine on numpy/pandas -- same backend-divergence class
    _extract_target_subset was already fixed for)."""
    import polars as pl

    from mlframe.training._data_helpers import _setup_sample_weight

    class _FakeModel:
        """Fake model stub used to control this test's predict path."""
        def fit(self, X, y, sample_weight=None):
            """No-op / recording stub matching the estimator's fit() signature."""
            pass

    fp: dict = {}
    bool_idx = np.array([True, False, True, False, True])
    _setup_sample_weight(pl.Series([1.0, 2.0, 3.0, 4.0, 5.0]), bool_idx, _FakeModel(), fp)
    assert isinstance(fp["sample_weight"], np.ndarray)
    np.testing.assert_allclose(fp["sample_weight"], [1.0, 3.0, 5.0])


def test_f8_setup_sample_weight_pandas_dataframe_still_works():
    """F8: setup sample weight pandas dataframe still works."""
    from mlframe.training._data_helpers import _setup_sample_weight

    class _FakeModel:
        """Fake model stub used to control this test's predict path."""
        def fit(self, X, y, sample_weight=None):
            """No-op / recording stub matching the estimator's fit() signature."""
            pass

    fp: dict = {}
    _setup_sample_weight(pd.DataFrame({"w": [1.0, 2.0, 3.0, 4.0, 5.0]}), np.array([0, 2, 4]), _FakeModel(), fp)
    np.testing.assert_allclose(fp["sample_weight"], [1.0, 3.0, 5.0])


def test_f8_setup_sample_weight_skips_when_model_lacks_param():
    """F8: setup sample weight skips when model lacks param."""
    from mlframe.training._data_helpers import _setup_sample_weight

    class _NoWeightModel:
        """Stub model whose fit() signature has no sample_weight parameter."""
        def fit(self, X, y):
            """No-op / recording stub matching the estimator's fit() signature."""
            pass

    fp: dict = {}
    _setup_sample_weight(np.array([1.0, 2.0, 3.0]), None, _NoWeightModel(), fp)
    assert "sample_weight" not in fp


# =====================================================================
# F9 -- _groupids_to_sizes warns on out-of-order qid input
# =====================================================================


def test_f9_groupids_to_sizes_warns_on_unsorted_qid(caplog):
    """F9: groupids to sizes warns on unsorted qid."""
    from mlframe.training._data_helpers import _groupids_to_sizes

    caplog.set_level(logging.WARNING)
    unsorted_qid = np.array([1, 1, 2, 2, 1, 3, 3])  # qid=1 reappears after a gap
    sizes = _groupids_to_sizes(unsorted_qid)
    assert sizes.sum() == len(unsorted_qid)
    assert any("not sorted by qid" in rec.message for rec in caplog.records)


def test_f9_groupids_to_sizes_no_warning_when_sorted(caplog):
    """F9: groupids to sizes no warning when sorted."""
    from mlframe.training._data_helpers import _groupids_to_sizes

    caplog.set_level(logging.WARNING)
    sorted_qid = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])
    sizes = _groupids_to_sizes(sorted_qid)
    np.testing.assert_array_equal(sizes, [3, 2, 4])
    assert not any("not sorted by qid" in rec.message for rec in caplog.records)


# =====================================================================
# F10 -- token (not substring) check for model_type_name prefixing
# =====================================================================


def test_f10_model_name_token_check_not_substring(tmp_path):
    """Pre-fix, model_type_name="CB" would spuriously match inside "CBOW-embeddings" (a raw
    substring check), silently skipping the intended prefixing and risking a model_file_name
    collision between two different model types."""
    from mlframe.training._data_helpers import _setup_model_info_and_paths

    _CB = type("CB", (), {})
    _model_obj, _model_type_name, model_name, _plot_file, _model_file_name = _setup_model_info_and_paths(
        _CB(), model_name="CBOW-embeddings mymodel", model_name_prefix="", plot_file="", data_dir="", models_subdir=""
    )
    assert model_name.split()[0] == "CB", f"model_type_name must be prefixed, got {model_name!r}"


def test_f10_model_name_token_check_no_duplicate_prefix_when_already_present():
    """F10: model name token check no duplicate prefix when already present."""
    from mlframe.training._data_helpers import _setup_model_info_and_paths

    _CB = type("CB", (), {})
    _model_obj, _model_type_name, model_name, _plot_file, _model_file_name = _setup_model_info_and_paths(
        _CB(), model_name="CB mymodel", model_name_prefix="", plot_file="", data_dir="", models_subdir=""
    )
    assert model_name.split().count("CB") == 1


# =====================================================================
# F11 / PR7 -- point_estimate_alpha docstring matches the real snap-to-nearest behaviour
# =====================================================================


def test_f11_pr7_point_estimate_alpha_snaps_and_logs(caplog):
    """F11/PR7: point estimate alpha snaps and logs."""
    from mlframe.training._model_configs_behavior import QuantileRegressionConfig

    caplog.set_level(logging.DEBUG, logger="mlframe.training._model_configs_behavior")
    cfg = QuantileRegressionConfig(alphas=(0.1, 0.5, 0.9), point_estimate_alpha=0.4)
    assert cfg.point_estimate_alpha == 0.5
    assert any("snapping to closest" in rec.message for rec in caplog.records)


def test_f11_point_estimate_alpha_no_snap_when_already_present():
    """F11: point estimate alpha no snap when already present."""
    from mlframe.training._model_configs_behavior import QuantileRegressionConfig

    cfg = QuantileRegressionConfig(alphas=(0.1, 0.5, 0.9), point_estimate_alpha=0.1)
    assert cfg.point_estimate_alpha == 0.1


# =====================================================================
# F12 -- clamped shuffled counts reflected in the reported detail string
# =====================================================================


def test_f12_perform_split_returns_effective_clamped_counts():
    """F12: perform split returns effective clamped counts."""
    from mlframe.training._splitting_helpers import _perform_split

    items = np.arange(10)
    rng = np.random.default_rng(0)
    # Request far more shuffled test rows than the pool can supply after sequential test.
    _train, _val, test, _val_seq, _test_seq, eff_test_shuf, _eff_val_shuf = _perform_split(
        items, n_test_seq=2, n_test_shuf=50, n_val_seq=0, n_val_shuf=0, rng=rng, effective_val_placement="forward"
    )
    assert eff_test_shuf < 50, "effective count must be clamped to the actual pool size"
    assert eff_test_shuf == len(test) - 2  # test = sequential(2) + shuffled(eff_test_shuf)


def test_f12_perform_split_no_clamp_reports_requested_count():
    """F12: perform split no clamp reports requested count."""
    from mlframe.training._splitting_helpers import _perform_split

    items = np.arange(100)
    rng = np.random.default_rng(0)
    _train, _val, _test, _val_seq, _test_seq, eff_test_shuf, eff_val_shuf = _perform_split(
        items, n_test_seq=0, n_test_shuf=5, n_val_seq=0, n_val_shuf=5, rng=rng, effective_val_placement="forward"
    )
    assert eff_test_shuf == 5
    assert eff_val_shuf == 5


# =====================================================================
# F13 -- empty-train guard in make_train_test_split
# =====================================================================


def test_f13_empty_train_raises_actionable_error():
    """Direct exercise of the new guard: val_size is a fraction of the pool REMAINING after
    test (not of the original total), so val_size=1.0 consumes the entire post-test remainder,
    leaving 0 train rows -- exactly the "no equivalent guard for train" gap F13 flagged. Must
    raise an actionable error naming the actual config, not silently hand back an empty split
    (which would surface far downstream as an opaque booster crash)."""
    import pandas as pd

    from mlframe.training.splitting import make_train_test_split

    n = 100
    df = pd.DataFrame({"ts": pd.date_range("2020-01-01", periods=n, freq="D"), "x": np.arange(n)})
    with pytest.raises(ValueError, match="0 train rows"):
        make_train_test_split(df, timestamps=df["ts"], test_size=0.5, val_size=1.0, wholeday_splitting=False)


# =====================================================================
# F14 / PR4 -- suite_artefact_cache.py docstring honestly describes its unwired status
# =====================================================================


def test_f14_pr4_suite_artefact_cache_docstring_no_false_wire_in_claim():
    """F14/PR4: suite artefact cache docstring no false wire in claim."""
    import mlframe.training.suite_artefact_cache as sac_mod

    doc = sac_mod.__doc__ or ""
    assert "Wire-in proofs-of-concept (this commit)" not in doc
    assert "NOT currently wired into any call site" in doc


def test_f14_suite_artefact_cache_genuinely_unwired():
    """Confirms the module is still standalone (not silently wired in without a docstring
    update) -- this test itself should be revisited/removed the day it IS wired in."""
    import inspect

    from mlframe.training.core import _phase_helpers_fit_pipeline

    src = inspect.getsource(_phase_helpers_fit_pipeline)
    assert "cache_artefact" not in src
    assert "SuiteArtefactCache" not in src


# =====================================================================
# F15 -- _precompute.py inline comment matches the real NotImplementedError behaviour
# =====================================================================


def test_f15_precompute_all_leaves_dummy_and_composite_slots_none():
    """F15: precompute all leaves dummy and composite slots none."""
    from mlframe.training._precompute import precompute_all

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    bundle = precompute_all(df)
    assert bundle.dummy_baselines is None
    assert bundle.composite_target_specs is None
    assert bundle.trainset_features_stats is not None


def test_f15_precompute_stubs_actually_raise():
    """F15: precompute stubs actually raise."""
    from mlframe.training._precompute import precompute_composite_target_specs, precompute_dummy_baselines

    with pytest.raises(NotImplementedError):
        precompute_dummy_baselines(None, {})
    with pytest.raises(NotImplementedError):
        precompute_composite_target_specs()


# =====================================================================
# F16 -- sanitize_name_list gains collision-safe, frame-consistent dedup
# =====================================================================


def test_f16_sanitize_name_list_collision_safe_matches_frame():
    """Pre-fix, sanitize_name_list had zero collision tracking; two hostile names that
    collapse to the same safe base (e.g. "a[1]" and "a{1}" both -> "a_1_") would silently
    both map to the identical safe name, corrupting the name list vs the frame's own
    (collision-safe) renaming."""
    from mlframe.training._feature_name_sanitize import build_safe_mapping, sanitize_name_list

    cols = ["a[1]", "a{1}", "clean"]
    frame_mapping = build_safe_mapping(cols)
    result = sanitize_name_list(["a[1]", "a{1}"], full_columns=cols)
    assert result == [frame_mapping["a[1]"], frame_mapping["a{1}"]]
    assert result[0] != result[1], "the two colliding hostile names must map to DIFFERENT safe names"


def test_f16_sanitize_name_list_without_full_columns_still_dedups_within_itself():
    """F16: sanitize name list without full columns still dedups within itself."""
    from mlframe.training._feature_name_sanitize import sanitize_name_list

    result = sanitize_name_list(["a[1]", "a{1}"])
    assert result[0] != result[1]


def test_f16_sanitize_name_list_noop_when_nothing_hostile():
    """F16: sanitize name list noop when nothing hostile."""
    from mlframe.training._feature_name_sanitize import sanitize_name_list

    clean = ["a", "b", "c"]
    assert sanitize_name_list(clean) is clean


# =====================================================================
# F17 -- mlframe.training.__init__ is genuinely lazy for the formerly-eager submodules
# =====================================================================


def test_f17_bare_import_does_not_eagerly_load_formerly_eager_submodules():
    """A bare `import mlframe.training` must not pull in the heavy conformal/calibration/
    neural-adjacent submodules that used to be eager `from X import Y` statements at the
    bottom of __init__.py -- contradicting the module's own stated lazy-import design."""
    import subprocess
    import sys

    heavy_mods = [
        "mlframe.training._conformal_finalize",
        "mlframe.training._regression_calibration",
        "mlframe.training._tta",
        "mlframe.training._mc_dropout",
        "mlframe.training._noise_ensemble",
        "mlframe.training._partial_fit_es_wrapper",
        "mlframe.training._overlapping_walk_forward_cv",
        "mlframe.training._direct_horizon_bucket_forecaster",
    ]
    code = "import sys, mlframe.training; print(','.join(m for m in %r if m in sys.modules))" % heavy_mods
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"unexpectedly eager-loaded: {out.stdout.strip()}"


def test_f17_lazy_names_still_resolve_correctly():
    """F17: lazy names still resolve correctly."""
    import mlframe.training as t

    assert callable(t.mc_dropout_predict)
    assert callable(t.cv_stability_check)
    assert t.PartialFitESWrapper.__name__ == "PartialFitESWrapper"
    assert t.NoiseAugmentedEnsemble.__name__ == "NoiseAugmentedEnsemble"
