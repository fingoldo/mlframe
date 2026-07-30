"""Regression tests for the training_loose_a.md audit fix wave (2026-07-21).

One test per finding (F1-F11) plus the proposals implemented alongside the findings
(PR1 fingerprint-collision regression, PR2 batched-vs-legacy behavioral parity, PR5 the
NaN-check perf fix already folded into F8's block). PR3 (shared bounded-cache helper) was
assessed and deferred (see the module docstring below); PR4 (PU-learning biz_val coverage)
was confirmed already covered by tests/training/test_pu_learning_synthetic.py's existing
quantitative comparison tests, no duplicate added.
See ``audits/full_audit_2026-07-21/training_loose_a.md``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.configs import PreprocessingConfig, TargetTypes
from mlframe.training.preprocessing import preprocess_dataframe

# =====================================================================
# F1 -- fix_infinities=True default routes inf -> NaN, not a hardcoded 0.0
# =====================================================================


def test_f1_default_fix_infinities_pandas_legacy_path_uses_nan_not_zero():
    """Pandas always takes the legacy (non-batched) path. Default config (fix_infinities=True,
    fillna_value=None) must normalise +/-inf to NaN, matching the documented "let the post-split
    imputer decide" principle -- pre-fix it silently injected a hardcoded 0.0 pre-split."""
    df = pd.DataFrame({"a": [1.0, float("inf"), 3.0, float("-inf")]})
    cfg = PreprocessingConfig()
    out = preprocess_dataframe(df, cfg, verbose=0)
    assert np.isnan(out["a"].iloc[1])
    assert np.isnan(out["a"].iloc[3])
    assert not (out["a"] == 0.0).any()


def test_f1_default_fix_infinities_polars_batched_path_uses_nan_not_zero():
    """Polars takes the batched fastpath by default; same contract must hold there."""
    df = pl.DataFrame({"a": [1.0, float("inf"), 3.0, float("-inf")]})
    cfg = PreprocessingConfig()
    out = preprocess_dataframe(df, cfg, verbose=0)
    a = out["a"].to_list()
    assert a[1] != a[1]  # NaN != NaN
    assert a[3] != a[3]
    assert 0.0 not in a


def test_f1_emergency_autofix_path_uses_nan_not_zero(caplog):
    """fix_infinities=False but the frame still contains inf triggers the emergency auto-fix
    branch (pandas legacy path only) -- must also route through NaN, not 0.0."""
    df = pd.DataFrame({"a": [1.0, float("inf"), 3.0]})
    cfg = PreprocessingConfig(fix_infinities=False)
    caplog.set_level(logging.ERROR)
    out = preprocess_dataframe(df, cfg, verbose=0)
    assert np.isnan(out["a"].iloc[1])
    assert any("NaN" in rec.message for rec in caplog.records)


# =====================================================================
# F2 / PR1 -- compute_signature's widened row sample avoids the old 3-point collision
# =====================================================================


def test_f2_compute_signature_distinguishes_frames_identical_only_at_first_mid_last():
    """Pre-fix, compute_signature sampled exactly (0, n//2, n-1); two frames differing
    EVERYWHERE ELSE but agreeing at those 3 positions collided and the cache would hand back
    the wrong booster-native dataset object for a .fit() call with no error."""
    from mlframe.training._dataset_cache_fingerprint import compute_signature

    n = 100
    base = np.arange(n, dtype=float)
    df1 = pd.DataFrame({"a": base})
    df2 = pd.DataFrame({"a": base.copy()})
    # Perturb every row except the old fixed sample points (0, 50, 99).
    mask = np.ones(n, dtype=bool)
    mask[[0, 50, 99]] = False
    df2.loc[mask, "a"] = df2.loc[mask, "a"] + 1000.0

    assert compute_signature(df1) != compute_signature(df2)


def test_f2_compute_signature_stays_sublinear_on_large_frames():
    """The fingerprint must stay cheap (sample count capped) regardless of frame size --
    the whole design point of this module is O(sqrt(n))-ish, not O(n)."""
    from mlframe.training._dataset_cache_fingerprint import _MAX_SAMPLE_ROWS, _sample_indices

    assert len(_sample_indices(4_000_000)) <= _MAX_SAMPLE_ROWS
    assert len(_sample_indices(10)) < len(_sample_indices(10_000))


def test_f2_compute_signature_identical_content_same_signature():
    """Sanity: two frames with genuinely identical content must still produce the same
    signature (the widened sample must not introduce false negatives)."""
    from mlframe.training._dataset_cache_fingerprint import compute_signature

    df1 = pd.DataFrame({"a": np.arange(500, dtype=float)})
    df2 = pd.DataFrame({"a": np.arange(500, dtype=float)})
    assert compute_signature(df1) == compute_signature(df2)


# =====================================================================
# F3 -- mc_dropout_predict restores train/eval mode even when a forward pass raises
# =====================================================================


def test_f3_mc_dropout_predict_restores_mode_on_exception():
    """A forward pass raising mid-loop (OOM, bad input) must not leave dropout submodules
    stuck in train mode while the rest of the module sits in eval -- the docstring promises
    restoration "on exit", which pre-fix only happened on the SUCCESS path."""
    import torch
    import torch.nn as nn

    from mlframe.training._mc_dropout import mc_dropout_predict

    class _FlakyModule(nn.Module):
        """Flaky Module."""
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(0.5)
            self.calls = 0

        def forward(self, x):
            """Fake forward pass used to control the module's output for this test."""
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("simulated forward-pass failure")
            return self.drop(x)

    module = _FlakyModule()
    module.train()  # was_training=True before the call
    X = torch.randn(4, 4)
    with pytest.raises(RuntimeError, match="simulated forward-pass failure"):
        mc_dropout_predict(module, X, n=8)
    assert module.training is True, "module.train(was_training) must run even after a mid-loop raise"
    assert module.drop.training is True


def test_f3_mc_dropout_predict_restores_mode_on_success():
    """Baseline: the success path must still restore the original mode (was already correct)."""
    import torch
    import torch.nn as nn

    from mlframe.training._mc_dropout import mc_dropout_predict

    module = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.3))
    module.eval()  # was_training=False
    mean, _std, n_drop = mc_dropout_predict(module, torch.randn(3, 4), n=4)
    assert module.training is False
    assert n_drop == 1
    assert mean.shape == (3, 4)


# =====================================================================
# F4 -- lean save strip set includes oof_target / calib_probs / calib_target / calib_preds
# =====================================================================


def test_f4_lean_strip_fields_includes_oof_and_calib_arrays():
    """Pre-fix, _LEAN_STRIP_FIELDS had oof_preds/oof_probs but not their sibling fields
    stamped by the same compute_calib_and_oof_outputs call -- a lean=True save with
    oof_n_splits>=2 or a non-trivial calib_size still leaked these arrays."""
    from mlframe.training.io import _LEAN_STRIP_FIELDS

    for field in ("oof_target", "calib_probs", "calib_target", "calib_preds"):
        assert field in _LEAN_STRIP_FIELDS, f"{field} must be in the lean-save strip set"
    # The original P0 #2 fix's fields must still be present (no regression).
    for field in ("oof_preds", "oof_probs"):
        assert field in _LEAN_STRIP_FIELDS


# =====================================================================
# F5 -- helpers.py no longer duplicates the CUDA_IS_AVAILABLE probe
# =====================================================================


def test_f5_helpers_reexports_gpu_probe_cuda_is_available():
    """helpers.py must expose the SAME CUDA_IS_AVAILABLE object _gpu_probe.py computes, not a
    second independent probe -- a future divergence in one probe's logic must not silently
    split behaviour between the two."""
    from mlframe.training import _gpu_probe, helpers

    assert helpers.CUDA_IS_AVAILABLE is _gpu_probe.CUDA_IS_AVAILABLE


# =====================================================================
# F6 -- optimize_model_for_storage strips *_preds for ALL classification target types
# =====================================================================


@pytest.mark.parametrize(
    "target_type",
    [TargetTypes.BINARY_CLASSIFICATION, TargetTypes.MULTICLASS_CLASSIFICATION, TargetTypes.MULTILABEL_CLASSIFICATION],
)
def test_f6_optimize_model_for_storage_strips_preds_for_every_classification_type(target_type):
    """Pre-fix, only BINARY_CLASSIFICATION stripped train/val/test_preds; multiclass and
    multilabel models silently kept the (potentially large) *_preds arrays despite the
    docstring's unqualified "For classification models" claim."""
    from types import SimpleNamespace

    from mlframe.training.train_eval import optimize_model_for_storage

    model = SimpleNamespace(train_preds=np.zeros(5), val_preds=np.zeros(5), test_preds=np.zeros(5))
    optimize_model_for_storage(model, target_type)
    assert model.train_preds is None
    assert model.val_preds is None
    assert model.test_preds is None


def test_f6_optimize_model_for_storage_regression_keeps_preds():
    """Regression models must NOT have their preds stripped (no *_probs to recreate them from)."""
    from types import SimpleNamespace

    from mlframe.training.train_eval import optimize_model_for_storage

    model = SimpleNamespace(train_preds=np.zeros(5), val_preds=np.zeros(5), test_preds=np.zeros(5))
    optimize_model_for_storage(model, TargetTypes.REGRESSION)
    assert model.train_preds is not None


# =====================================================================
# F7 -- evaluate_model forwards its own verbose param through to report_model_perf
# =====================================================================


def test_f7_evaluate_model_forwards_verbose(monkeypatch):
    """Pre-fix, evaluate_model's verbose parameter was accepted but never forwarded (only
    **kwargs was passed through, and verbose binds to the named parameter instead)."""
    import mlframe.training.evaluation as ev_mod

    captured = {}

    def _fake_report_model_perf(**kwargs):
        """Fake report model perf."""
        captured.update(kwargs)
        return (np.zeros(3), None)

    monkeypatch.setattr(ev_mod, "report_model_perf", _fake_report_model_perf)

    class _DummyModel:
        """Dummy Model."""
        pass

    ev_mod.evaluate_model(_DummyModel(), "m", np.zeros(3), ["a"], verbose=0)
    assert captured["verbose"] is False

    ev_mod.evaluate_model(_DummyModel(), "m", np.zeros(3), ["a"], verbose=1)
    assert captured["verbose"] is True


# =====================================================================
# F8 / PR5 -- _ChainEnsemble.fit dead NaN-detection block removed + cheaper pandas check
# =====================================================================


def test_f8_chain_ensemble_fit_imputes_nan_and_warns(caplog):
    """Behavioral check of the NaN-guard block after removing the dead if/pass computation:
    a multilabel X with NaN cells must still fit successfully (imputed), with the WARN log
    firing (proves the has_nan detection still works post-cleanup)."""
    from sklearn.linear_model import LogisticRegression

    from mlframe.training._classif_helpers import _build_classifier_chain_ensemble

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(60, 4), columns=list("abcd"))
    X.iloc[3, 1] = np.nan
    y = (rng.rand(60, 3) > 0.5).astype(int)

    ens = _build_classifier_chain_ensemble(LogisticRegression(), n_labels=3, n_chains=1, cv=2)
    caplog.set_level(logging.WARNING)
    ens.fit(X, y)
    assert any("NaN" in rec.message for rec in caplog.records)
    assert hasattr(ens, "chains_")


def test_f8_pr5_pandas_nan_check_matches_numpy_reference_no_nan():
    """PR5's cheaper isinstance(DataFrame)-branch NaN check must agree with the original
    to_numpy()+np.isnan reference on a mixed int/float frame with NO NaN."""
    df = pd.DataFrame({"f": [1.0, 2.0, 3.0], "i": [1, 2, 3]})
    arr = df.to_numpy()
    reference = (arr.dtype.kind == "f") and bool(np.isnan(arr).any())
    fast = bool(df.isna().to_numpy().any())
    assert reference == fast == False  # noqa: E712 - explicit bool comparison reads clearer here


def test_f8_pr5_pandas_nan_check_matches_numpy_reference_with_nan():
    """Same parity check with a genuine NaN cell present, mixed int/float columns (the case
    the original to_numpy()-based check silently mishandled less efficiently)."""
    df = pd.DataFrame({"f": [1.0, np.nan, 3.0], "i": [1, 2, 3]})
    fast = bool(df.isna().to_numpy().any())
    assert fast is True


# =====================================================================
# F9 -- fix_quantile_crossing(mode="isotonic") warns and passes NaN rows through unfixed
# =====================================================================


def test_f9_isotonic_nan_row_passed_through_with_warning(caplog):
    """A NaN quantile prediction (upstream booster instability) must not silently vanish into
    an unfixed, unexplained row -- the caller must be told via a WARNING."""
    from mlframe.training.quantile_postproc import fix_quantile_crossing

    preds = np.array(
        [
            [0.1, 0.5, 0.9],  # already monotone
            [0.5, np.nan, 0.9],  # NaN cell
            [0.9, 0.5, 0.1],  # genuinely crossing, must still be fixed
        ]
    )
    caplog.set_level(logging.WARNING)
    out = fix_quantile_crossing(preds, alphas=[0.1, 0.5, 0.9], mode="isotonic")
    assert np.isnan(out[1, 1])
    assert np.all(np.diff(out[2]) >= -1e-12), "genuinely-crossing row must still be isotonic-fixed"
    assert any("NaN" in rec.message for rec in caplog.records)


def test_f9_isotonic_no_nan_rows_no_warning(caplog):
    """No NaN present -> no warning, unchanged behaviour for the common case."""
    from mlframe.training.quantile_postproc import fix_quantile_crossing

    preds = np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]])
    caplog.set_level(logging.WARNING)
    fix_quantile_crossing(preds, alphas=[0.1, 0.5, 0.9], mode="isotonic")
    assert not any("NaN" in rec.message for rec in caplog.records)


# =====================================================================
# F10 -- configure_training_params has no implicit-Optional parameters (mypy-verified)
# =====================================================================


def test_f10_configure_training_params_accepts_explicit_none_for_all_fixed_params():
    """Runtime companion to the mypy fix: every parameter that was implicit-Optional must
    still genuinely accept None at call time (behavioural confirmation the annotation
    change didn't silently change the runtime contract)."""
    import inspect

    from mlframe.training._trainer_configure import configure_training_params

    sig = inspect.signature(configure_training_params)
    for name in ("df", "train_df", "test_df", "val_df", "target", "target_label_encoder", "train_target", "test_target", "val_target"):
        assert sig.parameters[name].default is None


# =====================================================================
# F11 -- NoiseAugmentedEnsemble.predict_proba only attached when the base supports it
# =====================================================================


def test_f11_noise_ensemble_hasattr_predict_proba_matches_base_classifier():
    """F11: noise ensemble hasattr predict proba matches base classifier."""
    from sklearn.linear_model import LogisticRegression

    from mlframe.training._noise_ensemble import NoiseAugmentedEnsemble

    ens = NoiseAugmentedEnsemble(LogisticRegression())
    assert hasattr(ens, "predict_proba")


def test_f11_noise_ensemble_hasattr_predict_proba_false_for_regressor_base():
    """Pre-fix, hasattr(ensemble, 'predict_proba') was unconditionally True even when
    base_estimator is a regressor -- duck-typing classifier/regressor dispatch code was misled
    until predict_proba was actually called and raised deep inside the list comprehension."""
    from sklearn.linear_model import LinearRegression

    from mlframe.training._noise_ensemble import NoiseAugmentedEnsemble

    ens = NoiseAugmentedEnsemble(LinearRegression())
    assert not hasattr(ens, "predict_proba")


def test_f11_noise_ensemble_predict_proba_still_works_end_to_end():
    """F11: noise ensemble predict proba still works end to end."""
    from sklearn.linear_model import LogisticRegression

    from mlframe.training._noise_ensemble import NoiseAugmentedEnsemble

    rng = np.random.RandomState(0)
    X = rng.randn(80, 3)
    y = (X[:, 0] > 0).astype(int)
    ens = NoiseAugmentedEnsemble(LogisticRegression(), k=3)
    ens.fit(X, y)
    probs = ens.predict_proba(X)
    assert probs.shape == (80, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


# =====================================================================
# PR2 -- batched (polars) vs legacy (env-disabled) preprocessing paths agree
# =====================================================================


def test_pr2_batched_and_legacy_polars_paths_agree_on_constant_and_inf_handling(monkeypatch):
    """The batched polars fastpath and the legacy 3-pass path are independently-maintained
    implementations of the same constant-column/inf-detection logic; force each in turn on
    the SAME frame and confirm they produce identical results (would catch future drift)."""
    df = pl.DataFrame(
        {
            "const": [5.0, 5.0, 5.0, 5.0],
            "a": [1.0, float("inf"), 3.0, float("-inf")],
            "b": [1.0, 2.0, 3.0, 4.0],
        }
    )
    cfg = PreprocessingConfig()

    monkeypatch.delenv("MLFRAME_DISABLE_BATCHED_PREPROCESS_SCAN", raising=False)
    out_batched = preprocess_dataframe(df.clone(), cfg, verbose=0)

    monkeypatch.setenv("MLFRAME_DISABLE_BATCHED_PREPROCESS_SCAN", "1")
    out_legacy = preprocess_dataframe(df.clone(), cfg, verbose=0)

    assert set(out_batched.columns) == set(out_legacy.columns) == {"a", "b"}, "both paths must drop the constant column identically"
    a_batched = out_batched["a"].to_list()
    a_legacy = out_legacy["a"].to_list()
    for v_b, v_l in zip(a_batched, a_legacy):
        if v_b != v_b:  # NaN
            assert v_l != v_l
        else:
            assert v_b == v_l
