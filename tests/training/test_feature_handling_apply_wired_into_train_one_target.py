"""Wave-3 wire-in: ``feature_handling_apply`` must be invoked by ``_train_one_target`` when ``ctx.feature_handling_config`` is set.

Before this wave, the ``feature_handling_config`` kwarg was accepted by ``train_mlframe_models_suite`` and validated in
``setup_configuration``, but the per-target training loop never actually called ``feature_handling_apply`` -- the kwarg was dead. These
tests pin the wire-in: a no-op default path when fhc is None, an active path when fhc is supplied, and the assembled result lands on
``ctx.artifacts["feature_handling_fitted"]`` so a future predict-side wave can replay handlers.

The tests target the small helper ``_maybe_run_feature_handling_apply`` directly with a hand-built ctx + frames; exercising the full
``_train_one_target`` (which pulls in ~50 ctx attributes, OD filtering, strategies, model loops) would be a fixture nightmare for a wire-
in check. The helper is the wire seam, so testing it surfaces the same regression.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import polars as pl
import pytest

from mlframe.training.core._phase_train_one_target import _maybe_run_feature_handling_apply


def _make_synth_frames(n: int = 64):
    """Make synth frames."""
    rng = np.random.RandomState(0)
    text_choices = [
        "great product fast shipping recommended",
        "terrible quality waste of money",
        "mid quality acceptable for the price",
        "amazing value highly recommended",
        "poor build quality returned it",
    ]
    df = pl.DataFrame(
        {
            "review": [text_choices[i % len(text_choices)] for i in range(n)],
            "x_num": rng.randn(n).astype(np.float32),
        }
    )
    return df


def _make_ctx(*, fhc, sorted_models=("xgb",)):
    """Minimal SimpleNamespace ctx -- _maybe_run_feature_handling_apply only reads a handful of attrs.

    Real TrainingContext uses slots=True so an ad-hoc test fixture cannot add a runtime attribute on it; SimpleNamespace lets the test
    poke arbitrary attrs without depending on the dataclass schema.
    """
    return SimpleNamespace(
        feature_handling_config=fhc,
        sorted_mlframe_models=list(sorted_models),
        mlframe_models=list(sorted_models),
        artifacts={},
    )


# =====================================================================
# (a) Default-off: fhc=None -> no call, no artifact written
# =====================================================================


def test_default_off_no_apply_call(monkeypatch):
    """Default off no apply call."""
    train_df = _make_synth_frames()
    target = np.zeros(len(train_df), dtype=np.int32)
    ctx = _make_ctx(fhc=None)

    apply_mock = mock.Mock()
    monkeypatch.setattr(
        "mlframe.training.feature_handling.feature_handling_apply",
        apply_mock,
    )

    result = _maybe_run_feature_handling_apply(
        ctx,
        cur_target_name="t1",
        train_df=train_df,
        val_df=None,
        test_df=None,
        current_train_target=target,
    )
    assert result is None
    apply_mock.assert_not_called()
    assert "feature_handling_fitted" not in ctx.artifacts


# =====================================================================
# (b) Wired-on: fhc set -> apply IS called, result lands in ctx.artifacts
# =====================================================================


def test_wired_on_apply_invoked_and_stashed():
    """Wired on apply invoked and stashed."""
    from mlframe.training.feature_handling import tfidf_only

    train_df = _make_synth_frames()
    val_df = _make_synth_frames(n=32)
    test_df = _make_synth_frames(n=16)
    target = np.random.RandomState(1).randint(0, 2, size=len(train_df)).astype(np.int32)
    fhc = tfidf_only(max_features=8)
    ctx = _make_ctx(fhc=fhc, sorted_models=("xgb",))

    result = _maybe_run_feature_handling_apply(
        ctx,
        cur_target_name="t1",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        current_train_target=target,
    )
    assert result is not None
    # FeatureHandlingResult carries an assembled train block.
    assert result.train is not None
    # And it landed on ctx.artifacts at the expected nesting.
    fitted = ctx.artifacts.get("feature_handling_fitted")
    assert fitted is not None
    assert "t1" in fitted
    assert fitted["t1"] is result


def test_wired_on_passes_frames_and_target_to_apply(monkeypatch):
    """The wire-in must forward train/val/test frames + train_target + fhc + a model_kind to feature_handling_apply.

    Asserts the call kwargs by mocking the apply target inside the module's local import binding.
    """
    from mlframe.training.feature_handling import tfidf_only

    train_df = _make_synth_frames()
    val_df = _make_synth_frames(n=32)
    test_df = _make_synth_frames(n=16)
    target = np.zeros(len(train_df), dtype=np.int32)
    fhc = tfidf_only(max_features=4)
    ctx = _make_ctx(fhc=fhc, sorted_models=("cb",))

    apply_mock = mock.Mock(return_value=SimpleNamespace(train=object(), val=None, test=None, feature_names=[]))
    # The helper does ``from mlframe.training.feature_handling import feature_handling_apply`` inside its body, so the patch must hit
    # the canonical module attribute that gets re-bound on import.
    monkeypatch.setattr(
        "mlframe.training.feature_handling.feature_handling_apply",
        apply_mock,
    )

    _maybe_run_feature_handling_apply(
        ctx,
        cur_target_name="tA",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        current_train_target=target,
    )
    apply_mock.assert_called_once()
    _, kwargs = apply_mock.call_args
    assert kwargs["train_df"] is train_df
    assert kwargs["val_df"] is val_df
    assert kwargs["test_df"] is test_df
    assert kwargs["train_target"] is target
    assert kwargs["fhc"] is fhc
    assert kwargs["model_kind"] == "cb"


# =====================================================================
# (c) sample_weight: accepted by the helper signature (forward-compat)
# =====================================================================


def test_sample_weight_kwarg_accepted_today():
    """Forward-compat: the helper accepts ``sample_weight`` even though apply.py does not yet consume it.

    Documents the plumbing contract so a later apply.py extension does not need a second wire-in patch here. The current behaviour is
    silent discard -- the helper must not propagate sample_weight into apply() until apply() grows the kwarg.
    """
    from mlframe.training.feature_handling import tfidf_only

    train_df = _make_synth_frames()
    target = np.zeros(len(train_df), dtype=np.int32)
    fhc = tfidf_only(max_features=4)
    ctx = _make_ctx(fhc=fhc)

    sw = {"recency": np.linspace(0.5, 1.5, len(train_df), dtype=np.float32)}
    # Must not raise.
    result = _maybe_run_feature_handling_apply(
        ctx,
        cur_target_name="t1",
        train_df=train_df,
        val_df=None,
        test_df=None,
        current_train_target=target,
        sample_weight=sw,
    )
    assert result is not None


# =====================================================================
# Edge: malformed config -> ValueError re-raised with named kwarg
# =====================================================================


def test_malformed_config_reraises_with_clear_kwarg_name(monkeypatch):
    """``feature_handling_apply`` raising ``ValueError`` on validate_against_models must be re-raised mentioning ``feature_handling_config``.

    Users greping ``feature_handling_config`` in the traceback should hit the wrapper, not the deep validator.
    """
    train_df = _make_synth_frames()
    target = np.zeros(len(train_df), dtype=np.int32)
    fhc = mock.Mock()  # any non-None
    ctx = _make_ctx(fhc=fhc, sorted_models=("xgb",))

    def _raise(**_kw):
        """Raise."""
        raise ValueError("synthetic: handler chain incompatible with xgb")

    monkeypatch.setattr(
        "mlframe.training.feature_handling.feature_handling_apply",
        _raise,
    )

    with pytest.raises(ValueError, match="feature_handling_config"):
        _maybe_run_feature_handling_apply(
            ctx,
            cur_target_name="t1",
            train_df=train_df,
            val_df=None,
            test_df=None,
            current_train_target=target,
        )


# =====================================================================
# Edge: helper exists at the expected import path (regression guard)
# =====================================================================


def test_helper_is_module_level_in_phase_train_one_target():
    """If a future refactor moves the helper, this fast-failing test surfaces the breakage before the wire-in goes dark."""
    from mlframe.training.core import _phase_train_one_target as pt

    assert hasattr(pt, "_maybe_run_feature_handling_apply"), (
        "_maybe_run_feature_handling_apply must remain a module-level symbol of _phase_train_one_target; the test rig + future "
        "predict-side wave both import it from there."
    )


def test_train_one_target_actually_calls_the_helper():
    """Pin the call site: the per-target training path must reference ``_maybe_run_feature_handling_apply`` in its compiled co_names.

    The compiled-bytecode introspection is behavioural (asks the interpreter what names the function actually resolves, not what the
    source string contains) so a future refactor that drops the call -- whether by deleting the line or by renaming through an
    alias -- breaks this assertion before the wire-in goes dark. Avoids ``inspect.getsource`` per the project rule against
    source-string assertions.

    The per-target body was carved into submodules; ``_train_one_target`` now delegates the model-setup seam (where the wire-in
    lives) to ``_setup_per_target_mlframe_models``. Walk the delegation chain so the sensor follows the call wherever it sits.
    """
    from mlframe.training.core._phase_train_one_target import _train_one_target
    from mlframe.training.core._phase_train_one_target_model_setup import (
        _setup_per_target_mlframe_models,
    )

    # _train_one_target delegates the model-setup seam (which owns the wire-in) to this function.
    assert "_setup_per_target_mlframe_models" in _train_one_target.__code__.co_names
    assert "_maybe_run_feature_handling_apply" in _setup_per_target_mlframe_models.__code__.co_names, (
        "the per-target model-setup path must invoke _maybe_run_feature_handling_apply; the wire-in lives there. If this fails "
        "after a refactor, the FHC kwarg is dead code again."
    )
