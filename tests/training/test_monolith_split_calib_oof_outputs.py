"""Sensor: trainer post-fit tail carve into ``_calib_oof_outputs.py``.

Verifies re-export identity AND calls into the moved bodies (calib/OOF outputs +
confidence-analysis dispatch) so a missing import would fail at runtime, not pass
an import-only check.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def test_calib_oof_reexport_identity():
    """Calib oof reexport identity."""
    from mlframe.training import _calib_oof_outputs as sib
    from mlframe.training import _trainer_train_and_evaluate as parent

    assert parent.compute_calib_and_oof_outputs is sib.compute_calib_and_oof_outputs
    assert parent.maybe_run_confidence_analysis is sib.maybe_run_confidence_analysis


def test_compute_calib_and_oof_outputs_no_calib_path():
    """Compute calib and oof outputs no calib path."""
    from mlframe.training._calib_oof_outputs import compute_calib_and_oof_outputs

    cp, ct, cpreds, op, opb, otg = compute_calib_and_oof_outputs(
        model=None,
        calib_df=None,
        calib_target=None,
        real_drop_columns=[],
        pre_pipeline=None,
        skip_pre_pipeline_transform=False,
        skip_preprocessing=False,
        fit_params={},
        model_type_name="X",
        model_name="m",
    )
    assert cp is None and ct is None and cpreds is None and op is None and opb is None and otg is None


def test_compute_calib_and_oof_outputs_mirrors_oof_without_predict_proba():
    """Compute calib and oof outputs mirrors oof without predict proba."""
    from mlframe.training._calib_oof_outputs import compute_calib_and_oof_outputs

    model = SimpleNamespace(oof_preds=np.array([1, 2, 3]), oof_probs=np.array([[0.1, 0.9]]), oof_target=np.array([1, 0, 1]))
    cp, ct, cpreds, op, opb, otg = compute_calib_and_oof_outputs(
        model=model,
        calib_df=object(),
        calib_target=None,
        real_drop_columns=[],
        pre_pipeline=None,
        skip_pre_pipeline_transform=False,
        skip_preprocessing=False,
        fit_params={},
        model_type_name="X",
        model_name="m",
    )
    # no predict_proba / no regression predict -> calib outputs None; oof mirrored from the model
    assert cp is None and ct is None and cpreds is None
    assert op is model.oof_preds and opb is model.oof_probs and otg is model.oof_target


def test_maybe_run_confidence_analysis_noop_paths(monkeypatch):
    """The analysis is skipped when the test split is off, confidence is excluded or there is no test frame, and runs for the enabled case."""
    from mlframe.training import _calib_oof_outputs as mod

    calls: list = []
    monkeypatch.setattr(mod, "run_confidence_analysis", lambda **kw: calls.append(kw))

    def invoke(run_test, include, test_df):
        """Call the dispatcher with an otherwise fixed argument set."""
        return mod.maybe_run_confidence_analysis(
            run_test=run_test,
            confidence=SimpleNamespace(include=include, model_kwargs=None, use_shap=False, max_features=5, cmap=None, alpha=0.5, title=None, ylabel=None),
            test_df=test_df,
            test_target="target",
            test_probs="probs",
            fit_params=None,
            model_type_name="X",
            figsize=None,
            verbose=0,
        )

    assert invoke(False, True, object()) is None
    assert invoke(True, False, object()) is None
    assert invoke(True, True, None) is None
    assert calls == []
    frame = object()
    assert invoke(True, True, frame) is None
    assert len(calls) == 1
    assert calls[0]["test_df"] is frame
    assert calls[0]["test_target"] == "target" and calls[0]["test_probs"] == "probs"
