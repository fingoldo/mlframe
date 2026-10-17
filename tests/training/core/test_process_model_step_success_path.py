"""Regression: ``_run_process_model_step`` must return the trained frames on success, not ``None``.

A ``return None`` once sat at function level after the ``try``/``except``, so every successful ``process_model`` call
was reported as a skipped weight and its model was never recorded. The caller then ``continue``-d past every model.
"""

from types import SimpleNamespace

from mlframe.training.core import _phase_train_one_target_steps as mod


def _state(continue_on_failure: bool = False) -> SimpleNamespace:
    """Minimal training state carrying only the attributes the step reads."""
    return SimpleNamespace(
        trainset_features_stats=None,
        train_df_transformed=None,
        behavior_config=SimpleNamespace(continue_on_model_failure=continue_on_failure),
        verbose=0,
        _non_neural_train_times=[],
        metadata={},
    )


def test_process_model_step_returns_frames_on_success(monkeypatch):
    """A successful process_model call yields its (pre_pipeline, val, test) and records the non-neural train time."""

    def fake_process_model(**kwargs):
        """Stand-in process_model that returns the three frames the success path must hand back."""
        return "stats", "pp_out", "train_t", "val_t", "test_t"

    monkeypatch.setattr(mod, "process_model", fake_process_model)
    st = _state()

    result = mod._run_process_model_step(st, "cb", "uniform", {}, "pp_in", False)

    assert result == ("pp_out", "val_t", "test_t")
    assert len(st._non_neural_train_times) == 1


def test_process_model_step_failure_is_tolerated_and_recorded(monkeypatch):
    """With continue_on_model_failure=True a raising process_model returns None and records the failed model."""

    def failing_process_model(**kwargs):
        """Stand-in process_model that raises, so the failure branch is exercised."""
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "process_model", failing_process_model)
    st = _state(continue_on_failure=True)

    result = mod._run_process_model_step(st, "cb", "uniform", {}, "pp_in", False)

    assert result is None
    assert st.metadata["failed_models"][0]["model"] == "cb"
    assert st._non_neural_train_times == []
