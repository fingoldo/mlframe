"""Audit 2026-11-20 finding 28-F11: the auto-wrap switch reaches the control config the training loop reads, so False is honoured."""

from mlframe.training._trainer_config_builders import _build_control_config


def _control(auto_wrap):
    """Build the control config with the given auto-wrap switch and every other input at its neutral default."""
    return _build_control_config(
        verbose=0, use_cache=False, just_evaluate=False, compute_trainset_metrics=False, compute_valset_metrics=False,
        compute_testset_metrics=False, pre_pipeline=None, skip_pre_pipeline_transform=False, skip_preprocessing=False,
        fit_params=None, callback_params=None, model_category="", slice_stable_es=None, row_wise_extensions_config=None,
        auto_wrap_partial_fit_es=auto_wrap,
    )


def test_false_switch_is_kept_on_the_control_config():
    """A user's False reaches the config the training loop reads; the old code always saw the default True."""
    assert _control(False).auto_wrap_partial_fit_es is False


def test_true_switch_is_kept_on_the_control_config():
    """The default True is still True on the config."""
    assert _control(True).auto_wrap_partial_fit_es is True
