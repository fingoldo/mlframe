"""Cluster 37 regression: knob overrides never mutate the caller's config object (audit 2026-11-20, finding 37-F11)."""

from types import SimpleNamespace

from pydantic import BaseModel

from mlframe.training.core._main_train_suite_target_distribution import _apply_knob_overrides


class _HpConfig(BaseModel):
    """Stand-in hyperparameter config with one per-knob slot."""

    lgb_kwargs: dict = {}


class _Report:
    """Fake distribution report that overrides one lgb knob."""

    knob_overrides = {"lgb_kwargs": {"learning_rate": 0.05}}

    def merge_into_config(self, cfg, override_existing=False):
        """Return a copy of ``cfg`` with the override merged into ``lgb_kwargs``."""
        out = dict(cfg)
        out["lgb_kwargs"] = {**(cfg.get("lgb_kwargs") or {}), "learning_rate": 0.05}
        return out


def test_caller_config_is_not_mutated():
    """The caller's config keeps its original slot value after an override is applied."""
    cfg = _HpConfig(lgb_kwargs={"max_depth": 3})
    ctx = SimpleNamespace()
    result = _apply_knob_overrides(_Report(), cfg, ctx)
    assert cfg.lgb_kwargs == {"max_depth": 3}, "caller's config was mutated"
    assert result.lgb_kwargs["learning_rate"] == 0.05
    assert ctx.hyperparams_config is result
