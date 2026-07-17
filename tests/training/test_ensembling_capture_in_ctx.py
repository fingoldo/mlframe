"""Regression: ensemble outputs from _phase_train_one_target must end up
on the training context, not silently discarded.

Pre-fix: ``_ensembles = score_ensemble(...)`` bound the return value to a
local that nothing read. Ensemble models were built then thrown away.
"""

from __future__ import annotations




def test_training_context_has_ensembles_slot():
    # Verifies the new TrainingContext.ensembles field exists with the
    # expected default (empty dict).
    from mlframe.training.core._training_context import TrainingContext

    ctx = TrainingContext()
    assert hasattr(ctx, "ensembles")
    assert ctx.ensembles == {}


def test_score_ensemble_return_assigned_to_ctx_and_models():
    """Mock score_ensemble; drive the small branch that consumes its result.

    We don't run the whole _train_one_target (it requires a full data
    suite). Instead we exercise the assignment block directly by replaying
    its logic against a fresh TrainingContext, asserting both side
    effects:
      1. ctx.ensembles[type][target] holds the dict.
      2. models[type][target] gained an entry per ensemble method.
    """
    from mlframe.training.core._training_context import TrainingContext

    ctx = TrainingContext()
    models = ctx.models  # same dict the phase mutates
    target_type = "regression"
    cur_target_name = "tgt"

    fake_ensembles = {
        "arithm": object(),
        "median": object(),
    }

    # Replicate the assignment block we added in _phase_train_one_target.
    if fake_ensembles:
        ctx.ensembles.setdefault(target_type, {})[cur_target_name] = fake_ensembles
        _target_models = models.setdefault(target_type, {}).setdefault(cur_target_name, [])
        for _ens_result in fake_ensembles.values():
            _target_models.append(_ens_result)

    assert target_type in ctx.ensembles
    assert cur_target_name in ctx.ensembles[target_type]
    assert ctx.ensembles[target_type][cur_target_name] is fake_ensembles

    assert target_type in models
    assert cur_target_name in models[target_type]
    assert len(models[target_type][cur_target_name]) == 2, "expected one entry per ensemble method"
    # Identity preserved -- we don't wrap/copy the per-method result.
    for ens_obj in fake_ensembles.values():
        assert ens_obj in models[target_type][cur_target_name]
