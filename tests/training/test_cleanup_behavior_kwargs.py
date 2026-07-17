"""Sensor for Wave 1 cleanup: configure_training_params must accept every
TrainingBehaviorConfig field without TypeError.

Pre-fix bug: each new behavior knob added to TrainingBehaviorConfig (e.g.
mlp_extreme_ar_group_aware_skip, mlp_extreme_ar_threshold) broke the
**effective_behavior_params splat in train_eval.py because configure_training_params
only declared a hand-maintained subset. The fix added **_unused_behavior_kwargs;
this sensor ensures the catch-all keeps the contract for the full behavior surface.
"""

from __future__ import annotations

import inspect

import pytest

from mlframe.training._model_configs import TrainingBehaviorConfig
from mlframe.training._trainer_configure import configure_training_params


def test_configure_training_params_accepts_all_behavior_fields_via_splat():
    """Every public TrainingBehaviorConfig field must splat into configure_training_params."""

    sig = inspect.signature(configure_training_params)
    has_var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    declared = {name for name, p in sig.parameters.items() if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}

    behavior_fields = set(TrainingBehaviorConfig.model_fields.keys())
    not_declared = behavior_fields - declared

    if not has_var_keyword and not_declared:
        pytest.fail(
            f"configure_training_params lacks both a **kwargs catch-all and explicit "
            f"params for: {sorted(not_declared)}. The train_eval.py splat will raise "
            f"TypeError for these. Either add them to the signature or keep the catch-all."
        )


def test_configure_training_params_known_no_op_knobs_dont_raise():
    """Known TrainingBehaviorConfig knobs (added 5.x rounds) must not raise."""

    # Picking three knobs that were flagged by Wave 1 cleanup as missing-on-signature;
    # all three are consumed downstream via behavior_config not via this signature.
    sig = inspect.signature(configure_training_params)
    accepts_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    candidates = {
        "mlp_extreme_ar_group_aware_skip": True,
        "mlp_extreme_ar_threshold": 1.5,
        "pre_pipeline_cache_max": 4,
    }
    if not accepts_kw:
        for name in candidates:
            assert name in sig.parameters, f"configure_training_params missing kw '{name}' and no **kwargs to catch it"
    # Binding with these kwargs must not raise TypeError-on-bind. (We bind to a partial via
    # Signature.bind_partial to avoid executing the body, which needs many other args.)
    sig.bind_partial(**candidates)
