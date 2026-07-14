"""Meta-test — every model-type alias accepted by ``VALID_MODEL_TYPES``
(or its linear-model sibling ``VALID_LINEAR_MODEL_TYPES``) must have a
corresponding entry in ``MODEL_STRATEGIES`` so the trainer's
``get_strategy(model_type)`` lookup returns a real ``ModelPipelineStrategy``
instance — not silently fall through to the default.

Catches the failure mode where a contributor adds ``"foo"`` to
``VALID_MODEL_TYPES`` so users can pass it in ``mlframe_models=["foo"]``,
but forgets to register the strategy. Trainer accepts the value at config
time, then much later raises a confusing KeyError / NoneType-attribute
error inside the suite loop. Same shape of bug as the config-field
consumption tests already police, but at the strategy-dispatch layer.

Also asserts the converse: every strategy registered in
``MODEL_STRATEGIES`` is reachable from at least one VALID_*_MODEL_TYPES
allow-list (so a strategy registered under a typo'd key is caught), with
a documented escape hatch ``_TRAINER_EXTRA_ALIASES`` for aliases the
trainer supports beyond the public config validator (e.g. ``"lstm"``,
``"transformer"`` registered for direct ``get_strategy()`` calls but not
through ``mlframe_models``).
"""

from __future__ import annotations

import pytest

from mlframe.training.configs import (
    VALID_LINEAR_MODEL_TYPES,
    VALID_MODEL_TYPES,
)
from mlframe.training.strategies import (
    MODEL_STRATEGIES,
    ModelPipelineStrategy,
)

# Strategy keys NOT in ``VALID_MODEL_TYPES`` / ``VALID_LINEAR_MODEL_TYPES``
# but still legitimate — typically called via ``get_strategy(name)`` from
# tests / advanced flows that bypass the public config validator.
# Each entry should have a one-line reasoning.
_TRAINER_EXTRA_ALIASES: dict[str, str] = {
    "logistic": "linear-model alias for binary logistic regression — accepted by get_strategy() directly; redundant with 'linear' for users who set mlframe_models",
    "lstm": "recurrent strategy alias used by neural test paths; not yet exposed via mlframe_models",
    "gru": "ditto for GRU",
    "rnn": "ditto for vanilla RNN",
    "transformer": "ditto for transformer recurrent variant",
    # Canonical long-form aliases for the short tree-strategy keys.
    # 2026-05-24: get_strategy('CATBOOST') was previously falling through
    # to the unknown-model UserWarning + TreeStrategy fallback. Adding
    # the long-form aliases removes the warning and routes correctly.
    "catboost": "long-form alias for 'cb' (CatBoost strategy)",
    "lightgbm": "long-form alias for 'lgb' (LightGBM tree strategy)",
    "xgboost": "long-form alias for 'xgb' (XGBoost strategy)",
    "histgradientboosting": "long-form alias for 'hgb' (sklearn HistGradientBoosting)",
    "lr": "common shorthand for 'linear' (logistic / ridge linear strategy); previously silently fell through to TreeStrategy + UserWarning",
    "gated_outlier": "opt-in gated-outlier regression variant (LGBMRegressor-backed); built via _trainer_configure.py's _should_create_model gate, routed through the tree strategy",
    "bagging": "opt-in feature-subset-bagging regression variant (LGBMRegressor-backed); built via _trainer_configure.py's _should_create_model gate, routed through the tree strategy",
    "composite_classification": "opt-in composite-target classification variant (LGBMClassifier-backed); built via _trainer_configure.py's _should_create_model gate, routed through the tree strategy",
}


def test_every_valid_model_type_has_a_strategy():
    """Every ``mlframe_models`` value the public validator accepts must
    resolve to a real strategy instance via ``MODEL_STRATEGIES``.
    """
    all_public_types = VALID_MODEL_TYPES | VALID_LINEAR_MODEL_TYPES
    missing: list[str] = []
    for model_type in sorted(all_public_types):
        if model_type not in MODEL_STRATEGIES:
            missing.append(model_type)
    if missing:
        pytest.fail(
            f"{len(missing)} model-type alias(es) accepted by VALID_MODEL_TYPES / "
            f"VALID_LINEAR_MODEL_TYPES but missing from MODEL_STRATEGIES — "
            f"trainer will fail at dispatch time. Either register the strategy "
            f"in strategies.py::MODEL_STRATEGIES, OR remove the alias from the "
            f"validator allow-list:\n  " + "\n  ".join(missing)
        )


def test_every_strategy_is_either_a_public_alias_or_a_documented_extra():
    """Every key in ``MODEL_STRATEGIES`` is reachable from at least one
    public allow-list, OR is documented in ``_TRAINER_EXTRA_ALIASES``.
    Catches a strategy registered under a typo'd key (e.g.
    ``"catboot"`` instead of ``"cb"``).
    """
    all_public_types = VALID_MODEL_TYPES | VALID_LINEAR_MODEL_TYPES
    orphan: list[str] = []
    for key in MODEL_STRATEGIES:
        if key in all_public_types:
            continue
        if key in _TRAINER_EXTRA_ALIASES:
            continue
        orphan.append(key)
    if orphan:
        pytest.fail(
            f"{len(orphan)} strategy key(s) in MODEL_STRATEGIES are not in "
            f"any public allow-list and not documented in "
            f"_TRAINER_EXTRA_ALIASES (typo or undocumented alias?):\n  " + "\n  ".join(orphan)
        )


def test_every_strategy_value_is_a_strategy_instance():
    """Sanity-check: each MODEL_STRATEGIES entry must actually be an
    instance of ``ModelPipelineStrategy`` — catches a half-finished
    refactor where a class object got wired in instead of an instance.
    """
    bad: list[str] = []
    for key, strategy in MODEL_STRATEGIES.items():
        if not isinstance(strategy, ModelPipelineStrategy):
            bad.append(f"{key!r}: {type(strategy).__name__}")
    if bad:
        pytest.fail("MODEL_STRATEGIES values must be instances of " "ModelPipelineStrategy:\n  " + "\n  ".join(bad))
