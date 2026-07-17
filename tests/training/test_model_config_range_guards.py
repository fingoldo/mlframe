"""Range guards on the per-model config dataclasses.

Each test pins a previously-silent out-of-range value that propagated to the
backend (sklearn / LightGBM / XGB / NGBoost) and either produced a degenerate
no-error run (iterations=0 -> zero-tree booster predicting the init constant)
or surfaced as a confusing error deep inside fit. The guards reject them at
construction with a clear ValidationError, matching the established
ModelHyperparamsConfig contract.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlframe.training.configs import (
    TreeModelConfig,
    LinearModelConfig,
    NGBConfig,
    MultilabelDispatchConfig,
)


# ---- TreeModelConfig -------------------------------------------------------


def test_tree_iterations_zero_raises():
    """Tree iterations zero raises."""
    with pytest.raises(ValidationError):
        TreeModelConfig(iterations=0)


def test_tree_learning_rate_negative_raises():
    """Tree learning rate negative raises."""
    with pytest.raises(ValidationError):
        TreeModelConfig(learning_rate=-0.1)


def test_tree_learning_rate_above_one_raises():
    """Tree learning rate above one raises."""
    with pytest.raises(ValidationError):
        TreeModelConfig(learning_rate=5.0)


def test_tree_max_depth_zero_raises():
    """Tree max depth zero raises."""
    with pytest.raises(ValidationError):
        TreeModelConfig(max_depth=0)


def test_tree_early_stopping_zero_is_auto_and_valid():
    # 0 means "auto (iterations // 3)" per docstring -- must stay valid.
    """Tree early stopping zero is auto and valid."""
    assert TreeModelConfig(early_stopping_rounds=0).early_stopping_rounds == 0


def test_tree_valid_config_unchanged():
    """Tree valid config unchanged."""
    cfg = TreeModelConfig(iterations=700, learning_rate=0.1, max_depth=6)
    assert cfg.iterations == 700 and cfg.max_depth == 6


# ---- LinearModelConfig -----------------------------------------------------


def test_linear_alpha_negative_raises():
    """Linear alpha negative raises."""
    with pytest.raises(ValidationError):
        LinearModelConfig(alpha=-1.0)


def test_linear_l1_ratio_above_one_raises():
    """Linear l1 ratio above one raises."""
    with pytest.raises(ValidationError):
        LinearModelConfig(model_type="elasticnet", l1_ratio=1.5)


def test_linear_l1_ratio_negative_raises():
    """Linear l1 ratio negative raises."""
    with pytest.raises(ValidationError):
        LinearModelConfig(l1_ratio=-0.2)


def test_linear_alpha_zero_is_ols_and_valid():
    """Linear alpha zero is ols and valid."""
    assert LinearModelConfig(alpha=0.0).alpha == 0.0


# ---- NGBConfig -------------------------------------------------------------


def test_ngb_n_estimators_zero_raises():
    """Ngb n estimators zero raises."""
    with pytest.raises(ValidationError):
        NGBConfig(n_estimators=0)


def test_ngb_minibatch_frac_above_one_raises():
    """Ngb minibatch frac above one raises."""
    with pytest.raises(ValidationError):
        NGBConfig(minibatch_frac=1.5)


def test_ngb_minibatch_frac_zero_raises():
    """Ngb minibatch frac zero raises."""
    with pytest.raises(ValidationError):
        NGBConfig(minibatch_frac=0.0)


# ---- MultilabelDispatchConfig ----------------------------------------------


def test_multilabel_n_chains_zero_raises():
    """Multilabel n chains zero raises."""
    with pytest.raises(ValidationError):
        MultilabelDispatchConfig(n_chains=0)


def test_multilabel_cv_one_raises():
    """Multilabel cv one raises."""
    with pytest.raises(ValidationError):
        MultilabelDispatchConfig(cv=1)


def test_multilabel_cv_none_is_valid():
    """Multilabel cv none is valid."""
    assert MultilabelDispatchConfig(cv=None).cv is None
