"""String-valued MRMR params reject typos: at construction on the config objects, at fit start on the flat kwargs.

The hybrid-orth consumers run inside a per-family try/except that turns their own ValueError into a warning, so a typo such as
``ensemble_aggregator="borda"`` used to disable that FE family for the whole fit instead of failing.
"""

from __future__ import annotations

import typing
import warnings

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters.mrmr import _mrmr_param_constants as pc
from mlframe.feature_selection.filters.mrmr._mrmr_config_dataclasses import DCDConfig, HybridOrthConfig, HybridOrthScorersConfig

_CONFIG_TYPOS = [
    (DCDConfig, "dcd_distance", "sux"),
    (DCDConfig, "dcd_swap_method", "pca"),
    (HybridOrthScorersConfig, "hsic_kernel", "rbg"),
    (HybridOrthScorersConfig, "ensemble_aggregator", "borda"),
    (HybridOrthScorersConfig, "ensemble_scorers", ("plug_in", "kgs")),
    (HybridOrthScorersConfig, "meta_force_scorer", "cmin"),
    (HybridOrthScorersConfig, "default_scorer", "plugin"),
    (HybridOrthConfig, "basis", "hermit"),
    (HybridOrthConfig, "cluster_basis_aggregator", "mean"),
]


@pytest.mark.parametrize("cls,field,bad", _CONFIG_TYPOS, ids=[f"{c.__name__}.{f}" for c, f, _ in _CONFIG_TYPOS])
def test_config_rejects_typo_at_construction(cls, field, bad):
    """A typo'd value must raise when the config object is built, not minutes into fit()."""
    with pytest.raises(ValidationError):
        cls(**{field: bad})


_CONFIG_ALLOWED = [
    (DCDConfig, "dcd_distance", "_VALID_DCD_DISTANCES"),
    (DCDConfig, "dcd_swap_method", "_VALID_DCD_SWAP_METHODS"),
    (HybridOrthScorersConfig, "hsic_kernel", "_VALID_FE_HYBRID_ORTH_HSIC_KERNELS"),
    (HybridOrthScorersConfig, "ensemble_aggregator", "_VALID_FE_HYBRID_ORTH_ENSEMBLE_AGGREGATORS"),
    (HybridOrthScorersConfig, "meta_force_scorer", "_VALID_FE_HYBRID_ORTH_META_FORCE_SCORERS"),
    (HybridOrthScorersConfig, "default_scorer", "_VALID_FE_HYBRID_ORTH_DEFAULT_SCORERS"),
    (HybridOrthConfig, "basis", "_VALID_FE_HYBRID_ORTH_BASES"),
    (HybridOrthConfig, "cluster_basis_aggregator", "_VALID_FE_HYBRID_ORTH_CLUSTER_BASIS_AGGREGATORS"),
]


@pytest.mark.parametrize("cls,field,allowed_name", _CONFIG_ALLOWED, ids=[f"{c.__name__}.{f}" for c, f, _ in _CONFIG_ALLOWED])
def test_config_literal_matches_flat_allow_list(cls, field, allowed_name):
    """The config Literal and the flat-param allow-list must accept exactly the same values, and every one must construct."""
    allowed = getattr(pc, allowed_name)
    ann = cls.model_fields[field].annotation
    args = [a for a in typing.get_args(ann) if a is not type(None)]
    literal_values = set(typing.get_args(args[0])) if args and typing.get_origin(args[0]) is typing.Literal else set(args)
    assert literal_values == set(allowed)
    for v in allowed:
        cls(**{field: v})


def test_ensemble_scorers_config_accepts_every_allowed_scorer():
    """Every scorer the ensemble consumer knows is accepted by the config."""
    cfg = HybridOrthScorersConfig(ensemble_scorers=pc._VALID_FE_HYBRID_ORTH_ENSEMBLE_SCORERS)
    assert set(cfg.ensemble_scorers) == set(pc._VALID_FE_HYBRID_ORTH_ENSEMBLE_SCORERS), "the config must keep every allowed scorer, not drop any"


def test_allow_lists_match_their_consumers():
    """The allow-lists must track what the consumers actually accept, or they reject valid values / pass bad ones."""
    from mlframe.feature_selection.filters._orthogonal_cluster_basis_fe import _VALID_AGGREGATORS
    from mlframe.feature_selection.filters._orthogonal_meta_scorer_fe import META_SCORER_NAMES
    from mlframe.feature_selection.filters._orthogonal_scorer_auto_fe import ENSEMBLE_AGGREGATORS, SCORER_NAMES
    from mlframe.feature_selection.filters.hermite_fe import _POLY_BASES

    assert set(pc._VALID_FE_HYBRID_ORTH_ENSEMBLE_AGGREGATORS) == set(ENSEMBLE_AGGREGATORS)
    assert set(pc._VALID_FE_HYBRID_ORTH_ENSEMBLE_SCORERS) == set(SCORER_NAMES)
    assert set(pc._VALID_FE_HYBRID_ORTH_META_FORCE_SCORERS) == set(META_SCORER_NAMES) | {"tc"}
    assert set(pc._VALID_FE_HYBRID_ORTH_BASES) == set(_POLY_BASES) | {"auto"}
    assert set(pc._VALID_FE_HYBRID_ORTH_CLUSTER_BASIS_AGGREGATORS) == set(_VALID_AGGREGATORS)


_FLAT_TYPOS = [
    ("fe_hybrid_orth_basis", "hermit"),
    ("fe_hybrid_orth_hsic_kernel", "rbg"),
    ("fe_hybrid_orth_ensemble_aggregator", "borda"),
    ("fe_hybrid_orth_ensemble_scorers", ("plug_in", "kgs")),
    ("fe_hybrid_orth_meta_force_scorer", "cmin"),
    ("fe_hybrid_orth_cluster_basis_aggregator", "mean"),
]


@pytest.mark.parametrize("name,bad", _FLAT_TYPOS, ids=[n for n, _ in _FLAT_TYPOS])
def test_flat_typo_raises_at_fit_start(name, bad):
    """The flat kwarg path must fail at fit() start with the parameter named, not silently drop an FE family."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0).astype(np.int32).to_numpy()
    m = MRMR(full_npermutations=3, baseline_npermutations=2, n_jobs=1, verbose=0, fe_max_steps=0, **{name: bad})
    with pytest.raises(ValueError, match=name):
        m.fit(X, y)


def test_defaults_and_case_insensitive_force_scorer_pass_validation():
    """Controls: the defaults are valid, and the meta force scorer keeps accepting any case, as its consumer lower-cases it."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # accepting with a warning would be a silent downgrade, not a pass
        MRMR()._validate_string_params()
        m = MRMR(fe_hybrid_orth_meta_force_scorer="CMIM")
        m._validate_string_params()
    assert m.fe_hybrid_orth_meta_force_scorer == "CMIM", "validation must not rewrite the caller's value"
