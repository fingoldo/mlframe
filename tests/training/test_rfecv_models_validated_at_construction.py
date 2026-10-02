"""``FeatureSelectionConfig.rfecv_models`` names are validated when the config is built, not deep inside the suite."""

import pytest
from pydantic import ValidationError

from mlframe.training.configs import FeatureSelectionConfig
from mlframe.training.fs_params.configs import RFECV_MODEL_NAMES


def test_unknown_rfecv_model_name_raises_at_construction():
    """Unknown rfecv model name raises at construction."""
    with pytest.raises(ValidationError, match="unknown RFECV model"):
        FeatureSelectionConfig(rfecv={"models": ["catboost_rfecv"]})


def test_unknown_rfecv_model_name_raises_on_assignment():
    """Assigning an rfecv model name raises: the config is frozen."""
    cfg = FeatureSelectionConfig(rfecv={"models": ["cb"]})
    with pytest.raises(ValidationError):
        cfg.rfecv.models = ("cb_rfec",)


@pytest.mark.parametrize("name", RFECV_MODEL_NAMES)
def test_canonical_rfecv_model_names_accepted(name):
    """Canonical rfecv model names accepted."""
    assert FeatureSelectionConfig(rfecv={"models": [name]}).rfecv.models == (name,)


def test_bare_backend_name_canonicalised_and_deduplicated():
    """Bare backend name canonicalised and deduplicated."""
    assert FeatureSelectionConfig(rfecv={"models": ["cb", "cb_rfecv", "lgb"]}).rfecv.models == ("cb_rfecv", "lgb_rfecv")
