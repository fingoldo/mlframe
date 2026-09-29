"""``FeatureSelectionConfig.rfecv_models`` names are validated when the config is built, not deep inside the suite."""

import pytest
from pydantic import ValidationError

from mlframe.training.configs import FeatureSelectionConfig
from mlframe.training._feature_selection_config import RFECV_MODEL_NAMES


def test_unknown_rfecv_model_name_raises_at_construction():
    with pytest.raises(ValidationError, match="unknown RFECV model"):
        FeatureSelectionConfig(rfecv_models=["catboost_rfecv"])


def test_unknown_rfecv_model_name_raises_on_assignment():
    cfg = FeatureSelectionConfig()
    with pytest.raises(ValidationError, match="unknown RFECV model"):
        cfg.rfecv_models = ["cb_rfec"]


@pytest.mark.parametrize("name", RFECV_MODEL_NAMES)
def test_canonical_rfecv_model_names_accepted(name):
    assert FeatureSelectionConfig(rfecv_models=[name]).rfecv_models == [name]


def test_bare_backend_name_canonicalised_and_deduplicated():
    assert FeatureSelectionConfig(rfecv_models=["cb", "cb_rfecv", "lgb"]).rfecv_models == ["cb_rfecv", "lgb_rfecv"]

