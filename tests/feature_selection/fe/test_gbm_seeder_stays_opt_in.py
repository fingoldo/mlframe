"""The surrogate-GBM interaction seeder stays opt-in: on the three-way XOR of the four-pillars frame it finds the true triple on 3 of 5 seeds (binary) and 0 of 5 (continuous)."""
import inspect

from mlframe.feature_selection.filters import _meta_fe_recommender as recommender
from mlframe.feature_selection.filters.mrmr import MRMR


def test_default_is_off():
    """The constructor default is False."""
    assert inspect.signature(MRMR.__init__).parameters["fe_gbm_seeder_enable"].default is False


def test_the_recommender_cannot_propose_it():
    """No flag the recommender can switch on is the seeder's."""
    proposable = set()
    for name, value in vars(recommender).items():
        if name.endswith("_FLAG") and isinstance(value, str):
            proposable.add(value)
        elif name.endswith("_FLAGS") and isinstance(value, (tuple, list, set, frozenset)):
            proposable.update(v for v in value if isinstance(v, str))
    assert proposable and "fe_gbm_seeder_enable" not in proposable
