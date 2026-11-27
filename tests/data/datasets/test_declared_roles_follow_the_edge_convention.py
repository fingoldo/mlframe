"""A declared edge kind marks its target, except an instrument, which marks its source (audit 2026-11-20, finding 04-F6)."""
from types import SimpleNamespace

from mlframe.data.datasets._scm import _declared_kind_roles
from mlframe.data.datasets.ground_truth import FeatureRole


def test_instrument_role_lands_on_the_instrument_not_the_treatment():
    """The instrument bed's arrow instrument -> x_cause must mark the instrument, and leave the treatment unmarked."""
    graph = SimpleNamespace(edges=[SimpleNamespace(source="instrument", target="x_cause", kind="instrument")])
    assert _declared_kind_roles(graph) == {"instrument": FeatureRole.INSTRUMENT}


def test_redundant_copy_role_lands_on_the_copy():
    """A redundant_noisy edge z -> copy marks the copy, the node that is redundant."""
    graph = SimpleNamespace(edges=[SimpleNamespace(source="z", target="copy", kind="redundant_noisy")])
    assert _declared_kind_roles(graph) == {"copy": FeatureRole.REDUNDANT_NOISY}
