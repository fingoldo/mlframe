"""Cluster 28 regression: a bare model-name string is one model, so its key does not collide with a character set (audit 2026-11-20, finding 28-F73)."""

from mlframe.training.suite_artefact_cache import SuiteKeyBuilder


def _key(models):
    """Build a suite key that differs only in the model selection."""
    return SuiteKeyBuilder.build(df_fp="fp", config_canonical={"a": 1}, mlframe_models=models, random_seed=0)


def test_string_model_does_not_collide_with_its_characters():
    """'cb' and {'b', 'c'} are different selections and must key differently (28-F73)."""
    assert _key("cb") != _key(["b", "c"])


def test_string_model_equals_its_one_element_list():
    """Control: 'cb' is the same selection as ['cb'] (28-F73)."""
    assert _key("cb") == _key(["cb"])
