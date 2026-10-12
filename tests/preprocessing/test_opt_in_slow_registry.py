"""Opt-in slow preprocessing functions resolve by name only when the config enables them.

Default config keeps every entry off, so the training pipeline is unchanged; enabling a name reaches the real function.
"""

import pytest

from mlframe.preprocessing.opt_in_slow import OPT_IN_SLOW, resolve_opt_in_slow


@pytest.mark.parametrize("name", sorted(OPT_IN_SLOW))
def test_registered_name_resolves_to_its_function_when_enabled(name):
    """Each registry name reaches the function it names in the module it is registered under."""
    fn = resolve_opt_in_slow(name, enabled=[name])
    assert callable(fn)
    assert fn.__name__ == name


@pytest.mark.parametrize("name", sorted(OPT_IN_SLOW))
def test_registered_name_refused_when_not_enabled(name):
    """Without the switch, a registered name raises instead of running the slow path."""
    with pytest.raises(PermissionError):
        resolve_opt_in_slow(name, enabled=None)


def test_unknown_name_raises_key_error():
    """A name missing from the registry is rejected outright."""
    with pytest.raises(KeyError):
        resolve_opt_in_slow("not_a_real_function", enabled=["not_a_real_function"])


def test_default_preprocessing_config_enables_nothing():
    """The default config leaves opt_in_slow unset, so no opt-in function is reachable from the pipeline."""
    from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig

    assert PreprocessingExtensionsConfig().opt_in_slow is None
