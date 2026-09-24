"""Every public config class that mlframe.training.configs defines or re-exports is in its __all__."""

import inspect

import mlframe.training.configs as configs


def test_star_import_brings_every_public_config_class():
    classes = [name for name, obj in vars(configs).items() if inspect.isclass(obj) and obj.__module__.startswith("mlframe") and not name.startswith("_")]
    assert len(classes) > 20
    missing = sorted(set(classes) - set(configs.__all__))
    assert missing == [], f"public config classes missing from __all__: {missing}"
