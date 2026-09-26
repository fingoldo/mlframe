"""Cross-package API of ``mlframe.training``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.training`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from . import (  # noqa: F401
    _patch_lgb_feature_names_in_setter as patch_lgb_feature_names_in_setter,
)
from .io import (  # noqa: F401
    _write_save_meta_sidecar as write_save_meta_sidecar,
)
from .trainer import (  # noqa: F401
    _build_configs_from_params as build_configs_from_params,
)

__all__ = [
    "build_configs_from_params",
    "patch_lgb_feature_names_in_setter",
    "write_save_meta_sidecar",
]
