"""Cross-package API of ``mlframe.feature_selection.filters``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.feature_selection.filters`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from ._boruta import (  # noqa: F401
    boruta_select,
)
from ._conditional_gate_fe import (  # noqa: F401
    apply_conditional_gate,
    apply_row_argmax,
    cheap_conditional_gate_scan,
    cheap_row_argmax_scan,
)
from ._fe_accuracy_gate import (  # noqa: F401
    bin_y_for_class_mi,
    class_mi_fe_applicable,
)
from ._integer_lattice_fe import (  # noqa: F401
    apply_integer_lattice,
    cheap_integer_lattice_scan,
)
from ._mrmr_fingerprints import (  # noqa: F401
    _hashable_params_signature as hashable_params_signature,
)
from ._pairwise_modular_fe import (  # noqa: F401
    _is_integer_col as is_integer_col,
    _mi as mi,
    apply_pairwise_modular,
    cheap_modular_scan,
    escalate_modulus,
)
from .group_aware import (  # noqa: F401
    _su_redundancy_matrix as su_redundancy_matrix,
)
from .permutation import (  # noqa: F401
    _perm_pvalue as perm_pvalue,
)

__all__ = [
    "apply_conditional_gate",
    "apply_integer_lattice",
    "apply_pairwise_modular",
    "apply_row_argmax",
    "bin_y_for_class_mi",
    "boruta_select",
    "cheap_conditional_gate_scan",
    "cheap_integer_lattice_scan",
    "cheap_modular_scan",
    "cheap_row_argmax_scan",
    "class_mi_fe_applicable",
    "escalate_modulus",
    "hashable_params_signature",
    "is_integer_col",
    "mi",
    "perm_pvalue",
    "su_redundancy_matrix",
]
