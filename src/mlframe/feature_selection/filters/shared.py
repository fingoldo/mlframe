"""Cross-package API of ``mlframe.feature_selection.filters``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.feature_selection.filters`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from ._boruta import (
    boruta_select,
)
from ._content_memo import (
    ContentMemo,
)
from ._conditional_gate_fe import (
    apply_conditional_gate,
    apply_row_argmax,
    cheap_conditional_gate_scan,
    cheap_row_argmax_scan,
)
from ._fe_accuracy_gate import (
    bin_y_for_class_mi,
    class_mi_fe_applicable,
)
from ._fe_gpu_batch._devices import (
    crit_float_dtype,
)
from ._fe_usability_signal import (
    _crit_np_dtype as crit_np_dtype,
)
from ._integer_lattice_fe import (
    apply_integer_lattice,
    cheap_integer_lattice_scan,
)
from ._mrmr_artifacts import (
    validate_artifact_dict,
)
from ._mrmr_fingerprints import (
    _hashable_params_signature as hashable_params_signature,
)
from ._pairwise_modular_fe import (
    _is_integer_col as is_integer_col,
    _mi as mi,
    apply_pairwise_modular,
    cheap_modular_scan,
    escalate_modulus,
)
from .correlated_features import (
    _su_redundancy_matrix as su_redundancy_matrix,
)
from .permutation import (
    _perm_pvalue as perm_pvalue,
)

from ._device_sort_safe import device_argsort, device_sort
from ._kernel_types import render
from ._fe_gpu_batch._devices import assert_kernel_buffers

from ._ksg_dispatch import ksg_mi_columns
from ._ksg_cd_fast import ksg_mi_discrete_target
from ._orthogonal_univariate_fe._orth_mi_backends import contingency_mi_nats
from ._modulus_perm_null import host_residue_path, perm_best_mis
from ._y_encoding import encode_y_for_classif_mi

__all__ = [
    "ContentMemo",
    "apply_conditional_gate",
    "apply_integer_lattice",
    "apply_pairwise_modular",
    "apply_row_argmax",
    "assert_kernel_buffers",
    "bin_y_for_class_mi",
    "boruta_select",
    "cheap_conditional_gate_scan",
    "cheap_integer_lattice_scan",
    "cheap_modular_scan",
    "cheap_row_argmax_scan",
    "class_mi_fe_applicable",
    "contingency_mi_nats",
    "crit_float_dtype",
    "crit_np_dtype",
    "device_argsort",
    "device_sort",
    "encode_y_for_classif_mi",
    "escalate_modulus",
    "hashable_params_signature",
    "host_residue_path",
    "is_integer_col",
    "ksg_mi_columns",
    "ksg_mi_discrete_target",
    "mi",
    "perm_best_mis",
    "perm_pvalue",
    "render",
    "su_redundancy_matrix",
    "validate_artifact_dict",
]
