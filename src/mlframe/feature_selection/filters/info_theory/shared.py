"""Cross-package API of ``mlframe.feature_selection.filters.info_theory``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.feature_selection.filters.info_theory`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from ._batch_kernel_selection import (  # noqa: F401
    select_batch_mi_kernel,
)
from ._batch_kernels import (  # noqa: F401
    check_joint_cardinality,
    joint_cardinality_cap,
)
from ._cmi_cuda import (  # noqa: F401
    _cpu_cmi_loop_parallel as cpu_cmi_loop_parallel,
    clear_cmi_resident_cache,
    conditional_mi_batched_dispatch,
    reset_cmi_gpu_circuit_breaker,
)
from ._group_mi import (  # noqa: F401
    group_blocked_mi,
    group_relevance_mi,
    prepare_group_segments,
)
from ._state_and_dispatch import (  # noqa: F401
    get_group_mi,
    set_group_mi,
)

__all__ = [
    "check_joint_cardinality",
    "clear_cmi_resident_cache",
    "conditional_mi_batched_dispatch",
    "cpu_cmi_loop_parallel",
    "get_group_mi",
    "group_blocked_mi",
    "group_relevance_mi",
    "joint_cardinality_cap",
    "prepare_group_segments",
    "reset_cmi_gpu_circuit_breaker",
    "select_batch_mi_kernel",
    "set_group_mi",
]
