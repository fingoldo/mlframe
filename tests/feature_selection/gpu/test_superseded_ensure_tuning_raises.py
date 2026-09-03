"""FS_BENCHMARKS_C-2 regression test: the 4 legacy ``ensure_*_tuning`` sweeps superseded by the
``pyutilz.performance.kernel_tuning`` registry must refuse to run instead of silently writing a
stale-schema region to the cache key the new registry owns.

The bug (fixed): ``ensure_batch_pair_mi_tuning`` (and 3 siblings -- ``ensure_cat_fe_perm_kernel_tuning``,
``ensure_unary_elementwise_tuning``, ``ensure_rff_matmul_tuning``) remained fully implemented and
importable after their kernels migrated to the new registry, even though ``cli.py``'s own docstring
documents that this exact legacy-write pattern "silently shadowed the new dispatcher" (writing regions
without a ``backend_choice``/``code_version`` to the same cache key). Direct import + call was still
possible for anyone bypassing the CLI. All 4 now raise ``RuntimeError`` unconditionally.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    "func_name",
    [
        "ensure_batch_pair_mi_tuning",
        "ensure_cat_fe_perm_kernel_tuning",
        "ensure_unary_elementwise_tuning",
        "ensure_rff_matmul_tuning",
    ],
)
def test_superseded_ensure_tuning_raises_instead_of_writing_cache(func_name):
    """Calling any of the 4 superseded legacy ensure_*_tuning sweeps must raise, not silently write."""
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import auto_tune

    fn = getattr(auto_tune, func_name)
    with pytest.raises(RuntimeError, match="superseded"):
        fn()
    with pytest.raises(RuntimeError, match="superseded"):
        fn(force=True)
