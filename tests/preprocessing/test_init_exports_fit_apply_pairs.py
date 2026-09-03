"""PREPROCESSING-8 regression test: mlframe.preprocessing must export the fit_*/apply_* leakage-safe
function pairs for rare_count_pruning, missing_indicator_pairing, and regime_conditioned_imputation, not
just their combined convenience wrappers.

The bug (fixed): the package __init__ omitted fit_rare_category_collapse/apply_rare_category_collapse,
fit_missing_indicator_imputation/apply_missing_indicator_imputation, and
fit_regime_conditioned_median/apply_regime_conditioned_median_fill -- even though each module's own
docstring recommends the fit-on-train/apply-to-test split, and the analogous pair
(align_feature_direction/apply_feature_direction) WAS exported.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    "name",
    [
        "fit_rare_category_collapse",
        "apply_rare_category_collapse",
        "fit_missing_indicator_imputation",
        "apply_missing_indicator_imputation",
        "fit_regime_conditioned_median",
        "apply_regime_conditioned_median_fill",
    ],
)
def test_fit_apply_pair_importable_from_package(name):
    """Each fit_*/apply_* function must be importable directly from mlframe.preprocessing."""
    import mlframe.preprocessing as pkg

    assert hasattr(pkg, name), f"mlframe.preprocessing should export {name}"
    assert name in pkg.__all__, f"{name} should be in mlframe.preprocessing.__all__"
