"""Regression: the stacked-discovery OOF factories must pass ``extra_base_columns``
through, so a multi-base spec's OOF comes from the SAME multi-base transform it represents.

Both ``fit_stacked`` and ``fit_stacked_on_residual`` built the OOF
``CompositeTargetEstimator`` with only ``base_column``, silently dropping
``extra_base_columns`` -> a multi-base spec's OOF column came from a different (single-base)
transform than the spec it feeds into pass-2 / the aggregate. The shared
``_spec_base_columns`` helper now supplies the full base tuple to both.
"""

from dataclasses import dataclass

from mlframe.training.composite.discovery._stacked import _spec_base_columns
from mlframe.training.composite.estimator import CompositeTargetEstimator


@dataclass
class _Spec:
    name: str
    transform_name: str
    base_column: str
    extra_base_columns: tuple = ()


def test_single_base_spec_yields_none():
    spec = _Spec("s", "linear_residual", "b0")
    assert _spec_base_columns(spec) is None


def test_multi_base_spec_yields_full_tuple():
    spec = _Spec("s", "linear_residual_multi", "b0", ("b1", "b2"))
    assert _spec_base_columns(spec) == ("b0", "b1", "b2")


def test_estimator_built_with_helper_resolves_all_bases():
    from sklearn.linear_model import Ridge

    spec = _Spec("s", "linear_residual_multi", "b0", ("b1", "b2"))
    est = CompositeTargetEstimator(
        base_estimator=Ridge(alpha=1e-3),
        transform_name=spec.transform_name,
        base_column=spec.base_column,
        base_columns=_spec_base_columns(spec),
    )
    # All three base columns must be visible to the estimator (pre-fix: only "b0").
    assert tuple(est._resolve_base_columns()) == ("b0", "b1", "b2")
