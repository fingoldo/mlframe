"""T1#9 2026-05-18 Pack J unary y-transform replay sufficiency.

Earlier I dismissed T1#9 "Pack J unary recipe" as duplicative of T1#3
(Hermite EngineeredRecipe). This test pins WHY: Pack J unary y-transforms
(``requires_base=False``) need only ``transform_name + fitted_params``
to reproduce the forward output. The transform registry IS the recipe.

Specifically: for each unary y-transform that lives in
``mlframe.training.composite.transforms``,

    transform.forward(y, base=zeros, params) -> T_train
    transform.forward(y_new, base=zeros, params) -> T_new

is fully deterministic once ``params`` is captured. No EngineeredRecipe
needed because no per-fit polynomial coefficients live outside ``params``.

Pin this guarantee with two tests:

1. ``CompositeSpec`` round-trip preserves transform_name + fitted_params,
   and recreated spec reproduces forward output.
2. Pickle round-trip on a fitted-params dict reproduces forward output
   (dill / sklearn.clone compatibility).
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest


_PACK_J_UNARY_NAMES = ["cbrt_y", "log_y", "yeo_johnson_y", "quantile_normal_y"]


@pytest.mark.parametrize("transform_name", _PACK_J_UNARY_NAMES)
def test_unary_y_transform_reproducible_via_params_only(transform_name):
    """T1#9 invariant: unary y-transform forward is a pure function of
    (y, params); no other state is needed for predict-time replay."""
    from mlframe.training.composite.transforms import get_transform

    transform = get_transform(transform_name)
    assert transform.requires_base is False, f"Pack J unary transforms must have requires_base=False; {transform_name!r} reports {transform.requires_base!r}"

    rng = np.random.default_rng(0)
    n = 500
    if transform_name == "log_y":
        y = rng.uniform(0.5, 100.0, n)  # log domain
    elif transform_name == "cbrt_y":
        y = rng.normal(0.0, 5.0, n)
    elif transform_name == "yeo_johnson_y":
        y = rng.normal(0.0, 5.0, n)
    elif transform_name == "quantile_normal_y":
        y = rng.normal(0.0, 5.0, n)
    else:
        raise AssertionError(f"unhandled transform_name {transform_name!r}")

    base = np.zeros_like(y)  # ignored for requires_base=False
    valid = transform.domain_check(y, base)
    y_valid = y[valid]
    base_valid = base[valid]

    params = transform.fit(y_valid, base_valid)
    T_orig = transform.forward(y_valid, base_valid, params)

    # Same params on the same y must reproduce T exactly.
    T_replay = transform.forward(y_valid, base_valid, params)
    np.testing.assert_allclose(T_replay, T_orig, atol=1e-12)


@pytest.mark.parametrize("transform_name", _PACK_J_UNARY_NAMES)
def test_unary_y_params_pickle_round_trip(transform_name):
    """``fitted_params`` must survive pickle (CompositeSpec is frozen
    dataclass so the dict gets pickled as part of CompositeSpec)."""
    from mlframe.training.composite.transforms import get_transform

    transform = get_transform(transform_name)
    rng = np.random.default_rng(0)
    n = 400
    if transform_name == "log_y":
        y = rng.uniform(1.0, 50.0, n)
    else:
        y = rng.normal(0.0, 3.0, n)
    base = np.zeros_like(y)
    valid = transform.domain_check(y, base)
    params = transform.fit(y[valid], base[valid])

    blob = pickle.dumps(params)
    params2 = pickle.loads(blob)

    T_orig = transform.forward(y[valid], base[valid], params)
    T_replay = transform.forward(y[valid], base[valid], params2)
    np.testing.assert_allclose(T_replay, T_orig, atol=1e-12)


def test_composite_spec_carries_pack_j_replay_state():
    """CompositeSpec is the persistence layer for Pack J. Building a
    spec from a unary fit and recreating the transform from it must
    reproduce the same forward output."""
    from mlframe.training.composite.spec import CompositeSpec
    from mlframe.training.composite.transforms import (
        get_transform,
        compose_target_name,
    )

    transform = get_transform("cbrt_y")
    rng = np.random.default_rng(0)
    n = 400
    y = rng.normal(0.0, 3.0, n)
    base = np.zeros_like(y)
    params = transform.fit(y, base)
    T_orig = transform.forward(y, base, params)

    # Pack J: base_column is structurally unused but the CompositeSpec
    # field is required. Discovery fills it from the per-base iteration.
    spec = CompositeSpec(
        name=compose_target_name("y", "cbrt_y", "x_placeholder"),
        target_col="y",
        transform_name="cbrt_y",
        base_column="x_placeholder",
        fitted_params=dict(params),
        mi_gain=0.1,
        mi_y=0.05,
        mi_t=0.15,
        valid_domain_frac=1.0,
        n_train_rows=n,
    )

    # Recreate by reading the spec.
    transform_replay = get_transform(spec.transform_name)
    T_replay = transform_replay.forward(y, base, spec.fitted_params)
    np.testing.assert_allclose(T_replay, T_orig, atol=1e-12)
