"""Sensor test for the ``composite_transforms`` monolith split (Wave 6a).

The parent ``mlframe.training.composite_transforms`` was carved into three
new siblings:

- ``_composite_transforms_simple`` (diff, additive_residual, median_residual,
  y_quantile_clip, ratio, rolling_quantile_ratio)
- ``_composite_transforms_registry`` (``_TRANSFORMS_REGISTRY`` + unary adapters)
- ``_composite_transforms_naming`` (compose_target_name + get_transform +
  list_transforms + is_composite_target_name + short-name aliases)

This sensor pins:

1. Identity preserved: every re-exported symbol on the parent is the SAME
   object as on the sibling (catches the canonical "redefinition in parent"
   regression where carve becomes a copy instead of a re-export).
2. Facade LOC budget: parent stays well under the 800-line target.
3. Smoke: one fit/forward/inverse call per sibling routed through the
   parent re-export, exercising at least one function body per carve.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_composite_transforms_simple_identity_preserved():
    from mlframe.training import composite_transforms as parent
    from mlframe.training import _composite_transforms_simple as simple

    for name in (
        "_diff_fit", "_diff_forward", "_diff_inverse", "_diff_domain",
        "_additive_residual_fit", "_additive_residual_forward",
        "_additive_residual_inverse", "_additive_residual_domain",
        "_median_residual_fit", "_median_residual_forward",
        "_median_residual_inverse", "_median_residual_domain",
        "_y_quantile_clip_fit", "_y_quantile_clip_forward",
        "_y_quantile_clip_inverse", "_y_quantile_clip_domain",
        "_ratio_fit", "_ratio_forward", "_ratio_inverse", "_ratio_domain",
        "_rolling_quantile_ratio_fit", "_rolling_quantile_ratio_forward",
        "_rolling_quantile_ratio_inverse", "_rolling_quantile_ratio_domain",
        "_MEDIAN_RESIDUAL_N_BINS", "_Y_QUANTILE_CLIP_LO",
        "_Y_QUANTILE_CLIP_HI", "_ROLLING_QUANTILE_DEFAULT_K",
    ):
        assert getattr(parent, name) is getattr(simple, name), (
            f"parent.{name} differs from simple sibling -- carve became a copy"
        )


def test_composite_transforms_registry_identity_preserved():
    from mlframe.training import composite_transforms as parent
    from mlframe.training import _composite_transforms_registry as registry

    assert parent._TRANSFORMS_REGISTRY is registry._TRANSFORMS_REGISTRY
    assert parent._make_unary_registry_adapter is registry._make_unary_registry_adapter
    for name in (
        "_cbrt_fit", "_log_fit_a", "_yj_fit_a", "_qn_fit_a",
        "_cbrt_forward", "_log_forward_a", "_yj_forward_a", "_qn_forward_a",
    ):
        assert getattr(parent, name) is getattr(registry, name), name


def test_composite_transforms_naming_identity_preserved():
    from mlframe.training import composite_transforms as parent
    from mlframe.training import _composite_transforms_naming as naming

    for name in (
        "TRANSFORM_NAME_SHORT", "_COMPOSITE_NAME_FRAGMENTS",
        "compose_target_name", "get_transform",
        "is_composite_target_name", "list_transforms",
    ):
        assert getattr(parent, name) is getattr(naming, name), name


def test_composite_transforms_facade_loc_budget():
    parent_path = (
        Path(__file__).resolve().parents[2]
        / "src" / "mlframe" / "training" / "composite_transforms.py"
    )
    n_lines = len(parent_path.read_text(encoding="utf-8").splitlines())
    # Plan target: <800; current carve lands ~300. Budget guard at 800 so
    # future additions trigger the split rule rather than re-grow the parent.
    assert n_lines <= 800, f"composite_transforms.py grew to {n_lines} lines"


def test_composite_transforms_smoke_simple_via_parent():
    from mlframe.training.composite_transforms import get_transform

    rng = np.random.default_rng(0)
    y = rng.normal(size=200).astype(np.float64)
    base = rng.normal(size=200).astype(np.float64)
    t = get_transform("diff")
    params = t.fit(y, base)
    fwd = t.forward(y, base, params)
    inv = t.inverse(fwd, base, params)
    assert np.allclose(inv, y)


def test_composite_transforms_smoke_registry_unary_via_parent():
    from mlframe.training.composite_transforms import get_transform

    rng = np.random.default_rng(1)
    y = np.abs(rng.normal(size=300)).astype(np.float64) + 0.1
    t = get_transform("cbrt_y")
    params = t.fit(y, None)
    fwd = t.forward(y, None, params)
    inv = t.inverse(fwd, None, params)
    assert np.allclose(inv, y, atol=1e-6)


def test_composite_transforms_smoke_naming_via_parent():
    from mlframe.training.composite_transforms import (
        compose_target_name,
        is_composite_target_name,
        list_transforms,
        TRANSFORM_NAME_SHORT,
    )

    name = compose_target_name("TVT", "linear_residual", "TVT_prev")
    assert name == "TVT-linres-TVT_prev"
    assert is_composite_target_name(name)
    assert not is_composite_target_name("raw_y")
    assert "linear_residual" in TRANSFORM_NAME_SHORT
    transforms = list_transforms()
    assert "diff" in transforms and "linear_residual" in transforms
