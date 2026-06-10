"""Composite-target naming helpers.

Public import path is ``mlframe.training.composite.transforms`` (e.g.
``from mlframe.training.composite.transforms import compose_target_name``);
the pre-split flat ``mlframe.training.composite_transforms`` module no
longer exists, so do not import from that path.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .registry import _TRANSFORMS_REGISTRY
from . import UnknownTransformError

if TYPE_CHECKING:
    from . import Transform


# Short-name aliases for composite-target naming. Used in
# ``compose_target_name`` to keep displayed target names compact;
# previously composites were named ``y__linear_residual__lag1``
# which read ugly in logs / report headings / dict keys. The dash
# separator + short aliases give us e.g. ``y-linres-lag1``.
#
# Order: declared transforms only -- if a transform is missing from
# this map we fall back to the full name in ``compose_target_name`` so
# adding a new transform never silently breaks naming.
TRANSFORM_NAME_SHORT: dict[str, str] = {
    "diff": "diff",
    "additive_residual": "addres",
    "median_residual": "medres",
    "y_quantile_clip": "yqclip",
    "ratio": "ratio",
    "logratio": "logr",
    "linear_residual": "linres",
    "linear_residual_robust": "linresR",
    "linear_residual_multi": "linresM",
    "linear_residual_grouped": "linresG",
    "quantile_residual": "qres",
    "monotonic_residual": "monres",
    "ewma_residual": "ewma",
    "rolling_quantile_ratio": "rqr",
    "frac_diff": "fdiff",
    # Pack J unary y-transforms (no base segment in the composite name).
    "cbrt_y": "cbrtY",
    "log_y": "logY",
    "yeo_johnson_y": "yjY",
    "quantile_normal_y": "qnY",
    # Pack K chain transforms.
    "chain_linres_cbrt": "linresCbrt",
    "chain_linres_yj": "linresYj",
    "chain_monres_cbrt": "monresCbrt",
    "chain_monres_yj": "monresYj",
    "chain_linres_cbrt_qn": "linresCbrtQn",
    # Pack L extended bivariate + multi-base transforms (2026-05-26).
    "asinh_residual": "asinhr",
    "centered_ratio": "cratio",
    "polynomial_residual_deg2": "poly2",
    "rank_residual": "rankr",
    "smoothing_spline_residual": "spline",
    "reciprocal_residual": "recipr",
    "geometric_mean_residual": "gmean",
    "pairwise_interaction_residual": "interact",
}


def compose_target_name(target_col: str, transform_name: str, base: str) -> str:
    """Build the canonical composite-target name from its three components.

    Uses ``-`` as the separator and the short transform alias from
    ``TRANSFORM_NAME_SHORT``. Falls back to the full transform name if
    the alias is missing (so brand-new transforms get a longer-but-correct
    name on day one instead of silent collision).

    Examples:
        compose_target_name('y', 'linear_residual', 'lag1')
            -> 'y-linres-lag1'
        compose_target_name('y', 'monotonic_residual', 'base')
            -> 'y-monres-base'
    """
    short = TRANSFORM_NAME_SHORT.get(transform_name, transform_name)
    return f"{target_col}-{short}-{base}"


# Reverse-lookup pattern fragments: ``f"-{short}-"`` and ``f"-{full}-"``
# both appear as substrings in canonical composite-target names. Used by
# ``is_composite_target_name`` to detect "this target name came from
# discovery, NOT a user-supplied column called ``y-base`` that happens
# to have one dash".
_COMPOSITE_NAME_FRAGMENTS: frozenset = frozenset(
    f"-{alias}-" for alias in TRANSFORM_NAME_SHORT.values()
) | frozenset(
    f"-{full}-" for full in TRANSFORM_NAME_SHORT.keys()
)


def is_composite_target_name(name: str) -> bool:
    """True if ``name`` matches the canonical composite-target naming
    convention ``{target}-{transform_short}-{base}`` for any registered
    transform.

    Used by per-target metric / chart helpers to switch their label from
    ``MTTR`` (raw mean target) to ``MTRESID`` (residual mean ~= 0 by
    construction). Robust to both the post-2026-05-13 short-alias format
    AND the legacy ``{target}__{transform}__{base}`` double-underscore
    format -- so loading a v1 suite-pickle still routes correctly.
    """
    if not name:
        return False
    if any(frag in name for frag in _COMPOSITE_NAME_FRAGMENTS):
        return True
    # Legacy double-underscore format (older pickles).
    for full in TRANSFORM_NAME_SHORT.keys():
        if f"__{full}__" in name:
            return True
    return False


def get_transform(name: str) -> "Transform":
    """Lookup helper. Raises :exc:`UnknownTransformError` for typos."""
    try:
        return _TRANSFORMS_REGISTRY[name]
    except KeyError as exc:
        raise UnknownTransformError(
            f"Unknown transform '{name}'. Registered: {sorted(_TRANSFORMS_REGISTRY)}"
        ) from exc


def list_transforms(*, tags: frozenset[str] | None = None) -> list[str]:
    """Return registered transform names, optionally filtered by tag
    intersection (any-of: a transform passes if it has at least one of
    the requested tags)."""
    if tags is None:
        return sorted(_TRANSFORMS_REGISTRY)
    return sorted(
        name for name, t in _TRANSFORMS_REGISTRY.items() if t.tags & tags
    )
