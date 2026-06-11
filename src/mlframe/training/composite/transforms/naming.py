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
# both appear as substrings in canonical composite-target names. Kept as a
# module global for back-compat (callers / tests import it) but the
# substring scan is NO LONGER the detection primitive -- it false-positives
# on plausible user columns (``price-diff-7d`` -> ``-diff-``,
# ``debt-ratio-q`` -> ``-ratio-``). ``is_composite_target_name`` now uses a
# strict structural parse (see below) that anchors the alias as a complete
# dash-delimited token inside a ``{target}-{alias}-{base}`` triple.
_COMPOSITE_NAME_FRAGMENTS: frozenset = frozenset(
    f"-{alias}-" for alias in TRANSFORM_NAME_SHORT.values()
) | frozenset(
    f"-{full}-" for full in TRANSFORM_NAME_SHORT.keys()
)

# Set of every recognised transform token (short alias + full name). A
# token must equal one of these *exactly* (whole dash-/underscore-delimited
# segment) to anchor a composite name -- this is what makes the parse strict
# instead of a substring scan.
_COMPOSITE_NAME_TOKENS: frozenset = frozenset(
    TRANSFORM_NAME_SHORT.values()
) | frozenset(TRANSFORM_NAME_SHORT.keys())


def _is_composite_via_separator(name: str, sep: str) -> bool:
    """Strict structural test for one separator (``-`` short-alias form or
    ``__`` legacy double-underscore form).

    A composite name is ``{target}{sep}{alias}{sep}{base}`` where ``alias``
    is a registered transform token occupying a *complete* segment, with a
    non-empty ``{target}`` segment before it and a non-empty ``{base}``
    segment after it. Target / base may themselves contain the separator
    (user column names with dashes), so we test every internal segment
    boundary, not just a fixed 3-way split.
    """
    parts = name.split(sep)
    if len(parts) < 3:
        return False
    # The alias must be an *internal* token: at least one segment before it
    # (non-empty target) and at least one segment after it (non-empty base).
    # parts[0] is the start of target; parts[-1] is the end of base. A valid
    # alias therefore sits at index 1..len-2 inclusive, and both the target
    # span (parts[:i]) and base span (parts[i+1:]) must be non-empty.
    for i in range(1, len(parts) - 1):
        if parts[i] not in _COMPOSITE_NAME_TOKENS:
            continue
        target_span = parts[:i]
        base_span = parts[i + 1:]
        # Non-empty, non-blank target and base (reject ``-linres-x`` /
        # ``y-linres-`` style malformed names that the old substring scan
        # silently accepted).
        if "".join(target_span) and "".join(base_span):
            return True
    return False


def is_composite_target_name(name: str) -> bool:
    """True if ``name`` matches the canonical composite-target naming
    convention ``{target}-{transform_short}-{base}`` for any registered
    transform.

    Used by per-target metric / chart helpers to switch their label from
    ``MTTR`` (raw mean target) to ``MTRESID`` (residual mean ~= 0 by
    construction). Robust to both the post-2026-05-13 short-alias format
    AND the legacy ``{target}__{transform}__{base}`` double-underscore
    format -- so loading a v1 suite-pickle still routes correctly.

    Detection is a **strict structural parse**, not a substring scan: the
    transform alias must be a complete dash- (or ``__``-) delimited token
    bracketed by a non-empty target segment and a non-empty base segment.
    This avoids the substring false-positives the old scan produced on
    plausible user columns -- e.g. ``price-diff-7d`` and ``debt-ratio-q``
    are 3-token names whose middle token IS a registered alias and remain
    structurally indistinguishable from real composites (so they still
    match -- there is no signal to separate them without the target list),
    but multi-segment columns that merely *contain* an alias substring
    (``quarterly-margin-ratio-of-revenue-2024``: alias ``ratio`` is an
    internal token, still matches; ``net-ratiometric-index``: ``ratio`` is
    only a substring of ``ratiometric``, NO longer matches) and malformed
    leading/trailing-separator forms (``-linres-x``, ``y-linres-``) are
    correctly rejected.
    """
    if not name:
        return False
    # Short-alias / full-name dash form (current discovery output).
    if _is_composite_via_separator(name, "-"):
        return True
    # Legacy double-underscore format (older pickles). Only full transform
    # names were ever emitted in that format, but accepting the union token
    # set here is harmless and keeps the two paths symmetric.
    if "__" in name and _is_composite_via_separator(name, "__"):
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
