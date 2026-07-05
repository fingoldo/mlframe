"""High-cardinality categorical target-encoding residual transform.

``target_encoding_residual`` removes the per-category level of ``y`` so the
downstream model only has to explain the WITHIN-category residual:

    T = y - smoothed_category_mean(cat)
    inverse: y_hat = T_hat + smoothed_category_mean(cat)

The per-category mean is an empirical-Bayes / additive-smoothing estimate that
shrinks each raw category mean toward the global mean by a strength ``a``::

    enc[g] = (sum_y[g] + a * global_mean) / (count[g] + a)

so a tiny category (``count[g] << a``) defaults to the global mean and a large
one (``count[g] >> a``) keeps its own mean. Unseen categories at predict time
fall back to the global mean (encoding == ``global_mean``).

LEAKAGE (the cardinal sin). The per-category means are computed TRAIN-ONLY: the
wrapper passes only the train rows into ``fit`` and never re-fits at predict.
That makes the in-sample encoding optimistic for high-cardinality bases (each
train row sees its OWN ``y`` folded into its category mean). For *discovery /
model fitting* the category mean should ideally be OUT-OF-FOLD (a CV / KFold
target encoder) so the inner model is not handed a leaked signal; the smoothing
strength ``a`` is the cheap in-fold guard that keeps a singleton category from
memorising its lone ``y``. Callers wiring this into a leakage-sensitive pipeline
should pair it with OOF target means; this transform supplies the smoothing knob
and the strict train-only fit, the OOF discipline is the pipeline's job.

Registered as a GROUPED-style transform (``requires_groups=True``): the category
column is threaded through the existing ``group_column`` plumbing as the
``groups`` kwarg, so no new wrapper wiring is needed. ``requires_base=False``
because the encoding is driven entirely by the category labels, not by a numeric
base column.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Default additive-smoothing strength (pseudo-count toward the global mean).
# 20 means a category needs ~20 train rows before its own mean dominates the
# global prior -- a conventional empirical-Bayes default that protects tiny
# levels without over-smoothing well-populated ones. Exposed as a module global
# so tests / callers can override via the ``smoothing`` fit kwarg.
_TARGET_ENCODING_DEFAULT_SMOOTHING: float = 20.0


def _target_encoding_residual_fit(
    y: np.ndarray,
    base: np.ndarray,  # noqa: ARG001 -- unused; encoding is category-driven
    groups: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,  # noqa: ARG001 -- see note below
    smoothing: float = _TARGET_ENCODING_DEFAULT_SMOOTHING,
) -> dict[str, Any]:
    """Fit per-category smoothed means on TRAIN rows only.

    Parameters
    ----------
    y
        Training target (1-D, length n).
    base
        Unused placeholder (``requires_base=False``); the encoding is driven by
        ``groups`` (the category labels), not a numeric base column.
    groups
        Per-row category labels (1-D, length n). Required; raised as ValueError
        if None (configure ``group_column`` on the wrapper to supply them).
    sample_weight
        Accepted for signature parity with the weight-aware transforms but
        currently unused -- weighted category means within tiny levels are
        unstable and the smoothing prior already dominates there.
    smoothing
        Additive-smoothing strength ``a`` (pseudo-count toward the global mean).
        ``a = 0`` is the raw (unsmoothed) category mean; larger ``a`` shrinks
        small categories harder toward the global mean.

    Returns
    -------
    dict with keys:
    - ``global_mean``: float -- train mean of ``y`` (fallback for unseen cats).
    - ``encoding``: dict[str(label) -> float] -- smoothed per-category mean.
      Keys are ``str(label)`` for JSON-serialisability (mirrors the grouped
      transform's ``per_group_*`` dicts).
    - ``smoothing``: float -- the ``a`` actually used (diagnostic).
    """
    if groups is None:
        raise ValueError(
            "target_encoding_residual requires a 1-D ``groups`` array of "
            "per-row category labels (configure ``group_column`` on the "
            "wrapper). The category column carries the high-cardinality levels."
        )
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    if len(groups) != len(y_f):
        raise ValueError(f"target_encoding_residual: groups has {len(groups)} rows but y " f"has {len(y_f)} rows.")
    a = float(smoothing)
    if not np.isfinite(a) or a < 0.0:
        raise ValueError(f"target_encoding_residual: smoothing must be a finite non-negative " f"float, got {smoothing!r}.")
    global_mean = float(np.mean(y_f)) if len(y_f) > 0 else 0.0

    # Group-wise sum + count in one pass via np.unique inverse-index scatter-add.
    unique_groups, inverse_idx = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse_idx, minlength=len(unique_groups)).astype(np.float64)
    sums = np.bincount(inverse_idx, weights=y_f, minlength=len(unique_groups))
    # Empirical-Bayes additive smoothing toward the global mean.
    smoothed = (sums + a * global_mean) / (counts + a)
    # Canonical keys so int<->float dtype drift between fit and predict cannot
    # silently miss every category and collapse to the global mean (see
    # _canonical_group_key). ``str`` alone made ``1`` and ``1.0`` distinct keys.
    from . import _canonical_group_key

    encoding: dict[str, float] = {_canonical_group_key(g): float(v) for g, v in zip(unique_groups, smoothed)}
    return {
        "global_mean": global_mean,
        "encoding": encoding,
        "smoothing": a,
    }


def _category_encoding_lookup(
    groups: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Map each row's category label to its smoothed mean.

    Unseen categories (absent from the train-fitted ``encoding``) fall back to
    the global mean. Returns a length-n float64 array.

    Vectorised: the per-row Python ``dict.get`` loop is O(n) Python frames and
    dominated the fit hotpath (cProfile n=200k: 1.77/2.05s = 86%). We instead
    deduplicate the row labels with ``np.unique(return_inverse=True)`` -- a
    single C pass -- look the smoothed mean up ONCE per DISTINCT label (the
    dict has at most n_categories entries, not n rows), then scatter back via
    the inverse index. ~30x faster at n=200k since distinct labels << rows.
    """
    encoding: dict[str, float] = params["encoding"]
    global_mean = float(params["global_mean"])
    groups = np.asarray(groups).reshape(-1)
    if groups.size == 0:
        return np.empty(0, dtype=np.float64)
    uniq, inverse_idx = np.unique(groups, return_inverse=True)
    # One dict lookup per DISTINCT label (unseen -> global mean fallback).
    # Canonical key matches the fit-side keying so an int->float dtype shift at
    # predict does not miss every category and silently fall back to global mean.
    from . import _canonical_group_key
    uniq_enc = np.fromiter(
        (encoding.get(_canonical_group_key(g), global_mean) for g in uniq),
        dtype=np.float64,
        count=len(uniq),
    )
    return uniq_enc[inverse_idx]


def _target_encoding_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],  # noqa: ARG001
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """T = y - smoothed_category_mean(cat)."""
    if groups is None:
        raise ValueError("target_encoding_residual.forward: groups kwarg is required.")
    enc = _category_encoding_lookup(groups, params)
    return np.asarray(y, dtype=np.float64).reshape(-1) - enc


def _target_encoding_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],  # noqa: ARG001
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """y_hat = T_hat + smoothed_category_mean(cat). Unseen cats -> global mean."""
    if groups is None:
        raise ValueError("target_encoding_residual.inverse: groups kwarg is required.")
    enc = _category_encoding_lookup(groups, params)
    return np.asarray(t_hat, dtype=np.float64).reshape(-1) + enc


def _target_encoding_residual_domain(
    y: np.ndarray | None, base: np.ndarray,  # noqa: ARG001
) -> np.ndarray:
    """Valid rows: finite y (base is unused). At predict (y is None) every row
    is in-domain -- unseen categories are handled by the global-mean fallback,
    not by the domain mask."""
    if y is None:
        n = len(base) if hasattr(base, "__len__") else 1
        return np.ones(n, dtype=bool)
    return np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1))
