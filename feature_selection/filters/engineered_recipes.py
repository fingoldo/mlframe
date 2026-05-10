"""Recipe-based replay of engineered features for ``MRMR.transform``.

Background
----------
Numeric ``_run_fe_step`` (and the upcoming categorical ``cat-FE`` path)
build new columns from input X during ``fit``. Historically those columns
were tracked only by name in ``_engineered_features_``; ``transform`` had
no way to recompute them on test data and silently dropped them from
the output (see line ~688 in ``mrmr.py``: ``# !TODO! failing when
fe_max_steps>1. need other source.``).

This module fixes that contract gap. A *recipe* is a frozen description
of how to recompute one engineered column from the original feature
matrix. ``MRMR.fit`` records one recipe per surviving engineered
feature; ``MRMR.transform`` replays each recipe against the test X and
appends the resulting columns to the output.

Recipe kinds
------------
``"unary_binary"``: numeric pair FE -- ``binary(unary_a(X[a]),
                    unary_b(X[b]))``, optionally discretized.
``"factorize"``:    cat-FE -- ``merge_vars`` of two ordinal-encoded
                    categorical columns (canonical XOR-style synergy
                    capture). Future PR.

The recipe is intentionally a small frozen dataclass (no
behaviour bound to ``self``) so it round-trips cleanly through pickle
and ``sklearn.base.clone``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]


def _extra_equal(a: dict, b: dict) -> bool:
    """``EngineeredRecipe.extra`` may contain numpy arrays (notably the
    factorize lookup table). Plain ``dict.__eq__`` calls ``arr1 == arr2``
    which returns an ndarray, not a bool, and breaks ``recipe1 == recipe2``.
    This helper compares each value with ``np.array_equal`` for arrays
    and ``==`` otherwise."""
    if a.keys() != b.keys():
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            if not (isinstance(va, np.ndarray) and isinstance(vb, np.ndarray)):
                return False
            if not np.array_equal(va, vb):
                return False
        else:
            if va != vb:
                return False
    return True


@dataclass(frozen=True, eq=False)
class EngineeredRecipe:
    """One frozen description of how to recompute an engineered column.

    Fields are picked so the recipe survives pickle and ``sklearn.clone``
    without holding any references to closures, fitted estimators, or
    captured numpy arrays.

    Parameters
    ----------
    name
        Engineered column name, e.g. ``"mul(log(c1),sin(c2))"``. Used
        as the key under which the column appears in transform output
        and ``get_feature_names_out``.
    kind
        Discriminator for the replay strategy.
        ``"unary_binary"`` -- numeric pair FE.
        ``"factorize"``     -- cat-FE k-way ordinal merge (future PR).
    src_names
        Original feature names this recipe consumes. Length 2 for
        unary_binary; length k for factorize. ``transform(X)`` looks
        these up by name in the input frame, so they must be a subset
        of ``feature_names_in_``.
    unary_names
        For ``"unary_binary"``: the two unary function names from
        ``feature_engineering.create_unary_transformations(preset)``,
        e.g. ``("log", "sin")``. ``"identity"`` means no transform.
    binary_name
        For ``"unary_binary"``: the binary function name from
        ``feature_engineering.create_binary_transformations(preset)``,
        e.g. ``"mul"``.
    unary_preset / binary_preset
        Preset names so we can rebuild the same registries at replay
        time. ``MRMR.fit`` snapshots these so a later import-time
        registry edit doesn't silently change replay semantics.
    quantization
        ``None`` if the recipe outputs a raw numeric column.
        Else a dict ``{"nbins": int, "method": str, "dtype": str}``
        captured at fit time so replay matches the fit-time discretization.
    factorize_nbins
        For ``"factorize"``: per-source nbins captured at fit time.
        Needed to feed ``merge_vars`` with the right shape, and to
        clip unseen test-time values to the highest trained bin (see
        ``unknown_strategy``).
    unknown_strategy
        For ``"factorize"``: how to handle test-time category values
        not seen at fit time. ``"clip"`` (default) caps them at the
        highest trained bin (no information leaks but unseen values
        collide with the largest bin); ``"sentinel"`` adds a separate
        bin (less collision but inflates cardinality); ``"raise"``
        errors out.
    """

    name: str
    kind: Literal["unary_binary", "factorize"]
    src_names: tuple[str, ...]
    unary_names: tuple[str, ...] = ()
    binary_name: str = ""
    unary_preset: str = "minimal"
    binary_preset: str = "minimal"
    quantization: dict | None = None
    factorize_nbins: tuple[int, ...] = ()
    unknown_strategy: Literal["clip", "sentinel", "raise"] = "clip"
    # `extra` is a free-form bucket for future recipe kinds (e.g. polynomial-
    # basis Hermite recipes carry `coef_a`, `coef_b`, `degree_a`, `degree_b`,
    # `bin_func_name`). Kept generic to avoid bloating the core dataclass.
    extra: dict = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        """Custom ``__eq__`` that handles ndarray values in ``extra``
        (factorize lookup tables). ``frozen=True, eq=False`` on the
        dataclass disables the auto-generated ``__eq__`` so this one
        wins."""
        if not isinstance(other, EngineeredRecipe):
            return NotImplemented
        if self.kind != other.kind:
            return False
        if self.name != other.name:
            return False
        if self.src_names != other.src_names:
            return False
        if self.unary_names != other.unary_names:
            return False
        if self.binary_name != other.binary_name:
            return False
        if self.unary_preset != other.unary_preset:
            return False
        if self.binary_preset != other.binary_preset:
            return False
        if self.quantization != other.quantization:
            return False
        if self.factorize_nbins != other.factorize_nbins:
            return False
        if self.unknown_strategy != other.unknown_strategy:
            return False
        return _extra_equal(self.extra, other.extra)

    def __hash__(self) -> int:
        # Frozen dataclasses with mutable fields (``extra: dict``,
        # ``unary_names: tuple`` is OK) typically opt out of __hash__.
        # Provide a name-based hash so recipes can be used as dict keys
        # within a single fitted MRMR (names are unique per fit).
        return hash((self.kind, self.name))


def _extract_column(X: Any, name: str) -> np.ndarray:
    """Pull a single column from X by name as a 1-D ndarray (no copy when
    possible). Supports pandas DataFrame, polars DataFrame, and numpy
    structured/recarray. Caller responsibility per ``mlframe/CLAUDE.md``:
    we never copy the whole frame, only narrow column pulls."""
    if pd is not None and isinstance(X, pd.DataFrame):
        # ``.values`` is zero-copy for numeric dtypes in modern pandas; for
        # categorical / object it materialises but only the single column.
        return X[name].to_numpy() if hasattr(X[name], "to_numpy") else X[name].values
    if pl is not None and isinstance(X, pl.DataFrame):
        return X[name].to_numpy()
    if isinstance(X, np.ndarray):
        if X.dtype.names is not None:
            return X[name]
        # Plain 2-D ndarray with no column names -- caller must have
        # passed a name we can't resolve. Raise rather than guess.
        raise KeyError(
            f"Cannot resolve column '{name}' on a plain 2-D ndarray. "
            "Pass a pandas / polars frame or a structured array."
        )
    raise TypeError(f"Unsupported X type for engineered-recipe replay: {type(X)!r}")


def apply_recipe(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Replay ``recipe`` against ``X`` and return the engineered column
    as a 1-D ndarray. The output dtype matches the recipe's recorded
    quantization dtype if discretized, else float32 (matching fit-time
    ``check_prospective_fe_pairs`` working buffer dtype).

    This function is hot in ``transform()``; keep it allocation-light.
    """
    if recipe.kind == "unary_binary":
        return _apply_unary_binary(recipe, X)
    if recipe.kind == "factorize":
        return _apply_factorize(recipe, X)
    raise ValueError(f"Unknown recipe kind: {recipe.kind!r}")


def _apply_unary_binary(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    if len(recipe.src_names) != 2 or len(recipe.unary_names) != 2:
        raise ValueError(
            f"unary_binary recipe '{recipe.name}' must have exactly 2 src_names "
            f"and 2 unary_names; got {len(recipe.src_names)} / {len(recipe.unary_names)}"
        )
    # Lazy import: avoids circular dependency (feature_engineering imports
    # from mrmr indirectly via _internals).
    from .feature_engineering import (
        create_unary_transformations,
        create_binary_transformations,
    )
    from .discretization import discretize_array

    unary_funcs = create_unary_transformations(preset=recipe.unary_preset)
    binary_funcs = create_binary_transformations(preset=recipe.binary_preset)

    name_a, name_b = recipe.src_names
    u_a, u_b = recipe.unary_names

    if u_a not in unary_funcs:
        raise KeyError(
            f"Unary function '{u_a}' not in '{recipe.unary_preset}' preset. "
            f"Replay requires the same preset that was active at fit time."
        )
    if u_b not in unary_funcs:
        raise KeyError(
            f"Unary function '{u_b}' not in '{recipe.unary_preset}' preset."
        )
    if recipe.binary_name not in binary_funcs:
        raise KeyError(
            f"Binary function '{recipe.binary_name}' not in "
            f"'{recipe.binary_preset}' preset."
        )

    vals_a = _extract_column(X, name_a)
    vals_b = _extract_column(X, name_b)

    transformed_a = unary_funcs[u_a](vals_a)
    transformed_b = unary_funcs[u_b](vals_b)
    out = binary_funcs[recipe.binary_name](transformed_a, transformed_b)

    # Match fit-time NaN/Inf scrubbing in
    # ``feature_engineering.check_prospective_fe_pairs`` (line ~209).
    out = np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if recipe.quantization is not None:
        q = recipe.quantization
        out = discretize_array(
            arr=out,
            n_bins=q["nbins"],
            method=q["method"],
            dtype=np.dtype(q["dtype"]),
        )
    return out


def _apply_factorize(recipe: EngineeredRecipe, X: Any) -> np.ndarray:
    """Cat-FE replay: look up each test row's ``(a, b)`` tuple in the
    fit-time lookup table and emit the post-prune class.

    The lookup table maps ``a_value + b_value * nbins_a`` (the
    pre-prune code) to the post-prune class produced by
    ``merge_vars`` at fit time. Test-time values that fall outside
    ``[0, nbins_a)`` / ``[0, nbins_b)`` are clipped (any test value
    above the fit-time max is treated as the max). Test-time
    combinations whose pre-prune code never appeared in training
    are resolved per ``recipe.unknown_strategy`` (already baked into
    the lookup at fit time, except for ``"raise"`` which keeps -1
    sentinels and surfaces here).
    """
    # K-way (k > 2) recipes don't yet ship a lookup table -- transform()
    # replay needs a chained-merge_vars approach that's deferred to a
    # future PR. Surface a clear error instead of silently producing
    # wrong codes.
    if recipe.extra.get("requires_refit_for_replay"):
        raise NotImplementedError(
            f"factorize recipe '{recipe.name}' is a k-way (order "
            f"{recipe.extra.get('kway_order', '?')}) engineered feature. "
            f"transform() replay for k > 2 is not implemented in v1 -- "
            f"pair recipes (order=2) replay fine. Disable k-way at fit "
            f"time via CatFEConfig(max_kway_order=2) if you need replay."
        )
    if len(recipe.src_names) != 2 or len(recipe.factorize_nbins) != 2:
        raise ValueError(
            f"factorize recipe '{recipe.name}' requires exactly 2 src_names "
            f"and 2 factorize_nbins; got {len(recipe.src_names)} / "
            f"{len(recipe.factorize_nbins)}"
        )
    if "lookup_table" not in recipe.extra:
        raise KeyError(
            f"factorize recipe '{recipe.name}' is missing the 'lookup_table' "
            f"in recipe.extra. This usually means the recipe was built before "
            f"the cat-FE replay PR landed; refit the MRMR estimator."
        )

    name_a, name_b = recipe.src_names
    nbins_a, nbins_b = recipe.factorize_nbins
    lookup: np.ndarray = recipe.extra["lookup_table"]

    vals_a = _extract_column(X, name_a)
    vals_b = _extract_column(X, name_b)
    # Cast to int64 for safe multiplication (pre-prune code can exceed int32
    # for high-cardinality combos, and the lookup is indexed by int64).
    vals_a_i = vals_a.astype(np.int64, copy=False)
    vals_b_i = vals_b.astype(np.int64, copy=False)

    # Clip out-of-range to nbins-1. Without this, a test value of
    # ``nbins_a + 1`` would index past the lookup buffer end. Per
    # ``unknown_strategy="clip"`` semantics (the default), unseen
    # values map to the highest seen class -- that's already encoded
    # in the lookup itself; here we just guard the buffer.
    vals_a_i = np.clip(vals_a_i, 0, nbins_a - 1)
    vals_b_i = np.clip(vals_b_i, 0, nbins_b - 1)

    pre_prune_codes = vals_a_i + vals_b_i * nbins_a
    out = lookup[pre_prune_codes]

    # ``raise`` strategy left -1 sentinels in the lookup; anything that
    # comes back negative here is a test combo never seen in training.
    if recipe.unknown_strategy == "raise" and (out < 0).any():
        n_unseen = int((out < 0).sum())
        raise ValueError(
            f"factorize recipe '{recipe.name}': {n_unseen} row(s) have "
            f"(X[{name_a}], X[{name_b}]) combinations not seen during fit. "
            f"Set unknown_strategy='clip' or 'sentinel' to handle these "
            f"silently."
        )
    return out


def build_unary_binary_recipe(
    *,
    name: str,
    src_a_name: str,
    src_b_name: str,
    unary_a_name: str,
    unary_b_name: str,
    binary_name: str,
    unary_preset: str,
    binary_preset: str,
    quantization_nbins: int | None,
    quantization_method: str | None,
    quantization_dtype: Any,
) -> EngineeredRecipe:
    """Build an ``EngineeredRecipe`` for the ``"unary_binary"`` kind.

    Encapsulates the convention that ``quantization`` is ``None`` when
    no discretization is recorded, else a dict with the three params.
    Stringifies the dtype so the recipe is JSON-friendly and pickle-safe
    across numpy versions.
    """
    if quantization_nbins is None:
        quantization = None
    else:
        quantization = {
            "nbins": int(quantization_nbins),
            "method": str(quantization_method) if quantization_method else "uniform",
            "dtype": np.dtype(quantization_dtype).str,
        }
    return EngineeredRecipe(
        name=name,
        kind="unary_binary",
        src_names=(src_a_name, src_b_name),
        unary_names=(unary_a_name, unary_b_name),
        binary_name=binary_name,
        unary_preset=unary_preset,
        binary_preset=binary_preset,
        quantization=quantization,
    )
