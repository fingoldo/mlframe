"""Replay + builders for the categorical encoding engineered-recipe kinds.

K-fold target encoding, count / frequency encoding, cat-x-num residual, and the
cat-pair / cat-triple cross builders. The ``apply_recipe`` dispatcher in the
parent ``engineered_recipes`` lazy-imports the ``_apply_*`` here; the builders
lazy-import the parent's ``EngineeredRecipe`` dataclass in-body to avoid a cycle.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]

from ._recipe_extract import _extract_column

if TYPE_CHECKING:
    from . import EngineeredRecipe


# ---------------------------------------------------------------------------
# Layer 33 (2026-05-31): K-fold target encoding for raw categorical columns
# ---------------------------------------------------------------------------
#
# The K-fold OOF mean-of-y per category. The recipe stores the FULL-DATA
# per-category mean (computed once at fit time) so transform() is a pure
# dict lookup with no y reference -- the OOF discipline is enforced at FIT
# only (the rows that fed into MRMR screening saw an out-of-fold encoded
# value, never their own y).
#
# extra layout:
# * kfold_target_encoded: {lookup: dict[str, float], global_mean: float,
#                          smoothing: float}
#   ``lookup`` maps the string-coerced source category to its smoothed
#   per-category mean. Categories not in the lookup at transform time map
#   to ``global_mean`` (no NaN propagation).
#
# Sibling to the cell-level ``target_encoding`` recipe above (which encodes
# MERGED k-way categorical CELLS inside the cat-FE pair-search kernel);
# this one encodes raw single columns and is the standard prod-ML pattern.


def _apply_kfold_target_encoded(recipe: EngineeredRecipe, X: Any, col_cache: "dict[str, np.ndarray] | None" = None) -> np.ndarray:
    """Replay a k-fold target-encoded column. Stateless given the stored
    lookup + global_mean; no y reference. Categories not present in the
    lookup map to ``global_mean``.

    ``col_cache``: optional dict shared across every recipe replayed in ONE ``transform()`` call (see
    ``apply_recipe``); forwarded to ``_extract_column`` on the polars / structured-array paths (the pandas
    fast path below already passes ``X`` through unmaterialised, so caching adds nothing there)."""
    if len(recipe.src_names) != 1:
        raise ValueError(f"kfold_target_encoded recipe '{recipe.name}' must have exactly " f"1 src_names; got {len(recipe.src_names)}")
    for key in ("lookup", "global_mean"):
        if key not in recipe.extra:
            raise KeyError(f"kfold_target_encoded recipe '{recipe.name}' missing '{key}' " f"in extra. Re-fit MRMR to regenerate.")
    # Lazy import to avoid circular dependency at module-load time.
    from .._target_encoding_fe import apply_target_encoding

    name = recipe.src_names[0]
    # Build a one-column frame view so apply_target_encoding's
    # column-name interface works on any X (pandas / polars / structured).
    if pd is not None and isinstance(X, pd.DataFrame):
        X_view = X
    elif pl is not None and pd is not None and isinstance(X, pl.DataFrame):  # pd required: X_view below is a pandas frame
        X_view = pd.DataFrame({name: _extract_column(X, name, col_cache=col_cache)})
    elif isinstance(X, np.ndarray) and X.dtype.names is not None:
        X_view = pd.DataFrame({name: _extract_column(X, name, col_cache=col_cache)})
    else:
        raise TypeError(
            f"kfold_target_encoded recipe '{recipe.name}': cannot extract "
            f"column {name!r} from X of type {type(X).__name__}. Pass a "
            f"pandas / polars frame or a structured ndarray."
        )
    # Hand off to the apply helper. recipe.extra is a MappingProxy; pass a
    # plain dict so older helpers that index into it without abstract-base-
    # class checks don't trip.
    return apply_target_encoding(
        X_view, name, {
            "lookup": dict(recipe.extra["lookup"]),
            "global_mean": float(recipe.extra["global_mean"]),
        },
    )


def build_kfold_target_encoded_recipe(
    *, name: str, src_name: str, lookup: dict, global_mean: float,
    smoothing: float,
) -> EngineeredRecipe:
    """Frozen recipe for a single-column K-fold target-encoded column.

    ``lookup`` is the per-category mean-of-y table built from the full
    training data with Micci-Barreca smoothing. ``global_mean`` is the
    unconditional mean of y (used as the prior in smoothing AND as the
    fallback for unseen categories at transform). ``smoothing`` is stored
    for diagnostics / round-trip but is NOT consulted at transform time
    (smoothing already baked into ``lookup`` values).
    """
    from . import EngineeredRecipe
    # Canonicalise keys (int<->float drift safe): the generators pre-canonicalise via _column_to_str so this is a
    # no-op today (canonical of an already-canonical string is idempotent), but a future caller that builds this
    # recipe from RAW category values would otherwise store str(1.0)="1.0" while the apply side queries "1" -- every
    # lookup then falls to the global fallback silently. Canonicalising IN the builder removes that footgun (EN-1).
    from .._internals import canonical_group_token
    lookup_clean = {canonical_group_token(k): float(v) for k, v in lookup.items()}
    return EngineeredRecipe(
        name=name,
        kind="kfold_target_encoded",
        src_names=(src_name,),
        extra={
            "lookup": lookup_clean,
            "global_mean": float(global_mean),
            "smoothing": float(smoothing),
        },
    )


# ---------------------------------------------------------------------------
# Layer 34 (2026-05-31): count / frequency encoding + cat x num residual.
# ---------------------------------------------------------------------------
#
# Companions to ``kfold_target_encoded``: these three kinds cover the
# remaining production-grade categorical pipelines that Layer 33 does not
# touch. The kernels live in ``_count_freq_interaction_fe.py``.
#
# extra layout:
# * count_encoded     : {lookup: dict[str, int],   default: int}
# * frequency_encoded : {lookup: dict[str, float], default: float}
# * cat_num_residual  : {lookup: dict[str, float], global_mean: float,
#                        smoothing: float, num_col: str}
#   src_names is (cat_col, num_col) -- both extracted at replay.
#
# All three replays are STATELESS given X (no y reference, no fold info).


def _apply_count_encoded(recipe: EngineeredRecipe, X: Any, col_cache: "dict[str, np.ndarray] | None" = None) -> np.ndarray:
    """Replay count encoding via the stored per-category lookup. ``col_cache``: see ``_apply_kfold_target_encoded``."""
    if len(recipe.src_names) != 1:
        raise ValueError(f"count_encoded recipe '{recipe.name}' must have exactly 1 " f"src_names; got {len(recipe.src_names)}")
    if "lookup" not in recipe.extra:
        raise KeyError(f"count_encoded recipe '{recipe.name}' missing 'lookup' in extra.")
    from .._count_freq_interaction_fe import apply_count_encoding

    name = recipe.src_names[0]
    if pd is not None and isinstance(X, pd.DataFrame):
        X_view = X
    elif pl is not None and pd is not None and isinstance(X, pl.DataFrame):  # pd required: X_view below is a pandas frame
        X_view = pd.DataFrame({name: _extract_column(X, name, col_cache=col_cache)})
    elif isinstance(X, np.ndarray) and X.dtype.names is not None:
        X_view = pd.DataFrame({name: _extract_column(X, name, col_cache=col_cache)})
    else:
        raise TypeError(f"count_encoded recipe '{recipe.name}': cannot extract column " f"{name!r} from X of type {type(X).__name__}.")
    return apply_count_encoding(
        X_view, name, {
            "lookup": dict(recipe.extra["lookup"]),
            "default": int(recipe.extra.get("default", 0)),
        },
    )


def _apply_frequency_encoded(recipe: EngineeredRecipe, X: Any, col_cache: "dict[str, np.ndarray] | None" = None) -> np.ndarray:
    """Replay frequency encoding via the stored per-category lookup. ``col_cache``: see ``_apply_kfold_target_encoded``."""
    if len(recipe.src_names) != 1:
        raise ValueError(f"frequency_encoded recipe '{recipe.name}' must have exactly 1 " f"src_names; got {len(recipe.src_names)}")
    if "lookup" not in recipe.extra:
        raise KeyError(f"frequency_encoded recipe '{recipe.name}' missing 'lookup' in extra.")
    from .._count_freq_interaction_fe import apply_frequency_encoding

    name = recipe.src_names[0]
    if pd is not None and isinstance(X, pd.DataFrame):
        X_view = X
    elif pl is not None and pd is not None and isinstance(X, pl.DataFrame):  # pd required: X_view below is a pandas frame
        X_view = pd.DataFrame({name: _extract_column(X, name, col_cache=col_cache)})
    elif isinstance(X, np.ndarray) and X.dtype.names is not None:
        X_view = pd.DataFrame({name: _extract_column(X, name, col_cache=col_cache)})
    else:
        raise TypeError(f"frequency_encoded recipe '{recipe.name}': cannot extract column " f"{name!r} from X of type {type(X).__name__}.")
    return apply_frequency_encoding(
        X_view, name, {
            "lookup": dict(recipe.extra["lookup"]),
            "default": float(recipe.extra.get("default", 0.0)),
        },
    )


def _apply_cat_num_residual(recipe: EngineeredRecipe, X: Any, col_cache: "dict[str, np.ndarray] | None" = None) -> np.ndarray:
    """Replay cat x num residual: ``X[num] - lookup.get(X[cat], global_mean)``. ``col_cache``: see
    ``_apply_kfold_target_encoded``."""
    if len(recipe.src_names) != 2:
        raise ValueError(f"cat_num_residual recipe '{recipe.name}' must have exactly 2 " f"src_names (cat_col, num_col); got {len(recipe.src_names)}")
    for key in ("lookup", "global_mean"):
        if key not in recipe.extra:
            raise KeyError(f"cat_num_residual recipe '{recipe.name}' missing {key!r} in extra.")
    from .._count_freq_interaction_fe import apply_cat_num_residual

    cat_name, num_name = recipe.src_names
    if pd is not None and isinstance(X, pd.DataFrame):
        X_view = X
    elif pl is not None and pd is not None and isinstance(X, pl.DataFrame):  # pd required: X_view below is a pandas frame
        X_view = pd.DataFrame({
            cat_name: _extract_column(X, cat_name, col_cache=col_cache),
            num_name: _extract_column(X, num_name, col_cache=col_cache),
        })
    elif isinstance(X, np.ndarray) and X.dtype.names is not None:
        X_view = pd.DataFrame({
            cat_name: _extract_column(X, cat_name, col_cache=col_cache),
            num_name: _extract_column(X, num_name, col_cache=col_cache),
        })
    else:
        raise TypeError(f"cat_num_residual recipe '{recipe.name}': cannot extract columns " f"from X of type {type(X).__name__}.")
    return apply_cat_num_residual(
        X_view, cat_name, num_name, {
            "lookup": dict(recipe.extra["lookup"]),
            "global_mean": float(recipe.extra["global_mean"]),
        },
    )


def build_count_encoded_recipe(
    *, name: str, src_name: str, lookup: dict, default: int = 0,
) -> EngineeredRecipe:
    """Frozen recipe for a count-encoded column. ``lookup`` maps each
    fit-time category (as str) to its observed count; unseen categories at
    replay map to ``default`` (default 0). No y reference at any stage."""
    from . import EngineeredRecipe
    lookup_clean = {str(k): int(v) for k, v in lookup.items()}
    return EngineeredRecipe(
        name=name,
        kind="count_encoded",
        src_names=(src_name,),
        extra={
            "lookup": lookup_clean,
            "default": int(default),
        },
    )


def build_frequency_encoded_recipe(
    *, name: str, src_name: str, lookup: dict, default: float = 0.0,
) -> EngineeredRecipe:
    """Frozen recipe for a frequency-encoded column (count / n_samples).
    Unseen categories at replay map to ``default`` (default 0.0)."""
    from . import EngineeredRecipe
    lookup_clean = {str(k): float(v) for k, v in lookup.items()}
    return EngineeredRecipe(
        name=name,
        kind="frequency_encoded",
        src_names=(src_name,),
        extra={
            "lookup": lookup_clean,
            "default": float(default),
        },
    )


def build_cat_num_residual_recipe(
    *, name: str, cat_name: str, num_name: str, lookup: dict,
    global_mean: float, smoothing: float,
) -> EngineeredRecipe:
    """Frozen recipe for a cat x num residual column. ``lookup`` is the
    smoothed per-category mean of ``num_name``; replay subtracts
    ``lookup.get(cat, global_mean)`` from the row's num value. Unseen
    categories fall back to ``global_mean`` (subtracting the unconditional
    mean is the natural fallback)."""
    from . import EngineeredRecipe
    lookup_clean = {str(k): float(v) for k, v in lookup.items()}
    return EngineeredRecipe(
        name=name,
        kind="cat_num_residual",
        src_names=(cat_name, num_name),
        extra={
            "lookup": lookup_clean,
            "global_mean": float(global_mean),
            "smoothing": float(smoothing),
            "num_col": str(num_name),
        },
    )


# ---------------------------------------------------------------------------
# Layer 89 (2026-06-01): cat x cat synergy cross. The (cat_i, cat_j) value-pair
# -> code mapping is stored as a list-of-(key, value) pairs so the frozen
# recipe's array-aware __eq__ / pickle round-trip handle the tuple keys cleanly
# (tuple-keyed dicts aren't JSON-friendly; a flat pair list is). ``encoding`` is
# "raw" (emit the integer cell code) or "target" (emit per-cell OOF mean-of-y
# from ``te_lookup``). The apply helper lives in ``_cat_pair_fe.py``.
# ---------------------------------------------------------------------------


def build_cat_pair_cross_recipe(
    *, name: str, cat_i: str, cat_j: str, mapping: dict,
    encoding: str = "raw", te_lookup: dict | None = None,
    global_mean: float = 0.0,
) -> EngineeredRecipe:
    """Frozen recipe for one cat x cat synergy cross.

    ``mapping`` maps each fit-time (str_i, str_j) value pair to its dense int
    cell code. ``encoding='raw'`` emits the cell code (unseen pairs -> sentinel
    bin); ``encoding='target'`` emits the per-cell smoothed mean-of-y from
    ``te_lookup`` (unseen pairs / codes -> ``global_mean``). No y reference is
    consumed at replay -- ``transform()`` is a pure function of X."""
    from . import EngineeredRecipe
    from .._internals import canonical_group_token
    # Canonicalise each key element (int<->float drift safe, idempotent on the already-canonical strings the
    # generator passes) so a future raw-value caller cannot desync the build keys from the canonical apply-side
    # lookup and silently route every row to the global fallback (EN-1).
    mapping_pairs = [[[canonical_group_token(a) for a in k], int(v)] for k, v in mapping.items()]
    extra: dict = {
        "mapping": mapping_pairs,
        "encoding": str(encoding),
    }
    if encoding == "target":
        extra["te_lookup"] = [[int(k), float(v)] for k, v in (te_lookup or {}).items()]
        extra["global_mean"] = float(global_mean)
    return EngineeredRecipe(
        name=name,
        kind="cat_pair_cross",
        src_names=(str(cat_i), str(cat_j)),
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Layer 94 (2026-06-01): cat x cat x cat TRIPLE synergy cross. Mirrors the
# Layer 89 pair recipe -- the value-TRIPLE -> code mapping is stored as a list-
# of-(key, value) pairs (tuple-keyed dicts aren't JSON-friendly; a flat pair
# list round-trips cleanly through the frozen recipe's array-aware __eq__ /
# pickle). ``encoding`` is "raw" (emit the integer cell code) or "target" (emit
# per-cell OOF mean-of-y from ``te_lookup``). The apply helper lives in
# ``_cat_triple_fe.py``.
# ---------------------------------------------------------------------------


def build_cat_triple_cross_recipe(
    *, name: str, cat_a: str, cat_b: str, cat_c: str, mapping: dict,
    encoding: str = "raw", te_lookup: dict | None = None,
    global_mean: float = 0.0,
) -> EngineeredRecipe:
    """Frozen recipe for one cat x cat x cat synergy cross.

    ``mapping`` maps each fit-time (str_a, str_b, str_c) value triple to its
    dense int cell code. ``encoding='raw'`` emits the cell code (unseen triples
    -> sentinel bin); ``encoding='target'`` emits the per-cell smoothed mean-of-y
    from ``te_lookup`` (unseen triples / codes -> ``global_mean``). No y
    reference is consumed at replay -- ``transform()`` is a pure function of X."""
    from . import EngineeredRecipe
    from .._internals import canonical_group_token
    # Canonicalise each key element (int<->float drift safe, idempotent on the already-canonical strings the
    # generator passes) so a future raw-value caller cannot desync the build keys from the canonical apply-side
    # lookup and silently route every row to the global fallback (EN-1).
    mapping_pairs = [[[canonical_group_token(a) for a in k], int(v)] for k, v in mapping.items()]
    extra: dict = {
        "mapping": mapping_pairs,
        "encoding": str(encoding),
    }
    if encoding == "target":
        extra["te_lookup"] = [[int(k), float(v)] for k, v in (te_lookup or {}).items()]
        extra["global_mean"] = float(global_mean)
    return EngineeredRecipe(
        name=name,
        kind="cat_triple_cross",
        src_names=(str(cat_a), str(cat_b), str(cat_c)),
        extra=extra,
    )
