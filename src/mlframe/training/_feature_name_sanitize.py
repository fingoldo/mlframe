"""Make model-facing feature names compatible with the GBM backends.

LightGBM rejects feature names that contain JSON-structural characters
(``,`` ``[`` ``]`` ``{`` ``}`` ``:`` ``"``) with the error *"Do not support
special JSON characters in feature name"*; XGBoost separately rejects ``[``
``]`` ``<``. Engineered interaction features are named like
``mul(log(f2),sin(f3))`` -- a parseable internal contract that the MRMR-FE
machinery splits on ``(`` to recover structure, so the *generation* format
must not change -- but the embedded comma trips LightGBM the moment such a
frame reaches a booster.

The fix renames only the *model-facing column labels* (the DataFrame handed to
``fit``/``predict``) through a pure, deterministic, idempotent map: the same
hostile name always maps to the same safe name, so the train frame and the
test frame -- transformed at different points by the same fitted pipeline --
map identically with no stored state, keeping fit and predict consistent. The
internal engineered name used for provenance / recipe replay is untouched.

Only the hostile characters are remapped; ``(`` and ``)`` are JSON-safe and
GBM-safe, so they are kept, which preserves the structural uniqueness of
engineered names (``mul(log(f2),sin(f3))`` -> ``mul(log(f2)_sin(f3))``) and
makes collisions effectively impossible.

The entry points are strict no-ops when no column name is hostile (the
overwhelmingly common case): a single membership scan and the original object
is returned unchanged.
"""
from __future__ import annotations

from typing import Optional, Sequence

# JSON-structural characters rejected by LightGBM plus XGBoost's ``< > [ ]``.
_HOSTILE_CHARS = ',[]{}":<>'
_TRANS = {ord(c): "_" for c in _HOSTILE_CHARS}


def _is_hostile(name) -> bool:
    """Whether ``name`` contains any GBM-hostile character (JSON structural chars rejected by LightGBM plus XGBoost's ``< > [ ]``)."""
    s = name if isinstance(name, str) else str(name)
    return any(ch in s for ch in _HOSTILE_CHARS)


def safe_feature_name(name) -> str:
    """Map one feature name to a GBM-compatible form (pure + idempotent)."""
    return (name if isinstance(name, str) else str(name)).translate(_TRANS)


def has_hostile_name(columns) -> bool:
    """True iff any column label contains a GBM-hostile character."""
    return any(_is_hostile(c) for c in columns)


def build_safe_mapping(columns) -> dict:
    """``{original: safe}`` for hostile columns only, deduping any safe-name
    collision against the full set of kept + remapped names. Empty dict when
    every name is already clean."""
    cols = list(columns)
    used = set()
    hostile = []
    for c in cols:
        if _is_hostile(c):
            hostile.append(c)
        else:
            used.add(str(c))
    mapping: dict = {}
    for c in hostile:
        base = safe_feature_name(c)
        cand = base
        i = 1
        while cand in used:
            cand = f"{base}_{i}"
            i += 1
        mapping[c] = cand
        used.add(cand)
    return mapping


def sanitize_frame_columns(df):
    """Return ``df`` with GBM-hostile column labels renamed (pandas or polars).

    Strict no-op -- returns the *same* object -- when no column is hostile.
    """
    if df is None:
        return df
    cols = getattr(df, "columns", None)
    if cols is None:
        return df
    cols_list = list(cols)
    if not has_hostile_name(cols_list):
        return df
    mapping = build_safe_mapping(cols_list)
    if not mapping:
        return df
    # pandas: ``rename(columns=...)``; polars: ``rename(dict)``.
    try:
        import pandas as _pd

        if isinstance(df, _pd.DataFrame):
            return df.rename(columns=mapping)
    except Exception:  # nosec B110 - optional dependency import guard
        pass
    try:
        return df.rename(mapping)  # polars.DataFrame
    except Exception:
        try:
            df.columns = [mapping.get(c, mapping.get(str(c), c)) for c in cols_list]
        except Exception:  # nosec B110 - best-effort path
            pass
        return df


def sanitize_name_list(names: Optional[Sequence], full_columns: Optional[Sequence] = None) -> Optional[Sequence]:
    """Remap a list of feature names (e.g. ``cat_features``) through the same
    pure map. Non-string entries (column indices) pass through unchanged.
    Returns the input unchanged when nothing is hostile.

    ``full_columns``, when given, should be the SAME column sequence passed to
    ``sanitize_frame_columns``/``build_safe_mapping`` for the associated frame --
    the collision-safe mapping is then built against that full universe so a
    hostile name here maps to EXACTLY the same safe name the frame itself was
    renamed to (a name list sanitised independently, without this shared
    universe, could otherwise dedupe differently than the frame and end up
    referencing a column that doesn't exist, or the wrong one). Without
    ``full_columns`` this at least dedupes within ``names`` itself -- better
    than no collision tracking at all, but not guaranteed frame-consistent.
    """
    if not names:
        return names
    if not any(isinstance(n, str) and _is_hostile(n) for n in names):
        return names
    mapping = build_safe_mapping(full_columns if full_columns is not None else names)
    return [mapping.get(n, safe_feature_name(n)) if isinstance(n, str) else n for n in names]
