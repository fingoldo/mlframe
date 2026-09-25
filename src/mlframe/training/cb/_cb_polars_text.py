"""CatBoost text features must reach it as strings when the frame is polars.

CatBoost 1.2.10 corrupts memory when a polars ``Categorical`` / ``Enum`` column is declared as a TEXT feature: building
a ``Pool`` from such a frame took down 6 of 6 fresh processes within 60 constructions, while the same column cast to
``String`` survived 6 of 6, as did a frame with no text feature. The pandas path already decategorised text columns
before a Pool; the polars path handed them over unchanged, and mlframe's own CatBoost fallback tests died with an access
violation inside ``Pool.__init__`` in about one run in four.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

_CATEGORICAL_DTYPE_NAMES = ("Categorical", "Enum")


def text_columns_as_strings(df: Any, text_features: Optional[Iterable[Any]]) -> Any:
    """``df`` with every polars Categorical / Enum column named in ``text_features`` cast to ``String``; otherwise ``df``.

    A no-op for pandas / numpy input, for an empty ``text_features`` and when no named column is categorical, so the
    common path returns the very same object and copies nothing.
    """
    if not text_features or not hasattr(df, "schema") or not hasattr(df, "with_columns"):
        return df
    names = {str(c) for c in text_features}
    to_cast = [c for c, dtype in df.schema.items() if c in names and type(dtype).__name__ in _CATEGORICAL_DTYPE_NAMES]
    if not to_cast:
        return df
    import polars as pl

    return df.with_columns([pl.col(c).cast(pl.String) for c in to_cast])


def model_text_feature_names(model: Any) -> list[str]:
    """Names of the text features a fitted CatBoost model expects; empty when it has none or cannot say."""
    try:
        indices = list(model.get_text_feature_indices() or [])
        names = list(getattr(model, "feature_names_", None) or [])
    except Exception:  # not a fitted CatBoost model: nothing to protect
        return []
    return [names[i] for i in indices if 0 <= i < len(names)]


def cb_text_features_as_strings(model_type_name: str, train_df: Any, fit_params: dict) -> Any:
    """For a CatBoost fit: ``train_df`` and every ``eval_set`` frame with their text-feature columns as strings.

    Applied once before the fit branches, so the Pool-reuse path, the val Pool and the plain ``model.fit`` fallback (the
    one a CatBoost build without ``Pool.set_label`` always takes) all receive safe input. ``fit_params["eval_set"]`` is
    rewritten in place only when a frame actually changed.
    """
    from mlframe.config import CATBOOST_MODEL_TYPES

    text = fit_params.get("text_features")
    if model_type_name not in CATBOOST_MODEL_TYPES or not text:
        return train_df
    eval_set = fit_params.get("eval_set")
    if isinstance(eval_set, (list, tuple)) and eval_set:
        single = isinstance(eval_set, tuple) and len(eval_set) == 2 and not isinstance(eval_set[0], tuple)
        entries = [eval_set] if single else list(eval_set)
        cast = [(text_columns_as_strings(e[0], text), e[1]) if isinstance(e, tuple) and len(e) == 2 else e for e in entries]
        if any(isinstance(e, tuple) and c[0] is not e[0] for c, e in zip(cast, entries)):
            fit_params["eval_set"] = cast[0] if single else cast
    return text_columns_as_strings(train_df, text)
