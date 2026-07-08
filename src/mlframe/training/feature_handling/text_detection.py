"""
Multi-criteria text-vs-categorical detector with anti-UUID guards.

Round-3 audits A10 + R2-21: the previous single-trigger
``cat_text_cardinality_threshold > 300`` mis-classifies UUID / hash
ID columns (UUID-v4 entropy ≈ 4.04 sat right at the prior 4.0
threshold). The new multi-criteria heuristic + anti-UUID guard
(min entropy 4.5, min mean tokens 2.0) catches text correctly AND
keeps ID columns out of TF-IDF.

Triggers (any -> text):
  1. ``mean_chars >= 100`` -- definitely-long text
  2. ``mean_chars >= 30`` AND ``mean_tokens >= 4`` -- medium with
     enough tokens
  3. ``unique_ratio >= 0.95`` AND ``mean_chars >= 15`` -- high-unique
     and meaningful length
  4. ``cardinality > 300`` AND ``mean_chars >= 30`` -- high-cardinality
     legacy trigger (still matches the original pre-2026 heuristic
     when the column has substance)

Anti-UUID guards (BOTH must hold for text classification):
  * ``alphabet_entropy >= 4.5`` -- excludes hex (4.0), base32 (5.0
    only if uniform), narrow alphabets
  * ``mean_tokens >= 2.0`` -- IDs have no spaces

User overrides:
  * ``explicit_text_columns`` -- always text, bypasses heuristic.
  * ``explicit_categorical_columns`` -- always categorical.
  * ``skip_columns`` -- excluded from analysis entirely.

The detector returns BOTH the text-column list AND a per-column
decision trace so ``fhc.describe()`` can surface "why was this
classified text".
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np

from mlframe.training.feature_handling.config import TextDetectionConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


# =====================================================================
# Decision dataclass
# =====================================================================


@dataclass
class TextDetectionDecision:
    """Per-column outcome for ``fhc.describe()`` introspection.

    ``rule_name`` is one of:
      * ``"explicit_categorical"`` -- user override
      * ``"explicit_text"`` -- user override
      * ``"skip_columns"`` -- user excluded
      * ``"anti_uuid_filter"`` -- failed entropy / token guard
      * ``"definite_long"`` -- trigger 1
      * ``"medium_with_tokens"`` -- trigger 2
      * ``"high_unique_ratio"`` -- trigger 3
      * ``"high_cardinality"`` -- trigger 4
      * ``"no_trigger_fired"`` -- categorical (default)
      * ``"non_string_dtype"`` -- not eligible
    """
    column: str
    rule_name: str
    decision: bool  # True = text, False = categorical
    stats: Dict[str, Any]


# =====================================================================
# Per-column statistics
# =====================================================================


def _shannon_entropy_of_chars(values: List[str]) -> float:
    """Average Shannon entropy over the alphabet of all characters
    used. UUID-v4 lowercase hex uses 16 symbols approximately
    uniform -> H ≈ 4.0; English text uses ~50 symbols non-uniform
    -> H ≈ 4.2-4.7 typically.

    Empty input returns 0.0.
    """
    char_counts: Dict[str, int] = {}
    total = 0
    for s in values:
        if s is None:
            continue
        for c in s:
            char_counts[c] = char_counts.get(c, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    h = 0.0
    for count in char_counts.values():
        p = count / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def _column_stats(values: List[str]) -> Dict[str, float]:
    """Compute the metrics the multi-criteria detector reads.

    NaN / None values excluded from char stats but counted toward
    cardinality / non_null totals correctly.
    """
    non_null_values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    non_null_count = len(non_null_values)
    n_unique = len(set(non_null_values))

    if non_null_count == 0:
        return {
            "mean_chars": 0.0,
            "mean_tokens": 0.0,
            "n_unique": 0,
            "non_null_count": 0,
            "unique_ratio": 0.0,
            "alphabet_entropy": 0.0,
        }

    char_counts = [len(s) for s in non_null_values if isinstance(s, str)]
    mean_chars = float(np.mean(char_counts)) if char_counts else 0.0

    token_counts = [len(s.split()) for s in non_null_values if isinstance(s, str)]
    mean_tokens = float(np.mean(token_counts)) if token_counts else 0.0

    string_values = [s for s in non_null_values if isinstance(s, str)]
    entropy = _shannon_entropy_of_chars(string_values)

    return {
        "mean_chars": mean_chars,
        "mean_tokens": mean_tokens,
        "n_unique": n_unique,
        "non_null_count": non_null_count,
        "unique_ratio": n_unique / max(1, non_null_count),
        "alphabet_entropy": entropy,
    }


def _column_to_string_list(df: Any, column: str, max_sample: int) -> List[str]:
    """Polars / pandas -> list[str | None], capped at ``max_sample``.
    Unknown dtypes return empty (caller treats as non-string).

    Sampling is via ``np.linspace`` rounded to int indices so the sampled rows are spread evenly
    across the FULL column. Pre-fix the integer-step formula (``step = max(1, n // max_sample)``
    then ``ser[::step][:max_sample]``) was either an exact stride OR silently truncated to the
    front of the column - e.g. ``n=25_000``, ``max_sample=10_000`` gave ``step=2`` and consumed
    only the first 20_000 rows, losing the trailing 20% of the column. For text vs UUID detection
    that systematically biased the entropy / token statistics on append-only frames where
    schema drift accumulates at the tail.
    """
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            ser = df[column]
            if ser.dtype not in (pl.Utf8, pl.String, pl.Categorical):
                # Could be Enum (a polars sub-type) -- try cast.
                try:
                    ser = ser.cast(pl.Utf8)
                except Exception as _e_cast:
                    # Pre-fix returned [] silently; the column then
                    # silently reclassified as non-text and CatBoost may
                    # later crash on long strings flowing through as
                    # cat / pass-through. Log so operators see the
                    # detection blind spot.
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "text_detection: cannot cast column %r (dtype=%s) "
                        "to Utf8 for sampling: %s. Returning empty sample "
                        "-> column will be treated as non-text downstream.",
                        column, ser.dtype, _e_cast,
                    )
                    return []
            n = len(ser)
            if n > max_sample:
                idx = np.linspace(0, n - 1, max_sample).round().astype(np.int64)
                idx = np.unique(idx)
                ser = ser[idx.tolist()]
            return ser.to_list()
    except ImportError:  # pragma: no cover
        pass

    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            ser = df[column]
            if not pd.api.types.is_string_dtype(ser) and not pd.api.types.is_object_dtype(ser):
                return []
            n = len(ser)
            if n > max_sample:
                idx = np.linspace(0, n - 1, max_sample).round().astype(np.int64)
                idx = np.unique(idx)
                ser = ser.iloc[idx]
            return list(ser.tolist())
    except ImportError:  # pragma: no cover
        pass

    return []


def _is_string_column(df: Any, column: str) -> bool:
    """Best-effort check whether a column is text-eligible."""
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            return df.schema[column] in (pl.Utf8, pl.String, pl.Categorical) or (hasattr(pl, "Enum") and isinstance(df.schema[column], pl.Enum))
    except ImportError:  # pragma: no cover
        pass
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            ser = df[column]
            return bool(pd.api.types.is_string_dtype(ser) or pd.api.types.is_object_dtype(ser))
    except ImportError:  # pragma: no cover
        pass
    return False


# =====================================================================
# Detector
# =====================================================================


def detect_text_columns(
    df: Any,
    *,
    candidate_columns: Optional[List[str]] = None,
    config: Optional[TextDetectionConfig] = None,
) -> Tuple[List[str], List[TextDetectionDecision]]:
    """Multi-criteria text detector.

    Returns ``(text_columns, decisions)`` where ``decisions`` is a
    list of :class:`TextDetectionDecision` -- one per evaluated
    column -- so ``fhc.describe()`` can surface the per-column
    rationale.

    ``candidate_columns`` defaults to all string-like columns in df.
    ``config`` defaults to :class:`TextDetectionConfig` defaults.
    """
    cfg = config or TextDetectionConfig()
    if candidate_columns is None:
        try:
            candidate_columns = list(df.columns)
        except Exception:  # pragma: no cover
            candidate_columns = []

    text_cols: List[str] = []
    decisions: List[TextDetectionDecision] = []

    for col in candidate_columns:
        # User overrides
        if col in cfg.skip_columns:
            decisions.append(TextDetectionDecision(
                column=col, rule_name="skip_columns", decision=False, stats={},
            ))
            continue
        if col in cfg.explicit_categorical_columns:
            decisions.append(TextDetectionDecision(
                column=col, rule_name="explicit_categorical", decision=False, stats={},
            ))
            continue
        if col in cfg.explicit_text_columns:
            text_cols.append(col)
            decisions.append(TextDetectionDecision(
                column=col, rule_name="explicit_text", decision=True, stats={},
            ))
            continue

        # Dtype guard: only string-like columns proceed.
        if not _is_string_column(df, col):
            decisions.append(TextDetectionDecision(
                column=col, rule_name="non_string_dtype", decision=False, stats={},
            ))
            continue

        # Round-3 user-confirmation (TextDetectionConfig.respect_explicit_categorical_dtype):
        # if column is already pl.Categorical / pl.Enum / pandas
        # category, the user has signalled intent -- treat as cat
        # regardless of cardinality. Heuristic only fires on raw
        # strings.
        if cfg.respect_explicit_categorical_dtype and _is_explicit_categorical(df, col):
            decisions.append(TextDetectionDecision(
                column=col, rule_name="explicit_categorical_dtype",
                decision=False, stats={},
            ))
            continue

        values = _column_to_string_list(df, col, cfg.sample_size_for_stats)
        stats = _column_stats(values)

        # Anti-UUID guard. UUIDs / hash IDs share TWO properties:
        # low alphabet entropy (narrow alphabet, often hex) AND no
        # whitespace (single-token strings). Real text usually fails
        # AT LEAST ONE -- has spaces OR rich alphabet. So we filter
        # only when BOTH ID-like signals fire.
        is_uuid_like = stats["alphabet_entropy"] < cfg.min_alphabet_entropy and stats["mean_tokens"] < cfg.min_mean_tokens_for_text
        if is_uuid_like:
            decisions.append(TextDetectionDecision(
                column=col, rule_name="anti_uuid_filter", decision=False, stats=stats,
            ))
            continue

        # Triggers (in order of specificity).
        if stats["mean_chars"] >= cfg.definite_text_mean_chars:
            text_cols.append(col)
            decisions.append(TextDetectionDecision(
                column=col, rule_name="definite_long", decision=True, stats=stats,
            ))
            continue

        if stats["mean_chars"] >= cfg.text_min_mean_chars and stats["mean_tokens"] >= cfg.text_min_mean_tokens:
            text_cols.append(col)
            decisions.append(TextDetectionDecision(
                column=col, rule_name="medium_with_tokens", decision=True, stats=stats,
            ))
            continue

        if stats["unique_ratio"] >= cfg.text_min_unique_ratio and stats["mean_chars"] >= 15:
            text_cols.append(col)
            decisions.append(TextDetectionDecision(
                column=col, rule_name="high_unique_ratio", decision=True, stats=stats,
            ))
            continue

        if stats["n_unique"] > cfg.text_min_cardinality and stats["mean_chars"] >= cfg.text_min_mean_chars:
            text_cols.append(col)
            decisions.append(TextDetectionDecision(
                column=col, rule_name="high_cardinality", decision=True, stats=stats,
            ))
            continue

        decisions.append(TextDetectionDecision(
            column=col, rule_name="no_trigger_fired", decision=False, stats=stats,
        ))

    return text_cols, decisions


def _is_explicit_categorical(df: Any, column: str) -> bool:
    """Is the column already cast to a categorical-flavoured dtype?"""
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            dtype = df.schema[column]
            if dtype == pl.Categorical:
                return True
            if hasattr(pl, "Enum") and isinstance(dtype, pl.Enum):
                return True
            return False
    except ImportError:  # pragma: no cover
        pass
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            return isinstance(df[column].dtype, pd.CategoricalDtype)
    except ImportError:  # pragma: no cover
        pass
    return False


def detect_cat_columns_by_dtype(
    df: Any,
    *,
    exclude_columns: Optional[List[str]] = None,
) -> List[str]:
    """Return cat-like columns by explicit dtype signal.

    Catches the columns a downstream cat handler (target_mean, WoE, m_estimate,
    james_stein, loo) should operate on by default:

    - **polars**: ``pl.Categorical``, ``pl.Enum``, and ``pl.String``/``pl.Utf8``
      (strings get the cat treatment unless ``detect_text_columns`` has claimed
      them; the caller passes the text-detected list via ``exclude_columns``).
    - **pandas**: ``CategoricalDtype`` and ``object``/``string`` dtypes (same
      string-vs-text exclusion rule via ``exclude_columns``).

    Counterpart to ``detect_text_columns`` so callers passing
    ``feature_handling_apply(candidate_cat_columns=None)`` get symmetric
    auto-detection instead of the pre-2026-05-20 "empty cat list silently drops
    every target-encoder handler" trap (docstring drift wave #4).

    Pure by-dtype: low-cardinality integer columns are NOT auto-promoted to cat
    here -- that path needs configurable thresholds + false-positive tests, kept
    as a separate follow-up. Caller can still pass an explicit list to override.

    Parameters
    ----------
    df : polars.DataFrame | pandas.DataFrame
        Input frame.
    exclude_columns : optional list
        Names to skip (typically the result of ``detect_text_columns`` so a
        text-promoted column doesn't also land in the cat list).

    Returns
    -------
    list[str]
        Column names whose dtype matches the categorical-like rule above,
        minus anything in ``exclude_columns``.
    """
    _excl = set(exclude_columns or [])
    out: List[str] = []
    # Polars path
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            _cat_dtypes = (pl.Categorical, pl.Utf8, pl.String)
            _enum_t = getattr(pl, "Enum", None)
            for _col, _dt in df.schema.items():
                if _col in _excl:
                    continue
                if isinstance(_dt, _cat_dtypes):
                    out.append(_col)
                    continue
                if _enum_t is not None and isinstance(_dt, _enum_t):
                    out.append(_col)
            return out
    except ImportError:  # pragma: no cover
        pass
    # Pandas path
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            for _col in df.columns:
                if _col in _excl:
                    continue
                _dt = df[_col].dtype
                if isinstance(_dt, pd.CategoricalDtype):
                    out.append(_col)
                    continue
                if pd.api.types.is_object_dtype(_dt) or pd.api.types.is_string_dtype(_dt):
                    out.append(_col)
            return out
    except ImportError:  # pragma: no cover
        pass
    return out


__all__ = [
    "TextDetectionDecision",
    "detect_text_columns",
    "detect_cat_columns_by_dtype",
]
