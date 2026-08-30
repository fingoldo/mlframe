"""Find, BEFORE the real fit, which text columns CatBoost cannot build a dictionary from.

CatBoost's text estimator raises ``Dictionary size is 0`` when a column's tokens all get filtered out of its
vocabulary, and the raise kills the whole fit -- not just that column. A production run lost a 40-second fit
this way and then silently retried with NO text features at all, so five deliberately promoted columns
stopped contributing.

The condition is NOT "too few non-null samples", which is what the old failure message asserted. Measured
against CatBoost 1.2.10:

===========================================  ========
column shape                                 result
===========================================  ========
1 token per row, each token repeats 300x     raises
2 tokens per row, each token repeats 24x     fits
2 tokens per row, each token repeats 2x      raises
3 tokens per row, each token repeats 3x      fits
1 token present in EVERY row + unique rest   raises
400 non-null rows, 400 distinct tokens       raises
===========================================  ========

So it depends on the token structure (how many tokens a row carries, and how their document frequencies land
against CatBoost's own vocabulary filters) in a way no simple row-count rule predicts -- and the exact filter
is CatBoost's internal business, free to change between versions. Rather than re-derive it, this probes the
installed CatBoost directly: a 1-iteration fit per candidate column, which either builds the dictionary or
raises exactly as the real fit would.

Probing is cheap against what it prevents: a couple of seconds per column versus a failed multi-minute fit
plus the silent loss of every text feature.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Rows the probe fits on. A sample can only SHRINK a token's document frequency, so a column that builds a
# dictionary on the sample builds one on the full data too -- the probe errs toward keeping a column, never
# toward passing one that then crashes the real fit.
PROBE_MAX_ROWS: int = 200_000
# The marker CatBoost uses for this failure. Any other error means the probe itself is broken (a bad frame, a
# missing dependency), which must not be read as "this column is unusable".
_EMPTY_DICT_MARKER: str = "Dictionary size is 0"


def _probe_one(frame: Any, y: np.ndarray, column: str) -> Tuple[bool, str]:
    """``(usable, reason)`` for one text column: can the installed CatBoost build a dictionary from it?"""
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        return True, f"catboost import failed ({exc}); probe skipped"
    try:
        CatBoostClassifier(iterations=1, depth=1, verbose=0, allow_writing_files=False).fit(frame, y, text_features=[column])
        return True, ""
    except Exception as exc:
        message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        logger.debug("text probe raised (%s: %s)", type(exc).__name__, message[:160])
        if _EMPTY_DICT_MARKER in str(exc):
            return False, "CatBoost cannot build a token dictionary from this column"
        if "Invalid type for text_feature" in str(exc):
            return False, "column holds nulls, which CatBoost rejects in a text feature"
        # Anything else is about the probe, not the column: keep the column and say what happened.
        return True, f"probe inconclusive ({type(exc).__name__}: {message[:120]})"


def _sample_rows(n_rows: int, seed: int = 0) -> Optional[np.ndarray]:
    """Row indices for the probe, or None when the frame is small enough to use whole."""
    if n_rows <= PROBE_MAX_ROWS:
        return None
    return np.random.default_rng(seed).choice(n_rows, size=PROBE_MAX_ROWS, replace=False)


def unusable_text_features(
    df: Any,
    y: Any,
    text_features: Sequence[str],
    *,
    verbose: bool = True,
) -> Dict[str, str]:
    """``{column: reason}`` for every text column the installed CatBoost would raise on.

    Each candidate is probed ALONE, because CatBoost reports the failure for the fit as a whole and names no
    column -- probing them together would only reproduce the original ambiguity that made the old handler drop
    all five columns when one was at fault.
    """
    cols = [c for c in (text_features or []) if c]
    if not cols:
        return {}
    try:
        import pandas as pd
    except ImportError:
        return {}

    y_arr = np.asarray(y).ravel()
    n_rows = int(y_arr.shape[0])
    if n_rows < 2:
        return {}
    idx = _sample_rows(n_rows)

    bad: Dict[str, str] = {}
    for col in cols:
        try:
            values = _column_values(df, col)
        except Exception as exc:
            logger.debug("text probe: could not read column %r (%s); leaving it in place", col, exc)
            continue
        if values is None:
            continue
        y_probe = y_arr if idx is None else y_arr[idx]
        vals = values if idx is None else values[idx]
        # A one-class slice makes CatBoost raise for a reason that has nothing to do with the text column.
        if np.unique(y_probe).size < 2:
            continue
        probe_frame = pd.DataFrame({"__probe_num__": np.arange(len(y_probe), dtype=np.float64), col: vals})
        usable, reason = _probe_one(probe_frame, y_probe, col)
        if not usable:
            bad[col] = reason
        elif reason and verbose:
            logger.debug("text probe on %r: %s", col, reason)

    if bad and verbose:
        logger.warning(
            "  CatBoost cannot use %d of %d text feature(s); dropping only these and keeping the rest: %s. "
            "Probed with a 1-iteration fit per column against the installed CatBoost, so this reflects what "
            "the real fit would do rather than a guess about token counts.",
            len(bad), len(cols), "; ".join(f"{c} ({r})" for c, r in bad.items()),
        )
    return bad


def _column_values(df: Any, column: str) -> Optional[np.ndarray]:
    """One column as an object ndarray of python strings, or None when the frame does not carry it."""
    cols = getattr(df, "columns", None)
    if cols is not None and column not in list(cols):
        return None
    series = df[column]
    to_numpy = getattr(series, "to_numpy", None)
    values = to_numpy() if callable(to_numpy) else np.asarray(series)
    # CatBoost rejects None in a text feature outright, so a null becomes the empty string here exactly as the
    # fit-time path does -- otherwise the probe would fail every column that merely has gaps.
    return np.array(["" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in values], dtype=object)


def usable_text_features(df: Any, y: Any, text_features: Sequence[str], *, verbose: bool = True) -> List[str]:
    """``text_features`` minus the columns the installed CatBoost would raise on, order preserved."""
    bad = unusable_text_features(df, y, text_features, verbose=verbose)
    return [c for c in (text_features or []) if c not in bad]


def unigram_text_processing() -> dict:
    """CatBoost ``text_processing`` that tokenizes into UNIGRAMS instead of the default bigrams.

    This is the actual remedy for ``Dictionary size is 0``, and it recovers the columns rather than discarding
    them. CatBoost's default text pipeline builds a word-BIGRAM dictionary, so a column whose rows hold a
    single token (a country code, a tag, a language, an id-like blob) can never contribute one bigram and its
    dictionary comes out empty -- which aborts the whole fit, not just that column.

    Measured against CatBoost 1.2.10 on a column of one token per row, each token present in ~120 of 600 rows:
    the default processing raises, the same data under a unigram dictionary fits. On a frame mixing that column
    with a three-token-per-row one, unigrams serve both; declaring unigrams AND bigrams together does not,
    because the bigram dictionary is still empty for the single-token column and any empty dictionary aborts
    the fit.

    Unigrams are a strictly weaker representation for genuinely multi-word text, so this is applied as a
    RESCUE after the default has already failed -- never preemptively.
    """
    return {
        "tokenizers": [{"tokenizer_id": "Space", "delimiter": " "}],
        "dictionaries": [{"dictionary_id": "Unigram", "gram_order": "1"}],
        "feature_processing": {
            "default": [
                {
                    "dictionaries_names": ["Unigram"],
                    "feature_calcers": ["BoW", "NaiveBayes"],
                    "tokenizers_names": ["Space"],
                }
            ]
        },
    }


def unigram_rescues_text_features(df, y, text_features, *, verbose: bool = True) -> bool:
    """Would a unigram dictionary let the installed CatBoost fit ALL of ``text_features`` together?

    Probes the real fit rather than reasoning about token counts, for the same reason the per-column probe
    does: the vocabulary filter is CatBoost's own business and free to change between versions.
    """
    cols = [c for c in (text_features or []) if c]
    if not cols:
        return False
    try:
        import pandas as pd
        from catboost import CatBoostClassifier
    except ImportError:
        return False

    y_arr = np.asarray(y).ravel()
    n_rows = int(y_arr.shape[0])
    if n_rows < 2:
        return False
    idx = _sample_rows(n_rows)
    y_probe = y_arr if idx is None else y_arr[idx]
    if np.unique(y_probe).size < 2:
        return False

    data = {"__probe_num__": np.arange(len(y_probe), dtype=np.float64)}
    for col in cols:
        values = _column_values(df, col)
        if values is None:
            return False
        data[col] = values if idx is None else values[idx]
    try:
        CatBoostClassifier(iterations=1, depth=1, verbose=0, allow_writing_files=False, text_processing=unigram_text_processing()).fit(
            pd.DataFrame(data), y_probe, text_features=cols
        )
    except Exception as exc:
        logger.debug("unigram rescue probe failed (%s: %s)", type(exc).__name__, str(exc).splitlines()[0][:120])
        return False
    return True


__all__ = [
    "PROBE_MAX_ROWS",
    "unigram_rescues_text_features",
    "unigram_text_processing",
    "unusable_text_features",
    "usable_text_features",
]
