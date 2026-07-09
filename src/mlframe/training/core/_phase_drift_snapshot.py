"""Pre-train cardinality + val/test category-drift logging, carved from _phase_helpers.

Pure side-effect diagnostics: surfaces categorical cardinality and warns on val/test categories
absent from train (XGB/CB val-DMatrix crash hazard). Re-exported from _phase_helpers for callers.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

_DRIFT_SKIP_CARD = 100_000
_DRIFT_MIN_ABS = 5
_DRIFT_MIN_FRAC = 0.05


def _log_cardinality_and_drift_snapshot(
    *,
    train_df: pl.DataFrame | pd.DataFrame | None,
    val_df: pl.DataFrame | pd.DataFrame | None,
    test_df: pl.DataFrame | pd.DataFrame | None,
    cat_features: list[str],
    text_features: list[str],
    embedding_features: list[str],
    ctx: Any | None = None,
) -> None:
    """Pre-train cardinality + val/test drift logging (pure side-effect).

    Cardinality surfaces the input shape before any native XGB/CB crash on high-cardinality
    categoricals. Drift detection: XGB 3.x on Windows can crash silently during val
    IterativeDMatrix construction when val/test contain categories absent from train; we emit
    a WARNING with a healing suggestion keyed on train-side cardinality. Columns with
    cardinality > 100k (free-text) are skipped.
    """
    all_cat_cols = list(cat_features or []) + list(text_features or []) + list(embedding_features or [])
    if not (all_cat_cols and train_df is not None):
        return
    try:
        is_polars = isinstance(train_df, pl.DataFrame)
        # Single lazy collect for ALL train cardinalities (was: N eager n_unique() calls -> N kernel launches).
        # Reference: helpers.py:1040-1047 -- the same implode-batch pattern that collapsed 14 collects to 1
        # in trainset_features_stats. Here on 100 cat cols this drops ~100 eager kernels to 1 collect.
        cols_present = [c for c in all_cat_cols if c in train_df.columns]
        if is_polars:
            if cols_present:
                _card_row = train_df.lazy().select([pl.col(c).n_unique().alias(c) for c in cols_present]).collect()
                pairs = [(c, int(_card_row[c][0])) for c in cols_present]
            else:
                pairs = []
        else:
            pairs = [(c, int(train_df[c].nunique(dropna=False))) for c in cols_present]  # type: ignore[union-attr]  # is_polars bool flag already excludes the pl.DataFrame arm here
        pairs.sort(key=lambda x: -x[1])
        summary = ", ".join(f"{c}:{n:_}" for c, n in pairs)
        logger.info("  Categorical cardinalities (train, n_unique, desc): %s", summary)

        # Drift log: val/test categories not seen in train.
        if is_polars and val_df is not None and test_df is not None and val_df.height > 0:
            # Per-col anti-join was 3 selects + 2 joins = ~5 eager passes; on 100 cols that's ~500 passes ~10-30 s.
            # Batched implode pattern: one lazy collect per frame yielding a 1-row frame whose cells are the
            # imploded unique-value lists. Anti-set is then a pure-Python set-difference on the materialised lists.
            drift_cols = [c for c, card in pairs if card <= _DRIFT_SKIP_CARD and c in val_df.columns and c in test_df.columns]
            drift_rows: list = []
            if drift_cols:
                # CAT-DRIFT-FULL-IMPLODE: cache the train-side ``unique().implode()`` result on
                # ctx keyed by (id(train_df), drift_cols_tuple). The drift snapshot is invoked up
                # to three times per suite (pre-split / post-split / pre-fit) on the same train
                # frame, each time paying a full lazy collect over hundreds of columns. Recompute
                # only when ctx is absent OR the train frame identity changes; val/test sides are
                # not cached because they're cheap relative to train and may rotate per pass.
                _cache_key = (id(train_df), tuple(drift_cols))
                _drift_cache = getattr(ctx, "_cat_drift_implode_cache", None) if ctx is not None else None
                _tr_sets_cached = _drift_cache.get(_cache_key) if isinstance(_drift_cache, dict) else None
                if _tr_sets_cached is None:
                    _tr_uniq = train_df.lazy().select([pl.col(c).drop_nulls().unique().implode().alias(c) for c in drift_cols]).collect()
                    # Materialise to per-col sets ONCE; the previous code recomputed the
                    # ``set(_tr_uniq[c][0].to_list())`` per column inside the row loop.
                    _tr_sets_cached = {c: set(_tr_uniq[c][0].to_list()) for c in drift_cols}
                    if isinstance(_drift_cache, dict):
                        _drift_cache[_cache_key] = _tr_sets_cached
                _v_uniq = val_df.lazy().select([pl.col(c).drop_nulls().unique().implode().alias(c) for c in drift_cols]).collect()
                _te_uniq = test_df.lazy().select([pl.col(c).drop_nulls().unique().implode().alias(c) for c in drift_cols]).collect()
                _card_by_col = dict(pairs)
                for c in drift_cols:
                    tr_set = _tr_sets_cached[c]
                    val_only = sum(1 for x in _v_uniq[c][0].to_list() if x not in tr_set)
                    test_only = sum(1 for x in _te_uniq[c][0].to_list() if x not in tr_set)
                    drift_rows.append((c, _card_by_col[c], val_only, test_only))

            if drift_rows:
                drift_rows.sort(key=lambda x: -x[2])
                drift_summary = ", ".join(f"{c}:val_only={v},test_only={t}" for c, _, v, t in drift_rows if v > 0 or t > 0) or "(none)"
                logger.info("  Category drift (val/test values missing from train): %s", drift_summary)

                # Test-side drift is reported above but NOT used in healing decisions
                # (would leak test info into training).
                for c, card_tr, v_only, t_only in drift_rows:
                    if v_only == 0 and t_only == 0:
                        continue
                    v_frac = v_only / max(card_tr, 1)
                    if v_only >= _DRIFT_MIN_ABS or v_frac >= _DRIFT_MIN_FRAC:
                        if card_tr >= 1000:
                            _healing = (
                                f"        suggested actions (pick one):\n"
                                f"          a) hash-bucket via FeatureHasher / target-encoding "
                                f"(card {card_tr:_} >= 1 000 -> model will memorize train-only "
                                f"values and generalize poorly on val/test);\n"
                                f"          b) drop '{c}' from cat_features and keep only the "
                                f"top-K most frequent (K=100-300) as one-hot, route the rest "
                                f"into an '__OTHER__' bucket;\n"
                                f"          c) drop '{c}' entirely if it's an identifier or "
                                f"free-text field -- promote to text_features via use_text_features=True "
                                f"so CatBoost handles it natively and other backends ignore it."
                            )
                        elif card_tr >= 100:
                            _healing = (
                                f"        suggested actions (pick one):\n"
                                f"          a) target-encoding (CatBoostEncoder) to collapse "
                                f"{card_tr:_} levels into a continuous feature;\n"
                                f"          b) keep top-K by train frequency, bucket the rest "
                                f"into '__OTHER__' before fit (K~=30-80)."
                            )
                        else:
                            _healing = (
                                "        suggested actions (pick one):\n"
                                "          a) add an explicit '__UNSEEN__' bucket in the "
                                "Enum domain so val values absent from train resolve to a "
                                "known category instead of raising;\n"
                                "          b) widen the training window (temporal split) so "
                                "val_only categories are observed at fit time."
                            )
                        logger.warning(
                            "  Category drift suspect: %s -- val has %s categories "
                            "(%s of train card %s) that train never saw. "
                            "XGB/CB may crash when constructing val DMatrix with ref=train.\n"
                            "%s",
                            c,
                            v_only,
                            f"{v_frac:.1%}",
                            f"{card_tr:_}",
                            _healing,
                        )
    except Exception as _e:
        logger.warning("  Failed to compute categorical cardinality/drift: %s", _e)
