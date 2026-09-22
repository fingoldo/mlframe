"""Row-wise summary-stats / top-k extreme-columns step of ``apply_preprocessing_extensions``.

Carved out of ``_pipeline_extensions.py`` (over the 900-line house limit), the way the PySR step already was.
"""

from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("mlframe.training.pipeline._pipeline_extensions")


def apply_row_wise_steps(train, val, test, config, verbose: int, out_row_wise_replay: Optional[dict] = None):
    """Run the row-wise steps on train/val/test and return ``(train, val, test, n_columns_before)``.

    ``out_row_wise_replay``, when given, receives what predict needs to reproduce these columns exactly:
    ``"columns"``, the pinned numeric column list the steps were computed over (fixed here, after the numeric filter
    and the all-null drop - re-deriving it from the serving frame's dtypes brought back a column that was all-null on
    train and so never entered any statistic, shifting every row_summary_* value with no warning), and
    ``"extreme_columns_reference"``, the fitted extremality reference as lists (it travels through persisted metadata;
    predict rebuilds float64 arrays). Without the reference predict re-ranks within the serving batch, where a single
    row is its own median and every extremality score is 0.0.
    """
    from mlframe.training.core import _elapsed_str

    # Row-wise summary stats / top-k extreme columns (step 1.5). Purely additive, generic per-row
    # aggregates over the already-numeric column subset -- no dataset-specific column names or entity
    # IDs required, so both default ON (see PreprocessingExtensionsConfig docstring + DEFAULTS_CHANGELOG.md).
    # Computed on the SAME pinned numeric column list across train/val/test (``_rw_cols``, fixed from
    # train BEFORE either step adds its own new columns) so all three splits stay schema-aligned.
    _pre_row_wise_ncols = train.shape[1]
    if isinstance(train, pd.DataFrame) and train.shape[1] > 0 and (
        getattr(config, "row_wise_summary_stats_enabled", False) or getattr(config, "row_wise_extreme_columns_enabled", False)
    ):
        _rw_cols = list(train.columns)
        if out_row_wise_replay is not None:
            out_row_wise_replay["columns"] = [str(_c) for _c in _rw_cols]

        def _rw_apply(_fn, _df):
            """Apply a row-wise-stats function to _df's pinned columns and join the result back on."""
            # ``keep_cols`` pinning upstream intersects with each split's own columns, so a raw
            # column genuinely missing from val/test (rather than typo'd) can leave that split with
            # fewer pinned columns than train -- re-intersect ``_rw_cols`` against the actual frame
            # here so a per-split KeyError never fires instead of raising into the caller.
            if _df is None or not isinstance(_df, pd.DataFrame) or _df.shape[1] == 0:
                return _df
            _new = _fn(_df, [c for c in _rw_cols if c in _df.columns])
            return _df.join(_new)

        if getattr(config, "row_wise_summary_stats_enabled", False):
            t0_row_wise_summary = timer()
            from mlframe.feature_engineering.row_wise_summary import row_wise_summary_stats
            _rw_stats_list = getattr(config, "row_wise_summary_stats_list", None)

            def _summary_stats_for(_d, _cols):
                """Compute row-wise summary stats for _d over _cols, using the configured stat list if set."""
                _kwargs: Dict = {"columns": _cols}
                if _rw_stats_list:
                    _kwargs["stats"] = _rw_stats_list
                return row_wise_summary_stats(_d, **_kwargs)

            try:
                # ATOMIC across train/val/test: computed into locals first and only reassigned once
                # ALL THREE succeed. A per-split exception (e.g. an all-NaN column unique to one
                # split, seen more often on a composite-discovered target's transformed y/X than the
                # original target's) previously left whichever split(s) executed before the raise
                # WITH the new columns and the rest WITHOUT them -- schema-drifting the fitted
                # pre_pipeline's imputer against a later split, surfaced live as sklearn's "Feature
                # names should match those that were passed during fit".
                _train_rw = _rw_apply(_summary_stats_for, train)
                _val_rw = _rw_apply(_summary_stats_for, val)
                _test_rw = _rw_apply(_summary_stats_for, test)
                train, val, test = _train_rw, _val_rw, _test_rw
            except Exception:  # best-effort: row-wise summary stats are an optional enhancement
                logger.warning("apply_preprocessing_extensions: row_wise_summary_stats step failed; skipping.", exc_info=True)
            if verbose:
                logger.info("    apply_preprocessing_extensions.row_wise_summary_stats done in %s", _elapsed_str(t0_row_wise_summary))

        if getattr(config, "row_wise_extreme_columns_enabled", False):
            t0_row_wise_extreme = timer()
            from mlframe.feature_engineering.row_wise_extremality import row_wise_top_k_extreme_columns
            _rw_k_raw = getattr(config, "row_wise_extreme_columns_k", None)
            _rw_k = int(_rw_k_raw) if _rw_k_raw is not None else 3
            # Rank against a reference fixed on TRAIN rather than within each split. Ranking within the
            # frame made the feature depend on which rows were present: train, val and test were each
            # ranked against themselves, and a single row scored at predict time is its own median, so
            # every score collapsed to 0.0 -- a train/serve skew in a default-on feature. Set
            # ``row_wise_extreme_columns_fit_reference=False`` for the historical batch-relative score.
            _rw_reference = None
            if getattr(config, "row_wise_extreme_columns_fit_reference", True) and isinstance(train, pd.DataFrame):
                try:
                    from mlframe.feature_engineering.row_wise_extremality_reference import fit_extremality_reference

                    _rw_reference = fit_extremality_reference(train, [c for c in _rw_cols if c in train.columns])
                    if out_row_wise_replay is not None and _rw_reference:
                        out_row_wise_replay["extreme_columns_reference"] = {
                            str(_c): np.asarray(_v, dtype=np.float64).tolist() for _c, _v in _rw_reference.items()
                        }
                except Exception:
                    logger.warning(
                        "apply_preprocessing_extensions: could not fit the extremality reference; falling back "
                        "to within-split ranking (scores stay batch-relative).", exc_info=True,
                    )

            def _extreme_scores_only(_df, _cols):
                """Numeric-only slice of ``row_wise_top_k_extreme_columns`` output -- drops the ``topK_column`` name columns (object dtype), which would otherwise break the numeric-only contract the sklearn-bridge enforces on every downstream step (scaler / kbins / polynomial / dim_reducer).

                ``row_wise_top_k_extreme_columns`` internally clips ``k`` to the number of columns it was
                actually given (``k = min(k, n_cols)``); since ``_cols`` is ``_rw_cols`` RE-INTERSECTED
                against THIS split's own columns (see ``_rw_apply``'s docstring), a split with fewer
                eligible columns than another split -- e.g. train missing a raw column that val/test
                still have -- silently produces FEWER ``topN_score`` columns than the other splits,
                schema-drifting the fitted pre_pipeline's imputer against a later split with more
                columns (surfaced live: "Feature names should match" at test-transform time, train fit
                without ``row_extreme_top3_score`` while test carried it). Always request the SAME
                ``_rw_k`` and pad any missing ``topN_score`` slot with NaN so every split's output width
                is identical regardless of how many of ``_rw_cols`` that split actually had.
                """
                _out = row_wise_top_k_extreme_columns(_df, columns=_cols, k=_rw_k, reference=_rw_reference)
                assert isinstance(_out, pd.DataFrame)  # return_column_summary not passed -> always the plain-DataFrame overload
                _score_cols = [c for c in _out.columns if c.endswith("_score")]
                _result = _out[_score_cols].add_prefix("row_extreme_")
                for _i in range(1, _rw_k + 1):
                    _expected = f"row_extreme_top{_i}_score"
                    if _expected not in _result.columns:
                        _result[_expected] = np.nan
                return _result[[f"row_extreme_top{_i}_score" for _i in range(1, _rw_k + 1)]]

            try:
                # ATOMIC across train/val/test -- see the matching comment in the summary-stats
                # block above for why a partial per-split application schema-drifts the fitted
                # pre_pipeline.
                _train_rw = _rw_apply(_extreme_scores_only, train)
                _val_rw = _rw_apply(_extreme_scores_only, val)
                _test_rw = _rw_apply(_extreme_scores_only, test)
                train, val, test = _train_rw, _val_rw, _test_rw
            except Exception:  # best-effort: row-wise extreme-columns are an optional enhancement
                logger.warning("apply_preprocessing_extensions: row_wise_top_k_extreme_columns step failed; skipping.", exc_info=True)
            if verbose:
                logger.info("    apply_preprocessing_extensions.row_wise_extreme_columns done in %s", _elapsed_str(t0_row_wise_extreme))

    return train, val, test, _pre_row_wise_ncols
