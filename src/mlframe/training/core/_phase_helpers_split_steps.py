"""Step helpers carved out of ``_phase_train_val_test_split``: stratify-mode logging, the splitter call, split outputs, sequences and cleanup."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..phases import phase

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

from ..preprocessing import save_split_artifacts
from ..utils import get_process_rss_mb, log_ram_usage, maybe_clean_ram_and_gpu
from ..splitting import make_train_test_split
from .._fixed_splits import has_pinned_splits, pinned_train_val_test_split, record_split_membership, split_dir_for

logger = logging.getLogger("mlframe.training.core._phase_helpers_fit_split")


def _log_group_stratify_mode(_groups: Any, _stratify_y: Any) -> None:
    """Log which splitter route a group-aware split with stratification labels takes."""

    # Group + bucket combination: ``make_train_test_split`` now routes
    # the 1-D case through sklearn's ``StratifiedGroupKFold`` (sklearn
    # >=1.0) which preserves the class / regression-bucket distribution
    # while keeping whole groups together -- both invariants honoured by
    # the same splitter. The 2-D multilabel case still requires
    # ``MultilabelStratifiedGroupKFold`` from iterative-stratification;
    # the splitter detects + falls back to GroupShuffleSplit when that
    # package is absent.
    if _groups is not None and _stratify_y is not None:
        _strat_arr = np.asarray(_stratify_y)
        if _strat_arr.ndim == 2:
            try:
                from iterstrat.ml_stratifiers import (  # noqa: F401
                    MultilabelStratifiedGroupKFold,
                )

                _has_multilabel_iterstrat = True
            except ImportError:
                _has_multilabel_iterstrat = False
            if not _has_multilabel_iterstrat:
                logger.warning(
                    "Bucket-stratify: 2-D (multilabel) stratify_y + groups; "
                    "iterative-stratification not installed. Splitter will "
                    "stratify on a derived 1-D composite label-combination id "
                    "(joint balance preserved) when feasible, else fall back to "
                    "GroupShuffleSplit with NO multilabel proportion guarantee. "
                    "pip install iterative-stratification for exact "
                    "MultilabelStratifiedGroupKFold.",
                )
        else:
            logger.info(
                "Bucket-stratify: groups + 1-D stratify_y; using "
                "StratifiedGroupKFold (both invariants honoured: whole "
                "groups stay in one split AND class/bucket proportions "
                "preserved across train/val/test).",
            )


def _run_split_data(
    *,
    df: Any,
    timestamps: Any,
    stratify_y: Any,
    groups: Any,
    split_config: Any,
    row_ids: Any,
    metadata: dict,
    verbose: bool,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """Run the main splitter (pinned or free) plus the purged embargoes; return ``(train_idx, val_idx, test_idx, train_details, val_details, test_details, calib_idx, calib_details)``."""
    from ._phase_helpers_fit_split import _apply_purged_embargoes

    with phase("split_data"):
        # Dynamically derive the kwargs the splitter accepts by inspecting
        # its signature, then drop any TrainingSplitConfig field not in
        # that set. Static exclude lists drift: observed in prod two
        # consecutive TypeErrors (composite_cardinality_cap
        # then bucket_stratify) because new caller-side fields shipped
        # without exclude updates. Caller-side fields documented:
        #   usegroups -- derives groups upstream
        #   calib_size -- carves a disjoint calibration slice from train (return_calib=True below)
        #   composite_cardinality_cap -- bucket-stratify gate (line ~139)
        #   bucket_stratify -- selects the bucket-stratify branch
        # Signature-derived filtering catches future additions automatically.
        import inspect as _inspect

        _splitter_kwargs = set(_inspect.signature(make_train_test_split).parameters)
        _explicit_kwargs = {"df", "timestamps", "stratify_y", "groups", "return_calib"}
        _cfg_dict = {k: v for k, v in split_config.model_dump().items() if k in _splitter_kwargs and k not in _explicit_kwargs}
        if has_pinned_splits(split_config):
            # Some splits are fixed by an id file / date windows; the rest is carved by the same splitter on the remaining rows.
            train_idx, val_idx, test_idx, train_details, val_details, test_details, calib_idx, calib_details, _pin_info = pinned_train_val_test_split(
                n_rows=len(df) if df is not None else 0,
                row_ids=row_ids,
                timestamps=timestamps,
                split_config=split_config,
                stratify_y=stratify_y,
                groups=groups,
                splitter=make_train_test_split,
                splitter_kwargs=_cfg_dict,
            )
            metadata["split_pinning"] = _pin_info
        else:
            train_idx, val_idx, test_idx, train_details, val_details, test_details, calib_idx, calib_details = make_train_test_split(
                df=df,
                timestamps=timestamps,
                stratify_y=stratify_y,
                groups=groups,
                return_calib=True,
                **_cfg_dict,
            )
        train_idx, val_idx = _apply_purged_embargoes(train_idx, val_idx, timestamps, split_config, verbose)
    return train_idx, val_idx, test_idx, train_details, val_details, test_details, calib_idx, calib_details


def _record_split_outputs(
    *,
    train_idx: Any,
    val_idx: Any,
    test_idx: Any,
    calib_idx: Any,
    train_details: Any,
    val_details: Any,
    test_details: Any,
    calib_details: Any,
    timestamps: Any,
    group_ids_raw: Any,
    artifacts: Any,
    split_config: Any,
    row_ids: Any,
    data_dir: str,
    models_dir: str,
    target_name: str,
    model_name: str,
    metadata: dict,
) -> None:
    """Persist the split artifacts, record id-column membership and stamp the split sizes/details into ``metadata``."""
    if data_dir:
        save_split_artifacts(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            timestamps=timestamps,
            group_ids_raw=group_ids_raw,
            artifacts=artifacts,
            data_dir=data_dir,
            models_dir=models_dir,
            target_name=target_name,
            model_name=model_name,
        )

    _id_column = getattr(split_config, "id_column", None)
    if _id_column and row_ids is not None:
        metadata["split_membership"] = record_split_membership(
            row_ids=row_ids,
            id_column=_id_column,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            calib_idx=calib_idx,
            timestamps=timestamps,
            split_dir=split_dir_for(data_dir, models_dir, target_name, model_name) if data_dir else None,
        )

    metadata.update(
        {
            "train_details": train_details,
            "val_details": val_details,
            "test_details": test_details,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        }
    )
    if calib_idx is not None and len(calib_idx) > 0:
        metadata["calib_details"] = calib_details
        metadata["calib_size_rows"] = len(calib_idx)


def _carve_calib_df(df: Any, calib_idx: Any) -> Any:
    """Carve the disjoint calib slice (calib_size>0) as its own raw frame, same schema as train_df, or None when there is none."""
    # Carve the disjoint calib slice (calib_size>0) as its own raw frame, same schema as train_df, while ``df`` is still
    # alive (the caller nulls ``ctx.df`` right after this phase). The slice is a small fraction of train; subsetting is a
    # format-native ``df[calib_idx]`` / ``.iloc`` view, never a full-frame clone. The trainer transforms it through the
    # same fitted pre_pipeline as test and runs predict_proba so finalize can auto-calibrate.
    calib_df = None
    if calib_idx is not None and len(calib_idx) > 0 and df is not None:
        if isinstance(df, pl.DataFrame):
            calib_df = df[calib_idx]
        else:
            calib_df = df.iloc[calib_idx]
    return calib_df


def _split_sequences(sequences: Any, train_idx: Any, val_idx: Any, test_idx: Any, verbose: bool) -> tuple[Any, Any, Any]:
    """Split sequences by train/val/test indices (for recurrent models)."""
    train_sequences, val_sequences, test_sequences = None, None, None
    if sequences is not None:
        train_sequences = [sequences[i] for i in train_idx]
        val_sequences = [sequences[i] for i in val_idx] if val_idx is not None else None
        test_sequences = [sequences[i] for i in test_idx]
        if verbose:
            logger.info("Split sequences: train=%d, val=%d, test=%d", len(train_sequences), len(val_sequences) if val_sequences else 0, len(test_sequences))
    return train_sequences, val_sequences, test_sequences


def _post_split_cleanup(df_size_mb: float, verbose: bool) -> float:
    """Release RAM after the split and return the refreshed baseline RSS (MB)."""
    # Refresh baseline so the next maybe_clean_ram_and_gpu in the caller sees the post-del state.
    _rss_before_mb = get_process_rss_mb()
    baseline_rss_mb = maybe_clean_ram_and_gpu(_rss_before_mb, df_size_mb, verbose=verbose, reason="post-split (del df)")
    if verbose:
        # Report what the cleanup ACTUALLY released rather than announcing an intention: a production log printed
        # "Deleting original DataFrame to free RAM..." and then the same 45.4GB it printed before, because the
        # split frames are views over the source buffers and the source name is dropped by the caller, not here.
        _freed_mb = _rss_before_mb - get_process_rss_mb()
        if _freed_mb >= 64.0:
            logger.info("Post-split cleanup released %.1fGB.", _freed_mb / 1024.0)
        else:
            logger.info(
                "Post-split cleanup released no measurable RAM (%.0fMB): the split frames reference the same "
                "buffers as the source, so dropping the source name alone frees nothing.",
                max(_freed_mb, 0.0),
            )
        log_ram_usage()
    return baseline_rss_mb
