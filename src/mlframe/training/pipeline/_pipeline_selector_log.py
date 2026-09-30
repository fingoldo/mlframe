"""One INFO line per feature-selector fit at the suite level: ``feature selector <Name>: N_in -> N_out columns (dropped D: [first names])``."""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from mlframe.feature_selection._selection_log import format_name_list

logger = logging.getLogger(__name__)


def log_selector_retention(selector: Any, pipeline: Any, input_cols: Optional[Sequence], train_df_out: Any) -> None:
    """Log input -> output width and which columns were dropped; never raises (a reporting failure must not fail the suite).

    Dropped names are only listed when the output frame carries column names and the drop set is a plain subset of the input columns
    (an FE selector such as MRMR adds engineered columns, in which case only the counts are reported).
    """
    try:
        shape = getattr(train_df_out, "shape", None)
        if shape is None or len(shape) != 2 or input_cols is None:
            return
        n_in, n_out = len(input_cols), int(shape[1])
        label = type(selector).__name__ if selector is not None else type(pipeline).__name__
        msg = f"feature selector {label}: {n_in:_} -> {n_out:_} columns"
        out_cols = getattr(train_df_out, "columns", None)
        if out_cols is not None:
            out_set = set(out_cols)
            dropped = [c for c in input_cols if c not in out_set]
            added = n_out - (n_in - len(dropped))
            if dropped:
                msg += f" (dropped {len(dropped):_}: [{format_name_list(dropped)}])"
            if added > 0:
                msg += f" (+{added:_} new columns)"
        else:
            msg += f" (dropped {max(n_in - n_out, 0):_})"
        logger.info(msg)
    except Exception as exc:
        logger.debug("selector retention log failed: %s", exc)
