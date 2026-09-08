"""Render the per-run targets quality frame at suite end and stash it on the metadata.

The suite already prints a per-target verdict against the dummy baselines. What it never produced is the
one table you need to compare two RUNS of it -- every target of every kind side by side, in a stable order,
with the metrics each one reports. That is what lands in ``metadata["targets_performance"]`` here, so a
caller can diff two runs that differ only in their features or hyperparameters with
:func:`mlframe.training.targets_performance.compare_targets_performance`.

Diagnostic, never load-bearing: a failure here is logged and the suite returns exactly what it would have.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger("mlframe.training.core._phase_targets_performance")

#: Widen only for THIS render, then restore. A quality frame with twelve metric columns wraps into
#: unreadable rubble at pandas' default 80-column display width.
_DISPLAY_OPTIONS = {"display.width": 250, "display.max_columns": 60, "display.max_colwidth": 28}


#: Suffix of the file the frame is written to beside the run's charts. CSV rather than a pickle: comparing
#: yesterday's run with today's must not require the classes that produced it to still be importable.
TARGETS_PERFORMANCE_SUFFIX = "_targets_performance.csv"


def _persist(frame: Any, plot_file: Optional[str], metadata: dict) -> None:
    """Write the frame beside the run's charts so a later run can be diffed against it without a re-run.

    Comparing runs is the whole point of the frame, and the runs being compared are usually not both in
    memory -- yesterday's against today's is the normal case. Failure to write is logged and otherwise
    ignored: the in-memory frame is still on the metadata.
    """
    if not plot_file:
        return
    import os

    root = os.path.splitext(str(plot_file))[0]
    path = f"{root}{TARGETS_PERFORMANCE_SUFFIX}"
    try:
        # ``dirname`` returns "" for a bare filename, and ``makedirs("")`` raises -- so the empty string is
        # genuinely "the current directory" here, not a caller value being overwritten by a default.
        parent = os.path.dirname(path)
        os.makedirs(parent if parent else ".", exist_ok=True)
        frame.to_csv(path, index=False)
        metadata["targets_performance_path"] = path
        logger.info("targets quality frame written to %s", path)
    except Exception as err:
        logger.warning("could not write the targets quality frame to %s (%s: %s)", path, type(err).__name__, err)


def render_targets_performance(
    models: Mapping[str, Any],
    metadata: dict,
    *,
    split: Optional[str] = None,
    plot_file: Optional[str] = None,
) -> None:
    """Build the quality frame, log it, store it on the metadata, and write it beside the charts."""
    try:
        import pandas as pd

        from ..targets_performance import DEFAULT_SPLIT, targets_performance_frame

        # Explicit None check: ``split or DEFAULT`` would silently turn an empty string -- a caller bug --
        # into the default split, and the frame would then report a split nobody asked for.
        chosen = DEFAULT_SPLIT if split is None else split
        frame = targets_performance_frame(models, metadata, split=chosen)
        if frame.empty or len(frame) <= 1:
            # Only the aggregate row, or not even that: nothing was trained that carries metrics, and an
            # empty table in the log reads as a failure rather than as "there was nothing to show".
            #
            # The key is NOT written in that case. A caller that ran no targets gets its metadata back
            # exactly as it passed it in -- which several callers rely on, and one contract test asserts
            # outright -- and "the key is absent" reads the same as "there was nothing to report" without
            # putting an empty frame in front of anyone.
            logger.debug("targets performance frame is empty for split %r; nothing to render", chosen)
            return

        metadata["targets_performance"] = frame
        _persist(frame, plot_file, metadata)
        with pd.option_context(*[kv for item in _DISPLAY_OPTIONS.items() for kv in item]):
            logger.info(
                "\n===== TARGETS QUALITY (%s split, %d target(s)) =====\n%s\n"
                "Compare runs with mlframe.training.targets_performance.compare_targets_performance().",
                chosen,
                len(frame) - 1,
                frame.to_string(index=False),
            )
    except Exception as err:  # never load-bearing
        logger.warning("targets performance frame failed (%s: %s); suite output is unaffected.", type(err).__name__, err)
