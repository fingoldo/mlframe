"""Sequential grid runner for ``train_mlframe_models_suite``.

Replaces the dropped ``TryAllMethods`` pattern with an explicit, minimal helper:
run the suite once per variant, collect results in a dict keyed by variant label.
Users compose variants themselves (no implicit sweep over hyperparameters).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


GridEntry = Union[Dict[str, Any], Tuple[str, Dict[str, Any]]]


def _label_for(entry: GridEntry, idx: int) -> Tuple[str, Dict[str, Any]]:
    if isinstance(entry, tuple):
        if len(entry) != 2 or not isinstance(entry[0], str) or not isinstance(entry[1], dict):
            raise TypeError(
                f"grid entry {idx} must be dict or (label, dict); got {type(entry).__name__}"
            )
        return entry[0], entry[1]
    if isinstance(entry, dict):
        return f"variant_{idx}", entry
    raise TypeError(f"grid entry {idx} must be dict or (label, dict); got {type(entry).__name__}")


def run_grid(
    base_kwargs: Dict[str, Any],
    grid: Iterable[GridEntry],
    *,
    suite_fn: Optional[Callable[..., Any]] = None,
    stop_on_error: bool = False,
) -> Dict[str, Any]:
    """Run ``suite_fn`` once per grid entry, merging entry over ``base_kwargs``.

    Parameters
    ----------
    base_kwargs : dict
        Keyword arguments shared by every run.
    grid : iterable of dict or (label, dict)
        Each entry is merged on top of ``base_kwargs``. Tuple form supplies an
        explicit label; bare dicts auto-label as ``variant_0``, ``variant_1``...
    suite_fn : callable, optional
        Defaults to ``mlframe.training.core.train_mlframe_models_suite``; pass
        a stub for testing.
    stop_on_error : bool
        If False (default), exceptions from a variant are logged and the result
        is stored as ``{"error": repr(exc)}``; other variants keep running.
    """
    if suite_fn is None:
        from .core import train_mlframe_models_suite
        suite_fn = train_mlframe_models_suite

    results: Dict[str, Any] = {}
    labels: List[str] = []
    for idx, entry in enumerate(grid):
        label, overrides = _label_for(entry, idx)
        if label in results:
            raise ValueError(f"duplicate grid label: {label!r}")
        labels.append(label)
        merged = {**base_kwargs, **overrides}
        logger.info("[run_grid] variant %d/%s — %s", idx + 1, label, sorted(overrides.keys()))
        try:
            results[label] = suite_fn(**merged)
        except Exception as exc:  # noqa: BLE001 — we deliberately keep going
            logger.exception("[run_grid] variant %s raised %s", label, type(exc).__name__)
            if stop_on_error:
                raise
            results[label] = {"error": repr(exc)}
    logger.info("[run_grid] completed %d variants: %s", len(labels), labels)
    return results


__all__ = ["run_grid"]
