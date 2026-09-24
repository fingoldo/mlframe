"""The train-vs-test adversarial-validation panel and its per-run cache, split out of ``diagnostics_dispatch``.

The panel depends only on the feature frames, so every target of a run shares one fitted classifier through the cache.
The parent's helpers are imported per call: the parent re-exports this module's names at its top level.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

# The parent module's logger name: these lines predate the split, and log filters select them by that name.
logger = logging.getLogger("mlframe.reporting.diagnostics_dispatch")


def _render_adversarial_panel(*, train_frame: Any, test_frame: Any, val_frame: Any, non_calendar: Any, calendar: list,
                              plot_outputs: str, base_path: str, charts: Optional[dict], seed: int) -> None:
    """The train-vs-test separability panel, cached across targets since it depends only on the feature frames."""
    from .diagnostics_dispatch import DIAG_MAX_FEATURES, _column_names, _record, _record_path, _save_spec, _with_caption_note
    # Imported per call, not at module import: the builder's real home is the patch point tests reach for.
    from mlframe.reporting.charts.drift import adversarial_validation as _adversarial_validation_fn

    try:
        # Its own LightGBM classifier fit cost scales with COLUMN count, not just row count -- unlike
        # every other builder in this dispatcher (all row/histogram capped), this one had no bound on a
        # very wide frame at all. Capped the same way this module's OWN dense-matrix builders already
        # are (DIAG_MAX_FEATURES), by restricting feature_names before the fit rather than after: the
        # underlying frame-reader already narrows to exactly the given names, so no extra frame slicing
        # is needed. Traced to a production profile alongside the (separately fixed) PDP categorical-
        # sweep cost -- the same "cost scales with an unbounded dimension" bug class.
        _adv_names = list(non_calendar) if non_calendar is not None else _column_names(train_frame)
        if _adv_names is not None and len(_adv_names) > DIAG_MAX_FEATURES:
            _adv_names = _adv_names[:DIAG_MAX_FEATURES]
        # The adversarial classifier depends only on the feature frames, not on the target: every target of a run
        # (raw and composite alike) re-fitted the same 3-fold LightGBM, ~15 s x 32 targets in one production log.
        _adv_key = _adversarial_cache_key(train_frame, test_frame, val_frame, _adv_names, seed)
        with _ADVERSARIAL_LOCK:
            spec = _ADVERSARIAL_CACHE.get(_adv_key) if _adv_key is not None else None
        if spec is None:
            spec = _adversarial_validation_fn(
                train_frame, test_frame if test_frame is not None else val_frame,
                val_frame=val_frame if test_frame is not None else None,
                feature_names=_adv_names, seed=seed,
            )
            if _adv_key is not None:
                with _ADVERSARIAL_LOCK:
                    while len(_ADVERSARIAL_CACHE) >= 8:
                        # evict-ok: memo; a miss recomputes the value
                        _ADVERSARIAL_CACHE.pop(next(iter(_ADVERSARIAL_CACHE)))
                    _ADVERSARIAL_CACHE[_adv_key] = spec
        if calendar:
            spec = _with_caption_note(spec, f"Excluded {len(calendar)} calendar feature(s) derived from the timestamp (an earlier and a later period differ in them by construction): {', '.join(calendar)}.")
        ok = _save_spec(spec, plot_outputs, base_path + "_adversarial")
        _record(charts, "adversarial", ok)
        if ok:
            _record_path(charts, base_path + "_adversarial")
    except Exception:
        logger.exception("diagnostics_dispatch: adversarial_validation failed; continuing.")
        _record(charts, "adversarial", False)


_ADVERSARIAL_CACHE: dict = {}
# Targets can render concurrently; the lock covers each read and each evict-then-insert, so the eviction loop never races an insert.
# The fit itself runs outside it: two threads missing the same key both fit, which costs time but never corrupts the cache.
_ADVERSARIAL_LOCK = threading.Lock()


def _adversarial_cache_key(train_frame: Any, test_frame: Any, val_frame: Any, names: Any, seed: int) -> Optional[tuple]:
    """Content key for an adversarial-validation figure: frame signatures (columns, shape, row-sample hash) + features."""
    try:
        from mlframe.training import compute_signature

        sig = tuple(compute_signature(f)[:4] if f is not None else None for f in (train_frame, test_frame, val_frame))
        return (*sig, tuple(str(n) for n in names) if names is not None else None, int(seed))
    except Exception as exc:  # an unkeyable frame only means "do not cache": the figure is recomputed, never wrong
        logger.debug("adversarial-validation cache key unavailable (%s); computing the figure uncached", exc)
        return None
