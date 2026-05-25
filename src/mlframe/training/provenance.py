"""Provenance trail helpers.

Each producer step in the suite (pre_screen, MRMR, RFECV, scaler/imputer/encoder,
calibration, analyzer, drift) is responsible for stamping a small record into
``metadata["provenance"][<step_name>]`` describing the SOURCE split used to fit
that step (train / train_only / train+val / oof / val / test), the number of
rows seen, the seed (if any), and a UTC timestamp.

The downstream honest-diagnostics aggregator (``training.honest_diagnostics``)
emits the full trail as a report table so a reviewer can verify at a glance
that every fit step touched only train-side data.

The helper is intentionally tiny and defensive: callers that already populated
the dict pass ``metadata=None`` shaped as ``{}`` and we no-op. Producers must
not crash the suite when provenance recording fails -- the trail is observability,
not correctness-critical.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping, Optional

logger = logging.getLogger(__name__)


VALID_SOURCES = ("train", "train_only", "train+val", "oof", "val", "test", "calib", "holdout")


def record_provenance(
    metadata: Optional[MutableMapping[str, Any]],
    step_name: str,
    *,
    source: str,
    n_rows: Optional[int] = None,
    seed: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Stamp a single provenance record under ``metadata["provenance"][step_name]``.

    Parameters
    ----------
    metadata
        The suite-level metadata dict (mutated in place). When ``None`` the call is a no-op,
        so producers can call ``record_provenance(metadata, ...)`` unconditionally without
        an outer ``if metadata is not None`` gate.
    step_name
        Stable producer identifier (e.g. ``"pre_screen"``, ``"mrmr"``, ``"rfecv"``,
        ``"preprocessing_pipeline"``, ``"post_calibrate"``, ``"target_distribution_analyzer"``).
        Re-stamping the same step replaces the prior record (last write wins).
    source
        Which split the producer FIT against. Must be one of ``VALID_SOURCES``; an unknown value
        is recorded verbatim under ``source`` with a warning so downstream readers surface the
        anomaly instead of silently dropping it.
    n_rows
        Row count of the split actually consumed by ``.fit``. Optional but strongly recommended;
        a missing count masks downstream "fit on tiny slice" warnings.
    seed
        Seed used by the producer (None when the producer is deterministic / not seeded).
    extra
        Additional, step-specific fields. Common keys: ``"cv_folds"`` for cross-validated
        producers, ``"n_features_in"`` for selectors.
    """
    if metadata is None:
        return
    try:
        if source not in VALID_SOURCES:
            logger.warning(
                "record_provenance: step %r recorded with unknown source %r; valid choices are %s",
                step_name, source, VALID_SOURCES,
            )
        record: dict[str, Any] = {
            "source": source,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if n_rows is not None:
            record["n_rows"] = int(n_rows)
        if seed is not None:
            record["seed"] = int(seed)
        if extra:
            for k, v in extra.items():
                if k in record:
                    continue
                record[k] = v
        store = metadata.setdefault("provenance", {})
        if not isinstance(store, dict):
            logger.warning(
                "record_provenance: metadata['provenance'] is %r, expected dict; skipping %r",
                type(store).__name__, step_name,
            )
            return
        store[step_name] = record
    except Exception as exc:
        logger.warning("record_provenance: failed to stamp %r: %s", step_name, exc)


def get_provenance(metadata: Optional[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the provenance trail (or an empty dict) without mutating ``metadata``."""
    if metadata is None:
        return {}
    store = metadata.get("provenance")
    if not isinstance(store, dict):
        return {}
    return dict(store)


def format_provenance_table(metadata: Optional[Mapping[str, Any]]) -> str:
    """Render the provenance trail as a human-readable table for the honest-diagnostics summary."""
    trail = get_provenance(metadata)
    if not trail:
        return "(no provenance recorded)"
    header = f"{'step':<40} {'source':<12} {'n_rows':>10} {'seed':>8} ts"
    lines = [header, "-" * len(header)]
    for step in sorted(trail):
        rec = trail[step]
        lines.append(
            f"{step:<40} {str(rec.get('source','?')):<12} "
            f"{str(rec.get('n_rows','-')):>10} {str(rec.get('seed','-')):>8} "
            f"{rec.get('ts','-')}"
        )
    return "\n".join(lines)
