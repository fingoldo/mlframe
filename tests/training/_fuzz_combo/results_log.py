"""Append-only JSONL results log for fuzz combo outcomes."""

from __future__ import annotations

import os
from pathlib import Path

import orjson

from .combo import FuzzCombo  # noqa: F401  (annotation strings under PEP 563)


# parents[1] keeps the log at tests/training/_fuzz_results.jsonl (next to the
# former flat module), unchanged by the subpackage carve.
RESULTS_LOG = Path(__file__).parents[1] / "_fuzz_results.jsonl"


def log_combo_outcome(
    combo: FuzzCombo,
    outcome: str,
    duration_s: float,
    error_class: str | None = None,
    error_summary: str | None = None,
    extra: dict | None = None,
) -> None:
    """Append one JSONL row with the combo's outcome.

    Columns: combo fields, outcome in {pass,fail,xpass,xfail,skip}, duration,
    error_class/error_summary (for fail/xpass rows), extra (free-form dict),
    and ``master_seed`` (Fix E: seed-rotation telemetry — the nightly cron
    passes a different ``FUZZ_SEED`` each run, we tag each row so failures
    stay attributable to their generating seed).
    """
    row: dict = {
        **combo.to_json(),
        "outcome": outcome,
        "duration_s": round(duration_s, 3),
        "master_seed": int(os.environ.get("FUZZ_SEED", "20260422")),
    }
    if error_class:
        row["error_class"] = error_class
    if error_summary:
        row["error_summary"] = error_summary[:300]
    if extra:
        row["extra"] = extra
    try:
        RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS_LOG.open("ab") as f:
            # orjson keeps non-ASCII as raw UTF-8 bytes (no \uXXXX escaping),
            # matching the prior ensure_ascii=False behaviour.
            f.write(orjson.dumps(row, option=orjson.OPT_SORT_KEYS) + b"\n")
    except OSError:
        pass  # never break a test because logging failed


def read_fail_summary() -> dict:
    """Return a summary of failures since the last run start marker."""
    if not RESULTS_LOG.exists():
        return {"fails": [], "totals": {}}
    totals: dict[str, int] = {}
    fails: list[dict] = []
    with RESULTS_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = orjson.loads(line)
            except Exception:
                continue
            totals[row.get("outcome", "?")] = totals.get(row.get("outcome", "?"), 0) + 1
            if row.get("outcome") == "fail":
                fails.append(row)
    return {"fails": fails, "totals": totals}
