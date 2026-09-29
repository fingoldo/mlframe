"""Shared formatting for the one-line "what did this selector keep" INFO summaries emitted at the end of a selector fit."""
from __future__ import annotations

from typing import Iterable

MAX_LOGGED_NAMES = 30


def format_name_list(names: Iterable, max_names: int = MAX_LOGGED_NAMES) -> str:
    """Comma-join ``names``, keeping the first ``max_names`` and summarising the rest as ``... (+K more)`` so a 5000-column selection stays one readable line."""
    names = [str(n) for n in names]
    if len(names) <= max_names:
        return ", ".join(names)
    return ", ".join(names[:max_names]) + f", ... (+{len(names) - max_names} more)"
