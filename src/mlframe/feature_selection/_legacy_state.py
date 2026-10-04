"""Backfill of fitted attributes that an estimator pickled by an older release does not carry."""

from __future__ import annotations

import copy
from collections.abc import Mapping

def backfill_legacy_state(state: dict, defaults: Mapping, fitted_marker: str) -> dict:
    """Return a copy of ``state`` with every missing key of ``defaults`` filled in, when ``state`` belongs to a fitted estimator.

    ``fitted_marker`` is the core fitted attribute (for example ``support_``) whose presence means the pickle came from a fitted estimator; an
    unfitted instance is returned untouched so it keeps reporting itself as unfitted. Mutable defaults are deep-copied per instance.
    """
    out = dict(state)
    if fitted_marker not in out:
        return out
    for name, default in defaults.items():
        if name not in out:
            out[name] = copy.deepcopy(default)
    return out
