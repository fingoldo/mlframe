"""Regression test for canonical_params_hash key-order stability (Phase-1 fix #3).

Pre-fix ``fingerprint.py:182`` called ``json.dumps(params, default=str)`` WITHOUT ``sort_keys=True``
on the non-dict branch. That violates user memory rules ``feedback_json_hash_sort_keys`` AND
``feedback_orjson_compile_regex``. The dict branch had sort_keys but stdlib json was still used.

Post-fix the function uses ``orjson.dumps(payload, option=orjson.OPT_SORT_KEYS, default=str)`` for
all non-pydantic inputs, so any dict-shaped payload (including ``OrderedDict`` or nested dicts)
hashes identically across insertion orders.
"""
from __future__ import annotations

from collections import OrderedDict

import pytest

from mlframe.training.feature_handling.fingerprint import canonical_params_hash


def test_canonical_hash_stable_under_dict_key_reordering():
    a = {"b": 1, "a": 2, "c": 3}
    b = {"c": 3, "a": 2, "b": 1}
    assert canonical_params_hash(a) == canonical_params_hash(b)


def test_canonical_hash_stable_for_ordered_dict_with_distinct_insertion_orders():
    """OrderedDict preserves insertion order; pre-fix the non-dict branch (isinstance(params, dict)
    is True for OrderedDict so this actually went via the sorted path -- but the failure mode
    triggered for any subclass not caught by ``isinstance(... dict)``). Post-fix orjson sorts
    recursively regardless of input subclass.
    """
    a = OrderedDict([("z", 1), ("a", 2), ("m", 3)])
    b = OrderedDict([("a", 2), ("m", 3), ("z", 1)])
    assert canonical_params_hash(a) == canonical_params_hash(b)


def test_canonical_hash_stable_for_nested_dicts():
    a = {"outer": {"b": 1, "a": 2}, "x": [1, 2, 3]}
    b = {"x": [1, 2, 3], "outer": {"a": 2, "b": 1}}
    assert canonical_params_hash(a) == canonical_params_hash(b)


def test_canonical_hash_handles_non_native_value_via_default_str():
    """Custom object values must coerce via ``default=str`` rather than raise (parity with the
    prior stdlib ``default=str`` behaviour).
    """
    class Custom:
        def __str__(self) -> str:
            return "custom_repr"

    h = canonical_params_hash({"v": Custom()})
    # Stable across two evaluations.
    assert h == canonical_params_hash({"v": Custom()})


def test_canonical_hash_distinguishes_distinct_payloads():
    """Sanity: different content => different hash."""
    assert canonical_params_hash({"a": 1}) != canonical_params_hash({"a": 2})


def test_canonical_hash_handles_numpy_scalars_natively():
    """Behavioural fingerprint of the orjson migration: stdlib json.dumps raises TypeError on
    numpy scalars (without an explicit ``default=``), while orjson serialises them natively. With
    the orjson migration the hash must succeed for a numpy scalar payload without crashing -- the
    ``default=str`` fallback would coerce to a string and still succeed too, but the value parity
    only holds if orjson handles numpy values natively as the same int they represent.
    """
    import numpy as np

    h_native = canonical_params_hash({"v": np.int64(7)})
    h_python = canonical_params_hash({"v": 7})
    # orjson serialises numpy.int64(7) as the bare integer 7 -- identical to the python int.
    assert h_native == h_python, (
        "orjson should serialise numpy scalars natively; pre-fix stdlib json would have called "
        "default=str and produced a quoted string instead, yielding a different hash"
    )
