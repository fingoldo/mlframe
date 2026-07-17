"""Coverage for ``mlframe.config`` enums: pickle round-trip identity + per-enum value uniqueness.

These enums are stamped into saved-model metadata / pipeline configs, so a pickle that does
not round-trip to the SAME singleton member (identity, not just equality) or an accidental
alias (two members sharing a value) would silently corrupt config replay.
"""

from __future__ import annotations

import pickle
from enum import Enum

import pytest

import mlframe.config as cfg


def _all_config_enums():
    out = []
    for name in dir(cfg):
        obj = getattr(cfg, name)
        if isinstance(obj, type) and issubclass(obj, Enum) and obj is not Enum:
            out.append(obj)
    return out


CONFIG_ENUMS = _all_config_enums()


def test_config_exposes_multiple_enums():
    # Guard against the discovery helper silently finding nothing (e.g. after a refactor).
    names = {e.__name__ for e in CONFIG_ENUMS}
    assert {"CategoricalsAssigning", "MissingHandling", "EarlyStopping"} <= names
    assert len(CONFIG_ENUMS) >= 10


@pytest.mark.parametrize("enum_cls", CONFIG_ENUMS, ids=lambda e: e.__name__)
def test_enum_values_are_unique_no_aliases(enum_cls):
    # __members__ includes aliases; list(enum_cls) does not. Equal length => no member is an
    # alias of another (every member owns a distinct value).
    members = list(enum_cls)
    values = [m.value for m in members]
    assert len(set(values)) == len(values), f"{enum_cls.__name__} has duplicate/alias values: {values}"
    assert len(members) == len(enum_cls.__members__), f"{enum_cls.__name__} contains alias members"


@pytest.mark.parametrize("enum_cls", CONFIG_ENUMS, ids=lambda e: e.__name__)
def test_enum_pickle_roundtrip_is_same_singleton(enum_cls):
    for member in enum_cls:
        restored = pickle.loads(pickle.dumps(member))
        assert restored is member, f"{enum_cls.__name__}.{member.name} did not round-trip to the same singleton"
