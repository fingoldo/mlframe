"""The flag-set key equals the oracle's order-independent key over the full master-flag universe, with absent flags read as off."""

from __future__ import annotations

import random

from mlframe.feature_selection.filters._meta_fe_recommender import ALL_FE_MASTER_FLAGS, _flags_to_combo_key
from mlframe.utils._param_oracle import _combo_key


def test_key_matches_oracle_key_on_full_flag_dict() -> None:
    """A full flag dict keys identically under the flag helper and the oracle's combo key."""
    rng = random.Random(0)
    for _ in range(50):
        full = {f: rng.random() < 0.5 for f in ALL_FE_MASTER_FLAGS}
        assert _flags_to_combo_key(full) == _combo_key(full)


def test_missing_flags_read_as_off() -> None:
    """A partial dict keys the same as the full dict with the missing flags set to False."""
    rng = random.Random(1)
    for _ in range(50):
        full = {f: rng.random() < 0.5 for f in ALL_FE_MASTER_FLAGS}
        partial = {f: v for f, v in full.items() if rng.random() < 0.7}
        normalised = {f: partial.get(f, False) for f in ALL_FE_MASTER_FLAGS}
        assert _flags_to_combo_key(partial) == _combo_key(normalised)
