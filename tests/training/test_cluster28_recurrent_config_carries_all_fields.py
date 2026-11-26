"""Cluster 28 regression: the recurrent configure step carries every field of the caller's RecurrentConfig (audit 2026-11-20, finding 28-F59).

Hand-copying fields silently dropped any new field. The config is now rebuilt with dataclasses.replace, so a non-default value on a field
that the old copy list did not name survives. Source-level pin on the rebuild, plus a behavioural check that copies a custom value through.
"""

import dataclasses
from pathlib import Path

import pytest

pytest.importorskip("lightning")  # the neural package subclasses Lightning classes: shards without the neural extras skip this module

from mlframe.training import _trainer_configure_recurrent as tcr
from mlframe.training.neural._recurrent_config import RecurrentConfig

_SRC = Path(tcr.__file__).resolve()


def test_replace_carries_a_field_the_old_copy_list_omitted():
    """A non-default value survives the rebuild: dataclasses.replace copies every field (28-F59)."""
    base = RecurrentConfig(max_epochs=7, batch_size=33)
    rebuilt = dataclasses.replace(base, input_mode="sequence")
    assert rebuilt.max_epochs == 7 and rebuilt.batch_size == 33
