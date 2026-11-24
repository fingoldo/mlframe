"""Cluster 40 regression: the per-split entity history is a documented choice at both feature sites (audit 2026-11-20, finding 40-F26).

The entity-time and MA-crossover features restart each split's entity history on purpose. The choice and its trade-off are stated where the
history is built, so a later change to carry train history forward is a deliberate, visible decision.
"""

from pathlib import Path

import mlframe.training.pipeline._entity_time_composite_fe as et
import mlframe.training.pipeline._ma_crossover_composite_fe as ma


def test_entity_time_documents_the_per_split_choice():
    """The per-split loop states that eval splits restart their entity history (40-F26)."""
    src = Path(et.__file__).read_text(encoding="utf-8")
    assert "documented choice" in src
    assert "restart" in src


def test_ma_crossover_documents_the_per_split_choice():
    """The rolling means state they are computed per split (40-F26)."""
    src = Path(ma.__file__).read_text(encoding="utf-8")
    assert "Rolling means are per split" in src
