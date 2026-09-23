"""The one place a discovered composite target enters ``target_by_type``.

A plain ``target_by_type[tt][name] = values`` overwrites whatever sat in that slot. A composite spec whose name equals an
existing target's (a raw column already routed to that type, or an earlier spec of the same run) therefore replaced that
target's values in silence: the suite went on training a target under a name whose data was no longer its own. The slot
dict has no owner, so a collision can only be caught where the write happens - here.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["insert_composite_targets"]


def insert_composite_targets(target_by_type: dict, kept: list[dict], metadata: dict) -> list[dict]:
    """Add each kept pending composite to its ``target_by_type`` slot, dropping the ones whose name is already taken.

    Parameters
    ----------
    target_by_type
        ``{target_type: {name: values}}``, the suite's target slots.
    kept
        Pending composite entries (``tt``, ``target``, ``name``, ``values``, ``gain``) that passed the gates.
    metadata
        Suite metadata; a dropped spec is removed from ``composite_target_specs`` and recorded in ``composite_target_failures``.

    Returns
    -------
    list[dict]
        The entries that were inserted, in order.
    """
    from ._phase_composite_discovery_dedup import forget_untrained_specs

    inserted, collided = [], []
    for item in kept:
        slot = target_by_type.setdefault(item["tt"], {})
        name = item["name"]
        if name in slot:
            collided.append(item)
            logger.warning(
                "[CompositeTargetDiscovery] composite target '%s' collides with an existing target in target_by_type[%s]; "
                "the spec is dropped rather than overwriting that target's values.", name, item["tt"],
            )
            continue
        slot[name] = item["values"]
        inserted.append(item)
        logger.info(
            "[CompositeTargetDiscovery] added composite target '%s' to target_by_type[%s] (honest gain %+.3f).",
            name, item["tt"], item["gain"],
        )
    if collided:
        forget_untrained_specs(metadata, collided, "its name already belongs to another target in target_by_type")
    return inserted
