"""Mix-offset companion targets for composite-target discovery, carved out of ``_phase_composite_discovery.py``.

The parent module re-exports ``_add_mix_offset_targets``, so existing imports keep working.
"""
from __future__ import annotations

import logging

import numpy as np

from ..configs import TargetTypes

logger = logging.getLogger("mlframe.training.core._phase_composite_discovery")


def _add_mix_offset_targets(target_by_type: dict, metadata: dict, inputs: dict) -> dict:
    """Add ``<name>__mixoff`` for each regression target whose mix base the gate accepts; record the base and slope in metadata.

    The companion is an ordinary target in ``target_by_type``: it is trained like any other, and its predictions are returned
    to the y scale by the add-back wrapper (wiring pending). A name collision leaves the existing target untouched."""
    from ...feature_selection.filters import build_mix_offset_target

    out = dict(target_by_type)
    reg = dict(out.get(TargetTypes.REGRESSION, {}))
    for tname, values in list(reg.items()):
        if np.asarray(values).ndim != 1:
            continue
        companion = f"{tname}__mixoff"
        if companion in reg:
            continue
        built = build_mix_offset_target(
            inputs["X"], values, inputs["train_idx"],
            selector=inputs["selector"], gate_regressor=inputs["gate_regressor"],
        )
        if built is None:
            continue
        reg[companion] = built["values"]
        metadata.setdefault("mix_offset_targets", {}).setdefault(str(TargetTypes.REGRESSION), {})[companion] = {
            "source": tname, "beta": built["beta"], "base": repr(built["base"]), "recipe": built["base"],
        }
        logger.info("[MixOffset] added companion target '%s' (beta %+.4f).", companion, built["beta"])
    out[TargetTypes.REGRESSION] = reg
    return out
