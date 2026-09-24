"""Every input of the cached discovery run is in its cache key, and the list of inputs is derived, not remembered.

The key once omitted the group ids, hint strengths, time order and val frame the phase hands to discovery, so a rerun under
a new split replayed specs selected on the old one. The required inputs are derived here from what discovery reads: every
parameter of ``CompositeTargetDiscovery.fit`` and every private attribute the suite phase injects onto the instance and
discovery reads. Each must be keyed (with a perturbation that moves the key) or listed as not affecting the result.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

import mlframe

_TRAINING = Path(mlframe.__file__).resolve().parent / "training"

# Inputs that reach the key: the name of the argument that carries them into ``discovery_inputs_digest`` or the lookup.
CACHE_KEY_INPUTS = {
    "df": "lookup: the discovery frame's content",
    "target_col": "lookup: the target column name",
    "feature_cols": "lookup: the feature column list",
    "time_ordering": "digest: disc_df's time-column values",
    "val_df": "digest: val_df",
    "val_y": "digest: val_y (y_full[val_idx] as the phase derives it)",
    "_group_ids_for_rerank": "digest: group_ids",
    "_hint_strengths_pct": "digest: hint_strengths",
}
# Inputs of fit that cannot change the specs the phase caches, with the reason.
NOT_RESULT_INPUTS = {
    "train_idx": "the phase always passes every row of the discovery frame (np.arange), so it is the frame",
    "val_idx": "the phase never passes it; the y-scale gate reads val_df / val_y",
    "test_idx": "the phase never passes it, and discovery never reads test rows",
}


def _required_inputs() -> set[str]:
    """``fit``'s parameters plus the private attributes the phase sets on the discovery instance that discovery reads."""
    from mlframe.training.composite.discovery import CompositeTargetDiscovery

    params = {p for p in inspect.signature(CompositeTargetDiscovery.fit).parameters if p != "self"}
    phase = ast.parse((_TRAINING / "core" / "_phase_composite_discovery.py").read_text(encoding="utf-8"))
    injected = {n.attr for n in ast.walk(phase) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                and n.attr.startswith("_") and isinstance(n.value, ast.Name) and n.value.id == "_disc_instance"}
    read: set[str] = set()
    for path in (_TRAINING / "composite" / "discovery").rglob("*.py"):
        for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self" and isinstance(n.ctx, ast.Load):
                read.add(n.attr)
            elif isinstance(n, ast.Call) and getattr(n.func, "id", None) == "getattr" and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant):
                read.add(str(n.args[1].value))
    return params | (injected & read)


def test_every_input_discovery_reads_is_keyed_or_declared_irrelevant():
    """A new fit parameter, or a new attribute the phase injects, must join the key or say why it cannot change the specs."""
    required = _required_inputs()
    assert {"_group_ids_for_rerank", "_hint_strengths_pct"} <= required, "the derivation lost the injected state"
    listed = set(CACHE_KEY_INPUTS) | set(NOT_RESULT_INPUTS)
    assert required == listed, f"unlisted: {sorted(required - listed)}; stale: {sorted(listed - required)}"


def test_the_lookup_inputs_move_the_key(tmp_path):
    """The frame's content, the target column and the feature list each change the key (the digest's inputs are covered in
    test_cache_store_identity.py)."""
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    from mlframe.training.core._phase_composite_discovery_gates import _discovery_cache_lookup, discovery_inputs_digest

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"b": rng.normal(size=200), "c": rng.normal(size=200), "y": rng.normal(size=200), "z": rng.normal(size=200)})
    cfg = CompositeTargetDiscoveryConfig(enabled=True)
    digest = discovery_inputs_digest()

    def key(frame=df, target="y", feats=("b", "c")):
        return _discovery_cache_lookup(cfg, frame, target, list(feats), tmp_path, inputs_digest=digest)[1]

    ref = key()
    variants = {"frame content": key(frame=df.assign(b=df["b"] + 1.0)), "target": key(target="z"), "features": key(feats=("b",))}
    assert all(v != ref for v in variants.values()), [k for k, v in variants.items() if v == ref]
