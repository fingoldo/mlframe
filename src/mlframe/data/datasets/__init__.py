"""Synthetic and showcase dataset generators used for benchmarking mlframe's feature-selection and training pipelines.

The synthetic half is structural-causal-model first: a ``spec.DatasetSpec`` declares the graph, and ground
truth (roles, redundancy structure, the three named target sets) is DERIVED from that graph rather than
hand-listed beside it, so the answer key cannot drift away from the generator. Randomness is addressed by
name (``_rng.stream_for``), not by position in a spawn sequence, so inserting a stream cannot silently move
every later stream's bytes.

Import cost: ``mlframe.data`` star-imports this package, so anything eager here is paid by every consumer of
``mlframe.data``. The two legacy loaders are cheap and load eagerly; the spec / ground-truth / graph surface
pulls pydantic and is served through PEP 562 ``__getattr__`` instead, so ``import mlframe.data`` stays as
cheap as it was before this package existed. Attribute access resolves them on first use, and importing by
path (``from mlframe.data.datasets.spec import DatasetSpec``) bypasses the indirection entirely.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from mlframe.data.datasets._sapp import get_sapp_dataset, indicator
from mlframe.data.datasets._showcase import showcase_pycaret_datasets

# Which submodule each lazily-exported name lives in. Kept beside ``__all__`` so a name can never be
# advertised without a resolvable home -- a test asserts the two agree.
_LAZY_HOME: Dict[str, str] = {
    "CausalGraph": "_scm",
    "Ceiling": "ground_truth",
    "CeilingTarget": "spec",
    "DatasetSpec": "spec",
    "EdgeSpec": "spec",
    "FeatureRole": "ground_truth",
    "FeatureSpec": "spec",
    "FeatureTruth": "ground_truth",
    "GateSpec": "spec",
    "GroundTruth": "ground_truth",
    "LatentSpec": "spec",
    "LinkSpec": "spec",
    "MIBundle": "ground_truth",
    "MIEstimate": "ground_truth",
    "NoiseSpec": "spec",
    "Prior": "spec",
    "RedundancyGroup": "ground_truth",
    "TargetSet": "ground_truth",
    "TargetSpec": "spec",
    "build_ground_truth": "_scm",
    "build_target_sets": "_scm",
    "derive_roles": "_scm",
    "seed_sequence_for": "_rng",
    "stable_name_hash": "_rng",
    "stream_for": "_rng",
    "stream_key": "_rng",
}


def __getattr__(name: str) -> Any:
    """Resolve a lazily-exported core type on first attribute access (PEP 562).

    Args:
        name: The attribute being looked up on this package.

    Returns:
        The requested object, imported from its home submodule.

    Raises:
        AttributeError: if the name is not part of this package's public surface.
    """
    home = _LAZY_HOME.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = __import__(f"{__name__}.{home}", fromlist=[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> Tuple[str, ...]:
    """Include the lazily-resolved names so tab-completion and `dir()` see the real surface."""
    return tuple(sorted(set(globals()) | set(_LAZY_HOME)))


# Literal list of string constants: vulture only recognises this exact form as a re-export marker,
# and mlframe.data star-imports this package, so an implicit surface would silently shadow mlframe.data.synthetic.
__all__ = [
    "CausalGraph",
    "Ceiling",
    "CeilingTarget",
    "DatasetSpec",
    "EdgeSpec",
    "FeatureRole",
    "FeatureSpec",
    "FeatureTruth",
    "GateSpec",
    "GroundTruth",
    "LatentSpec",
    "LinkSpec",
    "MIBundle",
    "MIEstimate",
    "NoiseSpec",
    "Prior",
    "RedundancyGroup",
    "TargetSet",
    "TargetSpec",
    "build_ground_truth",
    "build_target_sets",
    "derive_roles",
    "get_sapp_dataset",
    "indicator",
    "seed_sequence_for",
    "showcase_pycaret_datasets",
    "stable_name_hash",
    "stream_for",
    "stream_key",
]
