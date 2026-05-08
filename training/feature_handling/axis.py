"""
``Axis`` enum + ``HandlerSpec`` ABC + axis registry.

Why a registry instead of a closed Literal: Round 3 future-proofing audit
F1 flagged that the previous closed enumeration of axes (``cat | text |
numeric``) would cost a 12-touchpoint refactor when image/audio/sequence
axes arrive. The registry seam lets a new axis ship as a single new
file plus one ``register_handler_spec`` call, no edits to existing code.

This module is INTENTIONALLY tiny in phase M — it defines the
extensibility contract; concrete ``TextHandlerSpec`` / ``CatHandlerSpec``
land in phase A with their full per-method TypedDict params.

# FUTURE: IMAGE, AUDIO, VIDEO, SEQUENCE axes -- see plan §F1, F23
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Pattern,
    Type,
    Union,
)

if TYPE_CHECKING:
    from pydantic import BaseModel as _BaseModel  # noqa: F401  (type-only)


class Axis(str, Enum):
    """Feature axis enum. Values are stable strings used as dict keys
    in the model-axis support matrix and as cache-key components.

    NEW values added later (image, audio, video, sequence) MUST be
    appended; never reorder or rename to keep cache invariance.
    """

    CAT = "cat"
    TEXT = "text"
    NUMERIC = "numeric"
    # FUTURE additions reserved here so future PRs are append-only:
    # IMAGE = "image"     # see plan §F1
    # AUDIO = "audio"     # see plan §F1
    # VIDEO = "video"     # see plan §F1
    # SEQUENCE = "sequence"  # see plan §F23


# `apply_to` accepts list-of-names | regex | predicate (round 3 F8).
# The predicate form takes the input column list and returns a filtered
# subset — useful for late-binding pipelines that synthesize columns
# during preprocessing.
ApplyToColumns = Union[
    List[str],
    Pattern,
    Callable[[List[str]], List[str]],
    None,
]


class HandlerSpec(ABC):
    """Abstract base for per-axis handler specifications.

    Concrete subclasses (``TextHandlerSpec``, ``CatHandlerSpec``, ...)
    are pydantic ``BaseModel`` subclasses that ALSO subclass this ABC,
    so the registry can do ``issubclass(spec_cls, HandlerSpec)``
    routing. The ABC stays import-light so registering a spec at
    module-import-time doesn't drag pydantic in.

    Subclasses MUST:
      * declare ``axis`` as a classmethod returning their ``Axis``;
      * register themselves via :func:`register_handler_spec` at
        module load;
      * declare ``apply_to`` as a pydantic field of type
        :data:`ApplyToColumns` (round-3 F8: list-of-names | regex |
        callable for late-binding);
      * declare ``group_columns: Optional[List[str]] = None`` for
        group-aware encoding (round-3 F11: lifted to ABC contract so
        symmetric across cat and text).

    NOTE: contract via docstring rather than class attributes here so
    the ABC doesn't declare instance-level defaults that pydantic
    would warn-about-shadowing in the BaseModel-subclass.
    """

    @classmethod
    @abstractmethod
    def axis(cls) -> Axis:
        """Return the axis this spec describes. Subclass override."""


_AXIS_REGISTRY: Dict[Axis, Type[HandlerSpec]] = {}


def register_handler_spec(spec_cls: Type[HandlerSpec]) -> Type[HandlerSpec]:
    """Register a concrete ``HandlerSpec`` subclass for its axis.

    Idempotent: re-registering the same class is a no-op. Registering
    a *different* class for an axis already taken raises ``ValueError``
    so stray copies don't silently override each other (e.g. a stale
    backup file accidentally picked up by an editor's reload).
    """
    if not issubclass(spec_cls, HandlerSpec):
        raise TypeError(
            f"register_handler_spec expects a HandlerSpec subclass, got {spec_cls!r}"
        )
    axis = spec_cls.axis()
    existing = _AXIS_REGISTRY.get(axis)
    if existing is not None and existing is not spec_cls:
        raise ValueError(
            f"axis {axis!r} already registered to {existing.__name__}; "
            f"cannot reassign to {spec_cls.__name__}"
        )
    _AXIS_REGISTRY[axis] = spec_cls
    return spec_cls


def get_handler_spec_for_axis(axis: Axis) -> Type[HandlerSpec]:
    """Look up the concrete spec class for an axis. Raises if unregistered."""
    if axis not in _AXIS_REGISTRY:
        raise KeyError(
            f"no HandlerSpec registered for axis {axis!r}; "
            f"registered axes: {sorted(a.value for a in _AXIS_REGISTRY)}"
        )
    return _AXIS_REGISTRY[axis]
