"""Base of the strict parameter models generated from a callable's signature: only the fields the caller wrote are forwarded.

"Set" matters: a field the caller did not write keeps the callee's own default (or the suite's own, such as the shallow-merged MRMR
defaults) instead of the model's. ``model_dump`` therefore emits only the fields that were set, so ``Config(**config.model_dump())``
rebuilds an equal config with the same set fields (a full dump would mark every field as written), and ``to_kwargs`` returns exactly
what the caller wrote.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Tuple

from pydantic import BaseModel, ConfigDict, model_serializer


class SparseParamsModel(BaseModel):
    """Frozen, unknown-key-rejecting model whose ``to_kwargs`` / ``model_dump`` carry only the explicitly set fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    #: Fields that configure the suite's wiring, not the callee's signature; left out of ``to_kwargs``.
    SUITE_ONLY: ClassVar[Tuple[str, ...]] = ()

    @model_serializer(mode="wrap")
    def _dump_only_set_fields(self, handler: Any) -> Dict[str, Any]:
        """Serialize only the explicitly set fields."""
        data = handler(self)
        explicit = self.model_fields_set
        return {k: v for k, v in data.items() if k in explicit}

    def to_kwargs(self) -> Dict[str, Any]:
        """Explicitly set callee arguments, in declaration order."""
        explicit = self.model_fields_set
        return {name: getattr(self, name) for name in type(self).model_fields if name in explicit and name not in self.SUITE_ONLY}
