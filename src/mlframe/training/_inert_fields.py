"""A config field that is accepted but read by nothing says so when someone sets it.

A knob only exists to change something. When the wiring behind it was never finished, or moved to another field, the
config still validates the value, still shows it in a repr and still round-trips it through a saved bundle, so the user
has every reason to believe it took effect. The composite config already warned for its two such fields; this is the same
mechanism for every other one, declared next to the class rather than in a test's ledger.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, ClassVar, Optional

from pydantic import model_validator

logger = logging.getLogger(__name__)

__all__ = ["InertFieldsWarningMixin"]


class InertFieldsWarningMixin:
    """Mixin for a pydantic config: declare ``INERT_FIELDS = {field: why}`` and a non-default value warns.

    The value is still accepted, so a saved config from an older build keeps loading; the user just learns that the
    setting does nothing. A field whose default is a mutable container compares by value, as pydantic stores it.
    """

    INERT_FIELDS: ClassVar[dict[str, str]] = {}
    # Set on a class the suite never reads at all: constructing it warns (FutureWarning, shown by default) with this hint.
    UNCONSUMED_CLASS_HINT: ClassVar[Optional[str]] = None

    @model_validator(mode="after")
    def _warn_on_inert_field_values(self) -> Any:
        """Warn once per construction of an unconsumed class, and once per inert field set away from its default."""
        hint = type(self).UNCONSUMED_CLASS_HINT
        if hint:
            warnings.warn(f"{type(self).__name__} is not read by the training suite and will be removed: {hint}", FutureWarning, stacklevel=2)
        fields = getattr(type(self), "model_fields", {})
        for name, why in type(self).INERT_FIELDS.items():
            field = fields.get(name)
            if field is None:
                continue
            value = getattr(self, name, None)
            default = field.default
            try:
                unchanged = value == default or (value is None and default is None)
            except Exception:  # pragma: no cover - exotic comparison: say so, then treat the field as set (the warning below fires)
                logger.warning("%s.%s: could not compare the value with its default; treating it as set.", type(self).__name__, name)
                unchanged = False
            if unchanged:
                continue
            warnings.warn(
                f"{type(self).__name__}.{name}={value!r} has no effect: {why}",
                stacklevel=2,
            )
        return self
