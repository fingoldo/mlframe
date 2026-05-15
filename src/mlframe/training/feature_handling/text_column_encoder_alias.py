"""Tiny indirection module to break a circular import.

``apply.py`` needs a TextColumnEncoder builder; ``text_encoder.py``
imports handlers.py params types; if ``apply.py`` imported
text_encoder.py directly we'd risk cycles when text_encoder later
imports apply for tests. The function lives here to be safe.
"""

from __future__ import annotations

from typing import Union

from mlframe.training.feature_handling.handlers import (
    HashingParams,
    TfidfParams,
)
from mlframe.training.feature_handling.text_encoder import TextColumnEncoder


def build_text_encoder(
    column: str,
    params: Union[TfidfParams, HashingParams],
) -> TextColumnEncoder:
    return TextColumnEncoder(column=column, params=params)


__all__ = ["build_text_encoder"]
