"""Shim: residual-normality tests moved to ``pyutilz.stats.normality``.

Kept as a thin re-export so existing import paths inside mlframe stay
green. New code should import from ``pyutilz.stats.normality`` directly.
"""
from pyutilz.stats.normality import (
    anderson_darling_normal as _anderson_darling_normal,
    dagostino_k2 as _dagostino_k2,
    normality_verdict,
)

__all__ = [
    "_dagostino_k2",
    "_anderson_darling_normal",
    "normality_verdict",
]
