"""The suffix codes of engineered basis columns (``{src}__{code}{degree}``) and the basis each stands for.

The orthogonal polynomials carry short codes (``He``, ``L``, ``T``, ``LL``); fourier, rbf, sigmoid and pade are written with their full name, which cannot clash with a short code. Parsers try
:data:`BASIS_NAME_CODES` in order (the longer ``LL`` before ``L``) and map the match with :data:`CODE_TO_BASIS`.
"""

from __future__ import annotations

import re

BASIS_NAME_CODES = ("LL", "He", "T", "L", "fourier", "rbf", "sigmoid", "pade")
CODE_TO_BASIS = {
    "He": "hermite",
    "LL": "laguerre",
    "T": "chebyshev",
    "L": "legendre",
    "fourier": "fourier",
    "rbf": "rbf",
    "sigmoid": "sigmoid",
    "pade": "pade",
}

_BASIS_SUFFIX = re.compile(r"__(?:" + "|".join(BASIS_NAME_CODES) + r")\d+$")


def split_engineered_name(name: str) -> tuple:
    """``(source, suffix)`` of an engineered name ``{source}__{code}{degree}``.

    The suffix is the trailing ``{code}{degree}`` and the source is everything before it, so a source that holds ``__`` itself (a one-hot column ``city__NY``, an engineered column such as
    ``div(exp((a__ratio_log__b)),sign(c))``) stays whole. A name without a basis suffix keeps the legacy first-``__`` split, so a caller unpacking two values fails on a name with no ``__``
    exactly as before."""
    m = _BASIS_SUFFIX.search(name)
    if m is not None:
        return name[: m.start()], name[m.start() + 2 :]
    return tuple(name.split("__", 1))


def source_of_engineered_name(name: str) -> str:
    """The source part of :func:`split_engineered_name`; the name itself when it has no ``__``."""
    m = _BASIS_SUFFIX.search(name)
    if m is not None:
        return name[: m.start()]
    return name.split("__", 1)[0] if "__" in name else name
