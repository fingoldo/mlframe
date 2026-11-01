"""The suffix codes of engineered basis columns (``{src}__{code}{degree}``) and the basis each stands for.

The orthogonal polynomials carry short codes (``He``, ``L``, ``T``, ``LL``); fourier, rbf, sigmoid and pade are written with their full name, which cannot clash with a short code. Parsers try
:data:`BASIS_NAME_CODES` in order (the longer ``LL`` before ``L``) and map the match with :data:`CODE_TO_BASIS`.
"""

from __future__ import annotations

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
