"""Shared state of one ``MRMR.fit``: the values every stage of ``_fit_impl`` reads, held in two namespaces.

``_fit_impl`` used to keep ~60 of these as plain locals, so every piece carved out of it needed them all as parameters.
They are grouped instead: ``FERecipes`` holds the per-family recipe registries the feature-engineering stages fill and
``transform`` replays; ``FEParams`` holds the feature-engineering parameters resolved once at the start of the fit.
"""

from __future__ import annotations

from types import SimpleNamespace


class FERecipes(SimpleNamespace):
    """Per-family recipe registries of one fit (``recipes.hybrid_orth``, ``recipes.kfold_te``, ...), each a dict the
    family's FE stage fills and ``transform`` replays."""


class FEParams(SimpleNamespace):
    """Feature-engineering parameters resolved at the start of one fit (``fe.max_steps``, ``fe.min_pair_mi``, ...)."""
