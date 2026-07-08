"""Public enums for the RFECV wrapper module."""

from __future__ import annotations

from enum import Enum


class OptimumSearch(str, Enum):
    """Strategy RFECV uses to search for the feature-count optimum along the elimination curve."""

    ScipyLocal = "ScipyLocal"  # Brent
    ScipyGlobal = "ScipyGlobal"  # direct, diff evol, shgo
    ModelBasedHeuristic = "ModelBasedHeuristic"  # GaussianProcess or Catboost with uncertainty, or quantile regression
    ExhaustiveRandom = "ExhaustiveRandom"
    ExhaustiveDichotomic = "ExhaustiveDichotomic"


class VotesAggregation(str, Enum):
    """Voting rule used to aggregate per-fold feature-importance rankings into one consensus ranking."""

    Minimax = "Minimax"
    OG = "OG"
    Borda = "Borda"
    Plurality = "Plurality"
    Dowdall = "Dowdall"
    Copeland = "Copeland"
    AM = "AM"
    GM = "GM"
