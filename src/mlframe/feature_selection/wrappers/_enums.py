"""Public enums for the RFECV wrapper module."""

from __future__ import annotations

from enum import Enum


class OptimumSearch(str, Enum):
    """Strategy RFECV uses to search for the feature-count optimum along the elimination curve."""

    # W8: _suggest_scipy_local/_suggest_scipy_global (in _helpers.py) are thin aliases of
    # _suggest_dichotomic, not Brent/DIRECT/differential-evolution/SHGO -- a deliberate, well-reasoned
    # simplification (the argmax of a piecewise-linear interpolant always lands on an already-evaluated
    # breakpoint, so a real scipy optimizer call was redundant with dichotomic search), but these
    # per-member comments still promised the original scipy backends they no longer call. See
    # _helpers.py's docstring on those two functions for the full rationale.
    ScipyLocal = "ScipyLocal"  # alias of dichotomic search (no longer calls scipy.optimize)
    ScipyGlobal = "ScipyGlobal"  # alias of dichotomic search (no longer calls scipy.optimize)
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
