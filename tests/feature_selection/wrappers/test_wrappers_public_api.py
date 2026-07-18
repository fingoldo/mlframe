"""Public-API contract tests for ``mlframe.feature_selection.wrappers``.

These tests lock the import surface so a future refactor (removing an entry from
``__all__``, accidentally killing a ``# noqa: F401`` re-export, renaming a public
class) breaks loudly instead of silently shipping with a broken import contract.

Motivated by the 2026-05-14 cleanup which removed 4 dead imports from
``_rfecv.py`` (``random``, ``warnings``, ``Enum``, ``Leaderboard``) - this test
suite catches the inverse: someone removing an import that is NOT dead and IS
actually re-exported. Grep across the in-repo codebase did not find external
callers of the ``_rfecv``-path re-exports, but downstream consumers outside the
repo may depend on them.
"""

from __future__ import annotations

import inspect
from enum import Enum

import pytest

PUBLIC_NAMES = [
    # Permuted-y noise-floor post-hoc cut of an over-selected feature ranking (wrappers/_noise_floor.py).
    "select_features_noise_floor",
    "noise_floor_plateau",
    "OptimumSearch",
    "VotesAggregation",
    "RFECV",
    # 2026-05-28: grouped pydantic configs (post-audit ergonomics).
    "SearchConfig",
    "FIConfig",
    "RobustnessConfig",
    "split_into_train_test",
    "store_averaged_cv_scores",
    "get_feature_importances",
    "get_next_features_subset",
    "get_actual_features_ranking",
    "select_appropriate_feature_importances",
    "suppress_irritating_3rdparty_warnings",
    "make_gaussian_knockoffs",
    "knockoff_importance",
    "select_features_fdr",
]


class TestPublicAPI:
    """Top-level ``__all__`` contract."""

    def test_all_names_importable(self):
        """All names importable."""
        import mlframe.feature_selection.wrappers as pkg

        for name in PUBLIC_NAMES:
            assert hasattr(pkg, name), f"public API missing {name!r}"

    def test_all_matches_documented_set(self):
        """``__all__`` must match the documented set so ``from pkg import *`` callers
        get exactly the expected names."""
        import mlframe.feature_selection.wrappers as pkg

        assert set(pkg.__all__) == set(PUBLIC_NAMES), f"__all__ drift: pkg.__all__={set(pkg.__all__)}, expected={set(PUBLIC_NAMES)}"

    @pytest.mark.parametrize("name", PUBLIC_NAMES)
    def test_each_name_is_importable_via_from(self, name):
        """Each public name must be reachable through a ``from pkg import name``
        statement, not just via attribute access. Locks the binding in
        ``__init__.py``."""
        mod = __import__("mlframe.feature_selection.wrappers", fromlist=[name])
        obj = getattr(mod, name)
        assert obj is not None


class TestPublicAPITypes:
    """Each public name has the expected category (enum / class / callable)."""

    def test_optimum_search_is_enum(self):
        """Optimum search is enum."""
        from mlframe.feature_selection.wrappers import OptimumSearch

        assert issubclass(OptimumSearch, Enum)
        expected = {
            "ScipyLocal",
            "ScipyGlobal",
            "ModelBasedHeuristic",
            "ExhaustiveRandom",
            "ExhaustiveDichotomic",
        }
        assert {m.name for m in OptimumSearch} == expected

    def test_votes_aggregation_is_enum(self):
        """Votes aggregation is enum."""
        from mlframe.feature_selection.wrappers import VotesAggregation

        assert issubclass(VotesAggregation, Enum)
        expected = {"Minimax", "OG", "Borda", "Plurality", "Dowdall", "Copeland", "AM", "GM"}
        assert {m.name for m in VotesAggregation} == expected

    def test_rfecv_is_class_with_sklearn_contract(self):
        """Rfecv is class with sklearn contract."""
        from mlframe.feature_selection.wrappers import RFECV
        from sklearn.base import BaseEstimator, TransformerMixin

        assert inspect.isclass(RFECV)
        assert issubclass(RFECV, BaseEstimator)
        assert issubclass(RFECV, TransformerMixin)
        for method in ("fit", "transform", "get_feature_names_out", "get_params", "set_params"):
            assert callable(getattr(RFECV, method, None)), f"RFECV.{method} missing or not callable"

    @pytest.mark.parametrize(
        "name",
        [
            "split_into_train_test",
            "store_averaged_cv_scores",
            "get_feature_importances",
            "get_next_features_subset",
            "get_actual_features_ranking",
            "select_appropriate_feature_importances",
            "suppress_irritating_3rdparty_warnings",
            "make_gaussian_knockoffs",
            "knockoff_importance",
            "select_features_fdr",
        ],
    )
    def test_helper_is_callable(self, name):
        """Helper is callable."""
        import mlframe.feature_selection.wrappers as pkg

        obj = getattr(pkg, name)
        assert callable(obj), f"{name} must be callable, got {type(obj).__name__}"


class TestRfecvSubmoduleReExports:
    """The ``# noqa: F401`` directives in ``_rfecv.py`` declare that these names
    are kept as re-exports for downstream callers that import them via the
    ``_rfecv`` submodule path. Grep across the in-repo codebase did not find
    callers, but the directives signal a stability contract for external
    consumers; lock the contract here.

    If a future refactor strips one of these imports, this test fires before the
    silent break reaches downstream.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "set_numba_random_seed",
            "DummyClassifier",
            "DummyRegressor",
            "GroupShuffleSplit",
            "StratifiedShuffleSplit",
        ],
    )
    def test_rfecv_reexport_present(self, name):
        """Rfecv reexport present."""
        mod = __import__("mlframe.feature_selection.wrappers.rfecv", fromlist=[name])
        obj = getattr(mod, name, None)
        assert obj is not None, f"{name} must remain importable from mlframe.feature_selection.wrappers.rfecv (declared as # noqa: F401 re-export)"
