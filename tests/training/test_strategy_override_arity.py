"""A base signature narrower than its overrides makes the caller guess which subclass it is holding.

``prepare_polars_dataframe`` and ``get_classif_objective_kwargs`` each grew a keyword on some subclasses
and not on the base, so calling with that keyword worked on HGB/XGB/neural and raised TypeError on every
strategy that inherits the base implementation. The polars call site papered over it by branching on
whether a ``category_map`` existed at all, which is the call site doing the type's job.
"""

from __future__ import annotations

import inspect

import pytest

from mlframe.training.strategies.base import ModelPipelineStrategy


def _all_strategies():
    """Every concrete strategy class reachable from the base, so a new one cannot skip this."""
    seen, stack = {}, [ModelPipelineStrategy]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub.__name__ not in seen:
                seen[sub.__name__] = sub
                stack.append(sub)
    return seen


@pytest.fixture(scope="module", autouse=True)
def _import_every_strategy():
    """Subclasses only register once their module is imported."""
    import mlframe.training.strategies  # noqa: F401


@pytest.mark.parametrize("method,keyword", [("prepare_polars_dataframe", "category_map"), ("get_classif_objective_kwargs", "multilabel_config")])
def test_every_strategy_accepts_the_keyword_its_siblings_declare(method, keyword):
    """One call must be valid against the base type, not only against whichever subclass is in hand."""
    assert keyword in inspect.signature(getattr(ModelPipelineStrategy, method)).parameters, (
        f"the BASE {method} does not accept {keyword}; any strategy inheriting it raises TypeError on a call " "its siblings accept"
    )
    offenders = [name for name, cls in _all_strategies().items() if keyword not in inspect.signature(getattr(cls, method)).parameters]
    assert not offenders, f"{offenders} narrow {method} by dropping {keyword}"


def test_the_polars_prep_helper_passes_the_map_without_branching_on_it():
    """The two-arity branch was the workaround; passing None must now reach the strategy unchanged."""
    from mlframe.training.core._misc_helpers import _prep_polars_df

    seen = {}

    class _Spy(ModelPipelineStrategy):
        """Records what the helper passed through; the abstract members are inert stubs."""

        requires_encoding = False
        requires_imputation = False
        requires_scaling = False

        def cache_key(self, *args, **kwargs):
            """Unused by the helper under test."""
            return "spy"

        def prepare_polars_dataframe(self, df, cat_features, category_map=None):
            """Record the keyword and hand the frame back."""
            seen["category_map"] = category_map
            return df

    sentinel = object()
    assert _prep_polars_df(sentinel, _Spy(), [], None) is sentinel
    assert "category_map" in seen and seen["category_map"] is None
