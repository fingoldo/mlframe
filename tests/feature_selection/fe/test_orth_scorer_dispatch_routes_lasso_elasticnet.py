"""The MRMR default-scorer dispatcher routes ``lasso`` and ``elasticnet`` to their own recipe builders.

Both builders are reached only through ``_dispatch_default_scorer``'s ``from ... import ... as _fn`` branches. The
uncalled-functions gate keeps one alias per file, so it cannot see these call sites; this test pins the routing
directly by replacing each builder with a spy and checking that the dispatcher reaches exactly that builder.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

_HELPERS = "mlframe.feature_selection.filters._mrmr_fit_impl._helpers"


@pytest.mark.parametrize(
    ("scorer", "module", "builder"),
    [
        ("lasso", "mlframe.feature_selection.filters._orthogonal_lasso_fe", "hybrid_orth_mi_lasso_fe_with_recipes"),
        ("elasticnet", "mlframe.feature_selection.filters._orthogonal_elasticnet_fe", "hybrid_orth_mi_elasticnet_fe_with_recipes"),
    ],
)
def test_dispatch_default_scorer_routes_to_named_builder(monkeypatch, scorer, module, builder):
    """The dispatcher calls the builder named for the scorer, with the fit arguments unchanged."""
    helpers = importlib.import_module(_HELPERS)
    target = importlib.import_module(module)
    sentinel = object()
    calls = []

    def spy(X, y, *, cols, degrees, basis, top_k):
        """Record the keyword arguments the dispatcher passes and return a sentinel."""
        calls.append({"cols": cols, "degrees": degrees, "basis": basis, "top_k": top_k})
        return sentinel

    monkeypatch.setattr(target, builder, spy)
    X = pd.DataFrame({"a": np.linspace(0.0, 1.0, 40), "b": np.linspace(1.0, 2.0, 40)})
    y = np.linspace(0.0, 1.0, 40)

    out = helpers._dispatch_default_scorer(scorer, X=X, y=y, cols=["a", "b"], degrees=(2,), basis="legendre", top_k=3)

    assert out is sentinel
    assert calls == [{"cols": ["a", "b"], "degrees": (2,), "basis": "legendre", "top_k": 3}]
