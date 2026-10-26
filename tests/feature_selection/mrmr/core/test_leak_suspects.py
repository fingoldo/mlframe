"""A selected column that alone almost determines the target is reported as a possible leak and is not dropped."""
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._leak_suspects import leak_score
from mlframe.feature_selection.filters.mrmr import MRMR


def _frame(n=3000, seed=0):
    """A real signal, a leakage proxy of the binary target (target plus a tiny noise) and a pure-noise column."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = (x + 0.6 * rng.standard_normal(n) > 0).astype(int)
    X = pd.DataFrame({"x_real": x, "proxy": y + 0.01 * rng.standard_normal(n), "noise0": rng.standard_normal(n)})
    return X, y


def test_leak_score_separates_a_proxy_from_a_strong_feature():
    """The proxy scores above 0.995, a strong but honest feature well below."""
    X, y = _frame()
    assert leak_score(X["proxy"].to_numpy(), y) > 0.995
    assert leak_score(X["x_real"].to_numpy(), y) < 0.95


def test_a_proxy_is_reported_and_kept():
    """The fit warns naming the proxy, records it in ``leak_suspects_`` and still selects it."""
    X, y = _frame()
    with pytest.warns(UserWarning, match="proxy"):
        model = MRMR(verbose=0, random_state=0, fe_max_steps=0).fit(X, pd.Series(y))
    assert [name for name, _score in model.leak_suspects_] == ["proxy"]
    assert "proxy" in list(model.get_feature_names_out())


def test_no_warning_without_a_proxy():
    """A frame without a near-determining column gives no leak warning and an empty record."""
    X, y = _frame()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", message=r"(?!MRMR: selected column).*")
        model = MRMR(verbose=0, random_state=0, fe_max_steps=0).fit(X[["x_real", "noise0"]], pd.Series(y))
    assert model.leak_suspects_ == []
