"""The gradient optimizer reports a missing PyTorch as an explicit ImportError, not a raw import failure from inside the backend."""

from types import SimpleNamespace

import numpy as np
import pytest


def test_gradient_optimizer_without_torch_raises_clear_import_error(monkeypatch):
    """With PyTorch unavailable, selecting optimizer='gradient' must fail with a message naming PyTorch."""
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_gradient
    from mlframe.feature_selection.shap_proxied_fs._shap_proxied_fit_search import run_search

    monkeypatch.setattr(_shap_proxy_gradient, "torch_available", lambda: False)
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(60, 4))
    base = rng.normal(size=60)
    y = rng.normal(size=60)
    stub = SimpleNamespace(classification=False, metric="rmse", random_state=0, top_n=3, min_features=1, max_features=3)

    with pytest.raises(ImportError, match="PyTorch"):
        run_search(stub, "gradient", phi, base, y)
