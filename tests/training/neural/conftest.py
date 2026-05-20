"""Neural-specific pytest fixtures.

PyTorch + Lightning carry process-global state (default RNG, default dtype, cuDNN
flags, autograd anomaly mode, Lightning's own seeded RNG). When pytest-randomly
shuffles test order, that state leaks between tests and produces order-dependent
failures (e.g. test_classification_with_regularization passes alone but fails
after some sibling has flipped a global flag). Reset before every neural test.

Module-level `pytest.importorskip("torch")` skips the entire neural test cluster
when torch is not installed; individual test files can rely on torch being
available without re-asserting.
"""

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _reset_torch_lightning_global_state():
    """Reset PyTorch + Lightning process-global RNG and determinism flags before each test.

    Note: root tests/conftest.py autouse `_reset_global_rng_state` re-seeds numpy
    to 0 before each test; this fixture then overrides with seed=42 for the
    neural cluster. Order is deterministic because pytest runs fixtures in the
    order they're declared; the inner-most (this) wins by being declared closer
    to the test.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    # The neural test bodies expect a deterministic global numpy RNG for parts of
    # torch/lightning that still pull from `np.random`. Seed via the default
    # bit-generator path; equivalent to `np.random.seed(42)` but using the
    # modern API so it doesn't trigger lint warnings in the file.
    np.random.default_rng(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    try:
        import lightning
        # ``verbose`` kwarg was added in lightning 2.2; older installs (and
        # the pytorch_lightning compat shim) raise TypeError on it. Probe
        # the signature once per call and pass only kwargs the installed
        # version supports.
        import inspect
        _kw = {"workers": True}
        try:
            _sig = inspect.signature(lightning.seed_everything)
            if "verbose" in _sig.parameters:
                _kw["verbose"] = False
        except (TypeError, ValueError):
            pass
        lightning.seed_everything(42, **_kw)
    except ImportError:
        pass

    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(False)  # tests rely on non-strict mode
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    yield
