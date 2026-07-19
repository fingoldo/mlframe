"""Neural-specific pytest fixtures.

PyTorch + Lightning carry process-global state (default RNG, default dtype, cuDNN
flags, autograd anomaly mode, Lightning's own seeded RNG). When pytest-randomly
shuffles test order, that state leaks between tests and produces order-dependent
failures (e.g. test_classification_with_regularization passes alone but fails
after some sibling has flipped a global flag). Reset before every neural test.

Module-level `pytest.importorskip("torch")` skips the entire neural test cluster
when torch is not installed; individual test files can rely on torch being
available without re-asserting. ``lightning`` is also a hard requirement -- the
``mlframe.training.neural`` package's top-level symbols (callbacks, MLP
estimators) subclass Lightning classes, so importing the module without
Lightning installed raises at collection time. Skip the entire neural test
cluster on shards (e.g. sklearn-matrix) that don't install the neural extras.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")


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

    np.random.default_rng(42)
    # The neural test bodies expect a deterministic global numpy RNG for parts of
    # torch/lightning that still pull from `np.random`. Seed via the default
    # bit-generator path; equivalent to `np.random.seed(42)` but using the
    # modern API so it doesn't trigger lint warnings in the file.
    np.random.default_rng(42)
    np.random.seed(42)
    try:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except RuntimeError:
        # A prior test that crashed mid-kernel or exhausted GPU memory can leave the CUDA
        # context corrupted -- the next reseed then raises "CUDA error: an illegal memory
        # access was encountered" and, since this fixture is autouse, POISONS every downstream
        # neural test's setup phase (hundreds of unrelated "ERROR at setup" in one run).
        # Mirrors tests/conftest.py's _reset_global_rng_state guard for the same corruption
        # class -- the CPU-side seeds above already ran.
        pass

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
        # On lightning < 2.2 (no verbose kwarg) seed_everything still emits
        # ``INFO: Seed set to 42`` via the ``lightning.fabric.utilities.seed``
        # logger - one line per test. Temporarily raise that logger's level
        # to WARNING so the test log isn't flooded with one redundant info
        # line per case. Restore on the way out.
        import logging as _lg

        _seed_logger = _lg.getLogger("lightning.fabric.utilities.seed")
        _seed_prev_level = _seed_logger.level
        _seed_logger.setLevel(_lg.WARNING)
        try:
            lightning.seed_everything(42, **_kw)
        finally:
            _seed_logger.setLevel(_seed_prev_level)
    except (ImportError, RuntimeError):
        # RuntimeError: lightning.seed_everything also reseeds torch.cuda internally, hitting
        # the same corrupted-context failure mode as the raw torch seeding above.
        pass

    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(False)  # tests rely on non-strict mode
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    yield
