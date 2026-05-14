"""Neural-specific pytest fixtures.

PyTorch + Lightning carry process-global state (default RNG, default dtype, cuDNN flags, autograd anomaly mode, Lightning's own seeded RNG). When pytest-randomly shuffles test order, that state leaks between tests and produces order-dependent failures (e.g. test_classification_with_regularization passes alone but fails after some sibling has flipped a global flag). Reset before every neural test.
"""

import pytest


@pytest.fixture(autouse=True)
def _reset_torch_lightning_global_state():
    """Reset PyTorch + Lightning process-global RNG and determinism flags before each test."""
    import numpy as np
    import torch

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    try:
        import lightning
        lightning.seed_everything(42, workers=True, verbose=False)
    except ImportError:
        pass

    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(False)  # tests rely on non-strict mode
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    yield
