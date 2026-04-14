"""
Root pytest fixtures shared across all test modules.
"""

import gc
import warnings

import pytest

try:
    import psutil
except ImportError:
    psutil = None


@pytest.fixture(autouse=True)
def _reset_global_rng_state():
    import random as _random
    import numpy as _np
    _random.seed(0)
    _np.random.seed(0)
    try:
        import torch as _torch
        _torch.manual_seed(0)
    except ImportError:
        pass
    yield


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Clean up memory after each test to prevent OOM issues."""
    import os

    is_main_process = os.environ.get('PYTEST_CURRENT_TEST') is not None

    if psutil is not None:
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        if is_main_process:
            print(f"\n[MEM] Before: {mem_before:.0f} MB")
    else:
        process = None
        mem_before = 0.0

    yield

    gc.collect()

    # Clear GPU memory and destroy distributed process groups
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except ImportError:
        pass
    except Exception:
        pass

    if psutil is not None and process is not None:
        mem_after = process.memory_info().rss / 1024 / 1024
        if is_main_process:
            print(f"[MEM] After: {mem_after:.0f} MB (delta: {mem_after - mem_before:+.0f} MB)")


@pytest.fixture(autouse=True)
def suppress_convergence_warnings():
    """Suppress convergence warnings during tests."""
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
        warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
        warnings.filterwarnings("ignore", message=".*Objective did not converge.*")
        yield
