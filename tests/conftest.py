"""
Root pytest fixtures shared across all test modules.
"""

import gc
import warnings

import pytest


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Clean up memory after each test to prevent OOM issues."""
    import os
    import psutil

    is_main_process = os.environ.get('PYTEST_CURRENT_TEST') is not None

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    if is_main_process:
        print(f"\n[MEM] Before: {mem_before:.0f} MB")

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

    mem_after = process.memory_info().rss / 1024 / 1024
    if is_main_process:
        print(f"[MEM] After: {mem_after:.0f} MB (delta: {mem_after - mem_before:+.0f} MB)")


@pytest.fixture(autouse=True)
def suppress_convergence_warnings():
    """Suppress convergence warnings during tests."""
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
    warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
    warnings.filterwarnings("ignore", message=".*Objective did not converge.*")
    yield
    warnings.resetwarnings()
