"""
Root pytest fixtures shared across all test modules.
"""

import gc
import os
import warnings

import pytest

try:
    import psutil
except ImportError:
    psutil = None


# ---------------------------------------------------------------------------
# Fast-mode support
#
# Enable via `pytest --fast` or `MLFRAME_FAST=1 pytest`.
#
# Parametrized tests that cover many equivalent variants (10 scalers, every
# dim-reducer, every optimizer) can call `fast_subset(values)` when building
# their `@pytest.mark.parametrize` lists. In fast mode this returns a single
# representative value, so the whole code path is still exercised but runtime
# stays short. Outside fast mode it's an identity function.
#
# parametrize decorators run at import time (before fixtures), so fast mode is
# keyed off the env var, not a pytest fixture. The CLI flag just sets the env
# var in pytest_configure.
# ---------------------------------------------------------------------------

_FAST_ENV = "MLFRAME_FAST"


def is_fast_mode() -> bool:
    """Return True when tests should run in reduced-variant fast mode."""
    return os.environ.get(_FAST_ENV, "").strip() not in ("", "0", "false", "False")


def fast_subset(values, *, representative=None, keep: int = 1):
    """In fast mode return a single representative; otherwise return values unchanged.

    `values` may be any iterable of parametrize arguments (including
    `pytest.param(...)` entries). `representative` picks a specific entry when
    the first element isn't the desired one; if None, the first `keep` entries
    are kept.
    """
    if not is_fast_mode():
        return list(values)
    values = list(values)
    if not values:
        return values
    if representative is not None:
        for v in values:
            candidate = v.values[0] if hasattr(v, "values") else v
            if candidate == representative:
                return [v]
        # fall through if not found
    return values[:keep]


def pytest_addoption(parser):
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Fast mode: parametrized tests run one representative variant per group "
             f"(also enabled by {_FAST_ENV}=1).",
    )


def pytest_configure(config):
    if config.getoption("--fast"):
        os.environ[_FAST_ENV] = "1"
    config.addinivalue_line(
        "markers",
        "fast_only: test only runs in fast mode (smoke-style).",
    )
    config.addinivalue_line(
        "markers",
        "slow_only: test is skipped in fast mode.",
    )


def pytest_collection_modifyitems(config, items):
    if not is_fast_mode():
        return
    skip_slow = pytest.mark.skip(reason="skipped in --fast mode")
    for item in items:
        if "slow_only" in item.keywords or "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def fast_mode() -> bool:
    """Session-scoped boolean exposing fast-mode state to tests."""
    return is_fast_mode()


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
