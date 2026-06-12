"""Auxiliary helpers not worth their own modules."""

from __future__ import annotations


import os
import random
import numpy as np
from pyutilz.numbalib import set_numba_random_seed

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def set_random_seed(seed: int = 42, set_hash_seed: bool = False, set_torch_seed: bool = False):
    """Seed everything ml-related.

    Wave 49 (2026-05-20): this function INTENTIONALLY mutates the process-global
    RNG state across random / numpy / cupy / numba / torch. It is ONLY for use
    at top-of-script / notebook setup. NEVER call this from inside fit(), predict(),
    or any library code path -- it will break determinism for sibling code in
    the same process that already seeded its own local Generator.

    For library use, prefer ``np.random.default_rng(seed)`` / ``random.Random(seed)``
    / ``torch.Generator().manual_seed(seed)`` to keep RNG state local.
    """
    random.seed(seed)

    try:
        np.random.seed(seed)
    except (TypeError, ValueError):
        pass
    try:
        import cupy as cp

        cp.random.seed(seed)
    except (ImportError, ModuleNotFoundError):
        pass
    except Exception:
        # cupy installed but CUDA backend unusable on this host -- e.g.
        # ``CURAND_STATUS_INITIALIZATION_FAILED`` when libcurand can't be
        # opened (missing CUDA libs, GPU contention, container without
        # ``/dev/nvidia*``), or ``CUDARuntimeError`` on driver mismatch.
        # The CPU half of the seed pair (random / numpy / numba above)
        # is already set; silently degrade rather than poison every
        # downstream estimator that just wants reproducible CPU paths.
        pass
    try:
        set_numba_random_seed(seed)
    except (TypeError, ValueError, RuntimeError):
        pass

    if set_hash_seed:
        os.environ["PYTHONHASHSEED"] = str(seed)

    if set_torch_seed:

        try:
            import torch  # pylint: disable=import-outside-toplevel

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # type: ignore
        except (ImportError, ModuleNotFoundError, RuntimeError):
            pass


import contextlib
import functools


def rng_hygienic_fit(fit_method: Callable) -> Callable:
    """Decorator wrapping a selector ``fit`` so it does not leave the caller's
    process-global ``numpy`` / ``random`` RNG mutated. Within-fit determinism is
    bit-identical (any internal seeding still runs); the global stream is restored
    on return. Apply to selectors whose fit internally seeds or draws from the
    global RNG (the FS global-``np.random`` bug class)."""
    @functools.wraps(fit_method)
    def _wrapped(self, *args, **kwargs):
        with preserve_global_rng():
            return fit_method(self, *args, **kwargs)
    return _wrapped


def _restore_caller_frame_columns(X, original_cols):
    """Drop any columns a fit added to the caller's pandas DataFrame in place.

    Mutate-and-restore (CLAUDE.md RAM rule): a fit that temporarily materialises
    engineered columns (hinge legs, cat crosses, target-prefix probes) into the
    working frame must NOT leave them in the caller's reference -- on a 100 GB
    frame the leaked columns are both a memory leak and a silent schema corruption
    that breaks every downstream consumer iterating ``X.columns``. We restore by
    NAME-SET so a fit that legitimately reorders is untouched; only ADDED columns
    are dropped. No ``X.copy()`` -- we mutate the same reference back to its entry
    schema.
    """
    if original_cols is None:
        return
    try:
        added = [c for c in X.columns if c not in original_cols]
    except Exception:
        return
    if added:
        try:
            X.drop(columns=added, inplace=True)
        except Exception:
            pass


def hygienic_fit(fit_method):
    """Decorator wrapping a selector ``fit`` so it leaves the caller's environment
    untouched: (1) global numpy/random RNG restored (see :func:`rng_hygienic_fit`),
    and (2) the caller's input pandas DataFrame restored to its entry column schema
    (any columns the fit materialised in place are dropped on return). Apply to
    selectors whose fit engineers columns into the working frame (MRMR's hinge /
    cat-cross / target-prefix FE), so ``X`` in == ``X`` out."""
    @functools.wraps(fit_method)
    def _wrapped(self, X, *args, **kwargs):
        original_cols = None
        try:
            import pandas as _pd
            if isinstance(X, _pd.DataFrame):
                original_cols = list(X.columns)
        except Exception:
            original_cols = None
        with preserve_global_rng():
            try:
                return fit_method(self, X, *args, **kwargs)
            finally:
                _restore_caller_frame_columns(X, original_cols)
    return _wrapped


@contextlib.contextmanager
def preserve_global_rng() -> Iterator[None]:
    """Snapshot the process-global ``numpy`` + ``random`` RNG state on entry and
    restore it on exit (success OR exception).

    Library fit paths that internally call :func:`set_random_seed` (to make any
    sub-estimator with ``random_state=None`` reproducible WITHIN the fit) must
    not leave the caller's global RNG clobbered afterwards -- that is the exact
    hygiene violation ``set_random_seed``'s own docstring warns against. Wrap the
    fit body in this context manager: the internal seeding still runs (so the
    selection output is bit-identical), but the caller's ``np.random`` /
    ``random`` stream resumes untouched. numpy + random cover essentially all
    sibling-code RNG sharing; cupy/numba/torch global state is not restored here
    (rarely shared cross-library, and snapshotting it is disproportionately
    costly).
    """
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


def get_pipeline_last_element(clf) -> object:
    for elem_name, elem in clf.named_steps.items():
        pass
    return elem


def get_full_classifier_name(clf: object) -> str:
    clf_name = type(clf).__name__
    if clf_name == "TransformedTargetRegressor":
        regressor_name = get_full_classifier_name(clf.regressor)
        if clf.transformer:
            transformer_name = type(clf.transformer).__name__
            try:
                transformer_name += " " + clf.transformer.method
            except AttributeError:
                pass
            try:
                transformer_name += " " + clf.transformer.output_distribution  # QuantileTransformer
            except AttributeError:
                pass
        else:
            try:
                transformer_name = clf.func.__name__
            except AttributeError:
                transformer_name = "func"

        full_clf_name = " -> ".join([regressor_name, transformer_name])
    elif clf_name == "Pipeline":
        elem = get_pipeline_last_element(clf)
        return f"pipe[{get_full_classifier_name(elem)}]"
    elif clf_name == "MultiOutputRegressor":
        return f"MultiOutputRegressor[{get_full_classifier_name(clf.estimator)}]"

    else:
        if "Dummy" in clf_name:
            full_clf_name = clf_name + "[" + clf.strategy + "]"
        else:
            full_clf_name = clf_name

    return full_clf_name


def is_cuda_available() -> bool:
    from numba import cuda

    return cuda.is_available()


def check_cpu_flag(flag: str = "avx2") -> bool:

    import cpuinfo

    info = cpuinfo.get_cpu_info()
    # macOS / ARM builds of py-cpuinfo can omit the "flags" key entirely
    # (x86 feature flags are not the right abstraction on Apple Silicon or
    # macOS hosts where cpuinfo cannot read /proc/cpuinfo). Treat a missing
    # flag list as "feature not present" rather than raising KeyError.
    return flag in info.get("flags", [])
