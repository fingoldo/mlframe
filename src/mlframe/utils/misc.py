"""Auxiliary helpers not worth their own modules."""

from __future__ import annotations


import os
import random
import numpy as np
from pyutilz.numbalib import set_numba_random_seed


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
