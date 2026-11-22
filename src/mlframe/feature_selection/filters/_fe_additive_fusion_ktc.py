"""Host or device choice of the additive-fusion proposal (kernel-tuning entry ``fe_additive_fusion_backend``).

``propose_additive_fusions`` (numpy / njit) and ``propose_additive_fusions_gpu`` (resident binning, one float32 Gram matrix for the OLS separability test) admit the same fusions; the sweep times both on a
frame of six Hermite halves over the row axis and requires the same admitted set. Outside STRICT (which forces the device) the per-host tuning cache decides, else a row mark.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

_SWEEP_N_SAMPLES = [5_000, 20_000, 100_000, 400_000]
_HALVES = 6
_NBINS = 10
_SALT = 1
DEVICE_FALLBACK_ROWS = 10**12  # before any sweep: the host proposal (the sweep found it ahead at every size from 5000 to 400000 rows on the sm_61 host, 100000 rows: host 0.19 s, device 0.22 s)


def _fusion_inputs(dims: dict) -> tuple:
    """``(kwargs, mrmr)``: six squared-Hermite halves of independent normal columns and an 8-class target of their noisy sum."""
    import pandas as pd

    from .engineered_recipes import build_orth_univariate_recipe
    from .mrmr import MRMR

    n = int(dims["n_samples"])
    rng = np.random.default_rng(0)
    names = [chr(ord("a") + i) for i in range(_HALVES)]
    raw = {k: rng.standard_normal(n) for k in names}
    cont = {f"{k}__He2": raw[k] ** 2 - 1.0 for k in names}
    recipes = {nm: build_orth_univariate_recipe(name=nm, src_name=nm.split("__")[0], basis="hermite", degree=2) for nm in cont}
    y = sum(cont.values()) + 3.0 * rng.standard_normal(n)
    yd = pd.qcut(y, q=8, labels=False, duplicates="drop").astype(np.int64)
    kw = dict(engineered_recipes=recipes, engineered_continuous=cont, newly_engineered_names=list(cont), raw_name_set=set(names), cols=names, classes_y=yd, X=pd.DataFrame(raw, copy=False), nbins=_NBINS, seed=0)
    return (kw, MRMR(verbose=0))


def _admitted_vector(admitted: list) -> np.ndarray:
    """Comparison vector of an admitted set: the count, then the sorted names' hashes as indicators of the set."""
    return np.asarray([float(len(admitted))] + [float(sum(map(ord, a["name"])) % 100003) for a in sorted(admitted, key=lambda a: a["name"])], dtype=np.float64)  # Intentional float64: comparison vector


def _fusion_host(kw: dict, fs: Any) -> np.ndarray:
    """The numpy / njit proposal."""
    from ._fe_additive_fusion import propose_additive_fusions

    return _admitted_vector(propose_additive_fusions(fs, **kw)[0])


def _fusion_device(kw: dict, fs: Any) -> np.ndarray:
    """The resident proposal."""
    from ._fe_additive_fusion_gpu_resident import propose_additive_fusions_gpu

    return _admitted_vector(propose_additive_fusions_gpu(fs, **kw)[0])


def _fusion_sweep() -> list:
    """Time the host and the device proposal over the row axis; the admitted sets must be the same."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return cast(list, sweep_backend_grid({"host": _fusion_host, "device": _fusion_device}, {"n_samples": _SWEEP_N_SAMPLES}, _fusion_inputs, reference="host", repeats=2, equiv_rtol=0.0, equiv_atol=0.0))


from ._ktc_fallback import rows_fallback

_fallback_choice = rows_fallback(__name__, "DEVICE_FALLBACK_ROWS")

try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="fe_additive_fusion_backend",
        variant_fns=(),  # both backends are timed by the sweep callable
        tuner=_fusion_sweep,
        axes={"n_samples": list(_SWEEP_N_SAMPLES)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_additive_fusion_backend",
    )
except Exception as e:
    logger.warning("fe_additive_fusion_backend kernel_tuner registration failed: %s", e)
    _SPEC = None


def fusion_use_device(n: int) -> bool:
    """Whether the additive-fusion proposal over ``n`` rows runs resident (outside STRICT-resident, which forces it): the per-host tuning cache, else the row mark. ``MLFRAME_FE_FUSION_DEVICE=0|1`` forces a side."""
    import os

    forced = os.environ.get("MLFRAME_FE_FUSION_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    if _SPEC is None:
        return _fallback_choice(n) == "device"
    try:
        return str(_SPEC.choose(n_samples=int(n))) == "device"
    except Exception as exc:
        log_throttle(logger, "_fe_additive_fusion_ktc.fusion_use_device", logging.WARNING, "fusion_use_device: KTC choose() failed, using the row mark: %s", exc)
        return _fallback_choice(n) == "device"
