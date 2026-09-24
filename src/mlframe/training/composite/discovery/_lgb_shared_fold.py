"""One binned LightGBM dataset per (feature matrix, fold, thread), reused by every spec the rerank scores on it.

Within one base every spec's tiny model trains on the same feature rows per fold; only the label differs. The sklearn
``LGBMRegressor`` re-bins those rows on every fit, which at the tiny-model defaults (13.3k x 100, 60 trees, 15 leaves)
is 328 ms of a fit that otherwise takes about 0.9 s. Building the dataset once and swapping its label gives predictions
identical to the per-fit path (max abs diff 0.0 over 32 specs) and 1.32x on the fits.

Parity holds only when the fit uses every row of the fold: a row ``subset`` keeps the parent's bin boundaries, which a
fresh fit on fewer rows would place differently. Callers therefore use this path only for unmasked folds.

``set_label`` mutates the dataset, and the rerank scores specs from a thread pool, so the cache is per thread: each
worker builds its own copy once and reuses it for every spec it takes. Entries hold a weak reference to the feature
matrix, so a freed matrix whose ``id`` is reused can never hit a stale dataset.
"""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from typing import Any, Dict

import numpy as np

_MAX_ENTRIES = 64
"""Datasets kept across all threads; the oldest go first. Each holds one fold's binned rows only (about 1.3 MB at the defaults)."""

_CACHE: "OrderedDict[tuple, tuple[Any, Any]]" = OrderedDict()
_LOCK = threading.Lock()

_CONSTRUCT_LOCK = threading.Lock()
"""Serialises ``Dataset.construct()`` across every thread in the process.

LightGBM's binning runs in its own C++ layer and is not safe to enter from several threads at once, while training
boosters in parallel is. A production run died with an access violation and a heap corruption (0xc0000374) with 16
rerank threads inside ``Booster.update`` and 3 more inside a dataset construction at the same moment. The cache's
own lock only guards the dict, so before this the constructions ran concurrently.

Construction is a small share of the work it protects (about 328 ms against a 0.9 s fit at the tiny-model defaults)
and each thread builds its fold once, so the lock costs nothing measurable: 16 threads over the production fold shape
took 55.4 s with it against 58.9 s without (``_benchmarks/bench_lgb_shared_fold_locking.py``). Serialising
``lgb.train`` as well was measured and rejected -- 117.5 s, 1.99x, since training is where the rerank's time goes and
it is the part LightGBM runs in parallel safely.
"""


def lgb_params(*, num_leaves: int, learning_rate: float, random_state: int, deterministic: bool, num_threads: int) -> Dict[str, Any]:
    """The booster parameters ``_build_tiny_model('lgb', ...)`` hands to LightGBM through the sklearn wrapper."""
    params: Dict[str, Any] = {
        "objective": "regression",
        "num_leaves": int(num_leaves),
        "learning_rate": float(learning_rate),
        "seed": int(random_state),
        "num_threads": int(num_threads),
        "verbose": -1,
        "force_col_wise": True,
    }
    if deterministic:
        params["force_col_wise"] = False
        params["force_row_wise"] = True
        params["deterministic"] = True
    return params


def _fold_dataset(x: np.ndarray, rows: np.ndarray, params: Dict[str, Any]) -> Any:
    """This thread's constructed dataset for ``x[rows]``, built on first use."""
    import lightgbm as lgb

    key = (
        threading.get_ident(), id(x), x.shape, hash(np.ascontiguousarray(rows).tobytes()),
        tuple(sorted((k, v) for k, v in params.items() if k != "num_threads")),
    )
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and hit[0]() is x:
            _CACHE.move_to_end(key)
            return hit[1]
    # Only the binned rows are reused -- no subset, no re-construction -- so the raw feature copy is released once built.
    _raw = x[rows]  # materialised outside the lock: the copy is plain numpy and needs no serialisation
    with _CONSTRUCT_LOCK:
        ds = lgb.Dataset(_raw, label=np.zeros(rows.shape[0]), params=params, free_raw_data=True).construct()
    with _LOCK:
        _CACHE[key] = (weakref.ref(x), ds)
        while len(_CACHE) > _MAX_ENTRIES:
            _CACHE.popitem(last=False)
    return ds


def fit_on_shared_fold(x: np.ndarray, rows: np.ndarray, target: np.ndarray, *, params: Dict[str, Any], n_estimators: int) -> Any:
    """Train on every row of ``x[rows]`` with ``target`` as the label; returns the booster, which predicts like the model."""
    import lightgbm as lgb

    ds = _fold_dataset(x, rows, params)
    ds.set_label(np.asarray(target, dtype=np.float64))
    return lgb.train(params, ds, num_boost_round=int(n_estimators))
