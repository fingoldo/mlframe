"""Param-Oracle: a universal adaptive-dispatch / learning-to-optimize cache.

Meta-learning framing
---------------------
Almost every hot decision in this codebase -- which MI scorer to run
(``plug_in`` vs ``cmim``), which CUDA kernel variant, which block size,
which FE recipe -- has an *input-dependent* optimum. The right choice is a
function of the *shape and statistics* of the data, not a constant you can
hardcode in source. The classic fix is to benchmark once on the dev box and
bake a threshold; that threshold is wrong on every other machine and every
other dataset distribution.

Param-Oracle generalises that pattern into a small "learning to optimize"
loop. It:

1. **Fingerprints** an incoming call *cheaply and statistically* (n, p,
   dtype, sparsity, skew/kurtosis, cardinality, correlation) -- NEVER the
   raw data. The fingerprint is the feature vector of a tiny meta-model.
2. **Sweeps** the declared parameter space (in ``benchmark`` mode), timing
   and memory-profiling each combo, recording the objective.
3. **Recommends** the empirically-best parameter combo for a *new*
   fingerprint via exact-bucket match -> k-NN in fingerprint space ->
   global best -> caller default, gated by a confidence threshold.
4. Persists everything as an **append-only, host-keyed, stat-only** store on
   disk, so the meta-model gets better every run, across processes and
   across sessions.

Relationship to ``pyutilz.performance.kernel_tuning.cache``
------------------------------------------------------
This is an **additive sibling**, NOT a replacement, of
``pyutilz.performance.kernel_tuning.cache.KernelTuningCache``. We deliberately
*reuse* its proven patterns rather than reinvent them:

* ``hw_fingerprint()`` -- the exact same per-host key (CPU+GPU+cc) keys our
  rows, so a tuning learned on one box never pollutes another.
* ``cache_dir()`` -- same ``$PYUTILZ_KERNEL_CACHE_DIR`` / ``~/.pyutilz``
  on-disk layout convention (Param-Oracle uses a ``param_oracle/``
  subdir).
* **Log-scale bucketing** of continuous size axes -- KernelTuningCache
  buckets via ``axis_N_max`` region caps; we bucket via
  ``round(log10(max(v,1))*2)/2`` so n=1000 and n=1050 collapse to one
  region. Same intent (size-stable lookup), continuous form.
* **Atomic, concurrency-safe writes** -- KernelTuningCache merges under a
  ``filelock`` advisory lock around an atomic rename; we mirror that with
  per-process temp shards + an atomic filelock-guarded merge into a single
  parquet store.

Where it differs: KernelTuningCache stores a hand-emitted *region table* of
discrete ``(variant, block_size)`` choices keyed by integer size axes, and
returns the first matching region. Param-Oracle stores *raw observations*
(every benchmarked combo, with its objective and obs-count), aggregates
them by median, and does *continuous k-NN* over a richer statistical
fingerprint -- it learns the mapping rather than being told the regions.

Consumer roadmap
----------------
* **scorer-selection** (today's demo): wrap the MI-scorer choice; learn
  ``cmim`` wins on redundant-signal fingerprints, ``plug_in`` on clean ones.
* **Meta FE-recommender** (L99): wrap the FE-transformer shortlist; learn
  which engineered recipe pays off for which dataset fingerprint, turning
  the research-only transformer zoo into an auto-recommended pipeline.
* **HPO warm-start**: seed hyper-parameter search from the oracle's
  per-fingerprint best instead of a cold random start.

KEY CONSTRAINT: the store is **stat-only**. No raw arrays / DataFrames ever
touch disk -- only scalar fingerprint stats, the param combo, and the
objective. This keeps the store tiny, privacy-safe, and portable.
"""
from __future__ import annotations

import itertools
import logging
import math
import os
import random
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import orjson

logger = logging.getLogger(__name__)


# Continuous fingerprint dimensions used for k-NN distance. Categorical /
# string dims (dtype_kind) are matched exactly, not by euclidean distance.
_CONTINUOUS_FP_DIMS = (
    "n",
    "p",
    "sparsity",
    "mean_abs_skew",
    "mean_kurtosis",
    "cardinality_mean",
    "mean_abs_corr",
)

# Dims that span orders of magnitude (sizes / counts) use LOG bucketing.
# Dims bounded in [0, 1] (fractions / correlations) would collapse to a
# single bucket under log10 (everything < 1 -> 0.0), so they use LINEAR
# bucketing at 0.1 resolution instead. Unbounded-but-not-size dims
# (skew, excess kurtosis) also use linear bucketing since they straddle 0.
_LOG_BUCKET_DIMS = ("n", "p", "cardinality_mean")
_LINEAR_BUCKET_DIMS = ("sparsity", "mean_abs_corr", "mean_abs_skew", "mean_kurtosis")


# ---------------------------------------------------------------------------
# Host key + store path (reuse pyutilz.performance.kernel_tuning.cache patterns)
# ---------------------------------------------------------------------------

def _host_key() -> str:
    """Per-host key. Reuses ``kernel_tuning_cache.hw_fingerprint`` when
    pyutilz is importable (same CPU+GPU+cc string that keys the kernel
    cache); falls back to the node name otherwise so the oracle still
    works without pyutilz."""
    try:
        from pyutilz.performance.kernel_tuning.cache import hw_fingerprint
        return hw_fingerprint()
    except Exception:
        import platform
        return f"node_{platform.node() or 'unknown'}"


def default_store_dir() -> str:
    """Resolve the Param-Oracle store directory.

    Mirrors ``kernel_tuning_cache.cache_dir`` layout: honours
    ``$PYUTILZ_KERNEL_CACHE_DIR`` (then ``~/.pyutilz``) and lives in a
    ``param_oracle/`` subdir so it sits beside, but never inside, the
    kernel-tuning JSONs.
    """
    override = os.environ.get("PYUTILZ_KERNEL_CACHE_DIR", "").strip()
    base = override if override else os.path.join(os.path.expanduser("~"), ".pyutilz")
    path = os.path.join(base, "param_oracle")
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Fingerprint helpers (stat-only)
# ---------------------------------------------------------------------------

def log_bucket(v: float) -> float:
    """Log-scale bucketiser for size-like dims (n, p, cardinality).

    ``round(log10(max(v, 1)) * 2) / 2`` -- half-decade buckets. Maps
    n=1000 and n=1050 to the same bucket (3.0), which is the size-stability
    property KernelTuningCache gets from its ``axis_N_max`` region caps.

    NOTE: only correct for dims that span orders of magnitude. Bounded
    fractional dims (sparsity, correlation, all in [0, 1]) collapse to 0.0
    here -- use :func:`linear_bucket` for those.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return round(math.log10(max(abs(f), 1.0)) * 2.0) / 2.0


def linear_bucket(v: float, resolution: float = 0.1) -> float:
    """Linear bucketiser for bounded / signed dims (correlation, sparsity,
    skew, excess kurtosis) where log10 would destroy resolution.

    Rounds to the nearest ``resolution`` step. A redundant dataset
    (corr~0.99 -> 1.0) and a clean one (corr~0.02 -> 0.0) land in DISTINCT
    buckets, which is exactly what the scorer-selection demo needs.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return round(f / resolution) * resolution


def _as_2d_numeric(obj: Any):
    """Best-effort coercion of a DataFrame / Series / ndarray-like to a 2D
    numpy float array plus its raw column count. Returns (arr2d, n, p,
    dtype_kind) or None if the object is not array-like."""
    import numpy as np

    # polars / pandas DataFrame -> numpy
    arr = None
    dtype_kind = "f"
    n = p = 0
    # Duck-type pandas/polars without importing them.
    if hasattr(obj, "to_numpy") and hasattr(obj, "shape"):
        try:
            raw = obj.to_numpy()
        except Exception:
            raw = None
        if raw is not None:
            arr = raw
    if arr is None:
        if isinstance(obj, np.ndarray):
            arr = obj
        else:
            try:
                arr = np.asarray(obj)
            except Exception:
                return None
    if arr.dtype == object:
        # object arrays: treat as categorical-ish, don't try numeric stats.
        dtype_kind = "O"
    else:
        dtype_kind = arr.dtype.kind
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        n = int(arr.shape[0])
        p = 1
        arr2d = arr.reshape(-1, 1)
    else:
        n = int(arr.shape[0])
        p = int(arr.shape[1]) if arr.ndim >= 2 else 1
        arr2d = arr.reshape(n, -1)
    return arr2d, n, p, dtype_kind


def default_fingerprint(args: Sequence[Any], kwargs: Mapping[str, Any]) -> dict:
    """Stat-only fingerprint of the first array/DataFrame-like positional or
    keyword argument.

    Returns a dict of scalars ONLY -- never any raw data::

        {n, p, dtype_kind, sparsity, mean_abs_skew, mean_kurtosis,
         cardinality_mean, mean_abs_corr}

    Robust to non-numeric / object columns, NaNs, and tiny inputs. Used as
    the default ``fingerprint_fn``; callers with domain knowledge can pass
    their own.
    """
    import numpy as np

    candidate = None
    for a in list(args) + list(kwargs.values()):
        info = _as_2d_numeric(a)
        if info is not None and info[1] > 0:
            candidate = info
            break

    if candidate is None:
        return {
            "n": 0, "p": 0, "dtype_kind": "?", "sparsity": 0.0,
            "mean_abs_skew": 0.0, "mean_kurtosis": 0.0,
            "cardinality_mean": 0.0, "mean_abs_corr": 0.0,
        }

    arr2d, n, p, dtype_kind = candidate

    if dtype_kind == "O":
        # Object/categorical: only cheap structural stats.
        try:
            card = float(np.mean([len(np.unique(arr2d[:, j].astype(str)))
                                  for j in range(arr2d.shape[1])]))
        except Exception:
            card = 0.0
        return {
            "n": n, "p": p, "dtype_kind": "O", "sparsity": 0.0,
            "mean_abs_skew": 0.0, "mean_kurtosis": 0.0,
            "cardinality_mean": card, "mean_abs_corr": 0.0,
        }

    a = arr2d.astype(np.float64, copy=False)
    finite = np.isfinite(a)
    total = a.size or 1
    n_zero = int(np.sum(finite & (a == 0.0)))
    n_nan = int(np.sum(~finite))
    sparsity = float((n_zero + n_nan) / total)

    # Per-column moments, NaN-robust, computed VECTORISED across all columns
    # at once (a per-column Python loop was the cProfile hotspot). Columns
    # with < 3 finite values or ~zero variance contribute 0 to skew/kurtosis.
    finite_mask = np.isfinite(a)
    n_finite = finite_mask.sum(axis=0)  # per-column finite count
    a0 = np.where(finite_mask, a, 0.0)  # NaNs -> 0 so they don't poison sums
    safe_cnt = np.where(n_finite > 0, n_finite, 1)
    col_mean = a0.sum(axis=0) / safe_cnt
    dev = np.where(finite_mask, a - col_mean, 0.0)
    var = (dev ** 2).sum(axis=0) / safe_cnt
    sd = np.sqrt(var)
    valid = (n_finite >= 3) & (sd > 1e-12)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    z = dev / sd_safe
    skew_per_col = (z ** 3).sum(axis=0) / safe_cnt
    kurt_per_col = (z ** 4).sum(axis=0) / safe_cnt - 3.0  # excess kurtosis
    if valid.any():
        mean_abs_skew = float(np.mean(np.abs(skew_per_col[valid])))
        mean_kurtosis = float(np.mean(kurt_per_col[valid]))
    else:
        mean_abs_skew = 0.0
        mean_kurtosis = 0.0

    # Cardinality stays per-column (np.unique sorts each column); only count
    # columns with >= 3 finite values so degenerate cols don't skew the mean.
    cards = []
    for j in range(a.shape[1]):
        col = a[:, j]
        col = col[finite_mask[:, j]]
        if col.size < 3:
            continue
        cards.append(float(np.unique(col).size))
    cardinality_mean = float(np.mean(cards)) if cards else 0.0

    # Mean absolute off-diagonal correlation (redundancy signal). Guard for
    # p==1 and degenerate columns.
    mean_abs_corr = 0.0
    if a.shape[1] >= 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            filled = np.where(np.isfinite(a), a, np.nan)
            # Use pairwise complete via column-mean imputation (cheap, stat-only).
            col_means = np.nanmean(filled, axis=0)
            col_means = np.where(np.isfinite(col_means), col_means, 0.0)
            inds = np.where(np.isfinite(filled), filled, col_means)
            try:
                corr = np.corrcoef(inds, rowvar=False)
                if corr.ndim == 2 and corr.shape[0] >= 2:
                    iu = np.triu_indices_from(corr, k=1)
                    vals = corr[iu]
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        mean_abs_corr = float(np.mean(np.abs(vals)))
            except Exception:
                mean_abs_corr = 0.0

    return {
        "n": n,
        "p": p,
        "dtype_kind": str(dtype_kind),
        "sparsity": sparsity,
        "mean_abs_skew": mean_abs_skew,
        "mean_kurtosis": mean_kurtosis,
        "cardinality_mean": cardinality_mean,
        "mean_abs_corr": mean_abs_corr,
    }


def bucketize_fingerprint(fp: Mapping[str, Any]) -> dict:
    """Map a raw fingerprint dict to its log-bucketed form. Continuous dims
    are log-bucketed; categorical (dtype_kind) is passed through verbatim.
    This is the key used for exact-match lookup and persistence."""
    out: dict[str, Any] = {}
    for k, v in fp.items():
        if k == "dtype_kind":
            out[k] = str(v)
        elif k in _LOG_BUCKET_DIMS:
            out[k] = log_bucket(v)
        elif k in _LINEAR_BUCKET_DIMS:
            out[k] = round(linear_bucket(v), 6)
        else:
            # Unknown extra dim: linear-bucket if numeric (safe for both
            # bounded and size-ish unknowns at 0.1 res), else pass through.
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = round(linear_bucket(v), 6)
            else:
                out[k] = v
    return out


# ---------------------------------------------------------------------------
# Memory probe (best-effort, for objective rss_delta_mb)
# ---------------------------------------------------------------------------

def _rss_mb() -> Optional[float]:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Persistence (append-only, stat-only, concurrency-safe)
# ---------------------------------------------------------------------------

from ._param_oracle_store import (  # noqa: F401,E402
    SCHEMA_VERSION,
    _ParquetStore,
    _STORE_COLUMNS,
    _median,
    _stable_json,
    stable_json,
)


# ---------------------------------------------------------------------------
# ParamOracle
# ---------------------------------------------------------------------------

class ParamOracle:
    """Universal adaptive-dispatch / learning-to-optimize cache.

    Parameters
    ----------
    store_path:
        Path to the append-only parquet store. Defaults under
        :func:`default_store_dir` if a bare filename is given.
    fingerprint_fn:
        ``(args, kwargs) -> dict[str, scalar]``. Defaults to
        :func:`default_fingerprint`. MUST return scalars only (stat-only
        constraint).
    objective_fn:
        ``(output, elapsed_s, rss_delta_mb) -> dict[str, float]``. Declares
        the metrics to record. Defaults to ``{"elapsed_s": elapsed_s,
        "rss_delta_mb": rss_delta_mb}``.
    param_space:
        ``dict[str, list]`` -- the sweep grid (itertools.product over it).
    mode:
        ``"benchmark"`` (run every combo, record each), ``"inference"``
        (recommend only, no sweep), or ``"hybrid"`` (epsilon-greedy:
        exploit best w.p. 1-epsilon, explore a random combo w.p. epsilon).
    minimize / maximize:
        Name of the objective metric to optimise. Exactly one of the two
        should be set; ``minimize`` wins if both/neither given (safe
        default for the common ``elapsed_s`` case).
    epsilon:
        Exploration probability for ``hybrid`` mode.
    min_observations:
        Confidence gate -- a recommendation is only trusted if the chosen
        combo has at least this many observations; otherwise we fall back
        (k-NN -> global best -> caller default).
    """

    def __init__(
        self,
        store_path: str,
        fingerprint_fn: Optional[Callable[[Sequence[Any], Mapping[str, Any]], dict]] = None,
        objective_fn: Optional[Callable[[Any, float, Optional[float]], dict]] = None,
        *,
        param_space: Mapping[str, Sequence[Any]],
        mode: str = "hybrid",
        minimize: Optional[str] = None,
        maximize: Optional[str] = None,
        epsilon: float = 0.1,
        min_observations: int = 3,
        rng: Optional[random.Random] = None,
    ):
        if os.path.basename(store_path) == store_path:
            store_path = os.path.join(default_store_dir(), store_path)
        self.store = _ParquetStore(store_path)
        self.fingerprint_fn = fingerprint_fn or default_fingerprint
        self.objective_fn = objective_fn or _default_objective
        self.param_space = {k: list(v) for k, v in param_space.items()}
        if mode not in ("benchmark", "inference", "hybrid"):
            raise ValueError(f"mode must be benchmark/inference/hybrid, got {mode!r}")
        self.mode = mode
        if minimize is None and maximize is None:
            minimize = "elapsed_s"  # safe default
        if minimize is not None and maximize is not None:
            # minimize wins; documented in docstring.
            maximize = None
        self.minimize = minimize
        self.maximize = maximize
        self.epsilon = float(epsilon)
        self.min_observations = int(min_observations)
        self.host = _host_key()
        self.rng = rng or random.Random()

    # ----- param-space enumeration -----

    def _all_combos(self) -> list[dict]:
        keys = list(self.param_space.keys())
        if not keys:
            return [{}]
        grids = [self.param_space[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*grids)]

    # ----- objective comparison -----

    def _metric_name(self) -> str:
        return self.minimize if self.minimize is not None else self.maximize

    def _better(self, a: float, b: float) -> bool:
        """True iff objective value ``a`` is strictly better than ``b``."""
        if self.maximize is not None:
            return a > b
        return a < b

    def _score_of(self, objective: Mapping[str, float]) -> Optional[float]:
        v = objective.get(self._metric_name())
        if v is None or not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        if not math.isfinite(float(v)):
            return None
        return float(v)

    # ----- recording -----

    def record(self, fp_dict: Mapping[str, Any], params: Mapping[str, Any],
               objective: Mapping[str, float], ts: Optional[str] = None,
               fn_name: str = "<anon>") -> None:
        """Append one observation. ``ts`` is accepted as a parameter so
        callers can pin a deterministic timestamp; defaults to wall clock."""
        if ts is None:
            ts = _utc_now_iso()
        row = {
            "schema_version": SCHEMA_VERSION,
            "fn_name": fn_name,
            "host": self.host,
            "fp_bucket_json": _stable_json(bucketize_fingerprint(fp_dict)),
            "param_combo_json": _stable_json(dict(params)),
            "objective_json": _stable_json(dict(objective)),
            "n_obs": 1,
            "ts": ts,
        }
        self.store.append([row])

    # ----- kernel_tuning_cache import bridge (read-only) -----

    def read_ktc_regions(
        self,
        kernel_name: str,
        *,
        param_field: str,
        axis: str = "n_samples",
        fp_dim: str = "n",
        objective_metric: str = "elapsed_s",
        objective_field: str = "wall_ms",
        objective_scale: float = 1e-3,
        fixed_fp: Optional[Mapping[str, Any]] = None,
        n_obs: int = 1,
        ts: Optional[str] = None,
        fn_name: str = "<anon>",
        cache: Any = None,
    ) -> int:
        """Import a KernelTuningCache kernel's region table as cold-start
        observations so a freshly-migrated consumer inherits the tuning
        history instead of starting blind.

        READ-ONLY bridge: this reads ``kernel_tuning_cache`` and writes only
        into THIS oracle's own store. It never calls ``update``/``_save`` on
        the kernel cache.

        Each region in ``cache.get_regions(kernel_name)`` is turned into one
        observation:

        * the region's single size cap ``<axis>_max`` becomes the fingerprint
          value for ``fp_dim`` (a region capped at ``axis_max=N`` is treated as
          representative of arrays of size ``N``; a catch-all region with a
          ``None`` cap is skipped because it has no representative size);
        * the region's ``param_field`` value (e.g. ``"variant"`` /
          ``"backend"``) becomes the recorded param combo
          ``{param_field: <value>}``;
        * the region's ``objective_field`` (default ``"wall_ms"``) scaled by
          ``objective_scale`` (default ms -> s) becomes the objective metric
          ``objective_metric`` (default ``"elapsed_s"``).

        Regions lacking a usable cap, param value, or objective are skipped.
        Returns the number of observations imported.
        """
        if cache is None:
            from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
            cache = KernelTuningCache()
        regions = cache.get_regions(kernel_name)
        if not regions:
            return 0
        cap_key = f"{axis}_max"
        imported = 0
        for region in regions:
            cap = region.get(cap_key)
            if cap is None:
                # Catch-all region has no representative size -> nothing to
                # key a fingerprint on; skip (its variant still surfaces via
                # the size-capped regions that precede it).
                continue
            param_value = region.get(param_field)
            if param_value is None:
                continue
            raw_obj = region.get(objective_field)
            if not isinstance(raw_obj, (int, float)) or isinstance(raw_obj, bool):
                continue
            try:
                size = float(cap)
            except (TypeError, ValueError):
                continue
            fp = dict(fixed_fp or {})
            fp[fp_dim] = size
            objective = {objective_metric: float(raw_obj) * float(objective_scale)}
            self.record(
                fp, {param_field: param_value}, objective,
                ts=ts, fn_name=fn_name,
            )
            imported += int(n_obs)
            # If the caller asked for n_obs > 1 confidence weight, append the
            # remaining duplicate observations so the imported region clears
            # the min_observations gate.
            for _ in range(max(0, int(n_obs) - 1)):
                self.record(
                    fp, {param_field: param_value}, objective,
                    ts=ts, fn_name=fn_name,
                )
        return imported

    @classmethod
    def from_kernel_tuning_cache(
        cls,
        store_path: str,
        kernel_name: str,
        *,
        param_field: str,
        param_space: Mapping[str, Sequence[Any]],
        axis: str = "n_samples",
        fp_dim: str = "n",
        objective_metric: str = "elapsed_s",
        objective_field: str = "wall_ms",
        objective_scale: float = 1e-3,
        n_obs: int = 1,
        ts: Optional[str] = None,
        fn_name: str = "<anon>",
        cache: Any = None,
        **oracle_kwargs: Any,
    ) -> "ParamOracle":
        """Build a ParamOracle and seed it with ``kernel_name``'s KTC region
        table as cold-start observations. Read-only w.r.t. the kernel cache.
        Convenience wrapper over :meth:`read_ktc_regions`."""
        oracle = cls(store_path, param_space=param_space,
                     minimize=oracle_kwargs.pop("minimize", objective_metric),
                     **oracle_kwargs)
        oracle.read_ktc_regions(
            kernel_name, param_field=param_field, axis=axis, fp_dim=fp_dim,
            objective_metric=objective_metric, objective_field=objective_field,
            objective_scale=objective_scale, n_obs=n_obs, ts=ts,
            fn_name=fn_name, cache=cache,
        )
        return oracle

    # ----- recommend -----

    def recommend(self, fp_dict: Mapping[str, Any], fn_name: str = "<anon>") -> dict:
        """Best param combo for a fingerprint.

        Resolution order:
          1. exact ``fp_bucket`` match (best combo with >= min_observations)
          2. k-NN in continuous fingerprint space (nearest seen bucket's best)
          3. global best across all this fn's observations
          4. caller default = first combo of the param space

        Tie-break always prefers the cheaper (lower ``elapsed_s``) combo.
        Never raises; cold store -> caller default.
        """
        default = self._caller_default()
        rows = [r for r in self.store.read_rows()
                if r.get("fn_name") == fn_name and r.get("host") == self.host]
        if not rows:
            return default

        target_bucket = bucketize_fingerprint(fp_dict)
        target_key = _stable_json(target_bucket)

        # 1. Exact bucket match.
        exact = [r for r in rows if r.get("fp_bucket_json") == target_key]
        best = self._best_row(exact, require_confident=True)
        if best is not None:
            return _loads(best["param_combo_json"])

        # 2. k-NN: nearest distinct bucket (by continuous-dim euclidean) that
        #    has a confident best combo.
        knn_combo = self._knn_recommend(rows, target_bucket)
        if knn_combo is not None:
            return knn_combo

        # 3. Global best across all observations for this fn (confident).
        best = self._best_row(rows, require_confident=True)
        if best is not None:
            return _loads(best["param_combo_json"])

        # 3b. Global best ignoring confidence (still better than blind default).
        best = self._best_row(rows, require_confident=False)
        if best is not None:
            return _loads(best["param_combo_json"])

        # 4. Caller default.
        return default

    def _caller_default(self) -> dict:
        combos = self._all_combos()
        return combos[0] if combos else {}

    def _best_row(self, rows: Sequence[dict], *, require_confident: bool) -> Optional[dict]:
        """Pick the best row by the optimised metric, tie-broken by cheaper
        elapsed_s. Honours the confidence gate when requested."""
        best_row = None
        best_score = None
        best_tie = None
        for r in rows:
            if require_confident and int(r.get("n_obs", 0) or 0) < self.min_observations:
                continue
            obj = _loads(r.get("objective_json"))
            score = self._score_of(obj)
            if score is None:
                continue
            tie = obj.get("elapsed_s")
            tie = float(tie) if isinstance(tie, (int, float)) and not isinstance(tie, bool) else float("inf")
            if best_score is None or self._better(score, best_score) or (
                score == best_score and tie < (best_tie if best_tie is not None else float("inf"))
            ):
                best_row, best_score, best_tie = r, score, tie
        return best_row

    def _knn_recommend(self, rows: Sequence[dict], target_bucket: Mapping[str, Any],
                       k: int = 3) -> Optional[dict]:
        """Recommend via nearest buckets in continuous fingerprint space.

        Restricts neighbours to those sharing the categorical ``dtype_kind``
        (a float kernel choice shouldn't be transplanted onto object data),
        then ranks by euclidean distance over the bucketed continuous dims,
        and returns the best confident combo of the nearest neighbour(s).
        """
        # Group rows by bucket; keep the best confident row per bucket.
        from collections import defaultdict
        per_bucket: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            per_bucket[r.get("fp_bucket_json")].append(r)

        target_kind = target_bucket.get("dtype_kind")
        scored: list[tuple[float, dict]] = []
        for fpj, grp in per_bucket.items():
            bucket = _loads(fpj)
            if bucket.get("dtype_kind") != target_kind:
                continue
            dist = _euclidean_buckets(target_bucket, bucket)
            best = self._best_row(grp, require_confident=True)
            if best is None:
                continue
            scored.append((dist, best))
        if not scored:
            return None
        scored.sort(key=lambda t: t[0])
        # Among the k nearest, pick the closest; if the very nearest has
        # distance 0 it's effectively an exact match handled earlier, so this
        # path only fires on genuine neighbours.
        nearest = scored[:k]
        nearest.sort(key=lambda t: t[0])
        return _loads(nearest[0][1]["param_combo_json"])

    # ----- the sweep / call machinery -----

    def benchmark(self, fn: Callable, args: Sequence[Any] = (), kwargs: Optional[Mapping[str, Any]] = None,
                  ts: Optional[str] = None) -> dict:
        """Run EVERY param combo of ``fn(*args, **kwargs, **combo)``, timing
        and memory-profiling each, and record an observation per combo.
        Returns ``{param_combo_tuple: objective_dict}`` for inspection.
        Used directly, or internally by ``__call__`` in benchmark mode."""
        kwargs = dict(kwargs or {})
        fp = self.fingerprint_fn(args, kwargs)
        fn_name = getattr(fn, "__name__", "<anon>")
        results: dict = {}
        for combo in self._all_combos():
            obj, _out = self._run_one(fn, args, kwargs, combo)
            self.record(fp, combo, obj, ts=ts, fn_name=fn_name)
            results[_combo_key(combo)] = obj
        return results

    def _run_one(self, fn: Callable, args: Sequence[Any], kwargs: Mapping[str, Any],
                 combo: Mapping[str, Any]) -> tuple[dict, Any]:
        call_kwargs = dict(kwargs)
        call_kwargs.update(combo)
        rss_before = _rss_mb()
        t0 = time.perf_counter()
        out = fn(*args, **call_kwargs)
        elapsed = time.perf_counter() - t0
        rss_after = _rss_mb()
        rss_delta = (rss_after - rss_before) if (rss_before is not None and rss_after is not None) else None
        obj = self.objective_fn(out, elapsed, rss_delta)
        if not isinstance(obj, dict):
            raise TypeError(f"objective_fn must return a dict[str, float]; got {type(obj).__name__}")
        # Always ensure elapsed_s is present for cheaper-tie-break.
        obj.setdefault("elapsed_s", elapsed)
        return obj, out

    # ----- decorator -----

    def __call__(self, fn: Callable) -> Callable:
        """Decorate ``fn`` so each call is governed by the oracle's ``mode``.

        * ``inference``: pick the recommended combo, run it once, return its
          output. No recording.
        * ``benchmark``: sweep all combos, record each, return the output of
          the best (by objective) combo.
        * ``hybrid``: epsilon-greedy -- with prob ``epsilon`` explore a random
          combo, else exploit the recommendation; run it, record it, return
          output.
        """
        import functools

        fn_name = getattr(fn, "__name__", "<anon>")

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            fp = self.fingerprint_fn(args, kwargs)

            if self.mode == "inference":
                combo = self.recommend(fp, fn_name=fn_name)
                call_kwargs = dict(kwargs)
                call_kwargs.update(combo)
                return fn(*args, **call_kwargs)

            if self.mode == "benchmark":
                best_out = None
                best_score = None
                for combo in self._all_combos():
                    obj, out = self._run_one(fn, args, kwargs, combo)
                    self.record(fp, combo, obj, fn_name=fn_name)
                    score = self._score_of(obj)
                    if score is not None and (best_score is None or self._better(score, best_score)):
                        best_score, best_out = score, out
                return best_out

            # hybrid: epsilon-greedy.
            combos = self._all_combos()
            if self.rng.random() < self.epsilon and combos:
                combo = self.rng.choice(combos)
                wrapper._last_action = "explore"  # type: ignore[attr-defined]
            else:
                combo = self.recommend(fp, fn_name=fn_name)
                wrapper._last_action = "exploit"  # type: ignore[attr-defined]
            obj, out = self._run_one(fn, args, kwargs, combo)
            self.record(fp, combo, obj, fn_name=fn_name)
            return out

        wrapper._oracle = self  # type: ignore[attr-defined]
        wrapper._last_action = None  # type: ignore[attr-defined]
        return wrapper


# ---------------------------------------------------------------------------
# module-level helpers
# ---------------------------------------------------------------------------

def _default_objective(output: Any, elapsed_s: float, rss_delta_mb: Optional[float]) -> dict:
    obj = {"elapsed_s": float(elapsed_s)}
    if rss_delta_mb is not None:
        obj["rss_delta_mb"] = float(rss_delta_mb)
    return obj


def _utc_now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _combo_key(combo: Mapping[str, Any]) -> tuple:
    return tuple(sorted(combo.items()))


def _loads(s: Optional[str]) -> dict:
    if not s:
        return {}
    try:
        return orjson.loads(s)
    except Exception:
        return {}


def loads_json(s: Optional[str]) -> dict:
    """Public alias for the lenient JSON-object loader (returns ``{}`` on empty/invalid input); cross-package consumers import this instead of the private name."""
    return _loads(s)


def _euclidean_buckets(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    acc = 0.0
    for dim in _CONTINUOUS_FP_DIMS:
        av = a.get(dim, 0.0)
        bv = b.get(dim, 0.0)
        try:
            acc += (float(av) - float(bv)) ** 2
        except (TypeError, ValueError):
            continue
    return math.sqrt(acc)


__all__ = [
    "ParamOracle",
    "default_fingerprint",
    "bucketize_fingerprint",
    "log_bucket",
    "linear_bucket",
    "default_store_dir",
    "SCHEMA_VERSION",
]
