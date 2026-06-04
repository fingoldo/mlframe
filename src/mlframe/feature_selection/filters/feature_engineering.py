"""Pair-discovery feature engineering used by ``MRMR.fit``.

Public functions:

* ``check_prospective_fe_pairs`` -- given high-MI raw pairs, enumerate unary x binary transformation candidates, evaluate each via ``mi_direct``, recommend top-N replacements.
* ``compute_pairs_mis`` -- precompute MI for every pair candidate so the screening loop has a fast lookup.
* ``create_unary_transformations`` / ``create_binary_transformations`` -- preset registries of numpy / scipy.special / hermval functions.
* ``get_existing_feature_name`` / ``get_new_feature_name`` -- string formatters for engineered-feature names.
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from itertools import combinations
from timeit import default_timer as timer
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermval
from scipy import special as sp

from pyutilz.pythonlib import sort_dict_by_value
from pyutilz.system import tqdmu


# Wave 27 P1 (2026-05-20): ``check_prospective_fe_pairs`` is dispatched via
# ``parallel_run`` from mrmr.py with backend='threading'. The function
# accumulates per-binary-transform timings into a shared ``times_spent``
# defaultdict via ``+=``. Python's ``+=`` on a float is load-add-store and
# NOT atomic even under the GIL between threads; concurrent workers can
# drop updates silently, under-reporting the diagnostic at mrmr.py:1691.
# This module-level lock serialises the increment; threading workers
# synchronise correctly. Under loky/spawn each worker gets its own
# defaultdict copy (no shared state); the lock has no effect there but
# also doesn't break.
_TIMES_SPENT_LOCK = threading.Lock()

# CRITICAL: the hoisted shared buffer at
# ``check_prospective_fe_pairs`` allocates ``(n, max_n_combs * len(binary))``
# float32. With n=4M and the medium preset that's ~17.6 GiB -- production
# MRMR crashed with numpy.core._exceptions._ArrayMemoryError on a real run.
# The hoist landed in Wave Pack G (commit 068acdd) under small-n benchmarks
# and never measured peak RAM on million-row data.
#
# Two-strategy dispatch:
#   Fast path (current): if buffer < ``_FE_BUFFER_RAM_BUDGET_RATIO`` * available
#     RAM, allocate the shared buffer and use the hoist (cheapest if it fits).
#   Recompute fallback: drop the multi-column buffer, scratch into a fresh 1D
#     ``np.empty(n, float32)`` per inner iteration, and rebuild the ~10
#     survivor columns from their (transformations_pair, bin_func_name) metadata
#     after the inner loop. Extra recompute cost: ~K bin_func calls per pair
#     (K = num survivors, typically <= fe_max_pair_features + |leading|);
#     <= 1% of the ~max_combs*|binary| calls already done in the inner loop.
#
# Subsample path remains a separate opt-in (``subsample_n`` parameter); this
# memory dispatcher is the deterministic, accuracy-preserving fallback that
# auto-engages when the shared buffer would OOM.
_FE_BUFFER_RAM_BUDGET_RATIO: float = 0.4

# Shared subsample default across the two FE entry points. ``polynom_pair_fe``
# already uses 200_000 (validated 2026-05-18: 100k could lose a marginal hermite
# feature, 200k kept it). The accuracy bench for ``check_prospective_fe_pairs``
# at this n landed at jaccard=1.0 vs full -- see
# bench_fe_pair_subsample_accuracy.py. Keep both call sites pinned to ONE knob
# so a future re-tune lands consistently across the FE block.
FE_DEFAULT_SUBSAMPLE_N: int = 200_000


def _rebuild_full_survivor_col(
    config: tuple,
    X_full,
    original_cols: dict,
    unary_transformations: dict,
    binary_transformations: dict,
    prewarp_spec_by_var: dict | None = None,
) -> np.ndarray:
    """Rebuild a survivor column at full n from raw X by re-applying its unary + binary transforms.

    Used by every survivor-packing path (fast / fallback / subsample) so a single
    code path produces the full-n output the caller (mrmr.py) expects regardless
    of how the MI sweep was scratched. Cost: 2 unary + 1 binary ufunc per
    survivor; with len(this_pair_features) <= fe_max_pair_features this is
    bounded to ~K calls per pair, trivial vs the MI sweep work.
    """
    transformations_pair, bin_func_name, _i = config
    (var_a_idx, unary_a_name) = transformations_pair[0]
    (var_b_idx, unary_b_name) = transformations_pair[1]
    if isinstance(X_full, pd.DataFrame):
        vals_a = X_full.iloc[:, original_cols[var_a_idx]].values
        vals_b = X_full.iloc[:, original_cols[var_b_idx]].values
    else:
        vals_a = X_full[:, original_cols[var_a_idx]].to_numpy()
        vals_b = X_full[:, original_cols[var_b_idx]].to_numpy()
    # ``poly_*`` unary keys hold hermval coefficient arrays, not callables;
    # check_prospective_fe_pairs handles them via the same hermval(c=tr_func) path.
    # ``prewarp`` is the per-operand learned pseudo-unary (2026-06-02): its fitted
    # spec lives in ``prewarp_spec_by_var`` and replays closed-form from x alone.
    _pw = prewarp_spec_by_var or {}

    def _apply_unary(name, var_idx, vals):
        if name == "prewarp":
            from .hermite_fe import apply_operand_prewarp
            return apply_operand_prewarp(vals, _pw[var_idx])
        if "poly_" in name:
            return hermval(vals, c=unary_transformations[name])
        return unary_transformations[name](vals)

    param_a = _apply_unary(unary_a_name, var_a_idx, vals_a)
    param_b = _apply_unary(unary_b_name, var_b_idx, vals_b)
    col = binary_transformations[bin_func_name](param_a, param_b)
    np.nan_to_num(col, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return col.astype(np.float32, copy=False)


def _estimate_fe_shared_buffer_bytes(n_rows: int, max_n_combs: int, n_binary: int) -> int:
    """Bytes the hoisted shared buffer would consume at full precision."""
    return int(n_rows) * int(max_n_combs) * int(n_binary) * 4  # float32 = 4 bytes


def _can_hoist_shared_buffer(buffer_bytes: int, budget_ratio: float = _FE_BUFFER_RAM_BUDGET_RATIO) -> tuple[bool, int, int]:
    """Decide whether the shared scratch buffer fits in available RAM.

    Uses ``psutil.virtual_memory().available`` for a conservative "RAM I can
    take right now" reading -- ``total`` would include pages owned by other
    processes which we cannot evict cleanly. Falls back to a permissive yes
    when psutil is unavailable so single-test environments without it still
    take the historical fast path (and OOM loudly on truly large n if so).

    Returns (can_hoist, buffer_bytes, available_bytes).
    """
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
    except Exception:
        # No psutil: preserve legacy behaviour (always hoist, OOM is the signal).
        return True, buffer_bytes, -1
    return buffer_bytes < int(available * budget_ratio), buffer_bytes, available


# Domain-validity tags for unary transforms produced by ``create_unary_transformations(preset="maximal")``. Consumers that need to clip / reject inputs before
# applying these transforms can look them up here. Tag vocabulary: ``-1to1``, ``-pi/2topi/2``, ``1toinf``, ``-0.(9)to0.(9)``, ``pos``, ``nonzero``. Transforms not
# listed are unconstrained on real inputs (e.g. ``sin``, ``exp``, ``sqr``).
UNARY_INPUT_CONSTRAINTS: dict[str, str] = {
    # reverse trigonometric
    "arccos": "-1to1",
    "arcsin": "-1to1",
    "arctan": "-pi/2topi/2",
    # reverse hyperbolic
    "arccosh": "1toinf",
    "arctanh": "-0.(9)to0.(9)",
    # powers
    "sqrt": "pos",
    "log": "pos",
    "reciproc": "nonzero",
    "invsquared": "nonzero",
    "invqubed": "nonzero",
    "invcbrt": "nonzero",
    "invsqrt": "nonzero",
}

from ._internals import njit_functions_dict, smart_log
from .discretization import discretize_array
from .permutation import mi_direct

logger = logging.getLogger(__name__)




def compute_pairs_mis(
    all_pairs: Sequence,
    data,
    target_indices,
    nbins,
    classes_y,
    classes_y_safe,
    freqs_y,
    fe_min_nonzero_confidence,
    fe_npermutations,
    cached_confident_MIs: dict,
    cached_MIs: dict,
    fe_min_pair_mi: float,
    fe_min_pair_mi_prevalence: float,
):
    for raw_vars_pair in all_pairs:
        # check that every element of a pair has computed its MI with target
        for var in raw_vars_pair:
            if (var,) not in cached_confident_MIs and (var,) not in cached_MIs:
                mi, conf = mi_direct(
                    data,
                    x=(var,),
                    y=target_indices,
                    factors_nbins=nbins,
                    classes_y=classes_y,
                    classes_y_safe=classes_y_safe,
                    freqs_y=freqs_y,
                    min_nonzero_confidence=fe_min_nonzero_confidence,
                    npermutations=fe_npermutations,
                )
                cached_MIs[(var,)] = mi

        # ensure that pair as a whole has computed its MI with target
        if raw_vars_pair not in cached_confident_MIs and raw_vars_pair not in cached_MIs:
            ind_elems_mi_sum = cached_MIs[(raw_vars_pair[0],)] + cached_MIs[(raw_vars_pair[1],)]
            if ind_elems_mi_sum > fe_min_pair_mi:
                mi, conf = mi_direct(
                    data,
                    x=raw_vars_pair,
                    y=target_indices,
                    factors_nbins=nbins,
                    classes_y=classes_y,
                    classes_y_safe=classes_y_safe,
                    freqs_y=freqs_y,
                    min_nonzero_confidence=fe_min_nonzero_confidence,
                    npermutations=fe_npermutations,
                )

                # T3#24 2026-05-18 Pack #5 pair-MI cache: ALWAYS cache the computed pair MI, not only when it passes ``fe_min_pair_mi_prevalence``.
                # Pre-fix this branch dropped sub-threshold MIs on the floor; adaptive retry with relaxed prevalence (Pack #5) then re-computed them.
                # The downstream consumer in MRMR.fit applies the prevalence gate on the cached value, so caching unconditionally preserves filtering behaviour.
                cached_MIs[raw_vars_pair] = mi

    return cached_MIs


def get_existing_feature_name(fe_tuple: tuple, cols_names: Sequence) -> str:
    fname = cols_names[fe_tuple[0]]
    if fe_tuple[1] == "identity":
        return fname
    else:
        return f"{fe_tuple[1]}({fname})"


def get_new_feature_name(fe_tuple: tuple, cols_names: Sequence) -> str:
    return f"{fe_tuple[1]}({get_existing_feature_name(fe_tuple=fe_tuple[0][0],cols_names=cols_names)},{get_existing_feature_name(fe_tuple=fe_tuple[0][1],cols_names=cols_names)})"  # (((2, 'log'), (3, 'sin')), 'mul', 1016)


# GPU broadcast: per-function gpu_compatible flag. Many transformations have direct CuPy equivalents (``np.log`` -> ``cp.log`` etc.); some require ``scipy.special``
# (erfinv, gammaln, ...) which CuPy lacks. The FE driver groups columns into a CuPy tensor and applies the GPU-compatible transformation in one launch, falling back
# to per-column CPU dispatch for the rest.


def gpu_compatible_unary_names() -> set:
    """Names of unary transformations with a direct CuPy equivalent. Anything outside this set falls back to per-column CPU dispatch."""
    return {
        "identity", "sign", "neg", "abs", "rint",
        "sqr", "qubed", "reciproc", "invsquared", "invqubed",
        "cbrt", "sqrt", "invcbrt", "invsqrt",
        "log", "exp",
        "sin", "cos", "tan",
        "sinh", "cosh", "tanh",
    }


def gpu_compatible_binary_names() -> set:
    """Same idea for binary transformations."""
    return {
        "add", "sub", "mul", "div", "pow", "min", "max",
        "hypot", "atan2",
    }


def apply_gpu_unary_batched(
    cols_data,
    column_indices: Sequence[int],
    transformation_name: str,
):
    """Apply a unary transformation to a batch of columns on GPU. Returns a CuPy 2-D array with shape ``(n_samples, len(column_indices))``. Only safe for names in
    ``gpu_compatible_unary_names()``.

    Raises:
        ValueError: if the transformation is not GPU-compatible.
        ImportError: if CuPy is not installed.
    """
    import cupy as cp

    name = transformation_name
    if name not in gpu_compatible_unary_names():
        raise ValueError(f"transformation {name!r} is not GPU-compatible")

    if hasattr(cols_data, "to_numpy"):
        cols_np = cols_data.to_numpy() if hasattr(cols_data, "to_numpy") else np.asarray(cols_data)
    else:
        cols_np = np.asarray(cols_data)

    # Slice the requested columns.
    sub = cols_np[:, list(column_indices)] if cols_np.ndim == 2 else cols_np.reshape(-1, 1)
    sub_gpu = cp.asarray(sub.astype(np.float32))

    if name == "identity":
        return sub_gpu
    if name == "sign":
        return cp.sign(sub_gpu)
    if name == "neg":
        return -sub_gpu
    if name == "abs":
        return cp.abs(sub_gpu)
    if name == "rint":
        return cp.rint(sub_gpu)
    if name == "sqr":
        return cp.power(sub_gpu, 2)
    if name == "qubed":
        return cp.power(sub_gpu, 3)
    if name == "reciproc":
        return cp.power(sub_gpu, -1)
    if name == "invsquared":
        return cp.power(sub_gpu, -2)
    if name == "invqubed":
        return cp.power(sub_gpu, -3)
    if name == "cbrt":
        return cp.cbrt(sub_gpu)
    if name == "sqrt":
        return cp.sqrt(cp.abs(sub_gpu))
    if name == "invcbrt":
        return cp.power(sub_gpu, -1.0 / 3.0)
    if name == "invsqrt":
        return cp.power(sub_gpu, -0.5)
    if name == "log":
        # Smart log: avoid log of non-positive.
        x_min = cp.nanmin(sub_gpu)
        if x_min > 0:
            return cp.log(sub_gpu)
        return cp.log(sub_gpu + (1e-5 - x_min))
    if name == "exp":
        return cp.exp(sub_gpu)
    if name == "sin":
        return cp.sin(sub_gpu)
    if name == "cos":
        return cp.cos(sub_gpu)
    if name == "tan":
        return cp.tan(sub_gpu)
    if name == "sinh":
        return cp.sinh(sub_gpu)
    if name == "cosh":
        return cp.cosh(sub_gpu)
    if name == "tanh":
        return cp.tanh(sub_gpu)
    raise ValueError(f"GPU dispatch missing for {name!r} despite being in compatible set")


def apply_gpu_binary_batched(
    a_gpu,
    b_gpu,
    transformation_name: str,
):
    """Apply a binary transformation element-wise to two CuPy arrays already on the GPU. Names must be in ``gpu_compatible_binary_names()``."""
    import cupy as cp

    name = transformation_name
    if name not in gpu_compatible_binary_names():
        raise ValueError(f"transformation {name!r} is not GPU-compatible")

    if name == "add":
        return a_gpu + b_gpu
    if name == "sub":
        return a_gpu - b_gpu
    if name == "mul":
        return a_gpu * b_gpu
    if name == "div":
        return a_gpu / b_gpu
    if name == "pow":
        return cp.power(a_gpu, b_gpu)
    if name == "min":
        return cp.minimum(a_gpu, b_gpu)
    if name == "max":
        return cp.maximum(a_gpu, b_gpu)
    if name == "hypot":
        return cp.hypot(a_gpu, b_gpu)
    if name == "atan2":
        return cp.arctan2(a_gpu, b_gpu)
    raise ValueError(f"GPU dispatch missing for binary {name!r}")


_KNOWN_UNARY_PRESETS = ("minimal", "medium", "maximal")
_KNOWN_BINARY_PRESETS = ("minimal", "medium", "maximal")


def _resolve_preset(preset: str) -> str:
    """Canonicalise a preset name to one of {minimal, medium, maximal}.

    ``rich`` / ``full`` are treated as aliases for ``maximal`` (the richest
    tier) so callers using the historical loose vocabulary still get a
    well-defined registry. Any other value raises ValueError so a typo
    (``mininal``) surfaces loudly rather than silently aliasing to ``medium``.
    """
    p = (preset or "").strip().lower()
    if p in _KNOWN_UNARY_PRESETS:
        return p
    if p in ("rich", "full"):
        return "maximal"
    if p in ("basic", "default"):
        return "minimal"
    raise ValueError(
        f"unknown FE preset {preset!r}; expected one of "
        f"{_KNOWN_UNARY_PRESETS} (or aliases 'rich'/'full' -> 'maximal')"
    )


def create_unary_transformations(preset: str = "minimal"):
    # Domain-validity tags for each transform live in the module-level ``UNARY_INPUT_CONSTRAINTS`` dict so callers that need to clip / reject inputs can look them up
    # by transform name (e.g. ``arccos`` requires ``-1to1``).
    #
    # Preset ladder (monotone: minimal subset of medium subset of maximal):
    #   minimal -- non-degenerate workhorse set able to express the common
    #     algebraic targets (a**2/b, log(c)*sin(d), ...): identity + the
    #     elementary powers, sqrt, log, sin and their sign/neg/abs companions.
    #     Every tier MUST have >1 member; an identity-only "minimal" silently
    #     crippled MRMR's pair FE (it could only form mul(a,b)/add(a,b), never
    #     sqr(a) or log(c)), so the default fit found ZERO engineered cols.
    #   medium -- minimal + exp, reciprocal/inverse powers, cbrt, rint.
    #   maximal -- medium + trig/hyperbolic/special families (below).
    preset = _resolve_preset(preset)
    unary_transformations = {
        # simplest
        "identity": lambda x: x,
        # sign / magnitude companions
        "neg": np.negative,
        "abs": np.abs,
        # powers
        "sqr": lambda x: np.power(x, 2),
        "reciproc": lambda x: np.power(x, -1),
        "sqrt": lambda x: np.sqrt(np.abs(x)),
        # logarithms
        "log": smart_log,
        # trigonometric
        "sin": np.sin,
    }
    if preset != "minimal":
        unary_transformations.update(
            {
                "sign": np.sign,
                # outliers removal
                # Rounding
                "rint": np.rint,
                # np.modf Return the fractional and integral parts of an array, element-wise.
                # clip
                # powers
                "qubed": lambda x: np.power(x, 3),
                "invsquared": lambda x: np.power(x, -2),
                "invqubed": lambda x: np.power(x, -3),
                "cbrt": np.cbrt,
                "invcbrt": lambda x: np.power(x, -1 / 3),
                "invsqrt": lambda x: np.power(x, -1 / 2),
                # logarithms
                "exp": np.exp,
            }
        )

    if preset == "maximal":
        unary_transformations.update(
            {
                # math an
                "grad1": np.gradient,
                "grad2": lambda x: np.gradient(x, edge_order=2),
                # trigonometric
                "sinc": np.sinc,
                "cos": np.cos,
                "tan": np.tan,
                # reverse trigonometric
                "arcsin": np.arcsin,
                "arccos": np.arccos,
                "arctan": np.arctan,
                # hyperbolic
                "sinh": np.sinh,
                "cosh": np.cosh,
                "tanh": np.tanh,
                # reverse hyperbolic
                "arcsinh": np.arcsinh,
                "arccosh": np.arccosh,
                "arctanh": np.arctanh,
                # special
                #'psi':sp.psi, polygamma(0,x) is same as psi
                "erf": sp.erf,
                "dawsn": sp.dawsn,
                "gammaln": sp.gammaln,
                #'spherical_jn':sp.spherical_jn
            }
        )

    njit_functions_dict(unary_transformations)

    if preset == "maximal":
        for order in range(3):
            unary_transformations["polygamma_" + str(order)] = lambda x, _order=order: sp.polygamma(_order, x)
            unary_transformations["struve" + str(order)] = lambda x, _order=order: sp.struve(_order, x)
            unary_transformations["jv" + str(order)] = lambda x, _order=order: sp.jv(_order, x)

        """j0
        faster version of this function for order 0.

        j1
        faster version of this function for order 1.
        """

    return unary_transformations


def _safe_div(x, y):
    """Element-wise division with a sign-stable epsilon so ``x / 0`` does not
    blow up to +-inf (mirrors hermite_fe._safe_div). Required so the binary
    registry can express genuine ratio targets (e.g. a**2/b) directly rather
    than only via reciproc-then-multiply, which loses a representative on tight
    unary presets."""
    eps = 1e-9
    return x / (y + np.sign(y) * eps + eps)


def create_binary_transformations(preset: str = "minimal"):
    # Preset ladder (monotone: minimal subset of medium subset of maximal):
    #   minimal -- the elementary closed binary algebra: mul, add, sub,
    #     div, max, min. ``sub`` and ``div`` were absent from EVERY
    #     prior tier; division was only reachable as reciproc o mul, so a plain
    #     a/b ratio target could not be formed when the unary preset lacked
    #     reciproc. Every tier MUST have >1 member.
    #   medium -- minimal + abs_diff, hypot (richer magnitude combinations).
    #   maximal -- medium + the full numpy / scipy.special family below.
    preset = _resolve_preset(preset)

    binary_transformations = {
        # Basic
        "mul": np.multiply,
        "add": np.add,
        "sub": np.subtract,
        # Safe division (sign-stable eps; never +-inf on divide-by-zero).
        "div": _safe_div,
        # Extrema
        "max": np.maximum,
        "min": np.minimum,
    }

    if preset != "minimal":
        binary_transformations.update(
            {
                # Richer magnitude combinations available from medium up.
                "abs_diff": lambda x, y: np.abs(x - y),
                "hypot": np.hypot,
            }
        )

    if preset == "maximal":
        binary_transformations.update(
            {
                # All kinds of averages
                "hypot": np.hypot,
                "logaddexp": np.logaddexp,
                "agm": sp.agm,  # Compute the arithmetic-geometric mean of a and b.
                # Rational routines
                #'lcm':np.lcm, # requires int arguments #  ufunc 'lcm' did not contain a loop with signature matching types (<class 'numpy.dtype[float32]'>, <class 'numpy.dtype[float32]'>) -> None
                #'gcd':np.gcd, # requires int arguments
                #'mod':np.remainder, # requires int arguments
                # Powers
                "pow": np.power,  # non-symmetrical! may required dtype=complex for arbitrary numbers
                # Logarithms
                "logn": lambda x, y: np.emath.logn(x - np.min(x) + 0.1, y - np.min(y) + 0.1),  # non-symmetrical!
                # DSP
                # 'convolve':np.convolve, # symmetrical wrt args. scipy.signal.fftconvolve should be faster? SLOW?
                # Linalg
                "heaviside": np.heaviside,  # non-symmetrical!
                #'cross':np.cross, # symmetrical # incompatible dimensions for cross product (dimension must be 2 or 3)
                # Comparison
                "greater": lambda x, y: np.greater(x, y).astype(int),
                "less": lambda x, y: np.less(x, y).astype(int),
                "equal": lambda x, y: np.equal(x, y).astype(int),
                # special
                "beta": sp.beta,  # symmetrical
                "binom": sp.binom,  # non-symmetrical! binomial coefficient considered as a function of two real variables.
            }
        )

    njit_functions_dict(binary_transformations)

    return binary_transformations


# ----------------------------------------------------------------------
# Sibling-module re-export. The 505-LOC ``check_prospective_fe_pairs``
# body lives in ``_feature_engineering_pairs.py`` so this file stays
# below the 1k-LOC monolith threshold.
# ----------------------------------------------------------------------
from ._feature_engineering_pairs import check_prospective_fe_pairs  # noqa: E402,F401
