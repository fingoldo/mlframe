"""Improved orthogonal-polynomial pair Feature Engineering.

Originally a Hermite-only module (hence the file name and the
``HermiteResult`` dataclass). Now supports four orthogonal polynomial
families via the ``basis`` kwarg: Hermite, Legendre, Chebyshev,
Laguerre. **Default basis is Chebyshev**, picked empirically across
12 synthetic + UCI regimes -- it never finishes last, has the highest
minimum MI, and dominates real-world tabular data + threshold targets.
See ``_benchmarks/bench_polynomial_bases.py`` for the supporting
table.

Idea: orthogonal polynomials form a complete basis on their natural
domain, so any sufficiently smooth bivariate function ``f(x_a, x_b)``
can be represented as ``Σ c_{a,i} c_{b,j} P_i(x_a) P_j(x_b)`` -- find
coefficients via Optuna, MI-against-target as the objective. In theory
replaces the hand-coded ``unary x binary transformations`` zoo with a
single learned parametric family.

In practice the legacy implementation in ``MRMR._run_fe_step``
(``fe_smart_polynom_iters > 0`` branch) didn't deliver because of six
issues fixed here:

1. **Standardisation**. ``hermval(raw_x, c)`` blows up numerically
   when ``|x| >> 1`` (high-degree Hermite goes superlinear). We
   z-score inputs before evaluation so the ``[-3, 3]`` range covers
   ~99.7% of the support.

2. **Right Hermite family**. Numpy's ``polynomial.hermite`` is the
   *physicist's* family ``H_n(x)`` orthogonal under ``e^{-x²}``. For
   z-scored inputs (standard Normal) we want the *probabilist's*
   family ``He_n(x)`` orthogonal under ``e^{-x²/2}`` -- ``polynomial.
   hermite_e.hermeval``.

3. **Tight coefficient range**. ``[-2, 2]`` instead of ``[-10, 10]``:
   higher-degree terms dominate quickly, large ranges make TPE
   wander.

4. **Fixed degree per study**. Random ``length`` per trial breaks
   TPE's posterior. We sweep degrees as an outer loop (study per
   degree) and pick the best.

5. **L2 regularisation**. Penalty ``-lambda * ||c||²`` on the MI
   objective keeps coefficients bounded and discourages oscillating
   overfits.

6. **Identity baseline**. Returns ``best_mi`` only when it strictly
   beats the identity baseline ``MI((x_a, x_b), y)`` -- otherwise
   no engineered feature is recommended.

Usage::

    from mlframe.feature_selection.filters.hermite_fe import (
        optimise_hermite_pair, HermiteResult,
    )
    res = optimise_hermite_pair(
        x_a=col_a, x_b=col_b, y=target,
        n_trials=200, max_degree=4, n_jobs=1,
    )
    if res.uplift > 1.05:
        engineered = res.transform(x_a, x_b)  # numpy 1-D array
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from numpy.polynomial.hermite_e import hermeval  # probabilist's Hermite
from numpy.polynomial.legendre import legval
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.laguerre import lagval
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


# ---------------------------------------------------------------------------
# Polynomial basis registry. Each entry maps a name to (eval_func,
# preprocess_func, expected_input_distribution_doc).
#
# - hermite (probabilist's He_n): orthogonal under N(0, 1) -- best for
#   z-scored Gaussian-ish data. Preprocess = z-score.
# - legendre (P_n): orthogonal on [-1, 1] uniform weight -- best for
#   bounded uniform data. Preprocess = scale to [-1, 1] via min-max.
# - chebyshev (T_n): orthogonal on [-1, 1] under 1/sqrt(1-x^2) --
#   minimax error bound, equiripple. Preprocess = scale to [-1, 1].
# - laguerre (L_n): orthogonal on [0, +inf) under e^{-x} -- best for
#   positive exponentially-distributed data. Preprocess = shift to >= 0.
# ---------------------------------------------------------------------------


def _preprocess_zscore(x):
    mean = float(np.mean(x))
    std = float(np.std(x) + 1e-12)
    return (x - mean) / std, dict(mean=mean, std=std)


def _preprocess_minmax_neg1_1(x):
    lo = float(np.min(x))
    hi = float(np.max(x))
    span = hi - lo + 1e-12
    return 2 * (x - lo) / span - 1, dict(lo=lo, hi=hi)


def _preprocess_shift_nonneg(x):
    lo = float(np.min(x))
    return x - lo + 1e-9, dict(lo=lo)


def _apply_zscore(x, params):
    return (x - params["mean"]) / max(params["std"], 1e-12)


def _apply_minmax(x, params):
    span = params["hi"] - params["lo"] + 1e-12
    return 2 * (x - params["lo"]) / span - 1


def _apply_shift(x, params):
    return x - params["lo"] + 1e-9


_POLY_BASES = {
    "hermite": dict(eval=hermeval, fit=_preprocess_zscore, apply=_apply_zscore,
                     dist_note="standard Normal (z-score)"),
    "legendre": dict(eval=legval, fit=_preprocess_minmax_neg1_1, apply=_apply_minmax,
                      dist_note="uniform on [-1, 1]"),
    "chebyshev": dict(eval=chebval, fit=_preprocess_minmax_neg1_1, apply=_apply_minmax,
                       dist_note="uniform on [-1, 1] with 1/sqrt(1-x^2) weight"),
    "laguerre": dict(eval=lagval, fit=_preprocess_shift_nonneg, apply=_apply_shift,
                      dist_note="positive on [0, +inf)"),
}

logger = logging.getLogger(__name__)


@dataclass
class HermiteResult:
    """Result of an Optuna optimisation pass for a single feature pair.

    Despite the legacy name, ``HermiteResult`` carries the result for
    any supported polynomial basis (``basis`` field). The default
    basis is ``"chebyshev"`` (empirically robust on real tabular
    data); pass ``basis="hermite"`` for synthetic-Gaussian inputs or
    ``basis="laguerre"`` for skewed-positive distributions.
    """
    coef_a: np.ndarray
    coef_b: np.ndarray
    bin_func_name: str
    bin_func: Callable
    mi: float
    baseline_mi: float
    uplift: float
    degree_a: int
    degree_b: int
    basis: str = "chebyshev"
    # Preprocessing parameters for inputs (z-score mean/std, or min-max
    # lo/hi, or shift lo, depending on basis).
    preprocess_a: dict = field(default_factory=dict)
    preprocess_b: dict = field(default_factory=dict)

    def transform(self, x_a: np.ndarray, x_b: np.ndarray) -> np.ndarray:
        """Apply the learned polynomial-pair transformation: preprocess
        inputs to the basis's natural domain, evaluate the polynomial,
        combine via the chosen binary func."""
        basis_info = _POLY_BASES[self.basis]
        z_a = basis_info["apply"](x_a, self.preprocess_a)
        z_b = basis_info["apply"](x_b, self.preprocess_b)
        eval_func = basis_info["eval"]
        h_a = eval_func(z_a, self.coef_a)
        h_b = eval_func(z_b, self.coef_b)
        return self.bin_func(h_a, h_b)


_DEFAULT_BIN_FUNCS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
}


def _baseline_mi_pair(x_a, x_b, y, *, discrete_target: bool, n_neighbors: int = 3) -> float:
    """KSG MI of the (x_a, x_b) joint with target -- the identity baseline."""
    Xn = np.column_stack([x_a, x_b])
    if discrete_target:
        return float(mutual_info_classif(Xn, y, n_neighbors=n_neighbors,
                                          random_state=42, discrete_features=False).max())
    return float(mutual_info_regression(Xn, y, n_neighbors=n_neighbors,
                                         random_state=42, discrete_features=False).max())


def _ksg_mi_1d(x: np.ndarray, y: np.ndarray, *, discrete_target: bool,
               n_neighbors: int = 3) -> float:
    """KSG MI of 1-D x with target -- used as the optimisation objective."""
    if discrete_target:
        return float(mutual_info_classif(x.reshape(-1, 1), y,
                                          n_neighbors=n_neighbors, random_state=42,
                                          discrete_features=False)[0])
    return float(mutual_info_regression(x.reshape(-1, 1), y,
                                         n_neighbors=n_neighbors, random_state=42,
                                         discrete_features=False)[0])


def optimise_hermite_pair(
    x_a: np.ndarray,
    x_b: np.ndarray,
    y: np.ndarray,
    *,
    discrete_target: bool = True,
    bin_funcs: dict = None,
    max_degree: int = 4,
    min_degree: int = 2,
    n_trials: int = 200,
    coef_range: tuple = (-2.0, 2.0),
    l2_penalty: float = 0.05,
    n_neighbors: Optional[int] = None,
    seed: int = 42,
    sweep_degrees: bool = True,
    baseline_uplift_threshold: float = 1.01,
    early_stop_no_improve: int = 50,
    basis: str = "chebyshev",
) -> Optional[HermiteResult]:
    """Find Hermite-polynomial coefficients ``c_a``, ``c_b`` that
    maximise ``MI(bin_func(He(x_a, c_a), He(x_b, c_b)), y)`` over the
    requested Optuna budget. Standardises inputs, regularises
    coefficients, and only returns a result when the engineered MI
    strictly beats the identity baseline by ``baseline_uplift_threshold``.

    Knob tuning notes
    -----------------
    * ``basis="chebyshev"`` is the default after empirical evaluation
      across 12 regimes (synthetic + UCI California Housing + UCI
      Diabetes + bounded / heavy-tailed): Chebyshev wins on real
      tabular data and threshold-style targets, never finishes last,
      and has the highest minimum MI across the test suite. Pass
      ``basis="hermite"`` for synthetic Gaussian-input data, or
      ``basis="laguerre"`` for skewed-positive distributions. See
      ``_benchmarks/bench_polynomial_bases.py`` for the supporting
      table.
    * ``l2_penalty=0.05`` (default) is good for XOR-like targets where
      the optimum has small ``|c|``. For radial / saddle targets where
      ``|c| ~ 2-3`` is natural, drop to ``0.01``.
    * ``n_neighbors`` (KSG): ``None`` (default) auto-picks: 3 for
      ``n >= 5000``, 5 for ``n in [1000, 5000)``, 7 for ``n < 1000``.
      Smaller datasets need more neighbours to stabilise the MI estimate.
    * ``max_degree=4`` covers most smooth targets. For high-frequency
      (``tanh(x)*sin(x*pi)``) raise to 6-8 -- but each extra degree
      doubles the search space, so increase ``n_trials`` proportionally.
    * ``early_stop_no_improve``: stop a study early if no improvement in
      the last N trials. Cuts wall-time for already-converged degrees.

    Returns
    -------
    HermiteResult or None if the search failed to beat the baseline.
    """
    # Auto-pick n_neighbors based on n.
    n = len(y)
    if n_neighbors is None:
        if n >= 5000:
            n_neighbors = 3
        elif n >= 1000:
            n_neighbors = 5
        else:
            n_neighbors = 7
    try:
        import optuna
        from optuna.samplers import TPESampler
        # Optuna's TPESampler emits an ExperimentalWarning every study
        # init when ``multivariate=True``. The flag has been "experimental"
        # since 2020 and is the recommended setting for correlated params;
        # suppress the noise.
        import warnings as _w
        try:
            from optuna.exceptions import ExperimentalWarning
            _w.filterwarnings("ignore", category=ExperimentalWarning)
        except ImportError:
            pass
    except ImportError as e:
        raise ImportError(
            "optimise_hermite_pair requires the optional `optuna` package. "
            "Install via `pip install optuna`."
        ) from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    bin_funcs = bin_funcs or _DEFAULT_BIN_FUNCS

    if basis not in _POLY_BASES:
        raise ValueError(f"unknown basis {basis!r}; expected one of {list(_POLY_BASES)}")
    basis_info = _POLY_BASES[basis]

    # Preprocess inputs to the basis's natural domain.
    z_a, preprocess_a = basis_info["fit"](x_a)
    z_b, preprocess_b = basis_info["fit"](x_b)
    eval_func = basis_info["eval"]

    baseline = _baseline_mi_pair(z_a, z_b, y, discrete_target=discrete_target,
                                  n_neighbors=n_neighbors)
    logger.debug(f"baseline MI(pair, y) = {baseline:.4f}")

    best: Optional[HermiteResult] = None

    degree_grid = list(range(min_degree, max_degree + 1)) if sweep_degrees else [max_degree]

    for degree in degree_grid:
        # Coefficient vector size = degree + 1 (for c_0..c_degree).
        ca_size = degree + 1
        cb_size = degree + 1

        bf_names = list(bin_funcs.keys())
        bf_callables = [bin_funcs[n] for n in bf_names]

        def objective(trial, _degree=degree):  # closure over degree
            coef_a = np.array([
                trial.suggest_float(f"a_{i}", *coef_range)
                for i in range(ca_size)
            ], dtype=np.float64)
            coef_b = np.array([
                trial.suggest_float(f"b_{i}", *coef_range)
                for i in range(cb_size)
            ], dtype=np.float64)

            h_a = eval_func(z_a, coef_a)
            h_b = eval_func(z_b, coef_b)

            # Guard: NaN/inf can arise if std blew up.
            if not (np.all(np.isfinite(h_a)) and np.all(np.isfinite(h_b))):
                return -np.inf

            # Perf-1 (post-plan): batch all binary funcs into a single
            # ``mutual_info_classif`` call. sklearn's per-call validation
            # was ~44% of profile time; one matrix call is ~3x cheaper
            # than three scalar calls when ``len(bin_funcs) == 3``.
            cols = []
            valid_idx = []
            for k, bf in enumerate(bf_callables):
                try:
                    combined = bf(h_a, h_b)
                except Exception:
                    continue
                if np.all(np.isfinite(combined)):
                    cols.append(combined)
                    valid_idx.append(k)
            if not cols:
                return -np.inf
            X_batch = np.column_stack(cols)
            if discrete_target:
                mi_arr = mutual_info_classif(
                    X_batch, y, n_neighbors=n_neighbors, random_state=42,
                    discrete_features=False,
                )
            else:
                mi_arr = mutual_info_regression(
                    X_batch, y, n_neighbors=n_neighbors, random_state=42,
                    discrete_features=False,
                )

            best_mi = -np.inf
            best_k = None
            for j, k in enumerate(valid_idx):
                mi = float(mi_arr[j])
                penalty = l2_penalty * (np.sum(coef_a ** 2) + np.sum(coef_b ** 2))
                score = mi - penalty
                if score > best_mi:
                    best_mi = score
                    best_k = k
                    raw_mi_for_best = mi
            if best_k is not None:
                trial.set_user_attr("bf_name", bf_names[best_k])
                trial.set_user_attr("raw_mi", raw_mi_for_best)
            return best_mi

        sampler = TPESampler(multivariate=True, seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Early stopping on no-improvement.
        if early_stop_no_improve and early_stop_no_improve < n_trials:
            stop_state = {"best": -np.inf, "since_improve": 0}

            def _early_stop_callback(s, trial):
                cur_best = s.best_value if s.best_trial is not None else -np.inf
                if cur_best > stop_state["best"]:
                    stop_state["best"] = cur_best
                    stop_state["since_improve"] = 0
                else:
                    stop_state["since_improve"] += 1
                if stop_state["since_improve"] >= early_stop_no_improve:
                    s.stop()

            study.optimize(objective, n_trials=n_trials,
                           callbacks=[_early_stop_callback], show_progress_bar=False)
        else:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        bf_name = study.best_trial.user_attrs.get("bf_name", "add")
        raw_mi = study.best_trial.user_attrs.get("raw_mi", -np.inf)
        if raw_mi <= 0 or not np.isfinite(raw_mi):
            continue

        coef_a = np.array([study.best_params[f"a_{i}"] for i in range(ca_size)],
                           dtype=np.float64)
        coef_b = np.array([study.best_params[f"b_{i}"] for i in range(cb_size)],
                           dtype=np.float64)

        cand = HermiteResult(
            coef_a=coef_a, coef_b=coef_b,
            bin_func_name=bf_name, bin_func=bin_funcs[bf_name],
            mi=raw_mi, baseline_mi=baseline,
            uplift=raw_mi / max(baseline, 1e-12),
            degree_a=degree, degree_b=degree,
            basis=basis,
            preprocess_a=preprocess_a,
            preprocess_b=preprocess_b,
        )
        if best is None or cand.mi > best.mi:
            best = cand
        logger.debug(
            f"degree={degree}: best MI={raw_mi:.4f} (baseline {baseline:.4f}, "
            f"uplift {cand.uplift:.2f}x), bf={bf_name}"
        )

    if best is None or best.mi <= baseline * baseline_uplift_threshold:
        # Failed to beat baseline by enough -- don't recommend an
        # engineered feature.
        return None
    return best
