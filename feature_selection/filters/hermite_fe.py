"""Improved Hermite-polynomial pair Feature Engineering.

Idea: Hermite polynomials form a complete orthonormal basis on R under
the Gaussian weight, so any sufficiently smooth bivariate function
``f(x_a, x_b)`` can be represented as ``Σ c_{a,i} c_{b,j} H_i(x_a)
H_j(x_b)`` -- find coefficients via Optuna, MI-against-target as the
objective. In theory replaces the hand-coded
``unary x binary transformations`` zoo with a single learned
parametric family.

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
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

logger = logging.getLogger(__name__)


@dataclass
class HermiteResult:
    """Result of an Optuna optimisation pass for a single feature pair."""
    coef_a: np.ndarray
    coef_b: np.ndarray
    bin_func_name: str
    bin_func: Callable
    mi: float
    baseline_mi: float
    uplift: float
    degree_a: int
    degree_b: int
    mean_a: float
    std_a: float
    mean_b: float
    std_b: float

    def transform(self, x_a: np.ndarray, x_b: np.ndarray) -> np.ndarray:
        """Apply the learned Hermite-pair transformation (z-score
        standardise, evaluate, combine via the chosen binary func)."""
        z_a = (x_a - self.mean_a) / max(self.std_a, 1e-12)
        z_b = (x_b - self.mean_b) / max(self.std_b, 1e-12)
        h_a = hermeval(z_a, self.coef_a)
        h_b = hermeval(z_b, self.coef_b)
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
    n_trials: int = 200,
    coef_range: tuple = (-2.0, 2.0),
    l2_penalty: float = 0.05,
    n_neighbors: int = 3,
    seed: int = 42,
    sweep_degrees: bool = True,
) -> Optional[HermiteResult]:
    """Find Hermite-polynomial coefficients ``c_a``, ``c_b`` that
    maximise ``MI(bin_func(He(x_a, c_a), He(x_b, c_b)), y)`` over the
    requested Optuna budget. Standardises inputs, regularises
    coefficients, and only returns a result when the engineered MI
    strictly beats the identity baseline.

    Returns
    -------
    HermiteResult or None if the search failed to beat the baseline.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError as e:
        raise ImportError(
            "optimise_hermite_pair requires the optional `optuna` package. "
            "Install via `pip install optuna`."
        ) from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    bin_funcs = bin_funcs or _DEFAULT_BIN_FUNCS

    # Standardise inputs.
    mean_a, std_a = float(np.mean(x_a)), float(np.std(x_a) + 1e-12)
    mean_b, std_b = float(np.mean(x_b)), float(np.std(x_b) + 1e-12)
    z_a = (x_a - mean_a) / std_a
    z_b = (x_b - mean_b) / std_b

    baseline = _baseline_mi_pair(z_a, z_b, y, discrete_target=discrete_target,
                                  n_neighbors=n_neighbors)
    logger.debug(f"baseline MI(pair, y) = {baseline:.4f}")

    best: Optional[HermiteResult] = None

    degree_grid = list(range(2, max_degree + 1)) if sweep_degrees else [max_degree]

    for degree in degree_grid:
        # Coefficient vector size = degree + 1 (for c_0..c_degree).
        ca_size = degree + 1
        cb_size = degree + 1

        def objective(trial, _degree=degree):  # closure over degree
            coef_a = np.array([
                trial.suggest_float(f"a_{i}", *coef_range)
                for i in range(ca_size)
            ], dtype=np.float64)
            coef_b = np.array([
                trial.suggest_float(f"b_{i}", *coef_range)
                for i in range(cb_size)
            ], dtype=np.float64)

            h_a = hermeval(z_a, coef_a)
            h_b = hermeval(z_b, coef_b)

            # Guard: NaN/inf can arise if std blew up.
            if not (np.all(np.isfinite(h_a)) and np.all(np.isfinite(h_b))):
                return -np.inf

            best_mi = -np.inf
            for bf_name, bf in bin_funcs.items():
                try:
                    combined = bf(h_a, h_b)
                except Exception:
                    continue
                if not np.all(np.isfinite(combined)):
                    continue
                mi = _ksg_mi_1d(combined, y, discrete_target=discrete_target,
                                n_neighbors=n_neighbors)
                # L2 regularisation discourages high-magnitude coefficients.
                penalty = l2_penalty * (np.sum(coef_a ** 2) + np.sum(coef_b ** 2))
                score = mi - penalty
                if score > best_mi:
                    best_mi = score
                    trial.set_user_attr("bf_name", bf_name)
                    trial.set_user_attr("raw_mi", mi)
            return best_mi

        sampler = TPESampler(multivariate=True, seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
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
            mean_a=mean_a, std_a=std_a,
            mean_b=mean_b, std_b=std_b,
        )
        if best is None or cand.mi > best.mi:
            best = cand
        logger.debug(
            f"degree={degree}: best MI={raw_mi:.4f} (baseline {baseline:.4f}, "
            f"uplift {cand.uplift:.2f}x), bf={bf_name}"
        )

    if best is None or best.mi <= baseline * 1.01:
        # Failed to beat baseline by even 1% -- don't recommend an
        # engineered feature.
        return None
    return best
