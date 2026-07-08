"""Knockoff-based feature importance (Barber-Candes 2015 et seq.) for RFECV.

Carved out of _helpers.py in the Wave 5 sweep (2026-05-28) to keep _helpers.py below the 1k-LOC soft limit. Re-exported at the
parent's bottom so from mlframe.feature_selection.wrappers import knockoff_importance / select_features_fdr / make_gaussian_knockoffs
keep resolving exactly as before.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

# Lazy local-import shim to avoid a load-time circular dependency: ``_helpers.py``
# re-exports our public API at its bottom, and we need ``get_feature_importances``
# from ``_helpers.py`` for the W='gain'/'coef'/legacy paths.
def _get_feature_importances(*args, **kwargs):
    """Lazy-imported forward to ``_helpers.get_feature_importances`` (deferred to avoid a load-time circular import)."""
    from ._helpers import get_feature_importances as _gfi
    return _gfi(*args, **kwargs)


logger = logging.getLogger(__name__)


# E8 toggle: when True the Gaussian-knockoff builder raises ValueError instead
# of just warning when lambda_min(Sigma) is below 1e-4 (degenerate near-singular
# correlation matrix). Set via module-level globals().setdefault from the call
# site that explicitly opted into knockoffs.
_KNOCKOFFS_STRICT_LAM_MIN = False


def make_gaussian_knockoffs(X, random_state=None, sdp_solve: bool = False) -> np.ndarray:
    """Generate model-X Gaussian knockoffs (Candes et al. 2018, "Panning for Gold").

    This is the model-X construction (X assumed ~ N(mu, Sigma)), NOT the fixed-design Barber-Candes 2015 procedure:
    the FDR guarantee here holds in expectation over the Gaussian design, and degrades when X is non-Gaussian or
    near-collinear (s -> 0 makes the knockoff a near-copy with near-zero power; see the lambda_min guard below).

    For each X_j a knockoff X_tilde_j is produced that has the same
    correlation with X_{-j} as X_j does, but is independent of y. This
    lets us identify 'real' importance: a feature is selected if its
    importance >> its knockoff's importance under the same fitted model.

    Equicorrelated construction (default): s_j = s for all j, with
    s = min(2 * lambda_min(Sigma), 1) where Sigma = corr(X). This is the
    cheap closed-form path; the SDP-based optimal s requires cvxpy and
    gives slightly tighter knockoffs but is rarely worth the dependency.

    Parameters
    ----------
    X : ndarray (n, p)
        Numeric design matrix. Will be standardized internally; the
        returned X_tilde matches the standardized scale.
    random_state : int or None
        Seed for the noise injection.
    sdp_solve : bool
        Reserved for future SDP-based s; currently raises
        NotImplementedError if True.

    Returns
    -------
    X_tilde : ndarray (n, p)
        Standardized knockoff matrix, same shape as X.
    """
    if sdp_solve:
        raise NotImplementedError("SDP knockoffs not yet implemented; use equicorrelated default.")

    rng = np.random.default_rng(random_state)
    # E10 (Wave 4, 2026-05-28): explicit numeric check. ``np.asarray(X, dtype=float)`` would silently NaN-fill an object/string column;
    # detect those upstream and raise so the user fixes encoding rather than getting a useless all-NaN-knockoff path.
    if hasattr(X, "select_dtypes"):
        try:
            _non_numeric = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
        except (TypeError, AttributeError):
            _non_numeric = []
        if _non_numeric:
            raise ValueError(
                f"make_gaussian_knockoffs: non-numeric column(s) {_non_numeric[:10]} "
                f"would coerce to NaN via float conversion. Encode (e.g. target/ordinal) "
                f"before requesting knockoff-based importance."
            )
    try:
        X_arr = np.asarray(X, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"make_gaussian_knockoffs: failed to convert X to float ndarray ({exc}); " f"all knockoff columns must be numeric. Encode categoricals upstream."
        ) from exc
    n, p = X_arr.shape
    if n < 2 or p < 1:
        raise ValueError(f"X must have at least 2 rows and 1 column; got {X_arr.shape}")

    # Standardise X (zero mean, unit variance per column) so Sigma is correlation
    means = np.nanmean(X_arr, axis=0)
    stds = np.nanstd(X_arr, axis=0)
    stds = np.where(stds > 1e-12, stds, 1.0)
    X_std = (X_arr - means) / stds
    # Replace any NaNs (from constant columns) with 0
    X_std = np.where(np.isnan(X_std), 0.0, X_std)

    # Sigma = correlation matrix. Tiny ridge keeps it positive definite on near-collinear inputs.
    Sigma = (X_std.T @ X_std) / max(1, n - 1)
    Sigma = Sigma + 1e-8 * np.eye(p)

    # Equicorrelated s: s_j = s for all j, where s = min(2*lambda_min, 1).
    eigvals = np.linalg.eigvalsh(Sigma)
    lam_min = float(max(eigvals[0], 1e-8))
    # When Sigma is near-singular (anti-correlated pairs X_j = -X_k or 100% collinear copies),
    # lam_min ~ 1e-8 -> s_val ~ 2e-8 -> X_tilde becomes ~ X (self-corr ~ 1, useless as knockoff).
    # E8 (Wave 4, 2026-05-28): when ``strict_lam_min=True``, RAISE instead of just warning -- callers explicitly opting into knockoff
    # selection get a hard signal that the knockoffs are degenerate; otherwise they get an empty support_ with no obvious cause.
    if lam_min < 1e-4:
        _msg = (
            "make_gaussian_knockoffs: input correlation matrix has "
            f"lambda_min={lam_min:.2e} (near-singular); knockoffs will be near-copies "
            "of original features (self-corr ~ 1) and W statistics ~ 0. "
            "Reduce collinearity (drop duplicates / use feature_groups) "
            "or use stability_selection instead."
        )
        if globals().get("_KNOCKOFFS_STRICT_LAM_MIN", False):
            raise ValueError(_msg + " (strict_lam_min=True)")
        logger.warning(_msg)
    s_val = min(2.0 * lam_min, 1.0) * 0.99  # 0.99 buffer for numerical PSD
    s = np.full(p, s_val)

    # Knockoff construction:
    #   X_tilde = X_std (I - Sigma^{-1} diag(s)) + Z C^T
    # where C C^T = 2 diag(s) - diag(s) Sigma^{-1} diag(s) (must be PSD).
    # Pseudo-inverse is a safety fallback on near-singular Sigma.
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma)
    diag_s = np.diag(s)
    A = np.eye(p) - Sigma_inv @ diag_s

    M = 2.0 * diag_s - diag_s @ Sigma_inv @ diag_s
    # Ensure M is symmetric PSD numerically; add ridge if needed.
    M = 0.5 * (M + M.T)
    eigvals_M = np.linalg.eigvalsh(M)
    if eigvals_M[0] < 0:
        M = M + (-eigvals_M[0] + 1e-8) * np.eye(p)
    C = np.linalg.cholesky(M)

    Z = rng.standard_normal((n, p))
    X_tilde_std = X_std @ A + Z @ C.T

    # Return on the original scale; estimators don't typically standardise themselves.
    X_tilde = X_tilde_std * stds + means
    return np.asarray(X_tilde)


def select_features_fdr(W: dict, q: float = 0.1) -> list:
    """Barber-Candes FDR-controlled feature selection from a knockoff W
    statistic dict.

    Picks features with W_j >= tau, where
        tau = min{t > 0 : (1 + #{j: W_j <= -t}) / max(1, #{j: W_j >= t}) <= q}
    The probability that a noise feature is in the selected set is bounded
    by q (Barber & Candes 2015, Theorem 1). Returns [] if no threshold
    achieves the target FDR (typical on small n / weak signal).

    Low-power caveat (NOT a bug): on few positive W (e.g. one strong driver + one negative knockoff lead) the
    ``(1 + #neg)`` numerator offset cannot be beaten at small support, so even an obvious driver yields ``[]``. This
    is correct Barber-Candes behaviour -- the data-splitting "knockoff+" offset is the price of finite-sample FDR
    control -- not a defect. When the support is small, prefer a marginal screen (univariate-HT) for power.

    Parameters
    ----------
    W : dict
        Mapping feature_name -> W_j statistic (output of
        ``knockoff_importance``).
    q : float
        Target FDR in (0, 1). Lower = more conservative selection.

    Returns
    -------
    list of feature names with W_j >= tau, sorted by W_j desc.
    """
    if not W:
        return []
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0, 1); got {q}")
    abs_W = np.array([abs(v) for v in W.values()])
    candidates = sorted(set(abs_W[abs_W > 0]))
    tau = float("inf")
    for t in candidates:
        n_neg = sum(1 for v in W.values() if v <= -t)
        n_pos = sum(1 for v in W.values() if v >= t)
        ratio = (1 + n_neg) / max(1, n_pos)
        if ratio <= q:
            tau = t
            break
    if not np.isfinite(tau):
        return []
    selected = [(n, v) for n, v in W.items() if v >= tau]
    # Wave 58 (2026-05-20): secondary key on feature name; tied |W| (shrinkage
    # saturation) no longer makes downstream [:topN] slicing drift.
    selected.sort(key=lambda kv: (-kv[1], kv[0]))
    return [n for n, _ in selected]


def knockoff_importance(model_factory, X, y, current_features=None, random_state=None, importance_getter: str = "auto", w_statistic: str = "auto") -> dict:
    """Compute knockoff-based importance: W_j = imp(X_j) - imp(X_tilde_j).

    Builds Gaussian knockoffs X_tilde, fits a fresh model on [X, X_tilde]
    (2p columns), reads the importance of each REAL feature j and its
    KNOCKOFF j, returns the difference. Real features driving y will have
    W_j >> 0; noise features have W_j ~ N(0, sigma) symmetric around 0.

    Sign-symmetry (required for the Barber-Candes FDR guarantee): each feature's real and knockoff columns are placed in a
    RANDOM order (fair per-feature coin), so under the null the sign of W_j = imp(real) - imp(knockoff) is symmetric even when
    the importance source is non-negative (|SHAP| / gain). Without this swap, a non-negative importance gives no negative W_j,
    the FDR threshold's negative reference set is empty, and the procedure provides NO real control.

    L3 (Wave 5, 2026-05-28): ``w_statistic`` chooses the importance source for
    the W computation:
        - 'auto'   : tree models -> TreeSHAP mean(|shap|), else coef_ / FI (legacy default).
        - 'shap'   : force TreeSHAP (raises if shap import fails).
        - 'gain'   : feature_importances_ (split-gain for trees).
        - 'coef'   : |coef_|.
    KOBT (Jiang-Cheng-Zhao 2021) shows TreeSHAP-W has higher power on tree models with the same FDR floor.

    Parameters
    ----------
    model_factory : callable
        ``model_factory()`` must return a fresh unfitted estimator. We don't
        accept a pre-fit model because knockoffs need a fit on [X, X_tilde].
    X : DataFrame or ndarray (n, p)
    y : array-like (n,)
    current_features : list of feature names (optional)
        If None, defaults to X.columns or range(p).
    random_state : int or None
    importance_getter : same semantics as get_feature_importances
    w_statistic : 'auto' / 'shap' / 'gain' / 'coef'

    Returns
    -------
    Dict mapping feature_name -> W_j (knockoff statistic).
    """
    is_df = hasattr(X, "columns")
    X_arr = X.values if is_df else np.asarray(X)
    _n, p = X_arr.shape
    if current_features is None:
        current_features = list(X.columns) if is_df else list(range(p))

    X_tilde = make_gaussian_knockoffs(X_arr, random_state=random_state)

    # Flip-sign antisymmetry (Barber-Candes 2015, eq. for the swap statistic). The importance-difference W_j = imp(X_j) - imp(X_tilde_j)
    # only yields a valid FDR guarantee if W is ANTISYMMETRIC under swapping a null feature with its knockoff: swapping must flip the sign
    # of W_j. With real/knockoff in FIXED column positions and a NON-NEGATIVE importance source (|SHAP| / gain), imp(real) - imp(fake) is
    # NOT sign-symmetric under the null -- there is no mechanism producing negative W_j, so the (1 + #neg)/#pos threshold has an empty
    # negative reference set and gives no real control. We restore antisymmetry by drawing an independent fair coin per feature that
    # decides which of the two adjacent columns carries the REAL value and which carries the knockoff; the model cannot tell them apart,
    # so under the null the sign of (imp at real-slot - imp at knockoff-slot) is symmetric -> W is sign-symmetric as the theory requires.
    swap_rng = np.random.default_rng(random_state)
    swapped = swap_rng.random(p) < 0.5  # True -> real and knockoff columns are swapped for this feature
    # Column 2*i holds the value placed first, 2*i+1 the value placed second; ``swapped[i]`` records whether first==knockoff.
    cols = []
    for i in range(p):
        if swapped[i]:
            cols.append(X_tilde[:, i])
            cols.append(X_arr[:, i])
        else:
            cols.append(X_arr[:, i])
            cols.append(X_tilde[:, i])
    X_joint = np.column_stack(cols)
    # Slot names are position-based (_slotA_i / _slotB_i); which slot is real depends on swapped[i].
    real_names = [f"_slotA_{i}" for i in range(p)]
    fake_names = [f"_slotB_{i}" for i in range(p)]
    joint_names = []
    for i in range(p):
        joint_names.append(real_names[i])
        joint_names.append(fake_names[i])
    if is_df:
        X_joint_df = pd.DataFrame(X_joint, columns=joint_names, index=X.index)
    else:
        X_joint_df = X_joint

    model = model_factory()
    model.fit(X_joint_df, y)

    # L3 (Wave 5, 2026-05-28): pick the W-statistic source per ``w_statistic``.
    _w = w_statistic
    if _w == "auto":
        # Detect tree-family: any of feature_importances_ AND class-name contains "Forest" / "Boost" / "Tree".
        _name = type(model).__name__
        _is_tree = hasattr(model, "feature_importances_") and any(t in _name for t in ("Forest", "Tree", "Boost", "LGBM", "XGB", "CatBoost", "GBM"))
        _w = "shap" if _is_tree else ("coef" if hasattr(model, "coef_") else "gain")
    if _w == "shap":
        # TreeSHAP path.
        try:
            import shap as _shap
        except ImportError as _exc:
            raise ImportError(
                "w_statistic='shap' requires the optional ``shap`` package. " "Install via ``pip install shap`` or set w_statistic='gain'/'coef'."
            ) from _exc
        try:
            _expl = _shap.TreeExplainer(model)
            _vals = _expl.shap_values(X_joint_df)
            _arr = _vals if not isinstance(_vals, list) else np.stack(_vals)
            if _arr.ndim > 2:
                _arr = np.abs(_arr).mean(axis=tuple(range(2, _arr.ndim)))
            fi_vals = np.abs(_arr).mean(axis=0)
            fi = {n: float(v) for n, v in zip(joint_names, fi_vals)}
        except Exception as _exc:
            raise RuntimeError(f"TreeSHAP W-statistic failed for {type(model).__name__}: {_exc}. " f"Set w_statistic='gain' or 'coef' to fall back.") from _exc
    elif _w == "gain":
        fi = _get_feature_importances(
            model=model, current_features=joint_names,
            data=X_joint_df, target=y, importance_getter="feature_importances_",
        )
    elif _w == "coef":
        fi = _get_feature_importances(
            model=model, current_features=joint_names,
            data=X_joint_df, target=y, importance_getter="coef_",
        )
    else:
        fi = _get_feature_importances(
            model=model, current_features=joint_names,
            data=X_joint_df, target=y, importance_getter=importance_getter,
        )
    W = {}
    for i, fname in enumerate(current_features):
        imp_slotA = float(fi.get(f"_slotA_{i}", 0.0))
        imp_slotB = float(fi.get(f"_slotB_{i}", 0.0))
        # ``swapped[i]`` True -> slotA was the knockoff, slotB the real value. W_j = imp(real) - imp(knockoff). The per-feature coin
        # flip is what makes the sign of W_j symmetric under the null (Barber-Candes swap antisymmetry), so the FDR threshold's
        # negative reference set is well-defined.
        if swapped[i]:
            imp_real, imp_fake = imp_slotB, imp_slotA
        else:
            imp_real, imp_fake = imp_slotA, imp_slotB
        W[fname] = imp_real - imp_fake
    return W
