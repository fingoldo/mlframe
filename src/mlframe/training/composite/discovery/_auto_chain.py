"""Auto-discover 2-stage transform CHAINS beyond the hardcoded ``chain_*`` set.

The registry ships a small, hand-curated set of residual->tail-compress chains
(``chain_linres_cbrt``, ``chain_linres_yj``, ``chain_monres_cbrt``,
``chain_monres_yj`` + the 3-stage ``chain_linres_cbrt_qn``). Those cover the
combinations a human happened to write down. This module CLOSES that gap: given
the residual transforms that survived single-stage screening, it composes EVERY
``residual x tail-compress-unary`` pair on the fly, scores each composed chain
against the two singletons (the residual alone, the unary alone), and surfaces a
winning chain as a discovery candidate.

It reuses the existing chain machinery -- ``_make_chain_transform`` from
``transforms.nonlinear`` and the raw unary fit/forward/inverse helpers from
``transforms.unary`` -- so a discovered chain is a first-class ``Transform`` with
the same fit/forward/inverse/domain contract as a hardcoded one. No transform
math is reimplemented here; this module is purely the search + scoring layer.

WHY MI-GAIN IS THE WRONG SCORER HERE (a real finding, not a footnote)
---------------------------------------------------------------------
The natural instinct is to score a chain by ``mi_gain = MI(T_chain, X)``, the
metric the rest of discovery uses. It does NOT work for tail-compression chains:
the second stage (cbrt / yeo-johnson / signed-power) is a MONOTONE map, and the
binned MI estimator is monotone-INVARIANT by construction (that is precisely why
the project flipped ``mi_estimator`` default to ``"bin"`` -- "bias-free under
monotone transforms", CLAUDE.md R10b). So ``MI(cbrt(T1), X) == MI(T1, X)`` to the
bin resolution -- a chain's MI-gain is IDENTICAL to its residual single's, and MI
can never see the win. Measured: linres gain 0.1694, chain linres+cbrt gain
0.1694 (bit-equal). The tail compression helps the DOWNSTREAM REGRESSOR (an
RMSE/L2 learner is sensitive to the target's scale + tail shape), not the MI.

THE SCORER WE ACTUALLY USE: tiny-CV RMSE on the ORIGINAL y-scale
---------------------------------------------------------------
For each candidate transform we fit a tiny GBM on the transformed target ``T`` in
a small CV, invert each fold's prediction back to ``y`` via the transform's
``inverse``, and score RMSE on the ORIGINAL y-scale. This is the honest,
apples-to-apples comparison: every candidate (raw-y, single residual, single
unary, chain) is judged by how well a fixed tiny model recovers the TRUE y. A
chain wins only when its y-scale RMSE beats BOTH singles by ``min_rmse_margin``.

VERDICT (committed bench ``_benchmarks/bench_auto_chain_rmse.py`` + biz_value
test): on a synthetic whose generating process IS residual-then-tail-compress
(``y = alpha*base + z**3`` with ``z`` linear in the held-out features, so the
heavy-tailed residual ``z**3`` is exactly un-cubed by the chain's signed cbrt),
the auto-discovered ``linres+cbrt`` chain beats both the lone residual and the
lone cbrt on y-scale OOF RMSE across the MAJORITY of seeds. Where no chain clears
the margin, ``discover_chains`` returns an empty list and the caller keeps the
best single -- the search ADDS candidates, never removes them, so it is safe to
run by default.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..transforms import TRANSFORMS_REGISTRY, Transform
from ..transforms.nonlinear import _make_chain_transform
from ..transforms.unary import (
    cbrt_y_fit as _cbrt_fit,
    cbrt_y_forward as _cbrt_fwd,
    cbrt_y_inverse as _cbrt_inv,
    yeo_johnson_y_fit as _yj_fit,
    yeo_johnson_y_forward as _yj_fwd,
    yeo_johnson_y_inverse as _yj_inv,
    signed_power_y_fit as _sp_fit,
    signed_power_y_forward as _sp_fwd,
    signed_power_y_inverse as _sp_inv,
)
from ._screening_tiny import _build_tiny_model
from .screening import _mi_to_target

logger = logging.getLogger(__name__)


# The tail-compression unaries available as the SECOND stage of a chain. Each is
# a registry-shaped ``(fit, forward, inverse)`` triple over the residual T1.
# cbrt: fixed signed cube root (parameter-free, robust). yj: Yeo-Johnson MLE
# lambda, adapts to the residual's actual skew/tail. sp: signed-power with a
# learned exponent in (0, 1], a continuous family bracketing cbrt (1/3) and
# identity (1). Together they span fixed / adaptive-power / MLE-power tail
# compressors -- the natural second-stage menu for a heavy-residual target.
_TAIL_UNARIES: Dict[str, Tuple[Callable, Callable, Callable]] = {
    "cbrt": (_cbrt_fit, _cbrt_fwd, _cbrt_inv),
    "yj": (_yj_fit, _yj_fwd, _yj_inv),
    "sp": (_sp_fit, _sp_fwd, _sp_inv),
}

# Short-tag -> standalone registry transform name for the single-unary baseline.
_UNARY_REGISTRY_NAME: Dict[str, str] = {
    "cbrt": "cbrt_y",
    "yj": "yeo_johnson_y",
    "sp": "signed_power_y",
}

# Registry transforms that act as the FIRST (residual) stage. Restricted to the
# residual family -- a chain is "absorb the base, then compress the leftover
# tail", so the first stage must be a base-dependent residual and must NOT itself
# be a unary tail-compressor (that would just be two unaries).
_RESIDUAL_STAGE_NAMES: Tuple[str, ...] = (
    "linear_residual",
    "monotonic_residual",
)


@dataclass(frozen=True)
class ChainCandidate:
    """One auto-discovered chain + the measurements that justify it.

    ``rmse`` is the tiny-CV RMSE on the ORIGINAL y-scale (lower is better) -- the
    transform's predictions inverted back to y. ``residual_rmse`` / ``unary_rmse``
    / ``raw_rmse`` are the SAME metric for the single residual, the single unary,
    and untransformed raw-y. ``margin = min(residual_rmse, unary_rmse) - rmse`` is
    the RMSE improvement over the BETTER single that had to clear
    ``min_rmse_margin`` for this candidate to surface (positive = chain wins).
    ``mi_gain`` is carried for information only -- it is monotone-blind to the
    second stage and so cannot rank chains (see module docstring).
    """

    chain_name: str  # registry-style key, e.g. "chain_linear_residual_cbrt"
    short_name: str  # composed-name fragment, e.g. "linres+cbrt"
    residual_name: str
    unary_name: str
    transform: Transform  # ready-to-fit Transform built via _make_chain_transform
    fitted_params: Dict[str, Any]
    rmse: float
    residual_rmse: float
    unary_rmse: float
    raw_rmse: float
    margin: float
    mi_gain: float
    valid_domain_frac: float


def _short(residual_name: str, unary_name: str) -> str:
    """Build a short display name for a chained (residual + unary) transform, abbreviating the common residual kinds."""
    res_frag = {"linear_residual": "linres", "monotonic_residual": "monres"}.get(residual_name, residual_name)
    return f"{res_frag}+{unary_name}"


def build_chain_transform(residual_name: str, unary_name: str) -> Transform:
    """Compose ``residual_name`` (registry bivariate) with ``unary_name`` (tail unary).

    Pulls the bivariate's fit/forward/inverse/domain straight off the registry
    ``Transform`` and the unary's raw triple from ``_TAIL_UNARIES``, then hands
    both to the shared ``_make_chain_transform`` factory. The result is a full
    ``Transform`` indistinguishable from a hardcoded ``chain_*`` entry -- same
    contract, same params layout -- so downstream fit/predict/serialise code
    needs zero special-casing for an auto-discovered chain.
    """
    if residual_name not in TRANSFORMS_REGISTRY:
        raise KeyError(f"residual stage {residual_name!r} not in registry")
    if unary_name not in _TAIL_UNARIES:
        raise KeyError(f"unary stage {unary_name!r} not a tail-compress unary")
    biv = TRANSFORMS_REGISTRY[residual_name]
    u_fit, u_fwd, u_inv = _TAIL_UNARIES[unary_name]
    return _make_chain_transform(
        name=f"chain_{residual_name}_{unary_name}",
        short_name=_short(residual_name, unary_name),
        bivariate_fit=biv.fit,
        bivariate_forward=biv.forward,
        bivariate_inverse=biv.inverse,
        bivariate_domain=biv.domain_check,
        unary_fit=u_fit,
        unary_forward=u_fwd,
        unary_inverse=u_inv,
        description=(
            f"Auto-discovered chain: {residual_name} (residual) then "
            f"{unary_name} (tail compression). Surfaced by _auto_chain because "
            f"the composition beat both single stages on held-out y-scale RMSE."
        ),
    )


def reregister_auto_chain_transforms(transform_names: Iterable[str] | None) -> list[str]:
    """Re-register auto-discovered chain transforms (``chain_<residual>_<unary>``) that are referenced by name but
    absent from the registry -- the case after a discovery CACHE replay, where the in-process registration done by
    ``_run_auto_chain`` (``_TRANSFORMS_REGISTRY.setdefault`` + provenance) never ran, so a cached spec naming such a
    chain crashes ``get_transform`` (and predict-time inversion) with ``UnknownTransformError``.

    Parses each ``chain_<residual>_<unary>`` name against the known residual-stage / tail-unary menus, rebuilds the
    Transform via :func:`build_chain_transform`, and registers it + its provenance. Non-chain / already-registered /
    unparseable names are skipped. Returns the names actually re-registered.
    """
    from ..transforms.registry import _TRANSFORMS_REGISTRY
    from ..provenance import register_chain_provenance
    done: list[str] = []
    for nm in {n for n in (transform_names or []) if isinstance(n, str)}:
        if not nm.startswith("chain_") or nm in _TRANSFORMS_REGISTRY:
            continue
        body = nm[len("chain_") :]
        for un in _TAIL_UNARIES:
            suffix = "_" + un
            if body.endswith(suffix):
                res = body[: -len(suffix)]
                if res in _RESIDUAL_STAGE_NAMES:
                    try:
                        _TRANSFORMS_REGISTRY.setdefault(nm, build_chain_transform(res, un))
                        register_chain_provenance(nm, res, un)
                        done.append(nm)
                    except Exception as e:
                        logger.debug("swallowed exception in _auto_chain.py: %s", e)
                        pass
                break
    return done


def _y_scale_cv_rmse(
    transform: Optional[Transform],
    *,
    y: np.ndarray,
    base: np.ndarray,
    x_matrix: np.ndarray,
    cv_folds: int,
    random_state: int,
    family: str,
    n_estimators: int,
    num_leaves: int,
    learning_rate: float,
) -> Tuple[float, float]:
    """Tiny-CV RMSE on the ORIGINAL y-scale for one transform (``None`` = raw y).

    Per fold: fit ``transform`` on the train rows, forward to ``T``, fit a tiny
    GBM ``T ~ X`` on train, predict ``T_hat`` on val, invert ``T_hat -> y_hat``
    via ``transform.inverse``, and accumulate squared error against the TRUE
    ``y_val``. ``transform=None`` skips all transform steps and fits ``y ~ X``
    directly (the raw-y baseline). Returns ``(rmse, valid_domain_frac)``;
    ``rmse=inf`` when the transform fails or too few rows survive.

    Out-of-domain rows (per ``transform.domain_check``) are dropped from the
    train fit but kept on the val side scored against ``y_val`` directly -- the
    transform cannot improve a row it cannot represent, so dropping it from val
    would flatter the chain. Identical handling for every candidate keeps the
    comparison apples-to-apples.
    """
    from sklearn.model_selection import KFold

    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < cv_folds * 10:
        return float("inf"), 0.0
    valid_frac = 1.0
    if transform is not None:
        dom = np.asarray(transform.domain_check(y, base), dtype=bool)
        valid_frac = float(dom.mean()) if dom.size else 0.0
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    sse = 0.0
    cnt = 0
    for tr_idx, va_idx in kf.split(x_matrix):
        x_tr, x_va = x_matrix[tr_idx], x_matrix[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        b_tr, b_va = base[tr_idx], base[va_idx]
        try:
            if transform is None:
                target_tr = y_tr
                fit_mask = np.isfinite(y_tr)
            else:
                dom_tr = np.asarray(transform.domain_check(y_tr, b_tr), dtype=bool)
                if dom_tr.sum() < cv_folds * 5:
                    return float("inf"), valid_frac
                params = transform.fit(y_tr[dom_tr], b_tr[dom_tr])
                t_tr = np.asarray(transform.forward(y_tr, b_tr, params), dtype=np.float64)
                target_tr = t_tr
                fit_mask = dom_tr & np.isfinite(t_tr)
            if fit_mask.sum() < cv_folds * 5:
                return float("inf"), valid_frac
            model = _build_tiny_model(
                family, n_estimators=n_estimators, num_leaves=num_leaves,
                learning_rate=learning_rate, random_state=random_state,
                inner_n_jobs=1,
            )
            model.fit(x_tr[fit_mask], target_tr[fit_mask])
            pred = np.asarray(model.predict(x_va), dtype=np.float64)
            if transform is None:
                y_hat = pred
            else:
                y_hat = np.asarray(transform.inverse(pred, b_va, params), dtype=np.float64)
        except Exception as exc:
            logger.debug("y-scale CV fold failed for %s: %s", getattr(transform, "name", "raw"), exc)
            return float("inf"), valid_frac
        ok = np.isfinite(y_hat) & np.isfinite(y_va)
        if not ok.any():
            return float("inf"), valid_frac
        sse += float(np.sum((y_hat[ok] - y_va[ok]) ** 2))
        cnt += int(ok.sum())
    if cnt == 0:
        return float("inf"), valid_frac
    return float(np.sqrt(sse / cnt)), valid_frac


def _mi_gain_of(
    transform: Transform,
    *,
    y: np.ndarray,
    base: np.ndarray,
    x_matrix: np.ndarray,
    mi_y: float,
    mi_estimator: str,
    mi_nbins: int,
    mi_n_neighbors: int,
    random_state: int,
) -> float:
    """``MI(T, X) - mi_y`` for one transform (informational; see module docstring)."""
    dom = np.asarray(transform.domain_check(y, base), dtype=bool)
    if dom.sum() < 8:
        return float("nan")
    try:
        params = transform.fit(y[dom], base[dom])
        t = np.asarray(transform.forward(y, base, params), dtype=np.float64)
    except Exception:
        return float("nan")
    finite = np.isfinite(t) & dom
    if finite.sum() < 8:
        return float("nan")
    mi_t = _mi_to_target(
        x_matrix[finite], t[finite], n_neighbors=mi_n_neighbors,
        random_state=random_state, estimator=mi_estimator, nbins=mi_nbins,
    )
    return float(mi_t - mi_y)


def discover_chains(
    *,
    y: np.ndarray,
    base: np.ndarray,
    x_matrix: np.ndarray,
    residual_names: Optional[Sequence[str]] = None,
    unary_names: Optional[Sequence[str]] = None,
    min_rmse_margin: float = 0.0,
    min_valid_domain_frac: float = 0.5,
    cv_folds: int = 4,
    random_state: int = 0,
    family: str = "lgb",
    n_estimators: int = 60,
    num_leaves: int = 15,
    learning_rate: float = 0.1,
    compute_mi_gain: bool = True,
    mi_estimator: str = "bin",
    mi_nbins: int = 16,
    mi_n_neighbors: int = 3,
    top_k: int = 3,
) -> List[ChainCandidate]:
    """Search ``residual x unary`` chains; return those that beat BOTH single stages.

    Scoring is tiny-CV RMSE on the ORIGINAL y-scale (see module docstring for why
    MI-gain cannot rank tail-compression chains). A chain surfaces only when its
    y-scale RMSE beats ``min(residual_rmse, unary_rmse)`` by at least
    ``min_rmse_margin``.

    Parameters
    ----------
    y, base : 1-D arrays, same length. ``base`` is the single base column the
        residual stage regresses ``y`` on (the kept single-stage spec's base).
    x_matrix : (n, F) feature matrix the tiny model + MI use. The SAME matrix +
        CV folds score every candidate, so the RMSEs are directly comparable.
    residual_names : restrict the first-stage menu (default: ``linear_residual`` +
        ``monotonic_residual``). Pass the kept single-stage residual specs' names
        to search only what survived screening.
    unary_names : restrict the second-stage menu (default: cbrt / yj / sp).
    min_rmse_margin : a chain must beat the better single's RMSE by at least this
        absolute amount. ``0.0`` = strictly better.
    min_valid_domain_frac : chains whose residual stage is valid on fewer than
        this fraction of rows are dropped (mirrors discovery's domain gate).
    family : tiny-model family for the CV scorer (``"lgb"`` / ``"cb"`` /
        ``"ridge"``), passed to :func:`_build_tiny_model`.

    Returns the ``top_k`` winning ``ChainCandidate`` objects sorted by ASCENDING
    ``rmse`` (best first). Empty list = no chain beat its singles -> caller keeps
    the best single. NEVER removes single-stage candidates; it only proposes.
    """
    y = np.asarray(y, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    x_matrix = np.asarray(x_matrix, dtype=np.float64)
    if x_matrix.ndim == 1:
        x_matrix = x_matrix.reshape(-1, 1)
    res_names = tuple(residual_names) if residual_names else _RESIDUAL_STAGE_NAMES
    un_names = tuple(unary_names) if unary_names else tuple(_TAIL_UNARIES)

    cv_kw: Dict[str, Any] = dict(
        y=y, base=base, x_matrix=x_matrix, cv_folds=cv_folds,
        random_state=random_state, family=family, n_estimators=n_estimators,
        num_leaves=num_leaves, learning_rate=learning_rate,
    )

    raw_rmse, _ = _y_scale_cv_rmse(None, **cv_kw)

    # Score each single residual + single unary ONCE; reuse for every chain that
    # contains them (the "beats both singles" gate needs both numbers).
    residual_rmse: Dict[str, float] = {}
    for res in res_names:
        rtf = TRANSFORMS_REGISTRY.get(res)
        if rtf is None:
            continue
        residual_rmse[res], _ = _y_scale_cv_rmse(rtf, **cv_kw)

    unary_rmse: Dict[str, float] = {}
    for un in un_names:
        utf = TRANSFORMS_REGISTRY.get(_UNARY_REGISTRY_NAME.get(un, un))
        if utf is not None:
            unary_rmse[un], _ = _y_scale_cv_rmse(utf, **cv_kw)

    mi_y = float("nan")
    if compute_mi_gain:
        fy = np.isfinite(y)
        if fy.sum() >= 8:
            mi_y = _mi_to_target(
                x_matrix[fy], y[fy], n_neighbors=mi_n_neighbors,
                random_state=random_state, estimator=mi_estimator, nbins=mi_nbins,
            )

    candidates: List[ChainCandidate] = []
    for res in res_names:
        if res not in TRANSFORMS_REGISTRY:
            continue
        rr = residual_rmse.get(res, float("inf"))
        for un in un_names:
            chain_tf = build_chain_transform(res, un)
            cr, vf = _y_scale_cv_rmse(chain_tf, **cv_kw)
            ur = unary_rmse.get(un, float("inf"))
            best_single = min(rr, ur)
            margin = best_single - cr
            if not (np.isfinite(cr) and vf >= min_valid_domain_frac and margin > min_rmse_margin):
                continue
            mg = float("nan")
            if compute_mi_gain and np.isfinite(mi_y):
                # Fit the chain on the full in-domain rows just for the (informational) MI gain.
                mg = _mi_gain_of(
                    chain_tf, y=y, base=base, x_matrix=x_matrix, mi_y=mi_y,
                    mi_estimator=mi_estimator, mi_nbins=mi_nbins,
                    mi_n_neighbors=mi_n_neighbors, random_state=random_state,
                )
            # Fit once on all in-domain rows so the candidate carries usable params.
            dom = np.asarray(chain_tf.domain_check(y, base), dtype=bool)
            try:
                params = chain_tf.fit(y[dom], base[dom])
            except Exception:
                params = {}
            candidates.append(
                ChainCandidate(
                    chain_name=chain_tf.name,
                    short_name=_short(res, un),
                    residual_name=res,
                    unary_name=un,
                    transform=chain_tf,
                    fitted_params=params,
                    rmse=cr,
                    residual_rmse=rr,
                    unary_rmse=ur,
                    raw_rmse=raw_rmse,
                    margin=margin,
                    mi_gain=mg,
                    valid_domain_frac=vf,
                )
            )
    candidates.sort(key=lambda c: c.rmse)
    return candidates[:top_k]
