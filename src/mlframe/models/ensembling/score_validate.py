"""Input validation prelude for ``score_ensemble``.

Carved out of ``_ensembling_score.py`` to keep the parent below the 1k-line monolith threshold. The function defined here is called at the top of ``score_ensemble``; when its ``early_res`` return value is non-empty (single-member / no-members sentinel), the caller short-circuits with that dict.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger("mlframe.models.ensembling")


def _validate_score_ensemble_inputs(
    level_models_and_predictions: Sequence[Any],
    ensembling_methods: Any,
    ensure_prob_limits: bool,
    max_ensembling_level: int,
    verbose: bool,
) -> tuple[dict[str, Any], bool, Any, bool]:
    """Run the input-validation prelude of ``score_ensemble``.

    Returns
    -------
    early_res : dict
        Non-empty (``{"_reason": ..., "_n_members": ...}``) when caller should ``return`` immediately (single-member / no-members short-circuit). Empty dict otherwise.
    is_regression : bool
        Inferred from the first member's probs availability.
    ensembling_methods : Any
        The original list with ``"rrf"`` filtered out on the regression path.
    ensure_prob_limits : bool
        Possibly toggled to False on the regression path.
    """
    res: dict[str, Any] = {}

    # SINGLE-MEMBER: short-circuit when only one member is supplied. There is no ensemble to score;
    # historically the caller filtered K==1 but score_ensemble itself silently iterated every flavour
    # over a 1-member tensor (the rrf/median/harm reduction is a no-op). Returning a sentinel-only
    # dict ({"_reason": "single_member"}) signals "no ensemble built" to the caller without raising,
    # AND lets finalize / metadata distinguish "single-member suite" from "ensemble failed silently"
    # (Low-9). The sentinel key starts with ``_`` so it is filtered out by the ensemble-iteration
    # logic in callers that iterate ``res.items()`` for real flavours.
    if len(level_models_and_predictions) < 2:
        if verbose and len(level_models_and_predictions) == 1:
            logger.info("[ensemble] only one member supplied; nothing to ensemble. Returning sentinel result.")
        if len(level_models_and_predictions) == 1:
            res["_reason"] = "single_member"
            res["_n_members"] = 1
        else:
            res["_reason"] = "no_members"
            res["_n_members"] = 0
        return res, False, ensembling_methods, ensure_prob_limits

    # Uniformity gate: mixing a classifier (probs available) with a regressor (probs == None)
    # in one ensemble silently miscategorises the suite. The historical dispatch only
    # inspected member[0]; member[1] could disagree with no error. Validate up front.
    if level_models_and_predictions:
        def _has_probs(m) -> bool:
            # ``oof_probs`` MUST be inspected too: a member with val_probs=None but oof_probs
            # populated (rare: trainer stamped OOF but disabled val-metric computation; or
            # cross_val_predict-only fits) is classifier-like, not regressor-like. Pre-fix the
            # check skipped oof_probs and mis-classified those members as regression.
            return any(getattr(m, attr, None) is not None for attr in ("oof_probs", "val_probs", "test_probs", "train_probs"))

        _probs_flags = [_has_probs(m) for m in level_models_and_predictions]
        if len(set(_probs_flags)) > 1:
            _clf_idx = [i for i, f in enumerate(_probs_flags) if f]
            _reg_idx = [i for i, f in enumerate(_probs_flags) if not f]
            raise ValueError(
                "score_ensemble requires uniform member types: got a mix of classifier-like "
                f"(probs available, indices {_clf_idx}) and regressor-like (no probs, indices "
                f"{_reg_idx}) members. Split the suite into per-task lists before calling."
            )

    _first = level_models_and_predictions[0]
    if getattr(_first, "oof_probs", None) is not None or _first.val_probs is not None or _first.test_probs is not None or _first.train_probs is not None:
        is_regression = False
    else:
        is_regression = True
        ensure_prob_limits = False

    # RRF is a rank-fusion flavour that only makes sense on classifier probabilities (where per-row ranks across the n_samples axis encode "confidence ordering"). For regression there is no analogous per-sample rank operation, so drop "rrf" silently from the candidate list rather than fail late inside _process_single_ensemble_method.
    if is_regression and ensembling_methods:
        _pre = list(ensembling_methods)
        ensembling_methods = [m for m in ensembling_methods if m != "rrf"]
        if verbose and len(ensembling_methods) != len(_pre):
            logger.info("[ensemble] target_type=REGRESSION: skipping rrf candidate (rank-fusion only meaningful on classifier probabilities).")

    # Multi-level stacking requires OOF predictions on EVERY member: the level-2 (and deeper) meta-learner consumes
    # level-1 ensemble outputs as features, and if any member contributes an in-sample ``train_preds`` row instead of
    # a ``cross_val_predict`` OOF row the meta-learner sees leaked targets. Fail fast rather than silently fold the
    # leakage forward. Single-level (``max_ensembling_level == 1``) aggregation tolerates missing OOF by falling back
    # to ``train_*`` because no downstream meta-learner consumes the train slice in that case. Membership uses
    # ``isinstance(..., np.ndarray)`` for the same reason as ``_oof_or_train``: MagicMock test doubles fabricate
    # any attribute on access, so ``is None`` would never fire on a real-world stub.
    if max_ensembling_level > 1:
        _oof_attr = "oof_probs" if not is_regression else "oof_preds"
        _missing_oof = [i for i, m in enumerate(level_models_and_predictions) if not isinstance(getattr(m, _oof_attr, None), np.ndarray)]
        if _missing_oof:
            raise ValueError(
                f"score_ensemble(max_ensembling_level={max_ensembling_level}) requires {_oof_attr} on every member; "
                f"members at indices {_missing_oof} are missing OOF. Re-train with oof_n_splits>=2 so cross_val_predict "
                f"OOFs are stamped on each model, or call with max_ensembling_level=1."
            )

    return res, is_regression, ensembling_methods, ensure_prob_limits
