"""Recovery from CatBoost's "Dictionary size is 0" text-feature failure, split out of ``_training_loop``."""

from __future__ import annotations

import logging
from typing import Any

# The parent module's logger name: these lines predate the split, and log filters select them by that name.
logger = logging.getLogger("mlframe.training._training_loop")


def retry_params_after_empty_text_dictionary(model: Any, train_df: Any, train_target: Any, fit_params: dict) -> tuple[bool, dict]:
    """``(try_again, fit_params)`` after CatBoost raised "Dictionary size is 0" for at least one text column.

    Tries a unigram dictionary first, which keeps every text feature; otherwise drops only the columns the installed
    CatBoost cannot build a vocabulary for, rerouting them to ``cat_features``. With no text features in the params the
    error is an unexpected variant: ``try_again`` is False and the caller re-raises.
    """
    try_again = False
    text_feat = fit_params.get("text_features") or []
    if text_feat:
        try:
            from mlframe.training.cb import (
                unigram_rescues_text_features,
                unigram_text_processing,
                unusable_text_features,
            )

            # The default text pipeline builds BIGRAMS, and a column of one token per row can never
            # produce one -- which is what empties the dictionary. Switching to unigrams keeps every
            # text feature instead of discarding the columns the caller deliberately promoted, so try
            # that before considering any of them unusable.
            _rescued = False
            if unigram_rescues_text_features(train_df, train_target, text_feat, verbose=True):
                # ``text_processing`` is a CatBoost PARAMETER, not a ``fit()`` keyword -- passing it
                # through fit_params raises TypeError and takes the whole suite down instead of
                # rescuing the fit. The scaled-occurrence path in ``_training_loop`` already sets it
                # the supported way; do the same here.
                try:
                    model.set_params(text_processing=unigram_text_processing())
                    _rescued = True
                except Exception as _tp_exc:
                    logger.warning(
                        "unigram rescue could not be applied (%s: %s); falling back to probing each "
                        "text feature individually.", type(_tp_exc).__name__, _tp_exc,
                    )
            if _rescued:
                logger.warning(
                    "CatBoost raised 'Dictionary size is 0' because its DEFAULT text processing builds "
                    "word bigrams and %d text feature(s) %s carry a single token per row. Retrying with "
                    "a unigram dictionary, which keeps all of them rather than dropping any.",
                    len(text_feat), text_feat,
                )
                try_again = True
                _bad = {}  # the rescue keeps every column, so nothing is dropped
            else:
                _bad = unusable_text_features(train_df, train_target, text_feat, verbose=True)
        except Exception as _probe_exc:
            logger.debug("text-feature probe unavailable (%s); dropping all text features", _probe_exc)
            _bad = {c: "probe unavailable" for c in text_feat}
        _keep = [c for c in text_feat if c not in _bad]
        if _bad:
            logger.warning(
                "CatBoost raised 'Dictionary size is 0'. Probing each text feature individually against the "
                "installed CatBoost identified %d unusable of %d: %s. Retrying with the remaining %d text "
                "feature(s) %s instead of dropping them all.",
                len(_bad), len(text_feat), "; ".join(f"{c} ({r})" for c, r in _bad.items()) or "(none)",
                len(_keep), _keep or "(none)",
            )
            # A dropped text column is rerouted to cat_features so CB's categorical handling still sees it;
            # left out entirely, CB tries to cast its strings to float and raises "Cannot convert 'X' to float".
            _existing_cats = list(fit_params.get("cat_features") or [])
            _moved_to_cat = [c for c in _bad if c not in _existing_cats]
            if _keep:
                fit_params = dict(fit_params)
                fit_params["text_features"] = _keep
            else:
                fit_params = {k: v for k, v in fit_params.items() if k != "text_features"}
            if _moved_to_cat:
                fit_params["cat_features"] = _existing_cats + _moved_to_cat
            try_again = True
    return try_again, fit_params
