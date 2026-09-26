"""Constructor-string validation for ``mlframe.feature_selection.filters.mrmr``.

Carved out of ``_mrmr_validate_transform`` so that module stays under the 1k-LOC ceiling. Both functions are bound onto the ``MRMR`` class
at the package's module bottom exactly as before, and are re-exported from ``_mrmr_validate_transform`` so existing import sites keep
working. They validate only the constructor's string-valued parameters: nothing here reads data, so the pass is cheap enough to run at the
start of every fit.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


# Values the validator accepts because they are part of the documented menu, but which no implementation distinguishes yet: a fit asked for
# either of these runs the fleuret redundancy path and produces identical output. Silently aliasing is the problem; saying so is the fix.
_UNIMPLEMENTED_REDUNDANCY_ALGOS = {
    "pld_max": "max over the selected set",
    "pld_mean": "mean over the selected set",
}


def _validate_string_params(self):
    """Raise ValueError on bad constructor strings. Each branch lists the accepted values verbatim so the error message is actionable."""
    _checks = (
        ("quantization_method", self._VALID_QUANTIZATION_METHODS),
        ("nan_strategy", self._VALID_NAN_STRATEGIES),
        ("mrmr_relevance_algo", self._VALID_MRMR_RELEVANCE_ALGOS),
        ("mrmr_redundancy_algo", self._VALID_MRMR_REDUNDANCY_ALGOS),
        ("fe_unary_preset", self._VALID_FE_UNARY_PRESETS),
        ("fe_binary_preset", self._VALID_FE_BINARY_PRESETS),
        ("cluster_aggregate_mode", self._VALID_CLUSTER_AGGREGATE_MODES),
        ("nbins_strategy", self._VALID_NBINS_STRATEGIES),
        ("mi_correction", self._VALID_MI_CORRECTIONS),
        ("redundancy_aggregator", self._VALID_REDUNDANCY_AGGREGATORS),
        ("stability_selection_method", self._VALID_STABILITY_SELECTION_METHODS),
        # DCD distance / swap-method strings.
        ("dcd_distance", self._VALID_DCD_DISTANCES),
        ("dcd_swap_method", self._VALID_DCD_SWAP_METHODS),
        # additional_rfecv_selection_rule flows verbatim into RFECV's
        # n_features_selection_rule; validate it here so a typo fails at
        # fit() start, consistent with the other MRMR string params.
        ("additional_rfecv_selection_rule", self._VALID_RFECV_SELECTION_RULES),
    )
    # DCD range checks, gated on dcd_enable.
    if bool(getattr(self, "dcd_enable", False)):
        _d = getattr(self, "dcd_distance", "su")
        _tau_raw = getattr(self, "dcd_tau_cluster", 0.7)
        # Layer 47: ``dcd_tau_cluster='auto'`` opts into the
        # per-fit bimodality-detection calibration sweep in
        # ``make_dcd_state``. The string accepts only the literal lower-case
        # ``"auto"``; any other string is a configuration error.
        if isinstance(_tau_raw, str):
            if _tau_raw.lower() != "auto":
                raise ValueError(f"MRMR: dcd_tau_cluster must be a float in (0, 1] or the " f"literal string 'auto'; got {_tau_raw!r}.")
            # 'auto' string short-circuits the numeric range check.
            _tau = None
        else:
            _tau = float(_tau_raw)
            # Layer 46: ``"auto"`` (distance) returns max(SU, VI_sim) so the
            # score lives in [0, 1] just like SU; reuse the SU range check.
            if _d in ("su", "auto") and not (0.0 < _tau <= 1.0):
                raise ValueError(f"MRMR: dcd_tau_cluster must be in (0, 1] for " f"distance={_d!r}; got {_tau}.")
            if _d in ("vi", "sotoca_pla") and _tau <= 0.0:
                raise ValueError(f"MRMR: dcd_tau_cluster must be > 0 for distance={_d!r}; " f"got {_tau}.")
        # The lower bound is 1, not 2. The threshold counts
        # cluster MEMBERS (not anchor + members), so threshold=1 fires the
        # PC1 swap on the strict 2-feature redundancy case (anchor + 1
        # perfect duplicate); threshold=2 (the new default) fires only when
        # the cluster grew anchor + >=2 members. Both are sane settings.
        if int(getattr(self, "dcd_cluster_size_threshold", 2)) < 1:
            raise ValueError(f"MRMR: dcd_cluster_size_threshold must be >= 1; got " f"{self.dcd_cluster_size_threshold}.")
        if float(getattr(self, "dcd_swap_gain_threshold", 0.05)) < 0.0:
            raise ValueError(f"MRMR: dcd_swap_gain_threshold must be >= 0; got " f"{self.dcd_swap_gain_threshold}.")
        _alpha = float(getattr(self, "dcd_swap_alpha", 0.05))
        if not (0.0 < _alpha <= 1.0):
            raise ValueError(f"MRMR: dcd_swap_alpha must be in (0, 1]; got {_alpha}.")
        if bool(getattr(self, "dcd_postoc_compose", False)) and bool(getattr(self, "cluster_aggregate_enable", True)):
            import warnings as _w_dcd
            _w_dcd.warn(
                "MRMR: dcd_enable=True AND cluster_aggregate_enable=True AND "
                "dcd_postoc_compose=True will double-aggregate clusters. The "
                "post-hoc step will see almost no clusters DCD did not already "
                "process. Consider dcd_postoc_compose=False (the default).",
                UserWarning, stacklevel=3,
            )
    # AccuracyWarning for demoted nbins_strategy options.
    _demoted = getattr(self, "_DEMOTED_NBINS_STRATEGIES", ())
    _nbins_strat = getattr(self, "nbins_strategy", None)
    if _nbins_strat in _demoted:
        import warnings as _w
        _w.warn(
            f"MRMR: nbins_strategy={_nbins_strat!r} is DEMOTED to research-only. "
            f"F1-bench honest ranking by ``|err vs truth| + noise_floor`` "
            f"places these demoted methods last: Knuth (combined 0.213), "
            f"Bayesian Blocks (0.233), MAH/SCI (0.373, collapses to ~2 bins). "
            f"Recommended: 'mdlp' (combined 0.107, only TRUE zero noise floor) "
            f"for balanced production use; 'qs' (signal err 0.093 best, but "
            f"noise floor 0.123 inflates false positives) when no-signal "
            f"columns are absent. Opt-out via ``warnings.filterwarnings('ignore', "
            f"category=UserWarning, module='mlframe.feature_selection.filters')``.",
            UserWarning,
            stacklevel=3,
        )
    for _name, _valid in _checks:
        _val = getattr(self, _name, None)
        if _val is None:
            if None in _valid:
                continue
            raise ValueError(f"MRMR: {_name} cannot be None. Valid values: {_valid}.")
        if not isinstance(_val, str):
            raise ValueError(f"MRMR: {_name} must be a string; got {type(_val).__name__}={_val!r}. " f"Valid values: {_valid}.")
        if _val not in _valid:
            raise ValueError(f"MRMR: {_name}={_val!r} is not a recognised value. " f"Valid values: {_valid}.")
    # Validate the orth default-scorer routing flag.
    # Kept outside the ``_checks`` loop because the attribute lives on the
    # MRMR class as ``_VALID_FE_HYBRID_ORTH_DEFAULT_SCORERS`` (longer name
    # than the constants reused by the loop). Invalid value -> ValueError
    # listing every accepted scorer so the message is actionable.
    _default_scorer = getattr(self, "fe_hybrid_orth_default_scorer", None)
    if _default_scorer is not None:
        _valid_scorers = getattr(
            self, "_VALID_FE_HYBRID_ORTH_DEFAULT_SCORERS", None,
        )
        if _valid_scorers is not None:
            if not isinstance(_default_scorer, str):
                raise ValueError(
                    f"MRMR: fe_hybrid_orth_default_scorer must be a string; "
                    f"got {type(_default_scorer).__name__}={_default_scorer!r}. "
                    f"Valid values: {_valid_scorers}."
                )
            if _default_scorer not in _valid_scorers:
                raise ValueError(f"MRMR: fe_hybrid_orth_default_scorer={_default_scorer!r} " f"is not a recognised value. " f"Valid values: {_valid_scorers}.")
    _validate_hybrid_orth_string_params(self)
    # cluster_aggregate_methods is a sequence; validate each element.
    _methods = getattr(self, "cluster_aggregate_methods", None)
    if _methods is not None:
        for _m in _methods:
            if _m not in self._VALID_CLUSTER_AGGREGATE_METHODS:
                raise ValueError(
                    f"MRMR: cluster_aggregate_methods contains {_m!r}, not a recognised value. " f"Valid values: {self._VALID_CLUSTER_AGGREGATE_METHODS}."
                )

    _redundancy = getattr(self, "mrmr_redundancy_algo", None)
    if _redundancy in _UNIMPLEMENTED_REDUNDANCY_ALGOS:
        logger.warning(
            "mrmr_redundancy_algo=%r (%s) is accepted but not implemented: the fit runs the 'fleuret' redundancy path and the "
            "result is identical to mrmr_redundancy_algo='fleuret'. Set it explicitly if that is what you want.",
            _redundancy, _UNIMPLEMENTED_REDUNDANCY_ALGOS[_redundancy],
        )

def _validate_hybrid_orth_string_params(self) -> None:
    """Raise ValueError on an unrecognised hybrid-orth basis / kernel / aggregator / scorer string, listing the accepted values."""
    from mlframe.feature_selection.filters.mrmr.shared import (
        VALID_FE_HYBRID_ORTH_BASES as _VALID_FE_HYBRID_ORTH_BASES,
        VALID_FE_HYBRID_ORTH_CLUSTER_BASIS_AGGREGATORS as _VALID_FE_HYBRID_ORTH_CLUSTER_BASIS_AGGREGATORS,
        VALID_FE_HYBRID_ORTH_ENSEMBLE_AGGREGATORS as _VALID_FE_HYBRID_ORTH_ENSEMBLE_AGGREGATORS,
        VALID_FE_HYBRID_ORTH_ENSEMBLE_SCORERS as _VALID_FE_HYBRID_ORTH_ENSEMBLE_SCORERS,
        VALID_FE_HYBRID_ORTH_HSIC_KERNELS as _VALID_FE_HYBRID_ORTH_HSIC_KERNELS,
        VALID_FE_HYBRID_ORTH_META_FORCE_SCORERS as _VALID_FE_HYBRID_ORTH_META_FORCE_SCORERS,
    )

    for name, valid in (
        ("fe_hybrid_orth_basis", _VALID_FE_HYBRID_ORTH_BASES),
        ("fe_hybrid_orth_hsic_kernel", _VALID_FE_HYBRID_ORTH_HSIC_KERNELS),
        ("fe_hybrid_orth_ensemble_aggregator", _VALID_FE_HYBRID_ORTH_ENSEMBLE_AGGREGATORS),
        ("fe_hybrid_orth_cluster_basis_aggregator", _VALID_FE_HYBRID_ORTH_CLUSTER_BASIS_AGGREGATORS),
    ):
        if not hasattr(self, name):
            continue
        val = getattr(self, name)
        if not isinstance(val, str) or val not in valid:
            raise ValueError(f"MRMR: {name}={val!r} is not a recognised value. Valid values: {valid}.")
    scorers = getattr(self, "fe_hybrid_orth_ensemble_scorers", None)
    if scorers is not None:
        if isinstance(scorers, str) or not scorers:
            raise ValueError(f"MRMR: fe_hybrid_orth_ensemble_scorers must be a non-empty sequence of scorer names; got {scorers!r}.")
        bad = [s for s in scorers if s not in _VALID_FE_HYBRID_ORTH_ENSEMBLE_SCORERS]
        if bad:
            raise ValueError(f"MRMR: fe_hybrid_orth_ensemble_scorers contains {bad!r}. Valid values: {_VALID_FE_HYBRID_ORTH_ENSEMBLE_SCORERS}.")
    force = getattr(self, "fe_hybrid_orth_meta_force_scorer", None)
    # The meta scorer lower-cases the forced name, so accept any case here too.
    if force is not None and (not isinstance(force, str) or force.lower() not in _VALID_FE_HYBRID_ORTH_META_FORCE_SCORERS):
        raise ValueError(f"MRMR: fe_hybrid_orth_meta_force_scorer={force!r} is not a recognised value. Valid values: None or one of {_VALID_FE_HYBRID_ORTH_META_FORCE_SCORERS}.")
