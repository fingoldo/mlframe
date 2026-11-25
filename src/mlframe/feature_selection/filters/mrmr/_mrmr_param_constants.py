"""MRMR constructor-parameter validation constants (carved verbatim from ``_mrmr_class.py``).

These are the allow-lists (``_VALID_*``) and the demotion roster (``_DEMOTED_*``) that
``_mrmr_validate_transform._validate_inputs`` reads via ``self.<NAME>``. They were kept as
class attributes (rather than ``typing.Literal`` aliases) so the runtime check can produce
a richer error listing the valid options (fix audit row FS-P2-1). They hold only literal
data - no class refs - so they live in this leaf module and are re-bound onto the ``MRMR``
class body verbatim (``_VALID_QUANTIZATION_METHODS = _VALID_QUANTIZATION_METHODS`` etc.),
keeping ``self._VALID_*`` resolution byte-identical.
"""
from __future__ import annotations

# Allowed string values for the constructor params. Kept module-private rather
# than a typing.Literal alias so the runtime check can produce a richer error
# listing the valid options. fix audit row FS-P2-1.
_VALID_QUANTIZATION_METHODS = ("quantile", "uniform")
_VALID_NAN_STRATEGIES = ("separate_bin", "fillna_zero", "ffill_bfill", "propagate", "raise")
_VALID_CATEGORICAL_FOLD_STRATEGIES = ("frequency", "min_support", "target_stat")
_VALID_MRMR_RELEVANCE_ALGOS = ("fleuret", "pld")
_VALID_MRMR_REDUNDANCY_ALGOS = ("fleuret", "pld_max", "pld_mean")
# adaptive per-feature binning: the accepted values of ``binning_scheme`` and ``nbins_rule`` live in ``_binning_spec`` (BINNING_SCHEMES, NBINS_RULES).
# MRMR's MI computation stays exclusively on the integer-bin plug-in path (see bench_adaptive_nbins / bench_adaptive_nbins_mega). Alternative MI
# estimators (KSG, MINE, InfoNet, MIST, fastMI, aggregators) are intentionally NOT routed into the MRMR hot loop.
# opt-in validation sets.
_VALID_MI_CORRECTIONS = ("none", "miller_madow", "chao_shen")
_VALID_REDUNDANCY_AGGREGATORS = (None, "jmim", "auto")
_VALID_STABILITY_SELECTION_METHODS = (
    "classic", "cluster", "complementary_pairs",
)
_VALID_FE_UNARY_PRESETS = ("minimal", "medium", "maximal")
_VALID_FE_BINARY_PRESETS = ("minimal", "medium", "maximal")
_VALID_CLUSTER_AGGREGATE_MODES = ("augment", "replace")
# the cluster_aggregate method allow-list expands
# to include the four new combiners (``pca_pc2``, ``median_z``,
# ``signed_max_abs``, ``signed_l2_sum``) so direct ``cluster_aggregate_methods``
# API users can pin them individually, not just reach them via DCD ``auto``.
_VALID_CLUSTER_AGGREGATE_METHODS = (
    "mean_z", "mean_inv_var", "median", "pca_pc1", "factor_score",
    "pca_pc2", "median_z", "signed_max_abs", "signed_l2_sum",
)
# DCD validation constants. swap_methods alias the
# cluster_aggregate methods (Critic2/E fix: no duplicate constant).
# ``"auto"`` runs SU and VI in parallel per pair
# and returns the tighter redundancy score (``max(SU, VI_sim)``). Catches
# both linear-friendly duplicates (SU strong) and non-linear functional
# equivalences like y = f(x^2) (VI strong, SU silent).
_VALID_DCD_DISTANCES = ("su", "vi", "sotoca_pla", "auto")
# ``mi_normalization`` and ``group_mi_aggregate`` accepted values. The fit validator and the fs_params spec both read these,
# so a new value added here reaches the spec without a stale hardcoded copy (39-F66).
_VALID_MI_NORMALIZATIONS = ("none", "su")
_VALID_GROUP_MI_AGGREGATES = ("size", "equal")
# DCD ``dcd_swap_method`` accepts the same expanded combiner set
# so users can pin a single new method instead of relying on ``auto``.
_VALID_DCD_SWAP_METHODS = (
    "auto", "mean_z", "mean_inv_var", "median", "pca_pc1", "factor_score",
    "pca_pc2", "median_z", "signed_max_abs", "signed_l2_sum",
)
# additional_rfecv_selection_rule flows verbatim into RFECV's
# n_features_selection_rule (see _mrmr_fit_impl rescue block). Mirror
# RFECV's own accepted set (_rfecv.py constructor guard) so a typo fails
# at MRMR.fit() start with an actionable message instead of deep inside the
# RFECV fit.
_VALID_RFECV_SELECTION_RULES = ("auto", "argmax", "one_se_min", "one_se_max")

# accepted values for
# ``fe_hybrid_orth_default_scorer``. "plug_in" preserves Layer 21's
# behaviour byte-for-byte; the other 12 entries route the univariate
# basis-selection stage through the Layers listed alongside.
_VALID_FE_HYBRID_ORTH_DEFAULT_SCORERS = (
    "plug_in",  # Layer 21 (default)
    "cmim",  # Layer 74
    "jmim",  # Layer 72
    "tc",  # Layer 73
    "ksg",  # Layer 65
    "copula",  # Layer 66
    "dcor",  # Layer 67
    "hsic",  # Layer 71
    "auto",  # Layer 68
    "ensemble",  # Layer 69
    "meta",  # Layer 76
    "lasso",  # Layer 81
    "elasticnet",  # Layer 82
    "auto_oracle",  # Layer 100 (L76 cold-start + L68 bake-off + Param-Oracle learning)
)

# Accepted values of the remaining hybrid-orth string params, mirroring what each consumer accepts. The consumers run inside a
# broad per-family try/except that turns their own ValueError into a warning and drops the family, so without these a typo silently
# disables that FE family for the whole fit.
_VALID_FE_HYBRID_ORTH_HSIC_KERNELS = ("rbf",)
_VALID_FE_HYBRID_ORTH_ENSEMBLE_AGGREGATORS = ("mean_rank", "borda_count", "reciprocal_rank", "mutual_top_k")
_VALID_FE_HYBRID_ORTH_ENSEMBLE_SCORERS = ("plug_in", "ksg", "copula", "dcor", "hsic", "xi", "tail_dep")
_VALID_FE_HYBRID_ORTH_META_FORCE_SCORERS = ("plug_in", "ksg", "copula", "dcor", "hsic", "jmim", "cmim", "tc")
_VALID_FE_HYBRID_ORTH_BASES = ("auto", "hermite", "legendre", "chebyshev", "laguerre", "fourier", "rbf", "sigmoid", "pade")
_VALID_FE_HYBRID_ORTH_CLUSTER_BASIS_AGGREGATORS = ("mean_z", "median_z", "pc1")
