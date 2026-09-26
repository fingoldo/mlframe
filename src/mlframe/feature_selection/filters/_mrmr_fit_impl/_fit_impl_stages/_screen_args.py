"""Keyword arguments of the ``screen_predictors`` call in ``_fit_impl`` that are derived from estimator settings."""


def _screen_factors_names_to_use(self):
    """The pinned candidate names, extended with hybrid-orth and MI-greedy engineered columns so they reach the screening
    gates. When the caller did not pin factors_names_to_use, screen_predictors uses every column and needs no extension."""
    extra = list(self.hybrid_orth_features_ or []) + list(getattr(self, "mi_greedy_features_", None) or [])
    if self.factors_names_to_use and extra:
        return list(self.factors_names_to_use) + extra
    return self.factors_names_to_use


def _effective_max_confirmation_cand_nbins(self):
    """User-pinned max_confirmation_cand_nbins wins, else the formula default."""
    if self.max_confirmation_cand_nbins is not None:
        return self.max_confirmation_cand_nbins
    return self.quantization_nbins**self.interactions_max_order * 2


def _cat_fe_lineage(self):
    """Engineered lineage from the cat-FE step (None when cat-FE did not run); screen uses it to skip redundant
    (orig_parent, engineered_col) k-way candidates."""
    state = getattr(self, "_cat_fe_state_", None)
    return state.lineage if state is not None and state.lineage else None


def _dcd_config(self, X):
    """DCD config forwarded to screen as kwargs (not thread-local, for joblib parallel-backend safety); None unless dcd_enable."""
    if not getattr(self, "dcd_enable", False):
        return None
    return dict(
        enable=True,
        tau_cluster=self.dcd_tau_cluster,
        distance=self.dcd_distance,
        cluster_size_threshold=self.dcd_cluster_size_threshold,
        swap_gain_threshold=self.dcd_swap_gain_threshold,
        swap_method=self.dcd_swap_method,
        pairwise_cache_max=self.dcd_pairwise_cache_max,
        min_cluster_size=self.dcd_min_cluster_size,
        max_cluster_size=self.dcd_max_cluster_size,
        swap_alpha=self.dcd_swap_alpha,
        # the swap null draw count, decoupled from
        # full_npermutations. getattr fallback keeps old
        # pickles (lacking the attr) loading at the 199 default.
        swap_npermutations=getattr(self, "dcd_swap_npermutations", 199),
        warp_tiebreak_prefer_linear=getattr(self, "warp_tiebreak_prefer_linear", True),
        warp_twin_rank_corr=getattr(self, "warp_twin_rank_corr", 0.99),
        warp_linear_margin=getattr(self, "warp_linear_margin", 0.05),
        # Layer 47: forward the auto-tau
        # calibration knobs (number of sampled feature pairs
        # and RNG seed) so make_dcd_state can fingerprint
        # the calibration sweep deterministically.
        tau_calibration_n_pairs=getattr(
            self,
            "dcd_tau_calibration_n_pairs",
            100,
        ),
        tau_calibration_seed=getattr(
            self,
            "dcd_tau_calibration_seed",
            0,
        ),
        X_raw=X,
        quantization_method=self.quantization_method,
        quantization_nbins=self.quantization_nbins,
        quantization_dtype=self.quantization_dtype,
    )
