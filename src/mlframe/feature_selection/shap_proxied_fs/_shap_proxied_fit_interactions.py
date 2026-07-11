"""su_seeded_interactions / interaction_aware / proxy_mode=interaction candidate augmentation.

Carved out of ``ShapProxiedFitMixin.fit`` (``_shap_proxied_fit.py``) to keep that file under the
1k LOC ceiling. Holds the two cohesive "augment the search's candidate list with interaction-aware
subsets" steps: resolving the su_seeded synergistic operand pairs in proxy-column space (must run
BEFORE the importance prescreen so the operands can be rescued past it), and merging the
interaction_aware / proxy_mode="interaction" / su_seeded-sparse candidate expansions AFTER the
search returns its proxy-best subsets. Both take ``self`` explicitly (plain functions, not mixin
methods) so they stay decoupled from the fit-pipeline's class hierarchy.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


def _inject_operand_pairs(merged: dict, usable_pairs, name_to_phi_idx: dict, *, phi, base, y_phi, classification, metric) -> int:
    """Inject the bare operand pair of every surviving synergistic pair into ``merged``.

    ``merged`` maps a sorted phi-index tuple -> proxy loss (lower is better; the winner is the min).
    A pair that already carries a measured proxy loss (surfaced by the augmented search / expansion
    upstream) keeps it. A pair with NO prior entry gets its REAL additive proxy loss computed honestly
    on its operand phi-columns via ``subset_loss``. The pre-fix bug instead stamped such pairs with
    ``candidates[0][0]`` -- the BEST candidate's loss -- a fabricated optimistic stand-in that could
    sort to the front of the winner list and win selection on a number it never earned.

    Returns the count of pairs newly injected (those that had no prior entry).
    """
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import resolve_metric, subset_loss

    metric = resolve_metric(classification, metric)
    injected = 0
    for pair in usable_pairs:
        col_a, col_b = pair[-2], pair[-1]
        key = tuple(sorted((name_to_phi_idx[str(col_a)], name_to_phi_idx[str(col_b)])))
        if key not in merged:
            merged[key] = float(subset_loss(phi, base, y_phi, list(key), metric))
            injected += 1
    return injected


def resolve_su_seeded_pairs(
    self: Any,
    phi: np.ndarray,
    X_proxy: Any,
    y_phi: np.ndarray,
    unit_to_members: Any,
    working_cols: Any,
    X_cols: Any,
    report: dict,
    _stage: Callable,
) -> tuple:
    """Resolve su_seeded_interactions synergistic operand pairs in PROXY-column space.

    Must run BEFORE the importance pre-screen so the surviving operand columns (pure-interaction
    operands have ~0 mean|phi|, the regime the additive proxy is blind to) can be rescued past it.
    Reuses the prefilter-stage screen result (``self._su_seeded_pairs_orig``) when it already ran;
    otherwise runs the cheap pairwise-SU screen directly on ``X_proxy`` here. Returns
    ``(_su_kept_pairs, _su_screen_info, _su_rescue_proxy_idx)``.
    """
    _su_kept_pairs: list = []  # list of (synergy, proxy_col_name_a, proxy_col_name_b)
    _su_screen_info: dict = {}
    _su_rescue_proxy_idx: set[int] = set()
    if self.su_seeded_interactions and phi.shape[1] >= 2:
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import su_synergy_screen

        with _stage("su_seeded_interactions"):
            proxy_names = [str(c) for c in X_proxy.columns]
            # Build ORIGINAL-feature-name -> proxy-column index. Clustering renames proxy columns
            # to ``unitK`` whose members are WORKING-positions (unit_to_members); map each member
            # working-position -> original index -> name. When clustering is off X_proxy columns
            # ARE the post-prefilter original names, so the identity map applies.
            orig_name_to_proxy: dict[str, int] = {}
            if unit_to_members is not None:
                for _u, _members in enumerate(unit_to_members):
                    for _m in _members:
                        _orig_idx = int(working_cols[int(_m)])
                        orig_name_to_proxy[str(X_cols[_orig_idx])] = _u
            else:
                orig_name_to_proxy = {_nm: _u for _u, _nm in enumerate(proxy_names)}

            pairs_orig = getattr(self, "_su_seeded_pairs_orig", None)
            if pairs_orig is None:
                # Prefilter screen did not run (no prefilter narrowing) -> run it on X_proxy now.
                _kp, _su_screen_info = su_synergy_screen(
                    X_proxy, y_phi,
                    n_bins=self.su_seeded_n_bins, top_k=self.su_seeded_top_k,
                    max_screen_cols=self.su_seeded_max_screen_cols,
                    snr_z=self.su_seeded_snr_z,
                    snr_null_quantile=self.su_seeded_snr_null_quantile,
                    snr_abs_floor=self.su_seeded_snr_abs_floor,
                    n_permutations=self.su_seeded_n_permutations,
                    importance=np.abs(phi).mean(axis=0),
                    rng=np.random.default_rng(int(self.random_state) + 7919))
                # X_proxy column names are the proxy-space keys directly here.
                for _syn, _jsu, _ca, _cb in _kp:
                    _ia = proxy_names.index(str(_ca)) if str(_ca) in proxy_names else None
                    _ib = proxy_names.index(str(_cb)) if str(_cb) in proxy_names else None
                    if _ia is not None and _ib is not None and _ia != _ib:
                        _su_kept_pairs.append((float(_syn), str(_ca), str(_cb)))
                        _su_rescue_proxy_idx.add(_ia)
                        _su_rescue_proxy_idx.add(_ib)
            else:
                # Use the prefilter-stage screen result (operand ORIGINAL names) mapped to proxy.
                _su_screen_info = dict(getattr(self, "_su_seeded_screen_info", {}))
                for _syn, _ca, _cb in pairs_orig:
                    _ia = orig_name_to_proxy.get(str(_ca))
                    _ib = orig_name_to_proxy.get(str(_cb))
                    if _ia is not None and _ib is not None and _ia != _ib:
                        _su_kept_pairs.append((float(_syn), proxy_names[_ia], proxy_names[_ib]))
                        _su_rescue_proxy_idx.add(_ia)
                        _su_rescue_proxy_idx.add(_ib)
    return _su_kept_pairs, _su_screen_info, _su_rescue_proxy_idx


def augment_candidates_with_interactions(
    self: Any,
    candidates: list,
    phi: np.ndarray,
    base: Any,
    y_phi: np.ndarray,
    X_proxy: Any,
    proxy_cols_kept: Any,
    y_search: np.ndarray,
    model_template: Any,
    unit_to_members: Any,
    report: dict,
    _stage: Callable,
    _su_kept_pairs: list,
    _su_screen_info: dict,
) -> list:
    """Merge interaction_aware / proxy_mode="interaction" / su_seeded-sparse candidates into ``candidates``.

    Runs AFTER the search returns its proxy-best subsets. Each augmentation is independently opt-in
    (``interaction_aware``, ``proxy_mode="interaction"``, ``su_seeded_interactions``) and no-ops
    cleanly when its gate doesn't clear, so the additive default stays untouched. Returns the
    (possibly extended) ``candidates`` list; mutates ``report`` in place.
    """
    # Interaction-aware coalition (#5): for interaction-heavy targets the main-effect sum can't
    # see a pair's joint signal (XOR partners have ~0 main effect). Add candidates ranked by the
    # SHAP-interaction coalition value and let honest re-validation arbitrate. Bounded to a small
    # proxy width (post pre-screen); tensor is O(P^2).
    if self.interaction_aware and phi.shape[1] <= self.max_interaction_features:
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import compute_interaction_tensor, interaction_top_n

        X_proxy_kept = X_proxy.iloc[:, list(proxy_cols_kept)]
        Phi, ibase = compute_interaction_tensor(model_template, X_proxy_kept, y_search, classification=self.classification, rng=self._rng)
        icands = interaction_top_n(
            Phi, ibase, y_phi, classification=self.classification, metric=self.metric,
            min_card=self.min_features, max_card=self.max_features, top_n=self.top_n,
            exhaustive_max=self.max_interaction_features)
        merged = {tuple(sorted(c)): lo for lo, c in candidates}
        for lo, c in icands:
            merged.setdefault(tuple(sorted(c)), lo)
        candidates = sorted(((lo, c) for c, lo in merged.items()), key=lambda t: t[0])
        report["interaction_aware"] = dict(applied=True, n_proxy=int(phi.shape[1]), n_interaction_candidates=len(icands))

    # proxy_mode="interaction" (OPT-IN): re-score the additive search's candidates under the
    # INTERACTION-AWARE coalition proxy ``base + sum phi_j + 2*sum_{i<j in S} Phi_ij`` and add a
    # gated singleton/pair sweep, so a subset whose value comes from a non-additive PAIR (XOR /
    # multiplicative) earns credit the additive proxy denies it. The pairwise term is GATED to the
    # top-k features by mean |phi| so the cost is O(k^2) not O(P^2). Default stays "additive"
    # (bench_shap_interaction_proxy: interaction wins the competing-XOR bed by ~+0.24 AUC replicated
    # 3/3 seeds, but is only 1/6 beds and slightly regresses one additive-redundant seed -- not the
    # majority+no-regression win a default flip requires; kept opt-in). Tree models only (needs the
    # TreeSHAP interaction tensor); non-tree falls back to additive cleanly (compute_interaction_
    # tensor returns None-equivalent and the block no-ops).
    if str(getattr(self, "proxy_mode", "additive")).lower() == "interaction" and phi.shape[1] >= 2:
        with _stage("proxy_mode_interaction"):
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import compute_interaction_tensor
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interaction_proxy import interaction_proxy_top_n

            applied = False
            n_int_cands = 0
            try:
                X_proxy_kept = X_proxy.iloc[:, list(proxy_cols_kept)]
                Phi_ip, _ibase = compute_interaction_tensor(model_template, X_proxy_kept, y_search, classification=self.classification, rng=self._rng)
                icands = interaction_proxy_top_n(
                    phi, Phi_ip, base, y_phi, classification=self.classification, metric=self.metric,
                    min_card=self.min_features, max_card=self.max_features, top_n=self.top_n,
                    interaction_top_k=int(self.interaction_proxy_top_k),
                    candidate_subsets=[c for _l, c in candidates])
                merged = {tuple(sorted(c)): lo for lo, c in candidates}
                for lo, c in icands:
                    key = tuple(sorted(c))
                    if key not in merged or lo < merged[key]:
                        merged[key] = lo
                        n_int_cands += 1
                candidates = sorted(((lo, c) for c, lo in merged.items()), key=lambda t: t[0])
                applied = True
            except Exception as exc:  # unsupported model / tensor failure -> additive fallback
                logger.warning("proxy_mode=interaction fell back to additive: %s", exc)
            report["proxy_mode_interaction"] = dict(
                applied=applied, n_proxy=int(phi.shape[1]), interaction_top_k=int(self.interaction_proxy_top_k), n_added=int(n_int_cands)
            )

    # su_seeded_interactions merge (#5b, OPT-IN): the CHEAP sparse alternative to
    # ``interaction_aware``'s O(P^2) tensor. The synergy screen + SNR gate ran above (operands
    # already rescued past the prescreen, so they are present in ``phi``). Here the interaction
    # objective runs on ONLY the surviving K pairs -- a sparse product-column augmentation, never
    # the dense P x P tensor. Each surviving product candidate is expanded into its two OPERAND
    # proxy-columns so the merged coalition lives in plain phi-column space (selecting the product
    # == recovering both operands); honest re-validation downstream still retrains only on real
    # columns. Runs at ANY phi width (no max_interaction_features gate). When the screen kept no
    # pair this whole block is skipped (the SNR-gate no-op).
    if self.su_seeded_interactions:
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import sparse_interaction_candidates

        with _stage("su_seeded_interactions"):
            # Map proxy-column NAMES -> phi-column indices (X_proxy_kept aligns 1:1 with phi).
            X_proxy_kept = X_proxy.iloc[:, list(proxy_cols_kept)]
            name_to_phi_idx = {str(c): i for i, c in enumerate(X_proxy_kept.columns)}
            # Keep only pairs whose BOTH operand proxy-columns survived into phi (the prescreen
            # rescue should guarantee this for every kept pair; the filter is defensive).
            usable_pairs = [(s, a, b) for s, a, b in _su_kept_pairs if str(a) in name_to_phi_idx and str(b) in name_to_phi_idx]
            n_seed_cands = 0
            if usable_pairs:
                # ISOLATED rng (NOT self._rng): keep the sparse-interaction refit off the shared
                # RNG stream so the additive default stays byte-identical in the no-op case and the
                # win-path stays reproducible.
                icands_sparse, prod_to_operands = sparse_interaction_candidates(
                    model_template, X_proxy_kept, y_search, usable_pairs,
                    classification=self.classification, metric=self.metric,
                    min_card=self.min_features, max_card=self.max_features,
                    top_n=self.top_n,
                    rng=np.random.default_rng(int(self.random_state) + 7919))
                # Expand augmented-index candidates back into phi-column space: a product index is
                # replaced by its two operand phi-columns (selecting the product recovers both).
                merged = {tuple(sorted(c)): lo for lo, c in candidates}
                for lo, c in icands_sparse:
                    expanded: set[int] = set()
                    for idx in c:
                        idx = int(idx)
                        if idx in prod_to_operands:
                            a_i, b_i = prod_to_operands[idx]
                            expanded.add(int(a_i))
                            expanded.add(int(b_i))
                        elif idx < phi.shape[1]:
                            expanded.add(idx)
                    if not expanded:
                        continue
                    key = tuple(sorted(expanded))
                    if key not in merged or lo < merged[key]:
                        merged[key] = lo
                        n_seed_cands += 1
                # Always inject the bare operand pair of EVERY surviving synergistic pair as a
                # candidate coalition (so even when the augmented search prefers larger subsets,
                # the minimal interacting pair is still revalidated honestly on real columns).
                # Inject the bare operand pair of every surviving synergistic pair so honest
                # re-validation downstream can still revalidate the minimal interacting pair on
                # real columns; pairs with no measured proxy loss go in at +inf (cannot win).
                _inject_operand_pairs(
                    merged, usable_pairs, name_to_phi_idx, phi=phi, base=base, y_phi=y_phi, classification=self.classification, metric=self.metric
                )
                candidates = sorted(((lo, c) for c, lo in merged.items()), key=lambda t: t[0])
            report["su_seeded_interactions"] = dict(
                applied=True,
                n_proxy=int(phi.shape[1]),
                n_kept_pairs=len(_su_kept_pairs),
                n_usable_pairs=len(usable_pairs),
                n_seed_candidates=int(n_seed_cands),
                kept_pairs=[(round(float(s), 6), str(a), str(b)) for s, a, b in _su_kept_pairs],
                snr_gate=round(float(_su_screen_info.get("gate", float("nan"))), 6),
                best_synergy=round(float(_su_screen_info.get("best_synergy", float("nan"))), 6),
                n_screened_cols=int(_su_screen_info.get("n_screened_cols", 0)),
                n_pairs=int(_su_screen_info.get("n_pairs", 0)))

    return candidates
