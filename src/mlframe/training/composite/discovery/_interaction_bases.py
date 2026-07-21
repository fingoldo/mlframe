"""Interaction-aware base-pair discovery for ``CompositeTargetDiscovery``.

Detects pairs of candidate bases whose INTERACTION (product / ratio) explains
target structure that neither base explains alone, then surfaces the qualifying
pair as a synthetic interaction base (via the existing
``transforms.interaction_bases.generate_interaction_bases``) for the multi-base /
``pairwise_interaction_residual`` path.

Scorer (self-contained, ``_mi_pair_bin``-based, no leakage)
-----------------------------------------------------------
For an ordered pair ``(a, b)`` and op ``mul``/``div``::

    z          = a OP b
    mi_z       = MI(y, z)          interaction-feature MI
    mi_a, mi_b = MI(y, a), MI(y, b)   marginal MIs
    add_mi     = max(mi_a, mi_b)    best single base alone
    gain       = mi_z - add_mi      synergy NOT captured by either base alone

A pair is a SYNERGY candidate when ``gain >= min_synergy_gain`` AND the
interaction MI exceeds both marginals by the configured margin -- i.e. the
product/ratio carries structure that is genuinely interaction (the canonical
``y ~ a*b`` pure-interaction DGP, where ``MI(y,a) ~ MI(y,b) ~ 0`` but
``MI(y, a*b)`` is large).

All MI is estimated on the TRAIN screening sample only (the caller passes
``y_screen`` + the screening feature matrix); no test rows are read, so the
synthetic base is leakage-safe by construction.

Honest caveat (measured): the absolute synergy gate flags the product on an
ADDITIVE DGP too (``y = a + b``), because ``a*b`` correlates with ``(a+b)^2`` and
carries MI beyond either marginal. The RATIO ``mi_z / max(mi_a, mi_b)`` cleanly
separates the regimes -- huge for pure interaction (marginals ~0), small for
additive -- so results are ranked by ``gain`` and the additive case still
surfaces a (harmless, lower-ranked) synthetic that simply duplicates structure
the additive bases already cover. The OOS gate downstream rejects it when it
adds no holdout value; the pure-interaction case wins decisively (bench).

``discover_interaction_bases`` is wired into the discovery base-resolution path
whenever ``config.interaction_base_discovery_enabled`` is true, which is the
DEFAULT (``= True``) -- not opt-in. See the committed bench
``_benchmarks/bench_interaction_base_discovery.py`` + biz_value test for the
measured verdict on the pure-interaction synthetic.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..transforms.interaction_bases import generate_interaction_bases
from .screening import _mi_pair_bin

logger = logging.getLogger(__name__)

# Defaults: a pair must add at least this absolute MI beyond the best single
# base, and the interaction MI must beat each marginal by ``min_margin_ratio``
# (so a pair that only inherits one base's signal does not qualify).
_INTERACTION_MIN_SYNERGY_GAIN_DEFAULT: float = 0.02
_INTERACTION_MIN_MARGIN_RATIO_DEFAULT: float = 1.25
_INTERACTION_TOP_K_DEFAULT: int = 4
_INTERACTION_MAX_PAIRS_DEFAULT: int = 3
_INTERACTION_OPS_DEFAULT: Tuple[str, ...] = ("mul", "div")


def score_interaction_pairs(
    candidates: Dict[str, np.ndarray],
    y: np.ndarray,
    *,
    ops: Sequence[str] = _INTERACTION_OPS_DEFAULT,
    top_k: int = _INTERACTION_TOP_K_DEFAULT,
    nbins: int = 12,
    min_synergy_gain: float = _INTERACTION_MIN_SYNERGY_GAIN_DEFAULT,
    min_margin_ratio: float = _INTERACTION_MIN_MARGIN_RATIO_DEFAULT,
    train_mask: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Score interaction pairs by MI synergy beyond the additive marginals.

    Parameters
    ----------
    candidates
        Ordered ``name -> 1-D ndarray`` (assumed MI-sorted upstream). Pairs are
        formed from the first ``top_k`` entries.
    y
        Target on the SAME rows as the candidate arrays (train screening sample).
    ops
        Interaction ops to synthesise + score (``mul`` / ``div``). Uses
        ``generate_interaction_bases`` so the synthetic-column numerics (div eps
        floor, naming) match the production transform exactly.
    top_k
        How many leading candidates participate (pairs = ordered, no self).
    nbins
        Quantile bins for the ``_mi_pair_bin`` estimator.
    min_synergy_gain
        Absolute MI a pair must add beyond ``max(mi_a, mi_b)`` to qualify.
    min_margin_ratio
        Interaction MI must be ``>= ratio * max(mi_a, mi_b)`` (relative guard so
        a pair that merely re-expresses one strong base does not pass).
    train_mask
        Optional boolean row mask forwarded to ``generate_interaction_bases`` so
        the div eps floor is train-scale-derived (no test-scale leak).

    Returns
    -------
    list of dicts (sorted by ``gain`` desc), one per scored synthetic::

        {"synth_name", "parents", "op", "mi_z", "mi_a", "mi_b",
         "add_mi", "gain", "qualifies"}
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    names = list(candidates.keys())[:top_k]
    if len(names) < 2:
        return []
    # Marginal MI per candidate, computed once (reused across every pair it
    # appears in). Bit-identical to recomputing per pair, just cheaper.
    mi_marg: Dict[str, float] = {}
    for n in names:
        col = np.asarray(candidates[n], dtype=np.float64).reshape(-1)
        if col.shape != y.shape:
            mi_marg[n] = 0.0
            continue
        mi_marg[n] = _mi_pair_bin(col, y, nbins=nbins)
    synth, prov = generate_interaction_bases(
        {n: candidates[n] for n in names},
        ops=ops,
        top_k=top_k,
        forbid_self_pairs=True,
        train_mask=train_mask,
    )
    results: List[Dict[str, Any]] = []
    for synth_name, z in synth.items():
        meta = prov.get(synth_name, {})
        if meta.get("constant", False):
            continue
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if z.shape != y.shape:
            continue
        name_a, name_b = meta.get("parents", (None, None))
        mi_a = float(mi_marg.get(name_a, 0.0))
        mi_b = float(mi_marg.get(name_b, 0.0))
        add_mi = max(mi_a, mi_b)
        mi_z = _mi_pair_bin(z, y, nbins=nbins)
        gain = mi_z - add_mi
        qualifies = gain >= min_synergy_gain and mi_z >= max(min_margin_ratio * add_mi, 1e-12)
        results.append(
            {
                "synth_name": synth_name,
                "parents": (name_a, name_b),
                "op": meta.get("op"),
                "mi_z": mi_z,
                "mi_a": mi_a,
                "mi_b": mi_b,
                "add_mi": add_mi,
                "gain": gain,
                "qualifies": bool(qualifies),
            }
        )
    results.sort(key=lambda r: -r["gain"])
    return results


def discover_interaction_bases(
    candidates: Dict[str, np.ndarray],
    y: np.ndarray,
    *,
    ops: Sequence[str] = _INTERACTION_OPS_DEFAULT,
    top_k: int = _INTERACTION_TOP_K_DEFAULT,
    nbins: int = 12,
    min_synergy_gain: float = _INTERACTION_MIN_SYNERGY_GAIN_DEFAULT,
    min_margin_ratio: float = _INTERACTION_MIN_MARGIN_RATIO_DEFAULT,
    max_pairs: int = _INTERACTION_MAX_PAIRS_DEFAULT,
    train_mask: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """Surface the top qualifying interaction pairs as synthetic base columns.

    Runs :func:`score_interaction_pairs`, keeps only ``qualifies=True`` pairs,
    dedups op-symmetric duplicates (``a__mul__b`` == ``b__mul__a``), and returns
    up to ``max_pairs`` synthetic columns plus their scoring records (for
    provenance / report).

    Returns ``({synth_name -> ndarray}, [score_record, ...])``. Empty when no
    pair clears the synergy gate -- the caller then proceeds with raw bases only.
    """
    scored = score_interaction_pairs(
        candidates, y,
        ops=ops, top_k=top_k, nbins=nbins,
        min_synergy_gain=min_synergy_gain,
        min_margin_ratio=min_margin_ratio,
        train_mask=train_mask,
    )
    qualifying = [r for r in scored if r["qualifies"]]
    if not qualifying:
        return {}, []
    # Regenerate the synthetic arrays for the names we keep (cheap; the scorer
    # already proved them out). Dedup commutative ops by sorted-parent + op.
    synth_all, _prov = generate_interaction_bases(
        candidates, ops=ops, top_k=top_k,
        forbid_self_pairs=True, train_mask=train_mask,
    )
    out: Dict[str, np.ndarray] = {}
    kept_records: List[Dict[str, Any]] = []
    seen_keys: set = set()
    for rec in qualifying:
        if len(out) >= max_pairs:
            break
        op = rec["op"]
        pa, pb = rec["parents"]
        # mul/add are commutative -> dedup; div/sub are not.
        key: tuple
        if op in ("mul", "add"):
            key = (op, tuple(sorted((str(pa), str(pb)))))
        else:
            key = (op, str(pa), str(pb))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        name = rec["synth_name"]
        if name not in synth_all:
            continue
        out[name] = synth_all[name]
        kept_records.append(rec)
    if out:
        preview = ", ".join(f"{r['synth_name']}(mi_z={r['mi_z']:.3f} vs add={r['add_mi']:.3f}, " f"gain={r['gain']:.3f})" for r in kept_records)
        logger.info(
            "[CompositeTargetDiscovery] interaction-base discovery surfaced " "%d synthetic pair(s): %s",
            len(out),
            preview,
        )
    return out, kept_records
