"""Synthetic interaction-base generator: generate_interaction_bases combines candidate base columns pairwise under binary ops (mul / div / add / sub) to capture multiplicative DGPs that linear_residual on a single base misses. Pure numpy; no composite-internal deps."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

# ----------------------------------------------------------------------
# generate_interaction_bases (synthetic base column generator).
#
# Takes a small set of candidate base column NAMES + their numpy arrays and produces synthetic base columns by pairwise combination under the specified binary operations. Output names follow the convention ``<a>__<op>__<b>`` (mirroring the composite-target naming convention) so they're distinguishable from raw bases in downstream metadata. Synthetic bases are then fed into the standard linear_residual / ratio / etc transforms as inputs.
#
# Use case (multiplicative physics): when y is governed by ``y ~= lag_feature * porosity``, single-base linear-residual on lag_feature leaves the porosity interaction in T. Feeding the synthetic ``lag_feature__mul__porosity`` column as a base lets linear_residual capture the interaction directly.
#
# Safety:
# - Division by near-zero: replace base values with |b2| < eps with sign(b2) * eps (eps derived from train scale of b2). The downstream consumer should domain-check on the synthetic column anyway.
# - All-NaN columns / constant columns are not produced (caller usually filters those upstream; we still emit the synthetic but mark it ``constant=True`` in the returned diagnostic dict so downstream selection can skip them).
# - Output is a plain ``dict`` (synthetic_name -> ndarray). Provenance metadata lives in the second return value for downstream tracking (e.g. for the final spec).
# ----------------------------------------------------------------------

_INTERACTION_OPS_DEFAULT: tuple[str, ...] = ("mul", "div")


def generate_interaction_bases(
    candidates: dict[str, np.ndarray],
    *,
    ops: Sequence[str] = _INTERACTION_OPS_DEFAULT,
    top_k: int = 3,
    eps_div_floor_factor: float = 1e-6,
    forbid_self_pairs: bool = True,
    train_mask: Optional[np.ndarray] = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Greedy pairwise synthesis of interaction-base columns from ``candidates``.

    Parameters
    ----------
    candidates
        Ordered mapping of ``column_name -> 1-D ndarray``. Pairs are formed from the FIRST ``top_k`` items (preserving caller order; assumed to be MI-sorted upstream so pairs prioritise high-signal columns).
    ops
        Binary operations to apply. Each op generates one synthetic per ordered pair (a, b) where a != b. Supported: ``"mul"``, ``"div"``, ``"add"``, ``"sub"``.
    top_k
        How many top candidates participate. ``top_k * (top_k - 1) * len(ops)`` synthetics generated for ``forbid_self_pairs=True``.
    eps_div_floor_factor
        Factor of train-scale used to floor near-zero divisors. ``eps = factor * median(|b|)``; values with ``|b| < eps`` get replaced by ``sign(b) * eps``.
    forbid_self_pairs
        Skip pairs where a == b. Default True.
    train_mask
        Optional boolean mask over rows of each candidate; when supplied, the divide-by-zero epsilon
        floor (``median(|b|) * factor``) is computed from TRAIN rows only. ``None`` (default) keeps the
        legacy whole-array median, which leaks test-set scale into the eps; the leak is small in practice
        (median is robust) but the audited fix surfaces train-vs-test asymmetry explicitly when callers
        opt in. Caller is responsible for aligning the mask length with each candidate array.

    Returns
    -------
    (synthetics, provenance):
    - ``synthetics``: dict[synthetic_name -> ndarray]. Name convention ``"<a>__<op>__<b>"``.
    - ``provenance``: dict[synthetic_name -> {parents, op, n_finite, scale_eps, constant}]. For downstream filtering / metadata stamping.
    """
    if top_k < 2:
        return {}, {}
    valid_ops = {"mul", "div", "add", "sub"}
    bad_ops = [o for o in ops if o not in valid_ops]
    if bad_ops:
        raise ValueError(f"generate_interaction_bases: unsupported op(s) {bad_ops}. Valid: {sorted(valid_ops)}")
    selected_names = list(candidates.keys())[:top_k]
    selected_arrays = [np.asarray(candidates[n], dtype=np.float64).reshape(-1) for n in selected_names]
    synthetics: dict[str, np.ndarray] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for i, name_a in enumerate(selected_names):
        a = selected_arrays[i]
        for j, name_b in enumerate(selected_names):
            if forbid_self_pairs and i == j:
                continue
            b = selected_arrays[j]
            if a.shape != b.shape:
                # Length mismatch -- caller misuse; skip with provenance entry rather than raise so a single bad pair doesn't kill the whole batch.
                continue
            finite_b = np.isfinite(b) & (b != 0)
            if train_mask is not None and train_mask.shape == b.shape:
                # Train-only median for the eps floor; test rows still get divided but the safe_b floor is
                # train-scale-derived (no test stats leak into the synthetic feature value).
                _b_for_scale = b[finite_b & train_mask]
            else:
                _b_for_scale = b[finite_b]
            scale_b = float(np.median(np.abs(_b_for_scale))) if _b_for_scale.size else 1.0
            eps_b = max(scale_b * eps_div_floor_factor, 1e-12)
            for op in ops:
                synth_name = f"{name_a}__{op}__{name_b}"
                if op == "mul":
                    out = a * b
                elif op == "add":
                    out = a + b
                elif op == "sub":
                    out = a - b
                elif op == "div":
                    safe_b = np.where(np.abs(b) < eps_b, np.sign(b + 1e-300) * eps_b, b)
                    out = a / safe_b
                else:  # pragma: no cover -- guarded above
                    continue
                synthetics[synth_name] = out
                finite_out = np.isfinite(out)
                _ptp = float(np.ptp(out[finite_out])) if finite_out.any() else 0.0
                provenance[synth_name] = {
                    "parents": (name_a, name_b),
                    "op": op,
                    "n_finite": int(finite_out.sum()),
                    "scale_eps_b": eps_b,
                    "constant": _ptp < 1e-12,
                }
    return synthetics, provenance
