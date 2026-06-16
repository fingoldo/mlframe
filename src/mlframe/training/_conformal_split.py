"""Structure-aware carving of calib + conformal holdouts from the train portion.

Conformal validity needs the calib/conformal rows to be exchangeable with test at the right
UNIT: rows (iid), time-blocks (temporal), or whole groups (grouped). Carving a random 5% on a
temporal/grouped frame leaks. This module cuts two disjoint sub-slices out of the train indices
honoring the structure (see ``_conformal_finalize.infer_split_structure``), so the calib slice
fits the recalibration map and the conformal slice scores the recalibrated predictor on rows the
model never saw -- with a purge gap for temporal recurrences and whole-group assignment for groups.

Pure integer-index math (no frame copy); the splitter passes the carved indices to its
format-native ``.iloc``/``.filter`` at the call site. Returns ``(train_fit, calib, conformal)``.
"""

from __future__ import annotations

import numpy as np

CarveResult = tuple[np.ndarray, np.ndarray, np.ndarray]


def _resolve_counts(n: int, calib_frac: float, conformal_frac: float) -> tuple[int, int]:
    n_calib = int(np.floor(calib_frac * n)) if calib_frac and calib_frac > 0 else 0
    n_conf = int(np.floor(conformal_frac * n)) if conformal_frac and conformal_frac > 0 else 0
    return n_calib, n_conf


def carve_calib_conformal_iid(
    train_idx: np.ndarray,
    calib_frac: float,
    conformal_frac: float,
    *,
    seed: int = 0,
) -> CarveResult:
    """Random disjoint calib + conformal slices for iid/stratified data."""
    idx = np.asarray(train_idx)
    n = idx.size
    n_calib, n_conf = _resolve_counts(n, calib_frac, conformal_frac)
    if n_calib + n_conf >= n:
        raise ValueError(f"calib+conformal ({n_calib}+{n_conf}) leaves no train rows out of {n}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    conf = idx[perm[:n_conf]]
    calib = idx[perm[n_conf : n_conf + n_calib]]
    fit = idx[perm[n_conf + n_calib :]]
    return np.sort(fit), np.sort(calib), np.sort(conf)


def carve_calib_conformal_temporal(
    train_idx: np.ndarray,
    calib_frac: float,
    conformal_frac: float,
    *,
    time_values: np.ndarray | None = None,
    purge: int = 0,
) -> CarveResult:
    """Forward-walk carve: oldest rows fit, then calib, then the most-recent conformal block.

    Timeline within train: ``[ fit | purge | calib | purge | conformal ]`` (most recent = conformal,
    so the conformal residuals reflect predicting the near-future edge). ``time_values`` orders the
    rows when ``train_idx`` is not already chronological; ``purge`` rows are dropped at each block
    boundary to kill windowed-label / autocorrelation leakage (Lopez de Prado).
    """
    idx = np.asarray(train_idx)
    n = idx.size
    if time_values is not None:
        order = np.argsort(np.asarray(time_values), kind="stable")
        idx = idx[order]
    n_calib, n_conf = _resolve_counts(n, calib_frac, conformal_frac)
    purge = max(0, int(purge))
    # Lay out from the most-recent end backwards: conformal | purge | calib | purge | fit.
    conf_start = n - n_conf
    calib_stop = conf_start - purge
    calib_start = calib_stop - n_calib
    fit_stop = calib_start - purge
    if fit_stop <= 0:
        raise ValueError(f"temporal carve leaves no train-fit rows: n={n}, calib={n_calib}, conformal={n_conf}, purge={purge}")
    conf = idx[conf_start:]
    calib = idx[calib_start:calib_stop]
    fit = idx[:fit_stop]
    return np.sort(fit), np.sort(calib), np.sort(conf)


def carve_calib_conformal_grouped(
    train_idx: np.ndarray,
    calib_frac: float,
    conformal_frac: float,
    *,
    group_values: np.ndarray,
    seed: int = 0,
) -> CarveResult:
    """Whole-group carve: no group straddles fit/calib/conformal (group is the exchangeable unit).

    Groups are assigned (not rows): conformal gets ~``conformal_frac`` of the groups, calib the next
    ~``calib_frac``, the rest fit. Coverage is then group-conditional (Mondrian downstream).
    """
    idx = np.asarray(train_idx)
    groups = np.asarray(group_values)
    if groups.shape[0] != idx.shape[0]:
        raise ValueError(f"group_values {groups.shape} must align with train_idx {idx.shape}")
    uniq = np.unique(groups)
    g = uniq.size
    n_g_conf = int(np.floor(conformal_frac * g)) if conformal_frac and conformal_frac > 0 else 0
    n_g_calib = int(np.floor(calib_frac * g)) if calib_frac and calib_frac > 0 else 0
    if n_g_conf + n_g_calib >= g:
        raise ValueError(f"calib+conformal groups ({n_g_calib}+{n_g_conf}) leave no fit groups out of {g}")
    rng = np.random.default_rng(seed)
    perm = uniq[rng.permutation(g)]
    conf_groups = set(perm[:n_g_conf].tolist())
    calib_groups = set(perm[n_g_conf : n_g_conf + n_g_calib].tolist())
    is_conf = np.array([gv in conf_groups for gv in groups])
    is_calib = np.array([gv in calib_groups for gv in groups])
    is_fit = ~(is_conf | is_calib)
    return np.sort(idx[is_fit]), np.sort(idx[is_calib]), np.sort(idx[is_conf])


def carve_calib_conformal(
    train_idx: np.ndarray,
    calib_frac: float,
    conformal_frac: float,
    *,
    structure: str = "iid",
    time_values: np.ndarray | None = None,
    group_values: np.ndarray | None = None,
    purge: int = 0,
    seed: int = 0,
) -> CarveResult:
    """Dispatch to the structure-appropriate carver. ``structure`` from ``infer_split_structure``.

    ``temporal_grouped`` falls back to the grouped carve (whole-group disjointness is the stronger
    constraint); a group-blocked forward walk is a future refinement. Unknown structure -> iid.
    """
    if structure == "temporal":
        return carve_calib_conformal_temporal(
            train_idx,
            calib_frac,
            conformal_frac,
            time_values=time_values,
            purge=purge,
        )
    if structure in ("grouped", "temporal_grouped"):
        if group_values is None:
            raise ValueError(f"structure={structure!r} requires group_values")
        return carve_calib_conformal_grouped(
            train_idx,
            calib_frac,
            conformal_frac,
            group_values=group_values,
            seed=seed,
        )
    return carve_calib_conformal_iid(train_idx, calib_frac, conformal_frac, seed=seed)
