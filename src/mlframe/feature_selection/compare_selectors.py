"""Read-only diagnostic comparison of feature selectors.

``compare_selectors(X, y, selectors=...)`` is a single-call convenience report
that fits (or accepts pre-fitted) feature selectors -- MRMR, RFECV, BorutaShap,
ShapProxiedFS, HybridSelector, or any sklearn-style selector exposing a support
accessor -- aligns each one's selected support on a common feature index, and
emits three orthogonal views:

* an ``agreement`` matrix (features x selectors, boolean kept/not),
* pairwise ``jaccard`` overlaps between every pair of selectors,
* a ``consensus`` column = "picked by k of n" count per feature.

This is PURELY a display over already-computed selector decisions. It changes
no selection logic and mutates no selector. ``HybridSelector`` already COMBINES
its members into one decision and stashes the per-member name-lists at
``member_selections_`` but never surfaces them; this report is the orthogonal,
read-only consumer of exactly that data -- it DISPLAYS per-member disagreement,
it does NOT use it to change any decision.

Selectors that are unavailable (not installed / failed to fit) are skipped with
a recorded note rather than aborting the whole report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from mlframe.core.set_similarity import jaccard as _jaccard_similarity


def _selector_name(selector: Any, idx: int) -> str:
    """Human-readable label for a selector (class name, de-duplicated by caller)."""
    for attr in ("name", "_compare_name"):
        v = getattr(selector, attr, None)
        if isinstance(v, str) and v:
            return v
    return type(selector).__name__ or f"selector_{idx}"


def _is_fitted(selector: Any) -> bool:
    """Heuristic: a selector is fitted if it exposes any recognised support accessor with data."""
    if getattr(selector, "support_", None) is not None:
        return True
    for attr in ("selected_features_", "member_selections_", "feature_names_in_"):
        if getattr(selector, attr, None) is not None:
            return True
    return False


def _extract_selected(selector: Any, feature_names: Sequence[str]) -> list[str]:
    """Return the list of feature NAMES a fitted selector kept, aligned to ``feature_names``.

    Tries accessors in order of reliability: get_feature_names_out -> selected_features_ ->
    boolean/index support_ (paired with feature_names_in_) -> BorutaShap.accepted. Names are
    intersected with ``feature_names`` so engineered/extra columns don't pollute the matrix.
    """
    names_in = getattr(selector, "feature_names_in_", None)
    names_in = list(np.asarray(names_in, dtype=object)) if names_in is not None else list(feature_names)

    # 1) sklearn canonical
    gfno = getattr(selector, "get_feature_names_out", None)
    if callable(gfno):
        try:
            out = list(np.asarray(gfno(), dtype=object))
            if out:
                return [str(c) for c in out]
        except Exception:  # nosec B110 - best-effort path
            pass

    # 2) explicit name list
    sel = getattr(selector, "selected_features_", None)
    if sel is not None:
        return [str(c) for c in list(sel)]

    # 3) support_ as boolean mask or integer indices, paired with feature_names_in_
    support = getattr(selector, "support_", None)
    if support is not None:
        support = np.asarray(support)
        if support.dtype == bool or (support.size and isinstance(support.flat[0], (bool, np.bool_))):
            return [str(c) for c, keep in zip(names_in, support) if keep]
        # integer indices into names_in
        return [str(names_in[int(i)]) for i in support]

    # 4) BorutaShap-style accepted list
    accepted = getattr(selector, "accepted", None)
    if accepted is not None:
        return [str(c) for c in list(accepted)]

    raise AttributeError(
        f"{type(selector).__name__} exposes no recognised support accessor " "(get_feature_names_out / selected_features_ / support_ / accepted)"
    )


@dataclass
class SelectorComparison:
    """Read-only result of :func:`compare_selectors`.

    Attributes:
        agreement: DataFrame (index=features, columns=selector names), boolean kept/not.
        jaccard:   DataFrame (selector x selector) of pairwise Jaccard overlaps.
        consensus: Series (index=features) of "picked by k of n" counts.
        n_selectors: number of selectors that contributed (fitted successfully).
        skipped:   mapping selector-name -> reason it was skipped.
    """

    agreement: pd.DataFrame
    jaccard: pd.DataFrame
    consensus: pd.Series
    n_selectors: int
    skipped: dict[str, str] = field(default_factory=dict)

    def report(self, max_features: int | None = 25) -> str:
        """Compact text report (matrix + Jaccard + consensus), <1.5 screens by default."""
        lines: list[str] = []
        n = self.n_selectors
        lines.append(f"compare_selectors: {n} selector(s), {len(self.agreement)} feature(s)")
        if self.skipped:
            for nm, why in self.skipped.items():
                lines.append(f"  skipped {nm}: {why}")

        cons = self.consensus.sort_values(ascending=False)
        shown = cons.index if max_features is None else cons.index[:max_features]
        mat = self.agreement.loc[shown]
        cols = list(mat.columns)
        # Pull numpy arrays once and index positionally; per-cell pandas ``.at`` /
        # Series scalar lookups (``_get_value``) dominated the report wall time.
        mat_vals = mat.to_numpy()
        feat_index = list(mat.index)
        cons_vals = cons.reindex(feat_index).to_numpy()
        head = "  feature".ljust(24) + "  " + "  ".join(c[:10].rjust(10) for c in cols) + f"  {'k/n':>5}"
        lines.append("")
        lines.append("AGREEMENT (rows sorted by consensus):")
        lines.append(head)
        for ridx, feat in enumerate(feat_index):
            row = mat_vals[ridx]
            cells = "  ".join(("Y" if row[cidx] else ".").rjust(10) for cidx in range(len(cols)))
            lines.append("  " + str(feat).ljust(22) + "  " + cells + f"  {int(cons_vals[ridx]):>3}/{n}")
        if max_features is not None and len(self.agreement) > max_features:
            lines.append(f"  ... ({len(self.agreement) - max_features} more features, lower consensus)")

        lines.append("")
        lines.append("PAIRWISE JACCARD:")
        jc = self.jaccard
        jc_vals = jc.to_numpy()
        jc_cols = list(jc.columns)
        lines.append("  " + "".ljust(12) + "  " + "  ".join(c[:10].rjust(10) for c in jc_cols))
        for ridx, r in enumerate(jc.index):
            row = jc_vals[ridx]
            lines.append("  " + str(r)[:12].ljust(12) + "  " + "  ".join(f"{row[cidx]:>10.2f}" for cidx in range(len(jc_cols))))

        lines.append("")
        full = int((self.consensus == n).sum())
        none = int((self.consensus == 0).sum())
        lines.append(f"CONSENSUS: {full} feature(s) picked by ALL {n}; {none} picked by none.")
        return "\n".join(lines)


def compare_selectors(
    X: pd.DataFrame,
    y: Any = None,
    selectors: Sequence[Any] | Mapping[str, Any] | None = None,
    *,
    fit: bool | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> SelectorComparison:
    """Compare what each feature selector keeps -- read-only diagnostics.

    Args:
        X: feature frame; its columns define the common feature index for alignment.
        y: target, only needed when ``fit`` triggers fitting of un-fitted selectors.
        selectors: sequence of selector instances, or a name->selector mapping.
            Each may be pre-fitted (its support is read as-is) or un-fitted
            (fitted on ``(X, y)`` when ``fit`` allows). Any selector exposing one
            of get_feature_names_out / selected_features_ / support_ / accepted works.
        fit: if True, fit every selector; if False, never fit (read pre-fitted state,
            skip un-fitted ones with a note); if None (default), fit only selectors
            that are not already fitted.
        fit_kwargs: extra kwargs forwarded to each ``selector.fit``.

    Returns:
        SelectorComparison with .agreement (feature x selector bool matrix),
        .jaccard (pairwise overlaps), .consensus (picked-by-k-of-n), and .skipped.

    No selector is mutated beyond an explicit ``fit`` you authorise; selection
    logic is untouched. Unavailable/failing selectors are skipped, not fatal.
    """
    if selectors is None:
        raise ValueError("compare_selectors requires a non-empty `selectors` sequence or mapping")

    items: list[tuple[Optional[str], Any]]
    if isinstance(selectors, Mapping):
        items = list(selectors.items())
    else:
        items = [(None, s) for s in selectors]
    if not items:
        raise ValueError("`selectors` is empty")

    feature_names = [str(c) for c in X.columns]
    if not feature_names:
        # With no columns every selector trivially "agrees" (Jaccard of empty vs empty == 1.0), silently reporting full agreement on nothing; reject at entry.
        raise ValueError("compare_selectors requires X with >= 1 column; got an empty feature set.")
    fit_kwargs = dict(fit_kwargs or {})

    feature_set = set(feature_names)
    selected_by: dict[str, list[str]] = {}
    skipped: dict[str, str] = {}
    used_names: set[str] = set()

    for idx, (label, selector) in enumerate(items):
        name = label or _selector_name(selector, idx)
        base, k = name, 1
        while name in used_names:  # de-duplicate identical class names
            k += 1
            name = f"{base}#{k}"
        used_names.add(name)

        already = _is_fitted(selector)
        do_fit = fit if fit is not None else (not already)
        if do_fit:
            try:
                selector.fit(X, y, **fit_kwargs)
            except Exception as exc:
                # Broad by design: this is a best-effort cross-selector comparison harness, so any selector that cannot fit (missing optional dep, GPU-only path on a
                # CPU host, an input it rejects) must not abort the whole comparison. The failure is RECORDED in ``skipped`` with its type+message and surfaced to the
                # caller -- it is visible, not swallowed -- so the broad catch is the correct policy for an exploratory comparison rather than a masked error.
                skipped[name] = f"fit failed: {type(exc).__name__}: {exc}"
                continue
        elif not already:
            skipped[name] = "not fitted and fit=False"
            continue

        try:
            sel = _extract_selected(selector, feature_names)
        except Exception as exc:
            skipped[name] = f"no readable support: {exc}"
            continue

        kept = [c for c in sel if c in feature_set]
        selected_by[name] = kept

    if not selected_by:
        empty = pd.DataFrame(index=feature_names)
        return SelectorComparison(
            agreement=empty,
            jaccard=pd.DataFrame(),
            consensus=pd.Series(0, index=feature_names, dtype=int),
            n_selectors=0,
            skipped=skipped,
        )

    names = list(selected_by.keys())
    sets = {nm: set(selected_by[nm]) for nm in names}
    agreement = pd.DataFrame(
        {nm: [feat in sets[nm] for feat in feature_names] for nm in names},
        index=feature_names,
    )
    consensus = agreement.sum(axis=1).astype(int)
    consensus.name = "consensus"

    jaccard = pd.DataFrame(
        [[_jaccard_similarity(sets[a], sets[b]) for b in names] for a in names],
        index=names,
        columns=names,
    )

    return SelectorComparison(
        agreement=agreement,
        jaccard=jaccard,
        consensus=consensus,
        n_selectors=len(names),
        skipped=skipped,
    )
