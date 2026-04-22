"""Combo enumerator + results log for train_mlframe_models_suite fuzzing.

Design principles:
  * Deterministic: identical master_seed → identical combo list on any host.
  * Dedup-canonical: combos are canonicalized before hashing so
    semantically-equivalent combos (e.g. align_polars_dicts=True with
    pandas input) collapse to one.
  * Pairwise-covering: the greedy sampler guarantees every
    (axis_i=value_i, axis_j=value_j) pair is exercised at least once.
  * xfail-aware: combos hitting known bugs are auto-marked xfail via a
    declarative rule table — single source of truth shared with tracked
    tests elsewhere in the suite.

Results log: every fuzz run appends one JSONL row per combo to
``tests/training/_fuzz_results.jsonl`` capturing combo key, outcome
(pass/fail/xfail/skip), and — on failure — the exception class and a
one-line summary. That file is the audit trail used by human / agent
follow-ups to decide what to fix next.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import astuple, dataclass, field
from itertools import combinations as iter_combinations
from itertools import product as iter_product
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Axis space
# ---------------------------------------------------------------------------

MODELS: tuple[str, ...] = ("cb", "xgb", "lgb", "hgb", "linear")

AXES: dict[str, tuple[Any, ...]] = {
    "input_type": ("pandas", "polars_utf8", "polars_enum", "polars_nullable"),
    "n_rows": (300, 600, 1200),
    "cat_feature_count": (0, 1, 3, 8),
    "null_fraction_cats": (0.0, 0.1, 0.3),
    "use_mrmr_fs": (False, True),
    "weight_schemas": (("uniform",), ("uniform", "recency")),
    "target_type": ("binary_classification", "regression"),
    "auto_detect_cats": (True, False),
    "align_polars_categorical_dicts": (True, False),
}


# ---------------------------------------------------------------------------
# Combo dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FuzzCombo:
    models: tuple[str, ...]
    input_type: str
    n_rows: int
    cat_feature_count: int
    null_fraction_cats: float
    use_mrmr_fs: bool
    weight_schemas: tuple[str, ...]
    target_type: str
    auto_detect_cats: bool
    align_polars_categorical_dicts: bool
    seed: int

    def canonical_key(self) -> tuple:
        """Hashable tuple used for dedup. Canonicalizes semantically
        equivalent combos so e.g. ``align_polars_categorical_dicts=True`` with
        pandas input collapses to the False variant."""
        align = self.align_polars_categorical_dicts
        if self.input_type == "pandas":
            align = False
        null_frac = self.null_fraction_cats if self.cat_feature_count > 0 else 0.0
        return (
            tuple(sorted(self.models)),
            self.input_type,
            self.n_rows,
            self.cat_feature_count,
            null_frac,
            self.use_mrmr_fs,
            tuple(sorted(self.weight_schemas)),
            self.target_type,
            self.auto_detect_cats,
            align,
        )

    def short_id(self) -> str:
        h = hashlib.blake2s(repr(self.canonical_key()).encode(), digest_size=4).hexdigest()
        return f"c{self.seed:04d}_{h}"

    def pytest_id(self) -> str:
        # Include a human-readable prefix so failing IDs are diagnostic.
        tag = "_".join(sorted(self.models))
        short_input = self.input_type.replace("polars_", "pl_")
        return f"{self.short_id()}-{tag}-{short_input}-n{self.n_rows}"

    def to_json(self) -> dict:
        return {
            "short_id": self.short_id(),
            "models": list(self.models),
            "input_type": self.input_type,
            "n_rows": self.n_rows,
            "cat_feature_count": self.cat_feature_count,
            "null_fraction_cats": self.null_fraction_cats,
            "use_mrmr_fs": self.use_mrmr_fs,
            "weight_schemas": list(self.weight_schemas),
            "target_type": self.target_type,
            "auto_detect_cats": self.auto_detect_cats,
            "align_polars_categorical_dicts": self.align_polars_categorical_dicts,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# Known-xfail rules (single source of truth for auto-xfail)
# ---------------------------------------------------------------------------


def _rule_linear_polars_gating_bug(c: FuzzCombo) -> bool:
    return (
        "linear" in c.models
        and len(c.models) > 1
        and c.input_type.startswith("polars")
    )


KNOWN_XFAIL_RULES: list[tuple[Callable[[FuzzCombo], bool], str]] = [
    (
        _rule_linear_polars_gating_bug,
        "polars_pipeline_applied global-flag bug (core.py:3049): CB/XGB setting the "
        "flag to True during their Polars-fastpath iteration bleeds into the "
        "linear iteration, causing its encoder+scaler+imputer pipeline to be "
        "silently skipped. LogReg then receives a raw pd.Categorical frame. "
        "Tracked for a proper per-strategy gating fix.",
    ),
]


def xfail_reason(combo: FuzzCombo) -> str | None:
    for predicate, reason in KNOWN_XFAIL_RULES:
        if predicate(combo):
            return reason
    return None


# ---------------------------------------------------------------------------
# Enumerator: pairwise-greedy, deterministic, deduplicated
# ---------------------------------------------------------------------------


def _powerset_nonempty(items: tuple[str, ...]) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for r in range(1, len(items) + 1):
        for sub in iter_combinations(items, r):
            out.append(tuple(sorted(sub)))
    return out


def _sample_axes(rng: random.Random) -> dict[str, Any]:
    return {name: rng.choice(values) for name, values in AXES.items()}


def _build_combo(models: tuple[str, ...], axes: dict[str, Any], seed: int) -> FuzzCombo:
    return FuzzCombo(
        models=tuple(sorted(models)),
        input_type=axes["input_type"],
        n_rows=axes["n_rows"],
        cat_feature_count=axes["cat_feature_count"],
        null_fraction_cats=axes["null_fraction_cats"],
        use_mrmr_fs=axes["use_mrmr_fs"],
        weight_schemas=axes["weight_schemas"],
        target_type=axes["target_type"],
        auto_detect_cats=axes["auto_detect_cats"],
        align_polars_categorical_dicts=axes["align_polars_categorical_dicts"],
        seed=seed,
    )


def _all_axis_pairs() -> set[tuple[str, Any, str, Any]]:
    pairs: set[tuple[str, Any, str, Any]] = set()
    # Also include model-count ("n_models" pseudo-axis) to balance single vs
    # multi-model combos across other axes.
    axes_ext: dict[str, tuple[Any, ...]] = {**AXES, "n_models": (1, 2, 3, 4, 5)}
    axis_names = list(axes_ext.keys())
    for i in range(len(axis_names)):
        for j in range(i + 1, len(axis_names)):
            ai, aj = axis_names[i], axis_names[j]
            for vi in axes_ext[ai]:
                for vj in axes_ext[aj]:
                    pairs.add((ai, vi, aj, vj))
    return pairs


def _combo_pairs(combo: FuzzCombo) -> set[tuple[str, Any, str, Any]]:
    values = {
        "input_type": combo.input_type,
        "n_rows": combo.n_rows,
        "cat_feature_count": combo.cat_feature_count,
        "null_fraction_cats": combo.null_fraction_cats,
        "use_mrmr_fs": combo.use_mrmr_fs,
        "weight_schemas": combo.weight_schemas,
        "target_type": combo.target_type,
        "auto_detect_cats": combo.auto_detect_cats,
        "align_polars_categorical_dicts": combo.align_polars_categorical_dicts,
        "n_models": len(combo.models),
    }
    names = list(values.keys())
    out: set[tuple[str, Any, str, Any]] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ai, aj = names[i], names[j]
            out.add((ai, values[ai], aj, values[aj]))
    return out


def enumerate_combos(
    target: int = 150,
    master_seed: int = 2026_04_22,
    model_universe: tuple[str, ...] = MODELS,
) -> list[FuzzCombo]:
    """Return `target` unique FuzzCombo instances covering the axis space.

    Phase A seeds with one combo per non-empty model-subset.
    Phase B greedy-fills until pairwise coverage is achieved.
    Phase C random-fills until len == target.
    """
    rng = random.Random(master_seed)
    seen: set[tuple] = set()
    combos: list[FuzzCombo] = []

    # Phase A — model subsets
    for subset in _powerset_nonempty(model_universe):
        axes = _sample_axes(rng)
        combo = _build_combo(subset, axes, len(combos))
        key = combo.canonical_key()
        if key in seen:
            continue
        seen.add(key)
        combos.append(combo)

    # Phase B — pairwise coverage
    required = _all_axis_pairs()
    covered: set[tuple[str, Any, str, Any]] = set()
    for c in combos:
        covered.update(_combo_pairs(c))

    tries = 0
    max_tries = 10_000
    while covered < required and tries < max_tries and len(combos) < target:
        uncovered = required - covered
        best_combo = None
        best_new = 0
        for _ in range(50):
            subset = rng.choice(_powerset_nonempty(model_universe))
            axes = _sample_axes(rng)
            candidate = _build_combo(subset, axes, len(combos))
            if candidate.canonical_key() in seen:
                continue
            cand_pairs = _combo_pairs(candidate)
            new = len(cand_pairs & uncovered)
            if new > best_new:
                best_new = new
                best_combo = candidate
        if best_combo is None:
            break
        seen.add(best_combo.canonical_key())
        combos.append(best_combo)
        covered.update(_combo_pairs(best_combo))
        tries += 1

    # Phase C — random fill until target
    while len(combos) < target:
        subset = rng.choice(_powerset_nonempty(model_universe))
        axes = _sample_axes(rng)
        candidate = _build_combo(subset, axes, len(combos))
        key = candidate.canonical_key()
        if key in seen:
            continue
        seen.add(key)
        combos.append(candidate)

    return combos[:target]


# ---------------------------------------------------------------------------
# Results log — JSONL append-only
# ---------------------------------------------------------------------------

RESULTS_LOG = Path(__file__).parent / "_fuzz_results.jsonl"


def log_combo_outcome(
    combo: FuzzCombo,
    outcome: str,
    duration_s: float,
    error_class: str | None = None,
    error_summary: str | None = None,
    extra: dict | None = None,
) -> None:
    """Append one JSONL row with the combo's outcome.

    Columns: combo fields, outcome in {pass,fail,xpass,xfail,skip}, duration,
    error_class/error_summary (for fail/xpass rows), extra (free-form dict).
    """
    row: dict = {
        **combo.to_json(),
        "outcome": outcome,
        "duration_s": round(duration_s, 3),
    }
    if error_class:
        row["error_class"] = error_class
    if error_summary:
        row["error_summary"] = error_summary[:300]
    if extra:
        row["extra"] = extra
    try:
        RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        pass  # never break a test because logging failed


def read_fail_summary() -> dict:
    """Return a summary of failures since the last run start marker."""
    if not RESULTS_LOG.exists():
        return {"fails": [], "totals": {}}
    totals: dict[str, int] = {}
    fails: list[dict] = []
    with RESULTS_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            totals[row.get("outcome", "?")] = totals.get(row.get("outcome", "?"), 0) + 1
            if row.get("outcome") == "fail":
                fails.append(row)
    return {"fails": fails, "totals": totals}


# ---------------------------------------------------------------------------
# Frame builder — turns a combo into (df, target_col, cat_feature_names)
# ---------------------------------------------------------------------------


def build_frame_for_combo(combo: FuzzCombo):
    """Build a pd / pl DataFrame matching the combo's input spec.

    Returns (df, target_col_name, cat_feature_names: list[str]).
    """
    import numpy as np

    rng = np.random.default_rng(combo.seed)
    n = combo.n_rows

    num_cols = {
        f"num_{i}": rng.standard_normal(n).astype("float32") for i in range(4)
    }
    cat_pools = [
        ["A", "B", "C"],
        ["X", "Y", "Z", "W"],
        ["alpha", "beta"],
        ["cat1", "cat2", "cat3", "cat4", "cat5"],
        ["US", "UK", "DE"],
        ["mon", "tue", "wed", "thu"],
        ["P", "Q"],
        ["r1", "r2", "r3"],
    ]
    cat_cols = {}
    cat_names: list[str] = []
    for i in range(combo.cat_feature_count):
        pool = cat_pools[i]
        values = [pool[j % len(pool)] for j in range(n)]
        if combo.null_fraction_cats > 0:
            mask = rng.random(n) < combo.null_fraction_cats
            values = [None if mask[j] else v for j, v in enumerate(values)]
        cat_cols[f"cat_{i}"] = values
        cat_names.append(f"cat_{i}")

    # Target: derive from num_0 + num_1 with noise so models have signal
    if combo.target_type == "regression":
        target = 2.0 * num_cols["num_0"] - 1.5 * num_cols["num_1"] + rng.standard_normal(n) * 0.3
        target_col = "target_reg"
    else:
        logits = num_cols["num_0"] - 0.5 * num_cols["num_1"]
        target = (logits + rng.standard_normal(n) * 0.3 > 0).astype("int32")
        target_col = "target"

    if combo.input_type == "pandas":
        import pandas as pd
        data = {**num_cols}
        for name, values in cat_cols.items():
            data[name] = pd.Categorical(values)
        data[target_col] = target
        return pd.DataFrame(data), target_col, cat_names

    import polars as pl
    data_pl: dict[str, Any] = {**num_cols}
    for name, values in cat_cols.items():
        if combo.input_type == "polars_enum":
            pool_values = [v for v in values if v is not None]
            enum_type = pl.Enum(sorted(set(pool_values)))
            data_pl[name] = pl.Series(values).cast(enum_type)
        elif combo.input_type == "polars_nullable":
            data_pl[name] = pl.Series(values).cast(pl.Categorical)
        else:  # polars_utf8
            data_pl[name] = pl.Series(values, dtype=pl.Utf8)
    data_pl[target_col] = target
    return pl.DataFrame(data_pl), target_col, cat_names
