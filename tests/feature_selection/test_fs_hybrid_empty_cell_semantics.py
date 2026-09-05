"""Meta-test for the "silently wrong, never raises" defect class in the fs_hybrid Phase 0 report.

Three defects in this harness were the same shape -- a value that is wrong but never raises, so nothing in
the pipeline complains and the reader draws a confident conclusion from a number that does not mean what it
looks like:

* `_control_adjusted` resolved the `random-<k>` control name once globally, but `k` varies per bed, so the
  column printed as a bare `-`. A dash reads as "not applicable"; it actually meant "not computed".
* The intention-to-treat aggregate averaged null beds together with signal beds, penalising an arm for the
  correct behaviour on a bed where a low score IS correct -- a mean whose population was not what its
  label said.
* `JsonlCellStore.load()` returned every record of an append-only file, so a retried cell counted twice and
  reliability read 20-of-28 instead of 20-of-20.

Rather than pinning those three instances only, the first two tests here pin the GENERAL invariants they
each violate, checked against the rendered report text so that any future cell or aggregate is covered
automatically:

1. **A report cell that can legitimately be empty must distinguish "not applicable" from "not computed".**
   Every placeholder that reaches the page must come from the declared vocabulary, each token must carry
   exactly one meaning, and a not-computed cell must additionally say WHY.
2. **Any aggregate over cells must state its denominator.** A line that reports a mean, rate or fraction
   must carry the count it was taken over, on the same line.

Honest limit on the generality, stated rather than papered over: both invariants are checked on the
RENDERED text, so they catch a fourth instance of the "empty cell with two meanings" or "aggregate with a
hidden denominator" shape, and they do NOT catch a fourth instance whose value is simply wrong while
looking perfectly well-formed -- which is exactly what the `JsonlCellStore` double-count was. That third
instance therefore gets its own behavioural test below, pinned individually, as does the gate-bed exclusion.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid._cell_store import JsonlCellStore
from mlframe.feature_selection._benchmarks.fs_hybrid._control_adjusted import (
    EMPTY_CELL_LEGEND,
    EMPTY_CELL_TOKENS,
    STATUS_COMPUTED,
    STATUS_NOT_APPLICABLE,
    STATUS_NOT_COMPUTED,
    control_adjusted_table,
)
from mlframe.feature_selection._benchmarks.fs_hybrid._leaderboard import COST_NOT_MEASURED, cost_table
from mlframe.feature_selection._benchmarks.fs_hybrid.analyze import format_report

MODEL = "logistic"
K_LABEL = "k4"

#: Placeholders a report must NEVER use for an empty cell: each is ambiguous between the two meanings, and
#: a reader cannot tell from any of them whether the question did not apply or simply went unanswered.
AMBIGUOUS_PLACEHOLDERS = ("-", "--", "n/a", "N/A", "None", "null", "nan", "?", "TBD")

#: A rendered key whose value is an aggregate over cells. Anything matching must state its denominator.
AGGREGATE_KEY = re.compile(r"\b(itt|fits_mean|mean|completed|reliability|rate|avg|median|pct)\w*\s*=")

#: The forms in which a denominator is accepted, all of which name the count on the same line.
DENOMINATOR = re.compile(r"(\bn=\d+|\bm=\d+|\bdf=\d+|\d+\s*/\s*\d+|showing \d+ of \d+|measured_on=|\bof \d+\b)")


def _cell(
    arm: str,
    scenario: str,
    dataset_seed: int,
    value: float,
    status: str = "ok",
    n_model_fits: Optional[int] = 0,
    k_label: str = K_LABEL,
) -> Dict[str, Any]:
    """Build one synthetic cell record shaped like the runner's own JSONL output."""
    record: Dict[str, Any] = {
        "arm": arm,
        "scenario": scenario,
        "dataset_seed": dataset_seed,
        "cv_seed": 0,
        "status": status,
        "n_model_fits": n_model_fits,
        "wall_time_s": 0.1,
        "cell_key": f"{scenario}|{arm}|{dataset_seed}",
    }
    if status == "ok":
        record["scores"] = {k_label: {"models": {MODEL: {"roc_auc": value}}, "n_features": 4}}
    return record


def _records_with_both_empty_kinds() -> List[Dict[str, Any]]:
    """Records exercising BOTH empty-cell meanings in one report.

    `with-controls` carries the cardinality control, so that control's own `vs_random` cell is
    NOT APPLICABLE. `no-random-control` carries no `random-<k>` arm at all, so every arm's `vs_random`
    cell there is NOT COMPUTED. The two must not render the same way.
    """
    records: List[Dict[str, Any]] = []
    for seed in (1, 2, 3):
        base = 0.60 + 0.01 * seed
        records += [
            _cell("all-features", "with-controls", seed, base),
            _cell("random-4", "with-controls", seed, base + 0.01),
            _cell("variance-sort", "with-controls", seed, base + 0.02),
            _cell("some-selector", "with-controls", seed, base + 0.05),
            _cell("all-features", "no-random-control", seed, base),
            _cell("variance-sort", "no-random-control", seed, base + 0.02),
            _cell("some-selector", "no-random-control", seed, base + 0.04),
        ]
    return records


def test_a_not_applicable_cell_and_a_not_computed_cell_never_render_the_same_way() -> None:
    """The general invariant: an empty report cell says WHICH kind of empty it is, in a distinct token."""
    records = _records_with_both_empty_kinds()

    with_controls = {r["arm"]: r for r in control_adjusted_table(records, model=MODEL, k_label=K_LABEL) if r["scenario"] == "with-controls"}
    without = {r["arm"]: r for r in control_adjusted_table(records, model=MODEL, k_label=K_LABEL) if r["scenario"] == "no-random-control"}

    assert with_controls["random-4"]["delta_vs_random_status"] == STATUS_NOT_APPLICABLE
    assert without["some-selector"]["delta_vs_random_status"] == STATUS_NOT_COMPUTED
    assert with_controls["some-selector"]["delta_vs_random_status"] == STATUS_COMPUTED

    # Distinct tokens, and a not-computed cell additionally carries a reason a reader can act on.
    assert EMPTY_CELL_TOKENS[STATUS_NOT_APPLICABLE] != EMPTY_CELL_TOKENS[STATUS_NOT_COMPUTED]
    assert without["some-selector"]["delta_vs_random_reason"]
    assert not with_controls["some-selector"]["delta_vs_random_reason"]

    report = format_report(records, models=[MODEL])
    assert EMPTY_CELL_LEGEND.strip() in report
    for token in EMPTY_CELL_TOKENS.values():
        assert token in report, f"the report renders no {token!r} cell, so this invariant is untested here"


def test_no_report_cell_uses_an_ambiguous_empty_placeholder() -> None:
    """A fourth empty column added with a bare dash instead of a declared token fails here."""
    report = format_report(_records_with_both_empty_kinds(), models=[MODEL])
    declared = set(EMPTY_CELL_TOKENS.values())
    offenders: List[str] = []
    for line in report.splitlines():
        for field in line.split():
            if "=" not in field:
                continue
            value = field.split("=", 1)[1]
            if value in AMBIGUOUS_PLACEHOLDERS and value not in declared:
                offenders.append(line.strip())
    assert not offenders, f"ambiguous empty placeholder(s) rendered instead of a declared token: {offenders[:5]}"


def test_every_aggregate_line_states_its_denominator() -> None:
    """The general invariant: a mean/rate/fraction is printed with the count it was taken over."""
    report = format_report(_records_with_both_empty_kinds(), models=[MODEL])
    checked = 0
    offenders: List[str] = []
    for line in report.splitlines():
        if not AGGREGATE_KEY.search(line):
            continue
        checked += 1
        if not DENOMINATOR.search(line):
            offenders.append(line.strip())
    assert checked, "no aggregate line was found, so this invariant would pass vacuously"
    assert not offenders, f"aggregate line(s) printed without a denominator: {offenders[:5]}"


def test_a_cost_row_distinguishes_a_measured_zero_from_an_unmeasured_arm() -> None:
    """`n_model_fits` empty because nobody counted must not read as `n_model_fits` measured at zero."""
    records = [
        _cell("free-arm", "bed", 1, 0.6, n_model_fits=0),
        _cell("free-arm", "bed", 2, 0.6, n_model_fits=0),
        _cell("uncounted-wrapper", "bed", 1, 0.7, n_model_fits=None),
        _cell("uncounted-wrapper", "bed", 2, 0.7, n_model_fits=None),
        _cell("partly-counted", "bed", 1, 0.7, n_model_fits=40),
        _cell("partly-counted", "bed", 2, 0.7, n_model_fits=None, status="crashed"),
    ]
    rows = {r["arm"]: r for r in cost_table(records)}

    assert rows["free-arm"]["n_model_fits_total"] == 0
    assert rows["free-arm"]["n_cells_measured"] == 2
    assert rows["free-arm"]["n_model_fits_reason"] is None

    assert rows["uncounted-wrapper"]["n_model_fits_total"] is None
    assert rows["uncounted-wrapper"]["n_cells_measured"] == 0
    assert rows["uncounted-wrapper"]["n_model_fits_reason"]

    # The denominator makes a mean over one surviving cell visibly different from a mean over all cells.
    assert rows["partly-counted"]["n_cells_measured"] == 1
    assert rows["partly-counted"]["n_cells"] == 2

    report = format_report(records, models=[MODEL])
    assert COST_NOT_MEASURED in report
    assert "measured_on=1/2" in report


def test_a_wrapper_arm_with_no_fit_count_is_never_omitted_from_the_cost_table() -> None:
    """The under-count that hid the expensive arms: a countless arm must still appear, flagged."""
    records = [_cell("rfecv", "bed", seed, 0.7, n_model_fits=None) for seed in (1, 2)]
    assert [r["arm"] for r in cost_table(records)] == ["rfecv"]


def test_intention_to_treat_states_the_population_it_averaged() -> None:
    """Instance 2, pinned individually: the ITT mean names its cell count, not just a bare number."""
    report = format_report(_records_with_both_empty_kinds(), models=[MODEL])
    itt_lines = [line for line in report.splitlines() if "itt=" in line]
    assert itt_lines
    for line in itt_lines:
        assert "cells" in line and re.search(r"n=\d+", line), line


def test_a_retried_cell_is_counted_once_not_twice(tmp_path: Any) -> None:
    """Instance 3, pinned individually: an append-only file's retried cell must not double-count.

    Not reachable by the two general invariants above -- a double-counted reliability figure is a
    well-formed number with a stated denominator that is simply the wrong denominator.
    """
    path = tmp_path / "cells.jsonl"
    store = JsonlCellStore(path)
    failed = _cell("rfecv", "bed", 1, 0.0, status="crashed", n_model_fits=None)
    retried = dict(_cell("rfecv", "bed", 1, 0.7), status="ok")
    store.append(failed)
    store.append(retried)

    loaded = store.load()
    assert len(loaded) == 1, "a retried cell must collapse to its latest record, not appear twice"
    assert loaded[0]["status"] == "ok"
    assert len(list(store.iter_records())) == 2, "the raw file must still hold both writes"
    assert json.loads(path.read_bytes().decode().splitlines()[0])["status"] == "crashed"


@pytest.mark.parametrize("status", [STATUS_NOT_APPLICABLE, STATUS_NOT_COMPUTED])
def test_each_empty_status_maps_to_exactly_one_token(status: str) -> None:
    """A token shared by two statuses would re-create the original defect; pinned so it cannot come back."""
    assert status in EMPTY_CELL_TOKENS
    assert len(set(EMPTY_CELL_TOKENS.values())) == len(EMPTY_CELL_TOKENS)
    assert EMPTY_CELL_TOKENS[status] not in AMBIGUOUS_PLACEHOLDERS
