"""Getting data out of the generator, and getting the same data back months later.

Both of these fail quietly when they fail. A polars ``Categorical`` built without explicit levels compares
wrong against another frame's rather than raising; a replayed bed that was edited since produces data that
looks fine and is not what the recorded numbers describe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from mlframe.data.datasets import scenarios as scenario_registry
from mlframe.data.datasets.frames import feature_matrix, to_numpy, to_polars
from mlframe.data.datasets.generator import generate
from mlframe.data.datasets.replay import replay, replay_from_record, replay_report, spec_hash_for
from mlframe.data.datasets.scenarios._mixed_types import graded_cardinality_spec
from mlframe.data.datasets.scenarios._observation import missingness_trio_spec

polars = pytest.importorskip("polars", reason="the polars conversion is only testable where polars is installed")


def test_categoricals_become_an_enum_with_the_levels_the_spec_declared() -> None:
    """A polars Categorical resolves through a process-wide cache, so two frames' codes can mean different things."""
    spec = graded_cardinality_spec(seed=0, n_samples=400)
    frame = generate(spec).frame

    converted, _notes = to_polars(frame, spec)

    dtype = converted.schema["k5"]
    assert isinstance(dtype, polars.Enum), f"k5 came back as {dtype!r} rather than an Enum"
    assert len(dtype.categories) == 5


def test_converting_without_a_spec_says_the_levels_are_not_stable() -> None:
    """Levels read off one draw are correct for that frame and NOT comparable with another's."""
    frame = generate(graded_cardinality_spec(seed=0, n_samples=400)).frame

    _converted, notes = to_polars(frame)

    assert any("another draw" in note for note in notes)


def test_two_draws_of_the_same_bed_get_comparable_enums() -> None:
    """This is the whole point: codes from independent frames must mean the same thing."""
    spec = graded_cardinality_spec(seed=0, n_samples=400)
    first, _ = to_polars(generate(spec).frame, spec)
    second, _ = to_polars(generate(spec.model_copy(update={"root_seed": 7})).frame, spec)

    assert first.schema["k20"] == second.schema["k20"]


def test_masked_integers_stay_null_rather_than_becoming_a_sentinel() -> None:
    """A -999 in an integer column is a value some method treats as a very small number."""
    spec = missingness_trio_spec(seed=0, n_samples=600)
    spec = spec.model_copy(update={"features": tuple(feature.model_copy(update={"dtype": "int"}) if feature.name == "s0" else feature for feature in spec.features)})
    frame = generate(spec).frame

    converted, _notes = to_polars(frame, spec)

    assert converted["s0"].null_count() > 0
    assert converted["s0"].min() is None or int(converted["s0"].min()) > -900


def test_the_numpy_conversion_says_it_turned_a_nominal_column_ordinal() -> None:
    """Encoding categories as codes is a modelling choice, not a formatting detail, so it is stated."""
    spec = graded_cardinality_spec(seed=0, n_samples=300)

    matrix, names, notes = to_numpy(generate(spec).frame, spec)

    assert matrix.shape[0] == 300
    assert len(names) == matrix.shape[1]
    assert any("ordinal" in note for note in notes)


def test_masked_cells_stay_nan_in_the_matrix() -> None:
    """A sentinel would make a missingness bed an outlier bed."""
    frame = generate(missingness_trio_spec(seed=0, n_samples=400)).frame

    matrix, names, _notes = to_numpy(frame)

    assert np.isnan(matrix[:, names.index("s0")]).any()


def test_the_feature_matrix_follows_the_caller_s_column_order() -> None:
    """A selector returning a ranked list expects the matrix to follow that ranking, not the frame's order."""
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})

    matrix = feature_matrix(frame, ["c", "a"])

    np.testing.assert_allclose(matrix, np.array([[5.0, 1.0], [6.0, 2.0]]))


def test_the_feature_matrix_refuses_an_unknown_column() -> None:
    """A typo must fail here, not produce a matrix one column narrower than the caller believes."""
    frame = pd.DataFrame({"a": [1.0]})

    with pytest.raises(KeyError, match="not in the frame"):
        feature_matrix(frame, ["a", "typo"])


def test_a_replay_of_an_unedited_bed_is_exact() -> None:
    """The only state that needs no caveat: the library still builds the bed the result was measured on."""
    expected = spec_hash_for("linear_k5_p50", seed=3)

    result = replay("linear_k5_p50", seed=3, expected_hash=expected)

    assert result.is_exact()
    assert result.dataset is not None


def test_replaying_an_edited_bed_reports_the_change_and_still_returns_the_data() -> None:
    """Looking at the new data is usually how the difference gets diagnosed, so it is not withheld.

    What must not happen is the difference being invisible, which is exactly what a name-and-seed replay
    with no hash does.
    """
    result = replay("linear_k5_p50", seed=0, expected_hash="0" * 32)

    assert result.status == "changed"
    assert result.dataset is not None
    assert result.expected_hash == "0" * 32 and result.actual_hash != result.expected_hash


def test_a_replay_without_a_hash_is_unverified_rather_than_exact() -> None:
    """ "This is probably the right data" and "this is the right data" are different claims."""
    result = replay("linear_k5_p50", seed=0)

    assert result.status == "unverified"
    assert not result.is_exact()


def test_replaying_a_scenario_that_no_longer_exists_offers_no_partial_answer() -> None:
    """A removed bed cannot be rebuilt, and returning something would invite using it."""
    result = replay("a_bed_that_was_deleted", seed=0, expected_hash="abc")

    assert result.status == "absent"
    assert result.dataset is None


def test_replaying_from_a_stored_cell_reads_the_hash_the_runner_wrote() -> None:
    """The runner records the hash per cell precisely so a replay can certify itself."""
    record = {"scenario": "linear_k5_p50", "dataset_seed": 2, "spec_hash": spec_hash_for("linear_k5_p50", seed=2)}

    assert replay_from_record(record).is_exact()


def test_the_replay_report_names_every_bed_that_can_no_longer_be_rebuilt() -> None:
    """A results file whose beds have moved on is still readable; it just has to say so at the top."""
    records = [
        {"scenario": "linear_k5_p50", "dataset_seed": 0, "spec_hash": spec_hash_for("linear_k5_p50", seed=0)},
        {"scenario": "xor3", "dataset_seed": 0, "spec_hash": "0" * 32},
    ]

    lines = replay_report(records)

    assert any("xor3" in line and "changed" in line for line in lines)
    assert any("no longer produces" in line for line in lines)


def test_the_replay_report_says_so_when_there_is_nothing_to_rebuild() -> None:
    """An empty table must not read as "everything rebuilds"."""
    assert any("nothing to rebuild" in line for line in replay_report([]))


def test_every_registered_bed_can_be_replayed_exactly_against_its_own_hash() -> None:
    """The guarantee the generator's purity buys: same spec, same seed, same hash, on any machine."""
    failures: Any = []
    for name in sorted(scenario_registry.names()):
        expected = spec_hash_for(name, seed=1)
        if not replay(name, seed=1, expected_hash=expected).is_exact():
            failures.append(name)

    assert not failures, f"these beds do not rebuild to their own hash, so the generator is not a pure function of the spec: {failures}"
