"""`TrainingSplitConfig.conformal_size` was accepted, range-validated, counted in the split budget -- and read by nothing.

`grep -rn conformal_size src/` returns the field declaration, the sum validator, one docstring, and
`_conformal_split.py`'s own internals. The production split path is `splitting.make_train_test_split` ->
`_split_helpers._carve_calib_from_train`, which carves a calib slice and no conformal slice; the four
structure-aware carvers in `_conformal_split.py` are imported only by their own test module.

So a user setting `conformal_size=0.05` to get the documented behaviour -- "calib_size fits the recalibration
map g, conformal_size scores g(model) so the interval reflects what ships" -- got no conformal slice at all.
Finalize fell back to reusing the calib slice, which makes the residuals in-sample for g: precisely the
optimistic-coverage regime the field exists to avoid, with intervals narrower than the truth and nothing
warning. Refusing the setting is the honest behaviour until the carve is wired.
"""

from __future__ import annotations

import pytest

from mlframe.training._preprocessing_configs import TrainingSplitConfig


class TestTheUnwiredKnobIsRefused:
    """A setting that does nothing must not be accepted as though it did something."""

    def test_a_nonzero_conformal_size_raises(self):
        """The documented usage, which silently produced optimistic intervals."""
        with pytest.raises(ValueError, match="not wired"):
            TrainingSplitConfig(conformal_size=0.05)

    def test_the_message_says_what_would_have_happened(self):
        """A refusal with no explanation just moves the confusion."""
        with pytest.raises(ValueError, match="optimistically narrow"):
            TrainingSplitConfig(conformal_size=0.2)

    @pytest.mark.parametrize("value", [None, 0.0])
    def test_the_documented_defaults_are_accepted(self, value):
        """Unset and explicit zero both mean "use the calib-slice fallback", which is supported."""
        assert TrainingSplitConfig(conformal_size=value).conformal_size == value

    def test_the_existing_range_validation_still_applies(self):
        """`ge=0.0, lt=1.0` must fire before the wiring check, so a nonsense value reports as nonsense."""
        with pytest.raises(ValueError):
            TrainingSplitConfig(conformal_size=1.5)

    def test_the_budget_validator_still_fires_first_when_it_should(self):
        """A configuration that is BOTH over budget and unwired should not hide the budget error."""
        with pytest.raises(ValueError):
            TrainingSplitConfig(test_size=0.6, val_size=0.5, conformal_size=0.1)

    def test_calib_size_is_unaffected(self):
        """`calib_size` IS wired, and must keep working."""
        assert TrainingSplitConfig(calib_size=0.1).calib_size == 0.1


def test_nothing_in_src_reads_the_field():
    """The premise of the refusal. If someone wires the carve, this test tells them to lift it.

    AST rather than a text search: the field is legitimately NAMED in prose (its own comment, and the
    recalibration module's docstring explaining why the intervals are optimistic without it). What matters is
    whether any code READS it.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe"
    readers = []
    for path in src.rglob("*.py"):
        if path.name in ("_preprocessing_configs.py", "_conformal_split.py"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and node.attr == "conformal_size") or (isinstance(node, ast.Name) and node.id == "conformal_size"):
                readers.append(f"{path.relative_to(src).as_posix()}:{node.lineno}")
    assert not readers, "conformal_size now has a consumer; wire the carve and remove the refusal in TrainingSplitConfig: " + ", ".join(readers)
