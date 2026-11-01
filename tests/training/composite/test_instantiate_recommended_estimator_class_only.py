"""A recommendation can only construct an estimator class, not any attribute of an mlframe module (EXE-10, DYN2-2).

The module prefix was checked, but the attribute name was not: a function or a constant named in a recommendation was called with the
caller's kwargs. The name must now resolve to a class with a ``fit`` method, and private names are not looked up.
"""

import pytest

from mlframe.training.composite._estimator_dispatch import instantiate_recommended_estimator


def test_a_non_class_attribute_is_refused():
    """A module-level function or constant named by a recommendation is not called."""
    with pytest.raises(ValueError, match="not an estimator class"):
        instantiate_recommended_estimator({"module": "mlframe.utils.shared", "estimator": "sha256_of_file"})


def test_a_private_name_is_refused():
    """A leading underscore is never looked up through a recommendation."""
    with pytest.raises(ValueError, match="not an estimator class"):
        instantiate_recommended_estimator({"module": "mlframe.utils.shared", "estimator": "_private_helper"})
