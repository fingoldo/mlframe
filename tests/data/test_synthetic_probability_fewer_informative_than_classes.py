"""The probability generator accepts n_informative < n_classes (audit 2026-11-20, finding 22-F7)."""
from mlframe.data.synthetic import generate_modelling_data


def test_fewer_informative_columns_than_classes_generates():
    """n_informative=2 with n_classes=4 used to append four names for two columns and crash on assignment."""
    out = generate_modelling_data(
        n_samples=300, n_singly_correlated=0, n_mutually_correlated=0, n_informative=2, n_classes=4, generation="probability", random_state=0,
    )
    assert out is not None
