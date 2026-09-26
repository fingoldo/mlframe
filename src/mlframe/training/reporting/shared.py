"""Cross-package API of ``mlframe.training.reporting``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.training.reporting`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from ._reporting import (  # noqa: F401
    _canonical_multilabel_y as canonical_multilabel_y,
    _style_with_caption as style_with_caption,
)
from ._reporting_regression._sensor_ledger import (  # noqa: F401
    clear_sensor_trips,
    record_sensor_trip,
    sensor_trips_for,
)

__all__ = [
    "canonical_multilabel_y",
    "clear_sensor_trips",
    "record_sensor_trip",
    "sensor_trips_for",
    "style_with_caption",
]
