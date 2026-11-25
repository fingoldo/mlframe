"""Cluster 21 regression: the SHAP per-instance result documents its figure lifecycle and its n_background field (audit 2026-11-20, finding 21-F26).

Docstring pins: the figure is documented as closed on return, and n_background is documented as counting explained rows.
"""

from mlframe.reporting.charts import shap_per_instance as spi


def test_result_docs_say_the_figure_is_closed_on_return():
    """The result docstring no longer claims the figure is still open (21-F26)."""
    doc = spi.ShapPerInstanceResult.__doc__ or ""
    assert "still-open unless closed" not in doc
    assert "already CLOSED on return" in doc


def test_n_background_is_documented_as_explained_rows():
    """The field comment states that n_background counts explained rows (21-F26)."""
    import inspect

    src = inspect.getsource(spi.ShapPerInstanceResult)
    assert "number of EXPLAINED rows" in src
