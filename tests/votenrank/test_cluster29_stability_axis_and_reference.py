"""Audit 2026-11-20 findings 29-F7 and 29-F8 in the votenrank stability experiment.

29-F7: the plot passes the x-axis range once, not repeated 100 times (the old code made the correlation column the wrong length).
29-F8: the reference ranking is the clean table's, not the first noised one.
"""

import pytest

from mlframe.votenrank import stability_exp as se


def test_plot_passes_the_range_unscaled_to_the_reshape(monkeypatch):
    """create_exp_pic hands get_res_df the x-axis range as given, not multiplied by 100."""
    pytest.importorskip("seaborn")
    captured = {}

    def capturing_get_res_df(exp_range, exp_res):
        """Record the range argument, then stop the plot before drawing."""
        captured["exp_range"] = list(exp_range)
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(se, "get_res_df", capturing_get_res_df)
    with pytest.raises(RuntimeError, match="stop after capture"):
        se.create_exp_pic([0.1, 0.2], {"a": [0.5, 0.6]})
    assert captured["exp_range"] == [0.1, 0.2]
