"""``embed_website_to_mlflow`` refuses a URL whose scheme is not http or https, before it writes the file.

The embed iframe is sandboxed with ``allow-same-origin allow-scripts``. A ``javascript:`` or ``data:`` source would then run script in the
origin of the MLflow UI, so the helper must not write such an embed at all.
"""

from __future__ import annotations

import pytest

from mlframe.integrations import mlflow as mlflow_mod


@pytest.mark.parametrize("url", ["javascript:alert(document.cookie)", "data:text/html,<script>alert(1)</script>"])
def test_embed_refuses_script_scheme_and_writes_nothing(tmp_path, url):
    """A javascript: or data: URL raises ValueError and leaves the target directory empty."""
    target = tmp_path / "embed"

    with pytest.raises(ValueError, match="scheme"):
        mlflow_mod.embed_website_to_mlflow(url, fname=str(target))

    assert list(tmp_path.iterdir()) == []
