"""The embedding provider's cache signature and repr carry no credential from a URL under a non-secret key (SEC2-1).

``params={"base_url": "https://user:password@host"}`` used to hash the password into the cache key and print it in ``repr``.  # pragma: allowlist secret
"""

from mlframe.training.feature_handling.providers import EmbeddingProvider


def _provider(password):
    """A provider whose only credential-bearing value is a URL with the given password."""
    return EmbeddingProvider(kind="huggingface", model="stub-model", params={"base_url": f"https://user:{password}@host.example/v1"})


def test_signature_does_not_depend_on_the_url_password():
    """Two providers that differ only in the password under a non-secret key share one signature."""
    assert _provider("first-pass").signature == _provider("second-pass").signature


def test_repr_does_not_show_the_url_password():
    """The password is not printed by repr; the host still is."""
    text = repr(_provider("hunter2"))
    assert "hunter2" not in text
    assert "host.example" in text
