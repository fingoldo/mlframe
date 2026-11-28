"""`parsed_ast` parses with the cyclic collector paused and always restores it; the tree it returns is unchanged."""

from __future__ import annotations

import ast
import gc

from tests.test_meta import _shared_ast_cache as cache_module


def _write(tmp_path, name: str, text: str):
    """Write a module and drop the cached source so the parse sees it."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    cache_module.source_text.cache_clear()
    return path


def test_the_collector_is_off_during_the_parse_and_on_after(tmp_path, monkeypatch):
    """The parse itself runs with the collector off and it is back on once the call returns."""
    path = _write(tmp_path, "m1.py", "x = 1\n")
    seen: list[bool] = []
    real = ast.parse
    monkeypatch.setattr(cache_module.ast, "parse", lambda *a, **k: seen.append(gc.isenabled()) or real(*a, **k))
    tree = cache_module.parsed_ast(path)
    assert isinstance(tree, ast.Module)
    assert seen == [False]
    assert gc.isenabled()


def test_a_syntax_error_returns_none_and_restores_the_collector(tmp_path):
    """An unparsable module gives None and still hands the collector back."""
    path = _write(tmp_path, "m2.py", "def broken(:\n")
    assert cache_module.parsed_ast(path) is None
    assert gc.isenabled()


def test_a_collector_that_was_already_off_stays_off(tmp_path):
    """A caller that had the collector off finds it off afterwards."""
    path = _write(tmp_path, "m3.py", "y = 2\n")
    gc.disable()
    try:
        assert cache_module.parsed_ast(path) is not None
        assert not gc.isenabled()
    finally:
        gc.enable()


def test_the_tree_matches_a_plain_parse(tmp_path):
    """Pausing the collector changes nothing about the returned tree."""
    text = "def f(a, b=2):\n    return [a + b for _ in range(3)]\n"
    path = _write(tmp_path, "m4.py", text)
    assert ast.dump(cache_module.parsed_ast(path)) == ast.dump(ast.parse(text, filename=str(path)))
