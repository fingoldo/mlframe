"""The cross-process family-budget cache must never be readable half-written."""

import orjson

import mlframe.feature_selection.filters._fe_family_budget as fb


def test_the_final_file_appears_only_once_complete(tmp_path, monkeypatch):
    """A reader that catches the file mid-write falls back to equal-split budgets and loses the learned ROI."""
    monkeypatch.setattr(fb, "_BUDGET_CACHE_DIR", tmp_path)
    seen = {}
    real_replace = fb.os.replace

    def _replace(src, dst):
        # At the moment of publication the destination must not exist yet, and the payload must already be whole.
        seen["dst_existed"] = fb.os.path.exists(dst)
        seen["tmp_payload"] = open(src, encoding="utf-8").read()
        return real_replace(src, dst)

    monkeypatch.setattr(fb.os, "replace", _replace)
    fb.persist_budgets({"poly": 0.75, "trig": 0.25}, cache_key="t", fingerprint="fp")

    assert seen["dst_existed"] is False
    assert orjson.loads(seen["tmp_payload"]) == {"poly": 0.75, "trig": 0.25}
    assert fb.load_budgets(cache_key="t", fingerprint="fp") == {"poly": 0.75, "trig": 0.25}


def test_a_failed_write_leaves_no_temp_file_behind(tmp_path, monkeypatch):
    monkeypatch.setattr(fb, "_BUDGET_CACHE_DIR", tmp_path)

    def _boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(fb.os, "replace", _boom)
    fb.persist_budgets({"poly": 1.0}, cache_key="t2", fingerprint="fp")
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []
