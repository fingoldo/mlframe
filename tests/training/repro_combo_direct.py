"""Reliably reproduce one recorded fuzz combo from ``tests/training/_fuzz_results.jsonl``.

``enumerate_combos(target=150, master_seed=X)`` is deterministic in isolation (a plain
``random.Random(master_seed)``), but the SAME index has been observed to resolve to a DIFFERENT combo
across separate process launches with the identical master_seed (driver vs worker vs a manual repro
attempt all disagreeing) -- root cause not yet found. That makes ``pytest -k <short_id>`` selection
against a freshly-enumerated COMBOS list unreliable for reproducing a specific historical failure.

This script sidesteps enumeration entirely: it reads the EXACT recorded field values for a short_id
from the jsonl log and constructs a ``FuzzCombo`` directly via the dataclass constructor, then calls
the test body function as a plain Python call (stubbing ``tmp_path``/``request``). No dependency on
enumeration order or hash-seed behaviour.

Usage:
    python tests/training/repro_combo_direct.py <short_id>

Example:
    python tests/training/repro_combo_direct.py c0018_13625051
"""
import dataclasses
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, ".")
os.environ.setdefault("MLFRAME_FUZZ_FORCE_N_ROWS", "10000")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


class _FakeNode:
    def add_marker(self, *a, **kw):
        pass


class _FakeRequest:
    node = _FakeNode()


def _load_combo(short_id: str):
    from tests.training._fuzz_combo import FuzzCombo

    with open("tests/training/_fuzz_results.jsonl", encoding="utf-8") as f:
        rec = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("short_id") == short_id:
                rec = r
                break
    if rec is None:
        raise SystemExit(f"NOT FOUND: {short_id}")

    field_names = {f.name for f in dataclasses.fields(FuzzCombo)}
    kwargs = {}
    for k, v in rec.items():
        if k not in field_names:
            continue
        if k in ("models", "weight_schemas"):
            v = tuple(v)
        kwargs[k] = v
    return FuzzCombo(**kwargs)


def main() -> None:
    short_id = sys.argv[1]
    combo = _load_combo(short_id)
    print("Reconstructed combo short_id:", combo.short_id())
    print("n_rows override:", os.environ.get("MLFRAME_FUZZ_FORCE_N_ROWS"))

    import tests.training.fuzz.test_fuzz_suite as tmod

    tmp_path = Path(tempfile.mkdtemp(prefix="fuzzrepro_"))
    try:
        tmod.test_fuzz_train_mlframe_models_suite(combo, tmp_path, _FakeRequest())
        print("NO CRASH")
    except Exception as e:
        traceback.print_exc()
        print("CRASHED:", type(e).__name__, e)


if __name__ == "__main__":
    main()
