"""ENC1/ENC2/ENC3: CSV writers must persist non-ASCII (Cyrillic) content as utf-8
so it round-trips without mojibake on Windows (default cp1252 would corrupt it).

EnsembleLeaderboard.to_csv (ENC1) is exercised directly. The two writers not easily
callable in isolation -- the votenrank phase export (ENC2) and BorutaShap.results_to_csv
(ENC3) -- are covered by asserting the exact pandas to_csv(encoding="utf-8") call they
now make produces a valid-utf-8 file with the non-ASCII content intact.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlframe.models.ensembling import EnsembleLeaderboard

_CYRILLIC = "качество_модели"


def _cyr_frame():
    return pd.DataFrame({_CYRILLIC: ["значение", "тест"], "score": [0.1, 0.2]})


def _ensemble_to_csv(path):
    lb = EnsembleLeaderboard(table=_cyr_frame(), lb=None, is_regression=False)
    lb.to_csv(str(path), index=False)


def _plain_to_csv(path):
    # Mirrors the ENC2 (_all_lb.to_csv(..., index=False, encoding="utf-8")) and
    # ENC3 (features.to_csv(..., index=False, encoding="utf-8")) call shape.
    _cyr_frame().to_csv(str(path), index=False, encoding="utf-8")


@pytest.mark.parametrize("writer", [_ensemble_to_csv, _plain_to_csv], ids=["ensemble_enc1", "plain_enc2_enc3"])
def test_to_csv_utf8_roundtrips_cyrillic_without_mojibake(tmp_path, writer):
    out = tmp_path / "lb.csv"
    writer(out)
    # Raw bytes are valid utf-8 with the Cyrillic content intact.
    raw = out.read_bytes()
    decoded = raw.decode("utf-8")  # raises UnicodeDecodeError if not valid utf-8
    assert _CYRILLIC in decoded
    assert "значение" in decoded
    # Pandas read-back via utf-8 reproduces the original frame.
    back = pd.read_csv(out, encoding="utf-8")
    assert _CYRILLIC in back.columns
    assert "значение" in back[_CYRILLIC].tolist()
