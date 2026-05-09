"""Tests for ``CompositeProvenance`` + ``report_to_markdown`` (PR8).

Coverage map
------------
- :class:`CompositeProvenance.from_spec`: deterministic ``composite_id``
  (same spec yields same id; differing fitted_params change id).
- :meth:`to_dict` round-trip: every field appears, types are JSON-clean.
- :meth:`to_audit_trail`: paragraph contains the load-bearing
  numbers (mi_gain, n_train_rows, formula).
- :func:`_format_transform_formulas`: per-transform formula text
  pulls fitted parameters into the human-readable string.
- :func:`report_to_markdown`: kept + rejected sections render;
  ensemble section appears only when ensemble metadata supplied;
  no specs -> minimal "0 discovered" report.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Any

import pytest

from mlframe.training.composite import (
    CompositeProvenance,
    CompositeSpec,
    _format_transform_formulas,
    report_to_markdown,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_spec(
    transform_name: str = "linear_residual",
    base_column: str = "TVT_prev",
    target_col: str = "TVT",
    fitted_params: Dict[str, Any] = None,
    mi_gain: float = 0.42,
) -> CompositeSpec:
    fitted_params = fitted_params or {"alpha": 0.95, "beta": 0.5}
    return CompositeSpec(
        name=f"{target_col}__{transform_name}__{base_column}",
        target_col=target_col,
        transform_name=transform_name,
        base_column=base_column,
        fitted_params=fitted_params,
        mi_gain=mi_gain,
        mi_y=0.10,
        mi_t=0.10 + mi_gain,
        valid_domain_frac=0.99,
        n_train_rows=900,
    )


# ----------------------------------------------------------------------
# CompositeProvenance.from_spec
# ----------------------------------------------------------------------


class TestFromSpec:
    def test_basic_construction(self) -> None:
        spec = _make_spec()
        prov = CompositeProvenance.from_spec(spec, random_state=42)
        assert prov.target_col == "TVT"
        assert prov.transform_name == "linear_residual"
        assert prov.base_column == "TVT_prev"
        assert prov.mi_gain == pytest.approx(0.42)
        assert prov.n_train_rows == 900
        # composite_id is a 12-char sha256 prefix.
        assert len(prov.composite_id) == 12
        assert all(c in "0123456789abcdef" for c in prov.composite_id)

    def test_composite_id_deterministic(self) -> None:
        spec = _make_spec()
        a = CompositeProvenance.from_spec(spec, random_state=42)
        b = CompositeProvenance.from_spec(spec, random_state=42)
        assert a.composite_id == b.composite_id

    def test_composite_id_changes_with_fitted_params(self) -> None:
        a = CompositeProvenance.from_spec(
            _make_spec(fitted_params={"alpha": 0.95, "beta": 0.5}),
            random_state=42,
        )
        b = CompositeProvenance.from_spec(
            _make_spec(fitted_params={"alpha": 0.85, "beta": 0.5}),
            random_state=42,
        )
        assert a.composite_id != b.composite_id

    def test_timestamp_is_iso8601(self) -> None:
        prov = CompositeProvenance.from_spec(_make_spec(), random_state=42)
        # round-trip through datetime.fromisoformat shouldn't raise.
        dt = datetime.fromisoformat(prov.discovery_timestamp)
        # And the datetime is timezone-aware.
        assert dt.tzinfo is not None

    def test_optional_ensemble_fields(self) -> None:
        prov = CompositeProvenance.from_spec(
            _make_spec(), random_state=42,
            ensemble_weight=0.3, ensemble_strategy="oof_weighted",
        )
        assert prov.ensemble_weight == pytest.approx(0.3)
        assert prov.ensemble_strategy == "oof_weighted"


# ----------------------------------------------------------------------
# to_dict / to_audit_trail
# ----------------------------------------------------------------------


class TestSerialisation:
    def test_to_dict_json_clean(self) -> None:
        prov = CompositeProvenance.from_spec(_make_spec(), random_state=42)
        d = prov.to_dict()
        # Round-trip through JSON: catches any non-serialisable value.
        s = json.dumps(d)
        d2 = json.loads(s)
        assert d2["composite_id"] == prov.composite_id
        assert d2["mi_gain"] == pytest.approx(prov.mi_gain)
        assert d2["fitted_params"]["alpha"] == pytest.approx(0.95)

    def test_to_audit_trail_contains_numbers(self) -> None:
        prov = CompositeProvenance.from_spec(
            _make_spec(mi_gain=0.42),
            random_state=42,
            ensemble_weight=0.27, ensemble_strategy="oof_weighted",
        )
        para = prov.to_audit_trail()
        # Load-bearing facts must be present in the paragraph.
        assert "TVT" in para and "TVT_prev" in para
        assert "0.42" in para or "+0.42" in para  # mi_gain
        assert "900" in para  # n_train_rows
        assert "0.27" in para  # ensemble_weight
        assert "oof_weighted" in para
        # Formula presence:
        assert "T =" in para and "y_hat" in para


# ----------------------------------------------------------------------
# _format_transform_formulas
# ----------------------------------------------------------------------


class TestFormulas:
    def test_diff(self) -> None:
        fwd, inv, desc = _format_transform_formulas(
            "diff", base_column="x", target_col="y", fitted_params={},
        )
        assert "y - x" in fwd
        assert "T_hat + x" in inv
        assert desc

    def test_linear_residual_includes_alpha_beta(self) -> None:
        fwd, inv, desc = _format_transform_formulas(
            "linear_residual", base_column="x", target_col="y",
            fitted_params={"alpha": 0.95, "beta": -1.5},
        )
        assert "0.95" in fwd
        assert "-1.5" in fwd
        assert "0.95" in inv

    def test_logratio_includes_clip_params(self) -> None:
        fwd, inv, desc = _format_transform_formulas(
            "logratio", base_column="x", target_col="y",
            fitted_params={"median_t": 0.1, "mad_eff": 0.05},
        )
        assert "log(y)" in fwd
        assert "softcap" in inv
        assert "0.1" in inv

    def test_unknown_transform_falls_back_gracefully(self) -> None:
        fwd, inv, desc = _format_transform_formulas(
            "made_up", base_column="x", target_col="y", fitted_params={},
        )
        assert "made_up" in fwd
        assert "made_up" in inv


# ----------------------------------------------------------------------
# report_to_markdown
# ----------------------------------------------------------------------


class TestReportToMarkdown:
    def test_renders_specs_table(self) -> None:
        spec = _make_spec()
        md = report_to_markdown(target_col="TVT", specs=[spec])
        assert "# Composite-target discovery report" in md
        assert "## Discovered specs" in md
        assert "TVT__linear_residual__TVT_prev" in md
        assert "## Per-spec audit" in md

    def test_no_specs_minimal_report(self) -> None:
        md = report_to_markdown(target_col="TVT", specs=[])
        assert "0** discovered" in md or "0 discovered" in md
        # No specs section.
        assert "## Discovered specs" not in md

    def test_renders_failures(self) -> None:
        md = report_to_markdown(
            target_col="TVT", specs=[],
            failures=[{
                "base_column": "x1", "transform_name": "logratio",
                "reason": "valid_domain_frac=0.42 < 0.7",
            }],
        )
        assert "## Rejected candidates" in md
        assert "x1" in md
        assert "logratio" in md
        assert "valid_domain_frac" in md

    def test_renders_ensemble_section(self) -> None:
        spec = _make_spec()
        md = report_to_markdown(
            target_col="TVT",
            specs=[spec],
            ensemble_metadata={
                "strategy": "oof_weighted",
                "component_names": ["raw#0", f"{spec.name}#0"],
                "weights": [0.4, 0.6],
                "notes": {},
            },
        )
        assert "## Cross-target ensemble" in md
        assert "oof_weighted" in md
        assert "0.4" in md
        assert "0.6" in md
