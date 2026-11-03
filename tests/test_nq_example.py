"""The bundled NQ versus VXN study prices higher volatility at a higher premium."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "nq_realized_vs_vxn", ROOT / "examples" / "nq_realized_vs_vxn.py"
)
assert SPEC is not None and SPEC.loader is not None
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)


def test_atm_call_premium_rises_with_the_volatility_input() -> None:
    report = STUDY.build_report()
    rows = sorted(report["rows"], key=lambda row: row["vol"])
    calls = [row["call"] for row in rows]
    puts = [row["put"] for row in rows]
    assert calls[0] < calls[1] < calls[2]
    assert puts[0] < puts[1] < puts[2]
    assert report["futures"] == report["strike"]
    assert report["vxn_points"] > 3.0
    # Black-76 at-the-money call and put are close; both stay positive.
    for row in rows:
        assert row["call"] > 0.0
        assert row["put"] == pytest.approx(row["call"], rel=1e-12)
