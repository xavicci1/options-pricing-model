"""Command line smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from opm.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_price_command(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "price",
            "--spot",
            "100",
            "--strike",
            "100",
            "--time",
            "1",
            "--rate",
            "0.05",
            "--dividend-yield",
            "0.02",
            "--vol",
            "0.2",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "black-scholes-merton" in captured.out
    assert "price" in captured.out


def test_iv_command_round_trip(capsys: pytest.CaptureFixture[str]) -> None:
    from opm import black_scholes_price

    premium = black_scholes_price(100, 100, 1.0, 0.05, 0.2, 0.0, "put")
    code = main(
        [
            "iv",
            "--price",
            str(premium),
            "--spot",
            "100",
            "--strike",
            "100",
            "--time",
            "1",
            "--rate",
            "0.05",
            "--type",
            "put",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "0.20000000" in captured.out


def test_invalid_volatility_returns_one(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["price", "--spot", "100", "--strike", "100", "--time", "1", "--vol", "-0.1"])
    captured = capsys.readouterr()
    assert code == 1
    assert "volatility" in captured.err


def test_gex_command_prints_total(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "gex",
            "--chain",
            str(ROOT / "examples" / "sample_chain.csv"),
            "--spot",
            "100",
            "--time",
            "0.25",
            "--rate",
            "0.04",
            "--dividend-yield",
            "0.01",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "total" in captured.out
    assert "net dealer GEX" in captured.out
