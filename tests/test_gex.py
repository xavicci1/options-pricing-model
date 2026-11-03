"""Dealer gamma exposure sign, scale, and CSV input."""

from __future__ import annotations

from pathlib import Path

import pytest

from opm import ChainRow, analytic_greeks, net_gex_by_strike, read_chain_csv

ROOT = Path(__file__).resolve().parents[1]


def _expected(spot: float, strike: float, time: float, rate: float, div: float, vol: float, oi: float, mult: float, sign: float) -> float:
    gamma = analytic_greeks(spot, strike, time, rate, vol, div, "call").gamma
    return sign * gamma * oi * mult * spot * spot * 0.01


def test_call_positive_put_negative_and_same_strike_nets() -> None:
    spot, time, rate, div, vol, oi, mult = 100.0, 0.5, 0.02, 0.01, 0.25, 10.0, 100.0
    chain = [
        ChainRow(100.0, oi, vol, "call"),
        ChainRow(100.0, oi, vol, "put"),
        ChainRow(110.0, oi, vol, "call"),
    ]
    result = net_gex_by_strike(chain, spot, time, rate, div, mult)
    assert result.strikes.tolist() == [100.0, 110.0]
    assert result.net_gex[0] == pytest.approx(0.0, abs=1e-8)
    assert result.net_gex[1] == pytest.approx(
        _expected(spot, 110.0, time, rate, div, vol, oi, mult, 1.0)
    )
    assert result.total == pytest.approx(float(result.net_gex.sum()))
    assert result.net_gex[1] > 0.0


def test_put_only_is_negative() -> None:
    spot, time, rate, vol, oi = 50.0, 0.25, 0.01, 0.30, 4.0
    result = net_gex_by_strike(
        [{"strike": 45, "oi": oi, "iv": vol, "type": "put"}],
        spot,
        time,
        rate,
        multiplier=20.0,
    )
    assert result.net_gex[0] == pytest.approx(
        _expected(spot, 45.0, time, rate, 0.0, vol, oi, 20.0, -1.0)
    )
    assert result.total < 0.0


def test_zero_open_interest_contributes_nothing() -> None:
    result = net_gex_by_strike(
        [ChainRow(100.0, 0.0, 0.2, "call")],
        spot=100.0,
        time_to_expiry=0.3,
    )
    assert result.total == 0.0
    assert result.strikes.tolist() == [100.0]


def test_sample_chain_csv_reads() -> None:
    rows = read_chain_csv(ROOT / "examples" / "sample_chain.csv")
    assert len(rows) == 6
    assert {row.option_type for row in rows} == {"call", "put"}


def test_negative_open_interest_is_rejected() -> None:
    with pytest.raises(ValueError, match="open_interest"):
        net_gex_by_strike([ChainRow(100.0, -1.0, 0.2, "call")], 100.0, 0.2)
