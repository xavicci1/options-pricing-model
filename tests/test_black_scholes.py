"""Closed-form prices, parity, and boundary values."""

from __future__ import annotations

import math

import pytest

from opm import black76_price, black_scholes_price


def test_put_call_parity(atm: dict[str, float]) -> None:
    call = black_scholes_price(**atm, option_type="call")
    put = black_scholes_price(**atm, option_type="put")
    forward_value = atm["spot"] * math.exp(-atm["dividend_yield"] * atm["time_to_expiry"])
    strike_value = atm["strike"] * math.exp(-atm["rate"] * atm["time_to_expiry"])
    assert call - put == pytest.approx(forward_value - strike_value, abs=1e-12)


def test_black76_put_call_parity() -> None:
    futures, strike, time, rate, vol = 100.0, 105.0, 0.5, 0.04, 0.22
    call = black76_price(futures, strike, time, rate, vol, "call")
    put = black76_price(futures, strike, time, rate, vol, "put")
    assert call - put == pytest.approx(math.exp(-rate * time) * (futures - strike), abs=1e-12)


def test_black76_matches_bsm_with_dividend_equal_to_rate() -> None:
    premium = black76_price(25000.0, 25000.0, 30 / 365.25, 0.04, 0.18, "call")
    same = black_scholes_price(
        25000.0, 25000.0, 30 / 365.25, 0.04, 0.18, dividend_yield=0.04, option_type="call"
    )
    assert premium == pytest.approx(same, abs=1e-12)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_price_increases_in_volatility(atm: dict[str, float], option_type: str) -> None:
    low = black_scholes_price(**{**atm, "volatility": 0.10, "option_type": option_type})
    high = black_scholes_price(**{**atm, "volatility": 0.30, "option_type": option_type})
    assert high > low


def test_zero_volatility_is_discounted_forward_intrinsic() -> None:
    premium = black_scholes_price(100.0, 90.0, 1.0, 0.05, 0.0, 0.02, "call")
    forward = 100.0 * math.exp((0.05 - 0.02) * 1.0)
    expected = math.exp(-0.05) * (forward - 90.0)
    assert premium == pytest.approx(expected, abs=1e-12)


def test_expiry_is_intrinsic() -> None:
    assert black_scholes_price(100, 90, 0.0, 0.05, 0.2, 0.01, "call") == pytest.approx(10.0)
    assert black_scholes_price(100, 110, 0.0, 0.05, 0.2, 0.01, "put") == pytest.approx(10.0)
    assert black_scholes_price(100, 110, 0.0, 0.05, 0.2, 0.01, "c") == pytest.approx(0.0)


def test_option_type_aliases(atm: dict[str, float]) -> None:
    assert black_scholes_price(**atm, option_type="c") == black_scholes_price(**atm, option_type="call")
    assert black_scholes_price(**atm, option_type="P") == black_scholes_price(**atm, option_type="put")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"spot": -1}, "spot"),
        ({"strike": 0}, "strike"),
        ({"time_to_expiry": -0.1}, "time_to_expiry"),
        ({"volatility": -0.2}, "volatility"),
        ({"option_type": "straddle"}, "option_type"),
    ],
)
def test_rejects_bad_inputs(atm: dict[str, float], kwargs: dict, message: str) -> None:
    params = {**atm, "option_type": "call", **kwargs}
    with pytest.raises(ValueError, match=message):
        black_scholes_price(**params)
