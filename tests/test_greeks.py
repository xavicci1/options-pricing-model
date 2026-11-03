"""Analytic Greeks against identities and finite differences."""

from __future__ import annotations

import math

import pytest

from opm import analytic_greeks, black76_greeks, black76_price, numeric_greeks


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_analytic_greeks_match_finite_differences(atm: dict[str, float], option_type: str) -> None:
    params = {**atm, "option_type": option_type}
    closed = analytic_greeks(**params)
    bumped = numeric_greeks(
        **params,
        spot_bump=1e-3,
        vol_bump=1e-4,
        time_bump=1e-5,
        rate_bump=1e-4,
    )
    assert closed.delta == pytest.approx(bumped.delta, abs=1e-5)
    assert closed.gamma == pytest.approx(bumped.gamma, abs=1e-5)
    assert closed.vega == pytest.approx(bumped.vega, abs=1e-3)
    assert closed.theta == pytest.approx(bumped.theta, abs=1e-2)
    assert closed.rho == pytest.approx(bumped.rho, abs=1e-3)


def test_call_put_greek_identities(atm: dict[str, float]) -> None:
    call = analytic_greeks(**atm, option_type="call")
    put = analytic_greeks(**atm, option_type="put")
    discount_spot = math.exp(-atm["dividend_yield"] * atm["time_to_expiry"])
    assert call.delta - put.delta == pytest.approx(discount_spot, abs=1e-12)
    assert call.gamma == pytest.approx(put.gamma, abs=1e-12)
    assert call.vega == pytest.approx(put.vega, abs=1e-12)
    rho_gap = atm["strike"] * atm["time_to_expiry"] * math.exp(-atm["rate"] * atm["time_to_expiry"])
    assert call.rho - put.rho == pytest.approx(rho_gap, abs=1e-10)


def test_black76_rho_holds_the_futures_price_fixed() -> None:
    futures, strike, time, rate, vol = 100.0, 100.0, 1.0, 0.05, 0.20
    greeks = black76_greeks(futures, strike, time, rate, vol, "call")
    premium = black76_price(futures, strike, time, rate, vol, "call")
    assert greeks.rho == pytest.approx(-time * premium, abs=1e-12)
    bump = 1e-4
    finite = (
        black76_price(futures, strike, time, rate + bump, vol, "call")
        - black76_price(futures, strike, time, rate - bump, vol, "call")
    ) / (2.0 * bump)
    assert greeks.rho == pytest.approx(finite, abs=1e-4)
    partial = analytic_greeks(
        spot=futures,
        strike=strike,
        time_to_expiry=time,
        rate=rate,
        volatility=vol,
        dividend_yield=rate,
        option_type="call",
    )
    assert abs(partial.rho - greeks.rho) > 1.0


def test_greeks_require_positive_time_and_volatility(atm: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        analytic_greeks(**{**atm, "volatility": 0.0, "option_type": "call"})
    with pytest.raises(ValueError):
        analytic_greeks(**{**atm, "time_to_expiry": 0.0, "option_type": "put"})
