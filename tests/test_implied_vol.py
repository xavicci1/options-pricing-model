"""Implied volatility round trips and arbitrage rejections."""

from __future__ import annotations

import pytest

from opm import ImpliedVolatilityError, black_scholes_price, implied_volatility


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("volatility", [0.10, 0.20, 0.40])
@pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
def test_implied_vol_round_trip(option_type: str, volatility: float, strike: float) -> None:
    kwargs = dict(
        spot=100.0,
        strike=strike,
        time_to_expiry=1.0,
        rate=0.05,
        volatility=volatility,
        dividend_yield=0.02,
        option_type=option_type,
    )
    premium = black_scholes_price(**kwargs)
    solved = implied_volatility(
        premium,
        spot=100.0,
        strike=strike,
        time_to_expiry=1.0,
        rate=0.05,
        dividend_yield=0.02,
        option_type=option_type,
    )
    assert solved == pytest.approx(volatility, abs=1e-6)
    repriced = black_scholes_price(**{**kwargs, "volatility": solved})
    assert repriced == pytest.approx(premium, abs=1e-8)


def test_round_trip_with_poor_initial_guesses() -> None:
    kwargs = dict(
        spot=100.0,
        strike=130.0,
        time_to_expiry=1.0,
        rate=0.03,
        volatility=0.35,
        dividend_yield=0.01,
        option_type="call",
    )
    premium = black_scholes_price(**kwargs)
    for guess in (0.05, 1.5, 4.0):
        solved = implied_volatility(
            premium,
            spot=kwargs["spot"],
            strike=kwargs["strike"],
            time_to_expiry=kwargs["time_to_expiry"],
            rate=kwargs["rate"],
            dividend_yield=kwargs["dividend_yield"],
            option_type="call",
            initial_guess=guess,
        )
        assert solved == pytest.approx(kwargs["volatility"], abs=1e-5)


def test_deep_in_the_money_put_round_trip() -> None:
    kwargs = dict(
        spot=80.0,
        strike=120.0,
        time_to_expiry=0.5,
        rate=0.02,
        volatility=0.45,
        dividend_yield=0.0,
        option_type="put",
    )
    premium = black_scholes_price(**kwargs)
    solved = implied_volatility(
        premium, 80.0, 120.0, 0.5, 0.02, 0.0, "put", initial_guess=0.05
    )
    assert solved == pytest.approx(0.45, abs=1e-5)


def test_price_below_intrinsic_is_rejected() -> None:
    with pytest.raises(ImpliedVolatilityError, match="lower bound"):
        implied_volatility(19.0, spot=120.0, strike=100.0, time_to_expiry=1.0, rate=0.0, option_type="call")


def test_price_above_upper_bound_is_rejected() -> None:
    with pytest.raises(ImpliedVolatilityError, match="upper bound"):
        implied_volatility(120.5, spot=120.0, strike=100.0, time_to_expiry=1.0, rate=0.0, option_type="call")


def test_zero_time_value_returns_zero_volatility() -> None:
    # Rate and dividend are zero, so the intrinsic value is the zero-vol price.
    solved = implied_volatility(20.0, spot=120.0, strike=100.0, time_to_expiry=1.0, rate=0.0, option_type="call")
    assert solved == 0.0


def test_expiry_is_rejected() -> None:
    with pytest.raises(ValueError):
        implied_volatility(1.0, spot=100.0, strike=100.0, time_to_expiry=0.0, rate=0.0)
