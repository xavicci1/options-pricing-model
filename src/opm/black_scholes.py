"""Black-Scholes-Merton and Black-76 European prices.

The spot model is the Black-Scholes-Merton formula with a continuous dividend
yield. The futures model is Black (1976): the same formula with the dividend
yield set equal to the risk-free rate, so the forward is the futures price.
"""

from __future__ import annotations

import math

from scipy.stats import norm

from opm.validation import validate_contract


def _discounted_intrinsic(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    option_type: str,
) -> float:
    """European value when volatility is zero (or time has run out)."""

    forward = spot * math.exp((rate - dividend_yield) * time_to_expiry)
    discount = math.exp(-rate * time_to_expiry)
    if option_type == "call":
        return discount * max(forward - strike, 0.0)
    return discount * max(strike - forward, 0.0)


def _d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
) -> tuple[float, float, float, float, float]:
    sqrt_t = math.sqrt(time_to_expiry)
    vol_sqrt_t = volatility * sqrt_t
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry
    ) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    df_spot = math.exp(-dividend_yield * time_to_expiry)
    df_strike = math.exp(-rate * time_to_expiry)
    return d1, d2, df_spot, df_strike, sqrt_t


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
) -> float:
    """European Black-Scholes-Merton premium.

    Parameters
    ----------
    spot, strike:
        Underlying price and strike. Both must be positive.
    time_to_expiry:
        Year fraction. At zero the premium is the intrinsic value.
    rate:
        Continuously compounded risk-free rate.
    volatility:
        Annualized volatility in decimal form (``0.20`` for 20 percent).
        At zero the premium is the discounted forward intrinsic value.
    dividend_yield:
        Continuous dividend yield. Use the foreign risk-free rate for an
        FX option, or set it equal to ``rate`` to recover Black-76.
    option_type:
        ``'call'`` or ``'put'`` (``'c'`` and ``'p'`` are accepted).
    """

    spot, strike, time_to_expiry, rate, volatility, dividend_yield, option_type = validate_contract(
        spot,
        strike,
        time_to_expiry,
        rate,
        volatility,
        dividend_yield,
        option_type,
    )
    if time_to_expiry == 0.0 or volatility == 0.0:
        return _discounted_intrinsic(
            spot, strike, time_to_expiry, rate, dividend_yield, option_type
        )
    d1, d2, df_spot, df_strike, _sqrt_t = _d1_d2(
        spot, strike, time_to_expiry, rate, volatility, dividend_yield
    )
    if option_type == "call":
        return float(spot * df_spot * norm.cdf(d1) - strike * df_strike * norm.cdf(d2))
    return float(strike * df_strike * norm.cdf(-d2) - spot * df_spot * norm.cdf(-d1))


def black76_price(
    futures: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
) -> float:
    """European Black-76 premium for an option on a futures price.

    The futures price is held fixed. Discounting uses ``rate``. This is
    Black-Scholes-Merton with the dividend yield equal to ``rate``.
    """

    try:
        return black_scholes_price(
            spot=futures,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=rate,
            volatility=volatility,
            dividend_yield=rate,
            option_type=option_type,
        )
    except ValueError as exc:
        raise ValueError(str(exc).replace("spot", "futures", 1)) from exc
