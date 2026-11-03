"""Implied Black-Scholes-Merton volatility.

The solver is a safeguarded Newton iteration. Vega supplies the Newton step
when that step stays inside a bracket with the correct sign change. Otherwise
the solver bisects the bracket. The price is monotone in volatility for a
European vanilla, so the bracket shrinks to a volatility that reprices the
target.
"""

from __future__ import annotations

import math

from opm.black_scholes import black_scholes_price
from opm.greeks import analytic_greeks
from opm.validation import as_finite_float, as_positive_int, validate_contract


class ImpliedVolatilityError(ValueError):
    """No Black-Scholes-Merton volatility reproduces the supplied premium."""


def _initial_guess(
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    dividend_yield: float,
) -> float:
    """Rough starting volatility, clipped to a modest range.

    The scale is the Brenner-Subrahmanyam ATM approximation, widened by
    log-moneyness so a far-from-the-money premium does not start at a few
    basis points of volatility.
    """

    scale = spot * math.exp(-dividend_yield * time_to_expiry)
    rough = math.sqrt(2.0 * math.pi / time_to_expiry) * max(price, 0.0) / scale
    moneyness = abs(math.log(spot / strike)) / math.sqrt(time_to_expiry)
    return min(max(rough, 0.5 * moneyness, 0.05), 2.0)


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
    initial_guess: float | None = None,
    price_tolerance: float = 1e-8,
    max_iterations: int = 100,
    vol_lower: float = 1e-8,
    vol_upper: float = 5.0,
) -> float:
    """Volatility that reprices a European vanilla under Black-Scholes-Merton.

    ``price`` is the undiscounted-currency premium of one option (the same
    units as :func:`opm.black_scholes.black_scholes_price`). The solve is
    rejected when the premium is outside the no-arbitrage band

        call:  [max(0, S e^{-qT} - K e^{-rT}),  S e^{-qT}]
        put:   [max(0, K e^{-rT} - S e^{-qT}),  K e^{-rT}]

    A premium that matches the zero-volatility value returns ``0``. Volatility
    above 20 (2,000 percent) is not searched. ``initial_guess`` is optional;
    a poor guess falls back to bisection inside the bracket.
    """

    premium = as_finite_float("price", price)
    if premium < 0.0:
        raise ImpliedVolatilityError("price must be non-negative")
    spot, strike, time_to_expiry, rate, _ignored, dividend_yield, option_type = validate_contract(
        spot,
        strike,
        time_to_expiry,
        rate,
        0.0,
        dividend_yield,
        option_type,
        positive_time=True,
    )
    tolerance = as_finite_float("price_tolerance", price_tolerance)
    if tolerance <= 0.0:
        raise ValueError("price_tolerance must be positive")
    iterations = as_positive_int("max_iterations", max_iterations, minimum=1)
    lower_vol = as_finite_float("vol_lower", vol_lower)
    upper_vol = as_finite_float("vol_upper", vol_upper)
    if not 0.0 < lower_vol < upper_vol:
        raise ValueError("volatility bounds must satisfy 0 < vol_lower < vol_upper")

    def model(volatility: float) -> float:
        return black_scholes_price(
            spot,
            strike,
            time_to_expiry,
            rate,
            volatility,
            dividend_yield,
            option_type,
        )

    lower_price = model(0.0)
    if option_type == "call":
        upper_price = spot * math.exp(-dividend_yield * time_to_expiry)
    else:
        upper_price = strike * math.exp(-rate * time_to_expiry)
    slack = max(tolerance, 1e-12)
    if premium < lower_price - slack:
        raise ImpliedVolatilityError(
            "price is below the no-arbitrage lower bound for this European contract"
        )
    if premium > upper_price + slack:
        raise ImpliedVolatilityError(
            "price is above the no-arbitrage upper bound for this European contract"
        )
    if premium <= lower_price + slack:
        return 0.0
    if premium >= upper_price - slack:
        raise ImpliedVolatilityError(
            "price is at the infinite-volatility limit, so implied volatility is undefined"
        )

    lo = lower_vol
    hi = upper_vol
    if model(hi) < premium:
        for _ in range(24):
            if hi >= 20.0:
                break
            hi = min(20.0, hi * 1.6)
            if model(hi) >= premium:
                break
    if model(hi) < premium - slack:
        raise ImpliedVolatilityError(
            "price is not attained at a volatility of 20 or below"
        )
    if model(lo) > premium:
        hi = lo
        lo = 0.0

    if initial_guess is None:
        sigma = _initial_guess(premium, spot, strike, time_to_expiry, dividend_yield)
    else:
        sigma = as_finite_float("initial_guess", initial_guess)
        if sigma < 0.0:
            raise ValueError("initial_guess must be non-negative")
    if sigma <= lo or sigma >= hi:
        sigma = 0.5 * (lo + hi)
    step_limit = hi - lo

    for _ in range(iterations):
        value = model(sigma)
        gap = value - premium
        if abs(gap) <= tolerance:
            return float(sigma)
        if gap < 0.0:
            lo = sigma
        else:
            hi = sigma
        width = hi - lo
        if width <= 1e-14 * max(1.0, hi):
            return float(0.5 * (lo + hi))

        accepted = False
        if sigma > 0.0:
            vega = analytic_greeks(
                spot,
                strike,
                time_to_expiry,
                rate,
                sigma,
                dividend_yield,
                option_type,
            ).vega
            if math.isfinite(vega) and vega > 1e-10:
                delta = gap / vega
                nxt = sigma - delta
                if lo < nxt < hi and abs(delta) <= step_limit:
                    sigma = nxt
                    step_limit = abs(delta)
                    accepted = True
        if not accepted:
            step_limit = 0.5 * width
            sigma = 0.5 * (lo + hi)

    value = model(sigma)
    if abs(value - premium) <= max(tolerance * 100.0, 1e-6):
        return float(sigma)
    raise ImpliedVolatilityError("implied volatility did not converge")
