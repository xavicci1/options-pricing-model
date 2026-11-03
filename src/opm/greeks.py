"""Analytic and finite-difference Greeks for vanilla options."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from scipy.stats import norm

from opm.black_scholes import _d1_d2, black76_price, black_scholes_price
from opm.validation import as_finite_float, validate_contract


@dataclass(frozen=True)
class Greeks:
    """Partial derivatives of a vanilla premium.

    ``delta`` is the change in premium per unit change in the spot (or in the
    futures price, for a Black-76 contract).

    ``gamma`` is the change in delta per unit change in the spot.

    ``vega`` is the derivative with respect to volatility in decimal form.
    A move of one volatility point (0.01) changes the premium by about
    ``0.01 * vega``.

    ``theta`` is the derivative with respect to calendar time, in premium per
    year: as the clock advances and time-to-expiry falls, the premium changes
    at this rate. Divide by 365 for an approximate calendar-day figure.

    ``rho`` is the derivative with respect to the continuously compounded
    risk-free rate, holding the dividend yield fixed. For :func:`black76_greeks`
    the futures price is held fixed instead, and rho equals ``-T`` times the
    premium. A one basis-point move is about ``0.0001 * rho``.
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def as_dict(self) -> dict[str, float]:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }


def analytic_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
) -> Greeks:
    """Closed-form Black-Scholes-Merton Greeks.

    Requires a positive time to expiry and a positive volatility. At those
    boundaries gamma and theta are discontinuous, so the finite-difference
    helper is a better description of a bumped price.
    """

    spot, strike, time_to_expiry, rate, volatility, dividend_yield, option_type = validate_contract(
        spot,
        strike,
        time_to_expiry,
        rate,
        volatility,
        dividend_yield,
        option_type,
        positive_volatility=True,
        positive_time=True,
    )
    d1, d2, df_spot, df_strike, sqrt_t = _d1_d2(
        spot, strike, time_to_expiry, rate, volatility, dividend_yield
    )
    pdf_d1 = float(norm.pdf(d1))
    vol_sqrt_t = volatility * sqrt_t
    gamma = df_spot * pdf_d1 / (spot * vol_sqrt_t)
    vega = spot * df_spot * pdf_d1 * sqrt_t
    if option_type == "call":
        cdf_d1 = float(norm.cdf(d1))
        cdf_d2 = float(norm.cdf(d2))
        delta = df_spot * cdf_d1
        theta = (
            -spot * df_spot * pdf_d1 * volatility / (2.0 * sqrt_t)
            + dividend_yield * spot * df_spot * cdf_d1
            - rate * strike * df_strike * cdf_d2
        )
        rho = strike * time_to_expiry * df_strike * cdf_d2
    else:
        cdf_neg_d1 = float(norm.cdf(-d1))
        cdf_neg_d2 = float(norm.cdf(-d2))
        delta = -df_spot * cdf_neg_d1
        theta = (
            -spot * df_spot * pdf_d1 * volatility / (2.0 * sqrt_t)
            - dividend_yield * spot * df_spot * cdf_neg_d1
            + rate * strike * df_strike * cdf_neg_d2
        )
        rho = -strike * time_to_expiry * df_strike * cdf_neg_d2
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def black76_greeks(
    futures: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
) -> Greeks:
    """Greeks of a Black-76 futures option, holding the futures price fixed.

    Delta, gamma, vega, and theta match Black-Scholes-Merton with the dividend
    yield equal to ``rate``. Rho is ``-time_to_expiry * premium``, because a
    futures price does not change when the discount rate is bumped.
    """

    try:
        partial = analytic_greeks(
            spot=futures,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=rate,
            volatility=volatility,
            dividend_yield=rate,
            option_type=option_type,
        )
        premium = black76_price(
            futures, strike, time_to_expiry, rate, volatility, option_type
        )
    except ValueError as exc:
        raise ValueError(str(exc).replace("spot", "futures", 1)) from exc
    return Greeks(
        delta=partial.delta,
        gamma=partial.gamma,
        vega=partial.vega,
        theta=partial.theta,
        rho=-time_to_expiry * premium,
    )


def numeric_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
    pricer: Callable[..., float] | None = None,
    spot_bump: float | None = None,
    vol_bump: float = 1e-4,
    time_bump: float = 1e-4,
    rate_bump: float = 1e-4,
    **pricer_kwargs: object,
) -> Greeks:
    """Central-difference Greeks for a European or tree pricer.

    The default pricer is :func:`opm.black_scholes.black_scholes_price`.
    Pass ``pricer=binomial_price`` and ``steps=...`` to bump a tree. The
    pricer must accept ``spot``, ``strike``, ``time_to_expiry``, ``rate``,
    ``volatility``, ``dividend_yield``, and ``option_type`` as keywords.

    Theta is returned per year and uses the sign convention of
    :func:`analytic_greeks`: ``-(V(τ+h) - V(τ-h)) / (2h)``.
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
    if pricer is None:
        pricer = black_scholes_price

    def value(**overrides: float) -> float:
        params: dict[str, object] = {
            "spot": spot,
            "strike": strike,
            "time_to_expiry": time_to_expiry,
            "rate": rate,
            "volatility": volatility,
            "dividend_yield": dividend_yield,
            "option_type": option_type,
        }
        params.update(pricer_kwargs)
        params.update(overrides)
        return float(pricer(**params))

    h_s = spot * 1e-4 if spot_bump is None else spot_bump
    h_s = as_finite_float("spot_bump", h_s)
    if h_s <= 0.0:
        raise ValueError("spot_bump must be positive")
    if h_s >= spot:
        raise ValueError("spot_bump must be smaller than the spot")
    up = value(spot=spot + h_s)
    down = value(spot=spot - h_s)
    mid = value()
    delta = (up - down) / (2.0 * h_s)
    gamma = (up - 2.0 * mid + down) / (h_s * h_s)

    h_v = as_finite_float("vol_bump", vol_bump)
    if h_v <= 0.0:
        raise ValueError("vol_bump must be positive")
    if volatility > h_v:
        vega = (value(volatility=volatility + h_v) - value(volatility=volatility - h_v)) / (
            2.0 * h_v
        )
    elif volatility > 0.0:
        step = 0.5 * volatility
        vega = (
            value(volatility=volatility + step) - value(volatility=volatility - step)
        ) / (2.0 * step)
    else:
        vega = (value(volatility=h_v) - mid) / h_v

    h_r = as_finite_float("rate_bump", rate_bump)
    if h_r <= 0.0:
        raise ValueError("rate_bump must be positive")
    rho = (value(rate=rate + h_r) - value(rate=rate - h_r)) / (2.0 * h_r)

    h_t = as_finite_float("time_bump", time_bump)
    if h_t <= 0.0:
        raise ValueError("time_bump must be positive")
    if time_to_expiry == 0.0:
        theta = 0.0
    elif time_to_expiry > h_t:
        theta = -(
            value(time_to_expiry=time_to_expiry + h_t)
            - value(time_to_expiry=time_to_expiry - h_t)
        ) / (2.0 * h_t)
    else:
        theta = -(mid - value(time_to_expiry=0.0)) / time_to_expiry

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
