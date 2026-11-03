"""Cox-Ross-Rubinstein binomial tree for European and American vanillas."""

from __future__ import annotations

import math

import numpy as np

from opm.black_scholes import _discounted_intrinsic
from opm.validation import as_positive_int, normalize_exercise, validate_contract


def _zero_volatility_value(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    option_type: str,
    steps: int,
    exercise: str,
) -> float:
    """American or European value along a deterministic forward path."""

    if exercise == "european":
        return _discounted_intrinsic(
            spot, strike, time_to_expiry, rate, dividend_yield, option_type
        )
    times = np.linspace(0.0, time_to_expiry, steps + 1)
    spots = spot * np.exp((rate - dividend_yield) * times)
    if option_type == "call":
        intrinsic = np.maximum(spots - strike, 0.0)
    else:
        intrinsic = np.maximum(strike - spots, 0.0)
    return float(np.max(np.exp(-rate * times) * intrinsic))


def binomial_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
    steps: int = 200,
    exercise: str = "european",
) -> float:
    """Cox-Ross-Rubinstein premium.

    The up and down moves are ``exp(σ √Δt)`` and ``exp(-σ √Δt)``. The
    risk-neutral probability uses the continuous carry ``rate - dividend_yield``.
    American exercise replaces the continuation value with the intrinsic value
    when that is larger. European exercise does not.

    The tree needs enough steps for the up/down spread to cover the carry.
    If the risk-neutral probability would fall outside ``[0, 1]``, the function
    raises ``ValueError`` rather than returning a number from an arbitrage
    spread it cannot represent.

    Even step counts are usually a little closer to the Black-Scholes value
    than the neighbouring odd count. A few hundred steps are enough for a
    price; convergence is oscillatory and of order ``1 / steps``.
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
    step_count = as_positive_int("steps", steps, minimum=1)
    style = normalize_exercise(exercise)
    if time_to_expiry == 0.0:
        return _discounted_intrinsic(
            spot, strike, 0.0, rate, dividend_yield, option_type
        )
    if volatility == 0.0:
        return _zero_volatility_value(
            spot,
            strike,
            time_to_expiry,
            rate,
            dividend_yield,
            option_type,
            step_count,
            style,
        )

    dt = time_to_expiry / step_count
    log_u = volatility * math.sqrt(dt)
    if log_u > 700.0:
        raise ValueError("volatility is too large for this number of steps")
    growth = math.exp((rate - dividend_yield) * dt)
    down = math.exp(-log_u)
    up = math.exp(log_u)
    if growth < down or growth > up:
        raise ValueError(
            "risk-neutral probability is outside [0, 1]; increase steps so the "
            "tree can span the cost of carry"
        )
    probability = (growth - down) / (up - down)
    discount = math.exp(-rate * dt)

    nodes = np.arange(step_count + 1, dtype=np.float64)
    log_spot = np.log(spot) + (2.0 * nodes - step_count) * log_u
    np.clip(log_spot, -700.0, 700.0, out=log_spot)
    terminal_spot = np.exp(log_spot)
    if option_type == "call":
        values = np.maximum(terminal_spot - strike, 0.0)
    else:
        values = np.maximum(strike - terminal_spot, 0.0)

    stay = 1.0 - probability
    for step in range(step_count - 1, -1, -1):
        values = discount * (probability * values[1:] + stay * values[:-1])
        if style == "american":
            level = np.arange(step + 1, dtype=np.float64)
            log_level = np.log(spot) + (2.0 * level - step) * log_u
            np.clip(log_level, -700.0, 700.0, out=log_level)
            level_spot = np.exp(log_level)
            if option_type == "call":
                exercise_value = np.maximum(level_spot - strike, 0.0)
            else:
                exercise_value = np.maximum(strike - level_spot, 0.0)
            np.maximum(values, exercise_value, out=values)
    return float(values[0])
