"""European Monte Carlo prices with optional antithetic variates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from opm.validation import as_positive_int, validate_contract


@dataclass(frozen=True)
class MonteCarloResult:
    """Discounted Monte Carlo premium and its standard error.

    With antithetic variates, ``n_paths`` counts both members of each pair
    and ``standard_error`` is computed from the paired averages, which is
    the standard error of the estimator that was actually used.
    """

    price: float
    standard_error: float
    n_paths: int
    antithetic: bool


def monte_carlo_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
    n_paths: int = 100_000,
    antithetic: bool = True,
    seed: int | None = None,
) -> MonteCarloResult:
    """European premium by simulating the terminal spot under the risk-neutral measure.

    The spot follows geometric Brownian motion

        log S_T = log S_0 + (r - q - σ²/2) T + σ √T Z,   Z ~ N(0, 1)

    and the payoff is discounted at the continuous rate ``r``. The scheme
    prices European exercise only. American contracts should use the binomial
    tree.

    When ``antithetic`` is true, each draw ``Z`` is paired with ``-Z`` and the
    two payoffs are averaged before the sample mean is taken. ``n_paths`` is
    rounded down to an even number. ``seed`` initializes a NumPy Generator so
    a run can be repeated.
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
    paths = as_positive_int("n_paths", n_paths, minimum=2)
    if seed is not None:
        as_positive_int("seed", seed, minimum=0)
    rng = np.random.default_rng(seed)

    if antithetic:
        n_pairs = paths // 2
        if n_pairs < 1:
            raise ValueError("n_paths must be at least 2 when antithetic variates are used")
        normals = rng.standard_normal(n_pairs)
        shocks = np.concatenate([normals, -normals])
        used = n_pairs * 2
    else:
        shocks = rng.standard_normal(paths)
        used = paths

    drift = (rate - dividend_yield - 0.5 * volatility * volatility) * time_to_expiry
    diffusion = volatility * math.sqrt(time_to_expiry)
    terminal = spot * np.exp(drift + diffusion * shocks)
    if option_type == "call":
        payoff = np.maximum(terminal - strike, 0.0)
    else:
        payoff = np.maximum(strike - terminal, 0.0)

    if antithetic:
        paired = 0.5 * (payoff[: used // 2] + payoff[used // 2 :])
        sample = paired
    else:
        sample = payoff

    discount = math.exp(-rate * time_to_expiry)
    premium = discount * float(np.mean(sample))
    error = discount * float(np.std(sample, ddof=1) / math.sqrt(sample.size))
    return MonteCarloResult(
        price=premium,
        standard_error=error,
        n_paths=used,
        antithetic=antithetic,
    )
