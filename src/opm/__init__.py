"""Educational options pricing.

European Black-Scholes-Merton and Black-76 prices, a Cox-Ross-Rubinstein
tree for European and American exercise, European Monte Carlo with antithetic
variates, analytic and finite-difference Greeks, an implied-volatility solver,
and a dealer gamma-exposure summary.

This package is for study and research. It is not a trading system and it
does not produce financial advice.
"""

from opm.binomial import binomial_price
from opm.black_scholes import black76_price, black_scholes_price
from opm.gex import ChainRow, GexResult, net_gex_by_strike, read_chain_csv
from opm.greeks import Greeks, analytic_greeks, black76_greeks, numeric_greeks
from opm.implied_vol import ImpliedVolatilityError, implied_volatility
from opm.monte_carlo import MonteCarloResult, monte_carlo_price

__version__ = "0.1.0"

__all__ = [
    "ChainRow",
    "GexResult",
    "Greeks",
    "ImpliedVolatilityError",
    "MonteCarloResult",
    "analytic_greeks",
    "binomial_price",
    "black76_greeks",
    "black76_price",
    "black_scholes_price",
    "implied_volatility",
    "monte_carlo_price",
    "net_gex_by_strike",
    "numeric_greeks",
    "read_chain_csv",
]
