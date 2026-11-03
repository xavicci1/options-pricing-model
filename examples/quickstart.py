"""Small walk through the public pricing API.

Run from a checkout:

    python examples/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opm import (  # noqa: E402
    analytic_greeks,
    binomial_price,
    black_scholes_price,
    implied_volatility,
    monte_carlo_price,
    net_gex_by_strike,
)


def main() -> None:
    contract = dict(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        rate=0.05,
        volatility=0.20,
        dividend_yield=0.02,
    )
    call = black_scholes_price(**contract, option_type="call")
    put = black_scholes_price(**contract, option_type="put")
    greeks = analytic_greeks(**contract, option_type="call")
    recovered = implied_volatility(
        call,
        contract["spot"],
        contract["strike"],
        contract["time_to_expiry"],
        contract["rate"],
        contract["dividend_yield"],
        "call",
    )
    tree = binomial_price(**contract, option_type="put", steps=400, exercise="american")
    simulated = monte_carlo_price(**contract, option_type="call", n_paths=50_000, seed=1)
    exposure = net_gex_by_strike(
        [
            {"strike": 100, "open_interest": 250, "iv": 0.20, "type": "call"},
            {"strike": 100, "open_interest": 400, "iv": 0.22, "type": "put"},
        ],
        spot=100.0,
        time_to_expiry=0.25,
        rate=0.05,
        dividend_yield=0.02,
        multiplier=100.0,
    )

    print("Black-Scholes-Merton, S=K=100, T=1, r=0.05, q=0.02, vol=0.20")
    print(f"  call                 {call:.6f}")
    print(f"  put                  {put:.6f}")
    print(f"  implied vol of call  {recovered:.6f}")
    print(f"  call delta           {greeks.delta:.6f}")
    print(f"  call gamma           {greeks.gamma:.6f}")
    print(f"  call vega            {greeks.vega:.6f}")
    print(f"  call theta (per year) {greeks.theta:.6f}")
    print(f"  call rho             {greeks.rho:.6f}")
    print(f"  american put, 400-step CRR  {tree:.6f}")
    print(
        f"  monte carlo call     {simulated.price:.6f}"
        f"  (stderr {simulated.standard_error:.6f}, antithetic)"
    )
    print()
    print(exposure)


if __name__ == "__main__":
    main()
