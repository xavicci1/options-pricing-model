"""European Monte Carlo against the closed form."""

from __future__ import annotations

import pytest

from opm import black_scholes_price, monte_carlo_price


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_antithetic_price_is_within_sampling_error(atm: dict[str, float], option_type: str) -> None:
    params = {**atm, "option_type": option_type}
    closed = black_scholes_price(**params)
    result = monte_carlo_price(**params, n_paths=120_000, antithetic=True, seed=11)
    assert abs(result.price - closed) < 5.0 * result.standard_error
    assert result.price == pytest.approx(closed, abs=0.05)
    assert result.n_paths == 120_000


def test_antithetic_variates_reduce_the_standard_error(atm: dict[str, float]) -> None:
    params = {**atm, "option_type": "call"}
    plain = monte_carlo_price(**params, n_paths=40_000, antithetic=False, seed=19)
    paired = monte_carlo_price(**params, n_paths=40_000, antithetic=True, seed=19)
    assert paired.standard_error < plain.standard_error
    assert paired.n_paths == 40_000


def test_seed_repeats() -> None:
    kwargs = dict(
        spot=100.0,
        strike=110.0,
        time_to_expiry=0.4,
        rate=0.01,
        volatility=0.35,
        dividend_yield=0.0,
        option_type="put",
        n_paths=8_000,
        seed=7,
    )
    first = monte_carlo_price(**kwargs)
    second = monte_carlo_price(**kwargs)
    assert first.price == second.price
    assert first.standard_error == second.standard_error


def test_odd_path_count_is_rounded_down_for_antithetic_pairs(atm: dict[str, float]) -> None:
    result = monte_carlo_price(**atm, option_type="call", n_paths=1001, antithetic=True, seed=1)
    assert result.n_paths == 1000
