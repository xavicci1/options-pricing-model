"""Tree prices against Black-Scholes-Merton and early exercise."""

from __future__ import annotations

import math

import pytest

from opm import binomial_price, black_scholes_price


def test_european_tree_converges_to_black_scholes(atm: dict[str, float]) -> None:
    params = {**atm, "option_type": "call"}
    closed = black_scholes_price(**params)
    coarse = binomial_price(**params, steps=200, exercise="european")
    fine = binomial_price(**params, steps=1600, exercise="european")
    assert abs(fine - closed) < 0.02
    assert abs(fine - closed) < abs(coarse - closed)


def test_european_put_tree_is_close(atm: dict[str, float]) -> None:
    params = {**atm, "option_type": "put"}
    closed = black_scholes_price(**params)
    tree = binomial_price(**params, steps=800, exercise="european")
    assert tree == pytest.approx(closed, abs=0.03)


def test_european_tree_put_call_parity(atm: dict[str, float]) -> None:
    call = binomial_price(**atm, option_type="call", steps=500, exercise="european")
    put = binomial_price(**atm, option_type="put", steps=500, exercise="european")
    rhs = atm["spot"] * math.exp(-atm["dividend_yield"] * atm["time_to_expiry"])
    rhs -= atm["strike"] * math.exp(-atm["rate"] * atm["time_to_expiry"])
    assert call - put == pytest.approx(rhs, abs=0.05)


def test_american_call_without_dividends_matches_european() -> None:
    params = dict(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        rate=0.05,
        volatility=0.20,
        dividend_yield=0.0,
        option_type="call",
        steps=300,
    )
    european = binomial_price(**params, exercise="european")
    american = binomial_price(**params, exercise="american")
    assert american == pytest.approx(european, abs=1e-8)
    assert american + 1e-10 >= european


def test_american_put_is_worth_more_than_european() -> None:
    params = dict(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        rate=0.08,
        volatility=0.30,
        dividend_yield=0.0,
        option_type="put",
        steps=400,
    )
    european = binomial_price(**params, exercise="european")
    american = binomial_price(**params, exercise="american")
    assert american > european + 0.05


def test_american_call_with_dividend_can_be_exercised_early() -> None:
    params = dict(
        spot=100.0,
        strike=70.0,
        time_to_expiry=1.0,
        rate=0.01,
        volatility=0.25,
        dividend_yield=0.10,
        option_type="call",
        steps=500,
    )
    european = binomial_price(**params, exercise="european")
    american = binomial_price(**params, exercise="american")
    assert american + 1e-8 >= european
    assert american > european + 0.05


def test_zero_volatility_american_put_exercises_immediately() -> None:
    european = binomial_price(
        80.0, 100.0, 1.0, 0.08, 0.0, 0.0, "put", steps=50, exercise="european"
    )
    american = binomial_price(
        80.0, 100.0, 1.0, 0.08, 0.0, 0.0, "put", steps=50, exercise="american"
    )
    forward = 80.0 * math.exp(0.08)
    assert european == pytest.approx(math.exp(-0.08) * (100.0 - forward))
    assert american == pytest.approx(20.0)
    assert american > european


def test_tree_rejects_a_carry_it_cannot_span() -> None:
    with pytest.raises(ValueError, match="risk-neutral"):
        binomial_price(
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            rate=0.50,
            volatility=0.01,
            steps=1,
            option_type="call",
        )
