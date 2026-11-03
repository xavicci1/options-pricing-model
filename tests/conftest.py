"""Shared contract used by the pricing tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def atm() -> dict[str, float]:
    return {
        "spot": 100.0,
        "strike": 100.0,
        "time_to_expiry": 1.0,
        "rate": 0.05,
        "volatility": 0.20,
        "dividend_yield": 0.02,
    }
