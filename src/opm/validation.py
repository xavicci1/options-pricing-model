"""Shared checks for pricing inputs."""

from __future__ import annotations

import math


def as_finite_float(name: str, value: object) -> float:
    """Return ``value`` as a finite float.

    Booleans are rejected so that ``True`` is not accepted as ``1``.
    """

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def as_positive_int(name: str, value: object, minimum: int = 1) -> int:
    """Return ``value`` as an int greater than or equal to ``minimum``."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def normalize_option_type(option_type: object) -> str:
    """Return ``'call'`` or ``'put'``."""

    if not isinstance(option_type, str):
        raise ValueError("option_type must be 'call' or 'put'")
    label = option_type.strip().lower()
    if label in {"c", "call"}:
        return "call"
    if label in {"p", "put"}:
        return "put"
    raise ValueError("option_type must be 'call' or 'put'")


def normalize_exercise(exercise: object) -> str:
    """Return ``'european'`` or ``'american'``."""

    if not isinstance(exercise, str):
        raise ValueError("exercise must be 'european' or 'american'")
    label = exercise.strip().lower()
    if label not in {"european", "american"}:
        raise ValueError("exercise must be 'european' or 'american'")
    return label


def validate_contract(
    spot: object,
    strike: object,
    time_to_expiry: object,
    rate: object,
    volatility: object,
    dividend_yield: object,
    option_type: object,
    *,
    positive_volatility: bool = False,
    positive_time: bool = False,
) -> tuple[float, float, float, float, float, float, str]:
    """Validate a vanilla contract and return canonical numeric inputs."""

    spot_f = as_finite_float("spot", spot)
    strike_f = as_finite_float("strike", strike)
    time_f = as_finite_float("time_to_expiry", time_to_expiry)
    rate_f = as_finite_float("rate", rate)
    vol_f = as_finite_float("volatility", volatility)
    div_f = as_finite_float("dividend_yield", dividend_yield)
    kind = normalize_option_type(option_type)
    if spot_f <= 0.0:
        raise ValueError("spot must be positive")
    if strike_f <= 0.0:
        raise ValueError("strike must be positive")
    if time_f < 0.0:
        raise ValueError("time_to_expiry must be non-negative")
    if vol_f < 0.0:
        raise ValueError("volatility must be non-negative")
    if positive_time and time_f <= 0.0:
        raise ValueError("time_to_expiry must be positive")
    if positive_volatility and vol_f <= 0.0:
        raise ValueError("volatility must be positive")
    return spot_f, strike_f, time_f, rate_f, vol_f, div_f, kind
