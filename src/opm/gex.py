"""Dealer gamma exposure aggregated by strike.

The figure reported here is the change in dollar delta for a one percent
increase in the spot:

    sign * gamma * open_interest * multiplier * spot^2 * 0.01

Gamma is the Black-Scholes-Merton gamma. Calls are signed positive and puts
negative, which is the usual dealer-inventory assumption that the dealer is
long call gamma and short put gamma. It is an assumption about the book, not
an observed position. Negate the result for the opposite assumption.

``multiplier`` is the number of underlying units per contract: 100 for a
standard US equity option, 20 for the E-mini Nasdaq-100 futures.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opm.greeks import analytic_greeks
from opm.validation import as_finite_float, normalize_option_type


@dataclass(frozen=True)
class ChainRow:
    """One option contract in a chain snapshot."""

    strike: float
    open_interest: float
    implied_vol: float
    option_type: str


@dataclass(frozen=True)
class GexResult:
    """Net dealer gamma exposure summed onto each distinct strike."""

    strikes: np.ndarray
    net_gex: np.ndarray
    total: float
    spot: float
    multiplier: float

    def __str__(self) -> str:
        lines = [
            "net dealer GEX by strike (dollar-delta change for a +1% spot move)",
            f"{'strike':>14}  {'net_gex':>18}",
        ]
        for strike, exposure in zip(self.strikes, self.net_gex):
            lines.append(f"{strike:14.4f}  {exposure:18,.2f}")
        lines.append(f"{'total':>14}  {self.total:18,.2f}")
        return "\n".join(lines)


def chain_row_from_mapping(row: Mapping[str, object]) -> ChainRow:
    """Build a :class:`ChainRow` from a header-keyed record.

    Accepted names, case-insensitive: strike (or k); oi, open_interest;
    iv, implied_vol, volatility, vol; type, option_type, cp, right.
    Implied volatility is a decimal, not a percent.
    """

    lowered = {str(key).strip().lower(): value for key, value in row.items()}

    def pick(*keys: str) -> object:
        for key in keys:
            if key in lowered and lowered[key] not in (None, ""):
                return lowered[key]
        raise ValueError(f"missing one of {', '.join(keys)}")

    return ChainRow(
        strike=float(pick("strike", "k")),  # type: ignore[arg-type]
        open_interest=float(pick("oi", "open_interest", "openinterest")),  # type: ignore[arg-type]
        implied_vol=float(pick("iv", "implied_vol", "implied_volatility", "vol", "volatility")),  # type: ignore[arg-type]
        option_type=normalize_option_type(str(pick("type", "option_type", "cp", "right"))),
    )


def read_chain_csv(path: str | Path) -> list[ChainRow]:
    """Read a chain CSV with a header row. See :func:`chain_row_from_mapping`."""

    file_path = Path(path)
    with file_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{file_path.name} has no header")
        rows: list[ChainRow] = []
        for line_number, raw in enumerate(reader, start=2):
            try:
                rows.append(chain_row_from_mapping(raw))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{file_path.name}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"{file_path.name} has no contracts")
    return rows


def net_gex_by_strike(
    chain: Sequence[ChainRow | Mapping[str, object]],
    spot: float,
    time_to_expiry: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    multiplier: float = 100.0,
) -> GexResult:
    """Net dealer GEX at each strike.

    Strikes that round to the same 8 decimal places are aggregated. A contract
    with zero open interest or zero implied volatility contributes nothing:
    zero volatility puts a Dirac mass of gamma on the strike, which this
    helper does not spread into the bucket.
    """

    spot_f = as_finite_float("spot", spot)
    time_f = as_finite_float("time_to_expiry", time_to_expiry)
    rate_f = as_finite_float("rate", rate)
    div_f = as_finite_float("dividend_yield", dividend_yield)
    mult = as_finite_float("multiplier", multiplier)
    if spot_f <= 0.0:
        raise ValueError("spot must be positive")
    if time_f <= 0.0:
        raise ValueError("time_to_expiry must be positive")
    if mult <= 0.0:
        raise ValueError("multiplier must be positive")

    totals: dict[float, float] = {}
    labels: dict[float, float] = {}
    for index, item in enumerate(chain):
        row = item if isinstance(item, ChainRow) else chain_row_from_mapping(item)
        strike = as_finite_float("strike", row.strike)
        open_interest = as_finite_float("open_interest", row.open_interest)
        implied_vol = as_finite_float("implied_vol", row.implied_vol)
        option_type = normalize_option_type(row.option_type)
        if strike <= 0.0:
            raise ValueError(f"row {index}: strike must be positive")
        if open_interest < 0.0:
            raise ValueError(f"row {index}: open_interest must be non-negative")
        if implied_vol < 0.0:
            raise ValueError(f"row {index}: implied_vol must be non-negative")
        key = round(strike, 8)
        labels.setdefault(key, strike)
        totals.setdefault(key, 0.0)
        if open_interest == 0.0 or implied_vol == 0.0:
            continue
        gamma = analytic_greeks(
            spot_f,
            strike,
            time_f,
            rate_f,
            implied_vol,
            div_f,
            option_type,
        ).gamma
        sign = 1.0 if option_type == "call" else -1.0
        totals[key] += sign * gamma * open_interest * mult * spot_f * spot_f * 0.01

    keys = sorted(totals)
    strikes = np.array([labels[key] for key in keys], dtype=np.float64)
    net = np.array([totals[key] for key in keys], dtype=np.float64)
    total = float(net.sum()) if net.size else 0.0
    return GexResult(
        strikes=strikes,
        net_gex=net,
        total=total,
        spot=spot_f,
        multiplier=mult,
    )
