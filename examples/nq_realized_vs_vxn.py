"""Price an at-the-money Nasdaq-100 futures option from two volatility inputs.

The realized input is estimated from the bundled 5-minute NQ futures bars.
The implied input is the CBOE Nasdaq-100 Volatility Index (VXN) close on the
last date in that bar file. Premiums use the European Black-76 model. Listed
NQ options are American and follow the exchange contract calendar; the figure
is a research comparison of two volatility numbers, not a quote.

Sources, both via Yahoo Finance and included under data/:

- NQ_F_5m.csv: Nasdaq-100 futures (Yahoo Finance NQ=F), 5-minute bars,
  24 August 2026 through 2 November 2026.
- VXN_daily.csv: CBOE Nasdaq-100 Volatility Index (Yahoo Finance ^VXN),
  daily closes, 3 November 2025 through 2 November 2026. VXN is published by
  CBOE and is quoted in annualized percentage points. It refers to the
  Nasdaq-100 cash index, not to the futures.

Run from a checkout, with matplotlib installed:

    python examples/nq_realized_vs_vxn.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from opm.black_scholes import black76_price  # noqa: E402
from opm.market import (  # noqa: E402
    annualized_volatility,
    five_minute_log_returns,
    load_daily_closes,
    load_intraday_ohlcv,
    rolling_close_to_close_volatility,
    trailing_close_to_close_volatility,
    weekday_closes_at,
)

# A full electronic session in this file is 23 hours. At a 5-minute stamp
# that is 276 intervals. Weekday close-to-close vol is scaled by the usual
# 252 sessions. Neither constant is an exchange settlement convention.
FULL_SESSION_INTERVALS = 23 * 12
TRADING_DAYS = 252
YEAR_DAYS = 365.25
RATE = 0.04
TENOR_DAYS = 30
CONTRACT_MULTIPLIER = 20.0  # USD per index point, E-mini Nasdaq-100


def build_report() -> dict[str, object]:
    """Load the bundled files and price the model at-the-money contract."""

    timestamps, closes = load_intraday_ohlcv(ROOT / "data" / "NQ_F_5m.csv")
    vxn_dates, vxn_points = load_daily_closes(ROOT / "data" / "VXN_daily.csv")
    intraday_returns = five_minute_log_returns(timestamps, closes)
    intraday_vol = annualized_volatility(
        intraday_returns, TRADING_DAYS * FULL_SESSION_INTERVALS
    )
    weekday_dates, weekday_prices = weekday_closes_at(timestamps, closes, hour=15, minute=55)
    close_to_close_vol = trailing_close_to_close_volatility(
        weekday_dates, weekday_prices, lookback_days=TENOR_DAYS, periods_per_year=TRADING_DAYS
    )
    roll_dates, roll_vol = rolling_close_to_close_volatility(
        weekday_dates, weekday_prices, window=21, periods_per_year=TRADING_DAYS
    )

    as_of = weekday_dates[-1]
    match = np.flatnonzero(vxn_dates == as_of)
    if match.size != 1:
        raise RuntimeError(f"VXN file has no close on {as_of}")
    vxn_quote = float(vxn_points[match[0]])
    if vxn_quote < 3.0:
        raise RuntimeError("expected VXN in percentage points")
    vxn_vol = vxn_quote / 100.0

    futures = float(weekday_prices[-1])
    last_close = float(closes[-1])
    # Model at-the-money: strike equals the futures proxy, not a listed strike.
    strike = futures
    time_to_expiry = TENOR_DAYS / YEAR_DAYS
    rows = []
    for name, volatility in (
        ("5-minute realized", intraday_vol),
        ("30-day close-to-close", close_to_close_vol),
        ("VXN close", vxn_vol),
    ):
        call = black76_price(futures, strike, time_to_expiry, RATE, volatility, "call")
        put = black76_price(futures, strike, time_to_expiry, RATE, volatility, "put")
        rows.append(
            {
                "name": name,
                "vol": volatility,
                "call": call,
                "put": put,
                "call_usd": call * CONTRACT_MULTIPLIER,
                "put_usd": put * CONTRACT_MULTIPLIER,
            }
        )

    position = np.searchsorted(vxn_dates, roll_dates)
    in_range = position < vxn_dates.size
    clipped = np.clip(position, 0, vxn_dates.size - 1)
    matched = in_range & (vxn_dates[clipped] == roll_dates)
    return {
        "bar_start": timestamps[0],
        "bar_end": timestamps[-1],
        "last_close": last_close,
        "n_bars": int(closes.size),
        "n_intraday_returns": int(intraday_returns.size),
        "n_weekday_closes": int(weekday_dates.size),
        "as_of": as_of,
        "futures": futures,
        "strike": strike,
        "rate": RATE,
        "time_to_expiry": time_to_expiry,
        "vxn_points": vxn_quote,
        "rows": rows,
        "roll_dates": roll_dates[matched],
        "roll_vol": roll_vol[matched],
        "vxn_on_roll": vxn_points[clipped][matched],
    }


def _wall_clock(value: object) -> str:
    return str(np.datetime64(value, "s")).replace("T", " ")


def format_report(report: dict[str, object]) -> str:
    """Plain-text comparison. Premiums are index points unless marked USD."""

    lines = [
        "NQ futures volatility and a 30-day at-the-money Black-76 option",
        "",
        "Futures: data/NQ_F_5m.csv",
        "  Nasdaq-100 futures, 5-minute bars, Yahoo Finance (NQ=F).",
        f"  {_wall_clock(report['bar_start'])} through {_wall_clock(report['bar_end'])} (exchange-local wall time).",
        f"  bars: {report['n_bars']}; exact 5-minute returns: {report['n_intraday_returns']}.",
        "Volatility index: data/VXN_daily.csv",
        "  CBOE Nasdaq-100 Volatility Index, daily close, Yahoo Finance (^VXN).",
        f"  VXN on {report['as_of']}: {report['vxn_points']:.4f} percentage points.",
        "",
        "Realized volatility",
        "  5-minute: sample stdev of exact 5-minute log returns, times",
        f"  sqrt({TRADING_DAYS} * {FULL_SESSION_INTERVALS}). Gaps longer than 5 minutes",
        "  (the daily halt and the weekend) are omitted, so this number does not",
        "  include those jumps.",
        "  Close-to-close: last 15:55 America/New_York bar on each weekday, which is",
        "  the five-minute bar ending at the cash close if stamps are interval starts.",
        f"  The headline window is the last {TENOR_DAYS} calendar days, annualized with sqrt({TRADING_DAYS}).",
        f"  Weekday closes in the file: {report['n_weekday_closes']}.",
        "",
        "Model",
        "  European Black-76. The futures price is held fixed and the premium is",
        f"  discounted at r = {report['rate']:.2f}. Tenor is {TENOR_DAYS}/{YEAR_DAYS:.2f} years.",
        f"  Underlying: {report['futures']:.2f}, the 15:55 America/New_York close on {report['as_of']}.",
        f"  The last bar in the file is {report['last_close']:.2f} at {_wall_clock(report['bar_end'])}.",
        "  Strike equals that 15:55 close. This is model at-the-money, not a listed strike.",
        "  The rate is an illustrative input, not a market quote.",
        "  Listed NQ options are American. USD uses the $20 E-mini multiplier.",
        "  Call and put premiums match: Black-76 put-call parity is e^(-rT)(F-K),",
        "  which is zero when the strike equals the futures.",
        "",
        f"{'source':<24} {'vol':>8} {'call_pts':>10} {'put_pts':>10} {'call_usd':>12} {'put_usd':>12}",
    ]
    for row in report["rows"]:  # type: ignore[union-attr]
        lines.append(
            f"{row['name']:<24} {row['vol']:8.4f} {row['call']:10.2f} {row['put']:10.2f}"
            f" {row['call_usd']:12,.0f} {row['put_usd']:12,.0f}"
        )
    lines.extend(
        [
            "",
            "VXN is a 30-day implied volatility of the Nasdaq-100 cash index.",
            "It is not an NQ futures-option implied volatility. The comparison",
            "shows how two public volatility inputs change a Black-76 premium.",
            "It is not a forecast and it is not a recommendation to trade.",
        ]
    )
    return "\n".join(lines)


def plot_report(report: dict[str, object], output: Path) -> None:
    """Save the realized-vol history and the premium comparison."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = [str(day) for day in report["roll_dates"]]  # type: ignore[union-attr]
    from datetime import datetime

    axis_dates = [datetime.strptime(day, "%Y-%m-%d") for day in dates]
    roll_percent = np.asarray(report["roll_vol"], dtype=float) * 100.0
    vxn_percent = np.asarray(report["vxn_on_roll"], dtype=float)
    rows = report["rows"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    axes[0].plot(axis_dates, roll_percent, color="#1f4e79", linewidth=1.6, label="21-session close-to-close")
    axes[0].plot(axis_dates, vxn_percent, color="#b85c38", linewidth=1.6, label="VXN")
    axes[0].set_title("Realized volatility and VXN")
    axes[0].set_ylabel("Annualized volatility (%)")
    axes[0].legend(frameon=False)
    axes[0].grid(True, axis="y", linewidth=0.4, alpha=0.6)
    axes[0].tick_params(axis="x", labelrotation=30)

    labels = [row["name"] for row in rows]  # type: ignore[index, union-attr]
    calls = [row["call"] for row in rows]  # type: ignore[index, union-attr]
    puts = [row["put"] for row in rows]  # type: ignore[index, union-attr]
    positions = np.arange(len(labels))
    width = 0.36
    axes[1].bar(positions - width / 2, calls, width, color="#1f4e79", label="Call")
    axes[1].bar(positions + width / 2, puts, width, color="#7a8b99", label="Put")
    axes[1].set_xticks(positions, labels, rotation=15, ha="right")
    axes[1].set_ylabel("Premium (index points)")
    axes[1].set_title("30-day at-the-money Black-76")
    axes[1].legend(frameon=False)
    axes[1].grid(True, axis="y", linewidth=0.4, alpha=0.6)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.text(
        0.01,
        0.012,
        "NQ futures 5-minute bars and CBOE VXN via Yahoo Finance. "
        "European Black-76, r = 4%, 30-calendar-day at-the-money. Not a quote.",
        fontsize=8,
        color="#444444",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare NQ realized volatility with VXN.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "examples" / "output" / "nq_realized_vs_vxn.png",
        help="where to write the figure",
    )
    args = parser.parse_args(argv)
    report = build_report()
    print(format_report(report))
    try:
        plot_report(report, args.output)
    except ModuleNotFoundError:
        print("\nmatplotlib is not installed; the figure was not written.")
        return 0
    print(f"\nFigure: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
