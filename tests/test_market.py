"""Bar loaders and realized-volatility windows."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from opm.market import (
    annualized_volatility,
    five_minute_log_returns,
    load_daily_closes,
    load_intraday_ohlcv,
    rolling_close_to_close_volatility,
    trailing_close_to_close_volatility,
    weekday_closes_at,
)

ROOT = Path(__file__).resolve().parents[1]


def test_five_minute_returns_drop_longer_gaps() -> None:
    timestamps = np.array(
        [
            "2026-08-24T00:00:00",
            "2026-08-24T00:05:00",
            "2026-08-24T00:15:00",
        ],
        dtype="datetime64[ns]",
    )
    closes = np.array([100.0, 110.0, 121.0])
    returns = five_minute_log_returns(timestamps, closes)
    assert returns.shape == (1,)
    assert returns[0] == pytest.approx(math.log(1.1))


def test_annualized_volatility_uses_sample_standard_deviation() -> None:
    returns = np.array([0.01, -0.01, 0.01, -0.01])
    expected = float(np.std(returns, ddof=1) * np.sqrt(252.0))
    assert annualized_volatility(returns, 252.0) == pytest.approx(expected)


def test_weekday_close_uses_the_cash_bar_and_skips_sunday() -> None:
    timestamps = np.array(
        [
            "2026-08-28T15:50:00",
            "2026-08-28T15:55:00",
            "2026-08-28T16:00:00",
            "2026-08-30T15:55:00",
            "2026-08-31T15:50:00",
        ],
        dtype="datetime64[ns]",
    )
    closes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    days, prices = weekday_closes_at(timestamps, closes, hour=15, minute=55)
    assert days.astype(str).tolist() == ["2026-08-28", "2026-08-31"]
    assert prices.tolist() == [2.0, 5.0]


def test_trailing_window_includes_the_anchor_price() -> None:
    dates = np.arange(np.datetime64("2026-01-01"), np.datetime64("2026-01-11"))
    prices = np.array([100, 100, 100, 100, 100, 110, 110, 110, 110, 121], dtype=float)
    # Last date is 2026-01-10. A 4-calendar-day window starts after 2026-01-06,
    # and the anchor is the 2026-01-06 price of 110.
    vol = trailing_close_to_close_volatility(dates, prices, lookback_days=4, periods_per_year=252.0)
    chosen = np.array([110.0, 110.0, 110.0, 110.0, 121.0])
    expected = annualized_volatility(np.diff(np.log(chosen)), 252.0)
    assert vol == pytest.approx(expected)


def test_loader_round_trip(tmp_path: Path) -> None:
    intraday = tmp_path / "bars.csv"
    intraday.write_text(
        "Datetime,Adj Close,Close,High,Low,Open,Volume\n"
        "2026-08-24 15:55:00-04:00,100,100,100,100,100,1\n"
        "2026-08-24 16:00:00-04:00,101,101,101,101,101,1\n",
        encoding="utf-8",
    )
    timestamps, closes = load_intraday_ohlcv(intraday)
    assert closes.tolist() == [100.0, 101.0]
    assert str(timestamps[0].astype("datetime64[s]")) == "2026-08-24T15:55:00"

    daily = tmp_path / "vxn.csv"
    daily.write_text(",Close\n2026-11-02,30.5\n2026-11-01,29.5\n", encoding="utf-8")
    dates, levels = load_daily_closes(daily)
    assert dates.astype(str).tolist() == ["2026-11-01", "2026-11-02"]
    assert levels.tolist() == [29.5, 30.5]


def test_bundled_nq_and_vxn_samples() -> None:
    timestamps, closes = load_intraday_ohlcv(ROOT / "data" / "NQ_F_5m.csv")
    assert closes.size > 10_000
    assert closes[-1] == pytest.approx(28145.25)
    returns = five_minute_log_returns(timestamps, closes)
    intraday = annualized_volatility(returns, 252.0 * 276.0)
    days, daily = weekday_closes_at(timestamps, closes)
    assert 40 <= days.size <= 60
    realized = trailing_close_to_close_volatility(days, daily, lookback_days=30)
    roll_dates, roll_vol = rolling_close_to_close_volatility(days, daily, window=21)
    assert roll_dates.size == daily.size - 21
    assert 0.05 < intraday < 1.0
    assert 0.05 < realized < 1.0
    assert np.all((roll_vol > 0.02) & (roll_vol < 1.5))

    vxn_dates, vxn = load_daily_closes(ROOT / "data" / "VXN_daily.csv")
    assert str(vxn_dates[0]) == "2025-11-03"
    assert str(vxn_dates[-1]) == "2026-11-02"
    assert vxn[-1] == pytest.approx(30.89, abs=1e-3)
