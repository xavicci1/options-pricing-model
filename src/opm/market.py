"""Loaders and realized-volatility helpers for the bundled bar files.

These functions are deliberately small. They turn a CSV of prices into log
returns and an annualized standard deviation. They do not adjust for
corporate actions, and they do not know an exchange's holiday calendar beyond
Monday-Friday.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np


def load_intraday_ohlcv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a Yahoo-style intraday file.

    The file needs ``Datetime`` and ``Close`` columns. Timestamps are returned
    as naive ``datetime64[ns]`` wall times: the UTC offset in the file is
    applied by the parser and then dropped, which keeps the exchange-local
    clock (America/New_York in the bundled Nasdaq-100 futures file).
    Rows are sorted by time. Closes must be positive.
    """

    file_path = Path(path)
    stamps: list[np.datetime64] = []
    closes: list[float] = []
    with file_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{file_path.name} has no header")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        if "datetime" not in fields or "close" not in fields:
            raise ValueError(f"{file_path.name} needs Datetime and Close columns")
        time_key = fields["datetime"]
        close_key = fields["close"]
        for line_number, row in enumerate(reader, start=2):
            raw_time = (row.get(time_key) or "").strip()
            raw_close = (row.get(close_key) or "").strip()
            if not raw_time or not raw_close:
                continue
            try:
                # fromisoformat keeps the wall time; drop the offset afterwards.
                local = datetime.fromisoformat(raw_time).replace(tzinfo=None)
                close = float(raw_close)
            except ValueError as exc:
                raise ValueError(f"{file_path.name}:{line_number}: {exc}") from exc
            if close <= 0.0:
                raise ValueError(f"{file_path.name}:{line_number}: close must be positive")
            stamps.append(np.datetime64(local, "ns"))
            closes.append(close)
    if not stamps:
        raise ValueError(f"{file_path.name} has no bars")
    timestamps = np.array(stamps, dtype="datetime64[ns]")
    prices = np.array(closes, dtype=np.float64)
    order = np.argsort(timestamps, kind="mergesort")
    return timestamps[order], prices[order]


def load_daily_closes(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a daily file whose first column is a date and that has a Close column.

    Dates are returned as ``datetime64[D]``. Rows are sorted by date.
    """

    file_path = Path(path)
    days: list[np.datetime64] = []
    closes: list[float] = []
    with file_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{file_path.name} has no header")
        date_key = reader.fieldnames[0]
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        if "close" not in fields:
            raise ValueError(f"{file_path.name} needs a Close column")
        close_key = fields["close"]
        for line_number, row in enumerate(reader, start=2):
            raw_date = (row.get(date_key) or "").strip()
            raw_close = (row.get(close_key) or "").strip()
            if not raw_date or not raw_close:
                continue
            try:
                days.append(np.datetime64(raw_date, "D"))
                closes.append(float(raw_close))
            except ValueError as exc:
                raise ValueError(f"{file_path.name}:{line_number}: {exc}") from exc
    if not days:
        raise ValueError(f"{file_path.name} has no rows")
    dates = np.array(days, dtype="datetime64[D]")
    prices = np.array(closes, dtype=np.float64)
    if np.any(prices <= 0.0):
        raise ValueError(f"{file_path.name} has a non-positive close")
    order = np.argsort(dates, kind="mergesort")
    return dates[order], prices[order]


def five_minute_log_returns(
    timestamps: np.ndarray,
    closes: np.ndarray,
    bar_seconds: int = 300,
) -> np.ndarray:
    """Log returns between consecutive bars separated by exactly ``bar_seconds``.

    Longer gaps (session halts, weekends, missing bars) are dropped so they
    are not treated as five-minute increments.
    """

    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    prices = np.asarray(closes, dtype=np.float64)
    if ts.shape != prices.shape:
        raise ValueError("timestamps and closes must have the same length")
    if prices.size < 2:
        raise ValueError("need at least two bars")
    if np.any(prices <= 0.0):
        raise ValueError("closes must be positive")
    if bar_seconds <= 0:
        raise ValueError("bar_seconds must be positive")
    delta_seconds = np.diff(ts).astype("timedelta64[s]").astype(np.int64)
    returns = np.diff(np.log(prices))
    return returns[delta_seconds == bar_seconds]


def annualized_volatility(
    log_returns: np.ndarray,
    periods_per_year: float,
    ddof: int = 1,
) -> float:
    """Sample standard deviation of log returns, scaled by ``sqrt(periods_per_year)``.

    ``ddof=1`` is the sample standard deviation. ``periods_per_year`` is the
    number of returns that make one year under the chosen sampling: 252 for
    weekday close-to-close returns, or ``252 * intervals_per_session`` for
    intraday returns.
    """

    returns = np.asarray(log_returns, dtype=np.float64)
    if periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive")
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    if returns.size <= ddof:
        raise ValueError("not enough returns to estimate volatility")
    return float(np.std(returns, ddof=ddof) * np.sqrt(periods_per_year))


def weekday_closes_at(
    timestamps: np.ndarray,
    closes: np.ndarray,
    hour: int = 15,
    minute: int = 55,
) -> tuple[np.ndarray, np.ndarray]:
    """Last close at or before a local clock time on each Monday-Friday.

    For five-minute bars stamped at the start of the interval, 15:55 is the
    bar that ends at 16:00. That is a cash-close proxy for an equity-index
    futures tape. Holidays that fall on a weekday are kept: the filter is
    the weekday itself, not an exchange holiday calendar.
    """

    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    prices = np.asarray(closes, dtype=np.float64)
    if ts.shape != prices.shape:
        raise ValueError("timestamps and closes must have the same length")
    if ts.size == 0:
        raise ValueError("no bars")
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("hour and minute are out of range")
    days = ts.astype("datetime64[D]")
    time_of_day = ts - days
    target = np.timedelta64(hour, "h") + np.timedelta64(minute, "m")
    eligible = np.is_busday(days) & (time_of_day <= target)
    if not np.any(eligible):
        raise ValueError("no weekday bars at or before the cutoff")
    chosen_days = days[eligible]
    chosen_prices = prices[eligible]
    start = np.empty(chosen_days.shape[0], dtype=bool)
    start[0] = True
    start[1:] = chosen_days[1:] != chosen_days[:-1]
    start_index = np.flatnonzero(start)
    end_index = np.empty(start_index.shape[0], dtype=int)
    end_index[:-1] = start_index[1:] - 1
    end_index[-1] = chosen_days.shape[0] - 1
    return chosen_days[end_index], chosen_prices[end_index]


def trailing_close_to_close_volatility(
    dates: np.ndarray,
    prices: np.ndarray,
    lookback_days: int = 30,
    periods_per_year: float = 252.0,
) -> float:
    """Annualized close-to-close volatility ending on the last date.

    The window is ``lookback_days`` calendar days. The last price on or before
    the cutoff is included so the first return begins at the edge of the window.
    """

    day = np.asarray(dates, dtype="datetime64[D]")
    series = np.asarray(prices, dtype=np.float64)
    if day.shape != series.shape:
        raise ValueError("dates and prices must have the same length")
    if day.size < 2:
        raise ValueError("need at least two prices")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    order = np.argsort(day, kind="mergesort")
    day = day[order]
    series = series[order]
    cutoff = day[-1] - np.timedelta64(int(lookback_days), "D")
    before = np.flatnonzero(day <= cutoff)
    if before.size:
        chosen = series[before[-1] :]
    else:
        chosen = series[day > cutoff]
    if chosen.size < 2:
        raise ValueError("not enough prices inside the lookback window")
    return annualized_volatility(np.diff(np.log(chosen)), periods_per_year)


def rolling_close_to_close_volatility(
    dates: np.ndarray,
    prices: np.ndarray,
    window: int = 21,
    periods_per_year: float = 252.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling annualized close-to-close volatility.

    ``window`` is the number of log returns in each sample. The date attached
    to a point is the date of the last price in that window.
    """

    day = np.asarray(dates, dtype="datetime64[D]")
    series = np.asarray(prices, dtype=np.float64)
    if day.shape != series.shape:
        raise ValueError("dates and prices must have the same length")
    if window < 2:
        raise ValueError("window must be at least 2")
    returns = np.diff(np.log(series))
    if returns.size < window:
        raise ValueError("not enough returns for the requested window")
    vols = np.empty(returns.size - window + 1, dtype=np.float64)
    for index, end in enumerate(range(window - 1, returns.size)):
        vols[index] = annualized_volatility(
            returns[end + 1 - window : end + 1], periods_per_year
        )
    return day[window:], vols
