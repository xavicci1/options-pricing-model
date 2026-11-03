# options-pricing-model

Educational pricing for European and American vanilla options. The library
implements Black-Scholes-Merton with a continuous dividend yield, the Black-76
futures formula, a Cox-Ross-Rubinstein binomial tree, European Monte Carlo
with antithetic variates, analytic and finite-difference Greeks, an implied
volatility solver, and a dealer gamma-exposure summary.

The import package is `opm`.

## Disclaimer

This project is for education and research. It is not financial advice, not a
solicitation to buy or sell any instrument, and not a substitute for a
production risk system. The models omit jumps, stochastic volatility, discrete
dividends, margin, and microstructure. A computed premium is not a market
quote. Dealer gamma exposure depends on an inventory assumption that this
library does not verify. Nothing here is a forecast or a recommendation.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Runtime dependencies are NumPy and SciPy. Matplotlib is optional and is used
only by the Nasdaq-100 example. Python 3.10 or newer is required.

## Library

```python
from opm import (
    analytic_greeks,
    binomial_price,
    black76_price,
    black_scholes_price,
    implied_volatility,
    monte_carlo_price,
    net_gex_by_strike,
)

call = black_scholes_price(
    spot=100,
    strike=100,
    time_to_expiry=1.0,
    rate=0.05,
    volatility=0.20,
    dividend_yield=0.02,
    option_type="call",
)
greeks = analytic_greeks(
    spot=100,
    strike=100,
    time_to_expiry=1.0,
    rate=0.05,
    volatility=0.20,
    dividend_yield=0.02,
    option_type="call",
)
volatility = implied_volatility(
    call, spot=100, strike=100, time_to_expiry=1.0, rate=0.05, dividend_yield=0.02
)
```

Volatility is a decimal: `0.20` means 20 percent. Time is a year fraction.

`examples/quickstart.py` prices that contract, recovers the volatility, runs a
400-step American tree and an antithetic Monte Carlo, and prints a two-line
gamma-exposure example.

### Models

**Black-Scholes-Merton.** European call and put with continuous dividend yield
`q`. At expiry the value is intrinsic. At zero volatility the value is the
discounted forward intrinsic value.

**Black-76.** European option on a futures price. Implemented as
Black-Scholes-Merton with the dividend yield set equal to the discount rate,
so the forward equals the futures price. `black76_greeks` holds the futures
price fixed: rho is minus tenor times the premium. The other Greeks match the
spot formula under that yield.

**Cox-Ross-Rubinstein.** Recombining tree with up move `exp(σ √Δt)` and down
move `exp(-σ √Δt)`. `exercise="european"` or `"american"`. American nodes take
the larger of continuation and intrinsic value. The tree raises if a step is
too coarse for the carry to sit between the down and up moves. Even step
counts are usually a little closer to the European closed form. Convergence is
oscillatory and on the order of `1 / steps`.

**Monte Carlo.** One-step geometric Brownian motion under the risk-neutral
measure, discounted at the continuous rate. European exercise only. Antithetic
variates pair each normal draw with its negative. The returned standard error
is the standard error of that estimator. Pass `seed` to repeat a run. American
early exercise is not simulated; use the tree.

### Greeks

Analytic delta, gamma, vega, theta, and rho use the Black-Scholes-Merton
partial derivatives.

| Greek | Meaning |
| --- | --- |
| delta | Change in premium per unit spot (or per futures point, for Black-76) |
| gamma | Change in delta per unit spot |
| vega | Derivative with respect to volatility in decimal form. One volatility point is about `0.01 * vega` |
| theta | Derivative with respect to calendar time, premium per year. Divide by 365 for an approximate calendar day |
| rho | Derivative with respect to the continuous risk-free rate, holding the dividend yield fixed, except for `black76_greeks` as noted above. One basis point is about `0.0001 * rho` |

`numeric_greeks` bumps the same inputs with central differences and can take
another pricer, for example `pricer=binomial_price, steps=400`. Analytic
Greeks require positive time and positive volatility, because gamma and theta
are discontinuous on those boundaries.

### Implied volatility

`implied_volatility` solves for the Black-Scholes-Merton volatility of a
European premium. Newton steps use vega when the step stays inside a bracket
that surrounds the root; otherwise the solver bisects. A premium outside the
no-arbitrage band raises `ImpliedVolatilityError`. A premium equal to the
zero-volatility value returns `0`. The solver does not look above a volatility
of 20.

For a Black-76 premium, call it with `dividend_yield` equal to `rate`, which
is what the `opm iv --model black76` command does.

### Dealer gamma exposure

`net_gex_by_strike` takes contracts `(strike, open interest, implied vol, type)`
and returns the net exposure at each strike:

```text
sign * gamma * open_interest * multiplier * spot^2 * 0.01
```

The unit is the change in dollar delta for a one percent rise in the spot.
Calls are signed positive and puts negative. That is the common assumption
that a dealer is long call gamma and short put gamma. It is an assumption, not
a measured position; negate the result for the opposite book. `multiplier` is
the number of underlying units per contract (100 for a standard US equity
option, 20 for the E-mini Nasdaq-100). Implied volatility is a decimal.
Contracts with zero open interest or zero implied volatility contribute
nothing.

`examples/sample_chain.csv` is a small chain for the command line.

## Command line

```bash
opm price --spot 100 --strike 100 --time 1 --rate 0.05 --dividend-yield 0.02 --vol 0.20
opm greeks --spot 100 --strike 100 --time 1 --rate 0.05 --dividend-yield 0.02 --vol 0.20
opm greeks --method numeric --spot 100 --strike 100 --time 1 --rate 0.05 --vol 0.20
opm iv --price 9.22700551 --spot 100 --strike 100 --time 1 --rate 0.05 --dividend-yield 0.02
opm binomial --spot 100 --strike 100 --time 1 --rate 0.05 --vol 0.20 --type put --exercise american --steps 400
opm monte-carlo --spot 100 --strike 100 --time 1 --rate 0.05 --vol 0.20 --paths 50000 --seed 1
opm gex --chain examples/sample_chain.csv --spot 100 --time 0.25 --rate 0.04 --dividend-yield 0.01
opm price --model black76 --spot 28137.25 --strike 28137.25 --time 0.082135 --rate 0.04 --vol 0.3089
```

`opm mc` is an alias of `opm monte-carlo`. `--time` is a year fraction.
`--vol` is a decimal. The Black-76 commands treat `--spot` as the futures price.

On the contract in the first command the premium is `9.22700551`. Passing that
premium to `opm iv` recovers a volatility of `0.20000000`.

## Worked example: NQ realized volatility and VXN

```bash
python examples/nq_realized_vs_vxn.py
```

The script reads `data/NQ_F_5m.csv` and `data/VXN_daily.csv`, estimates
realized volatility from the futures bars, and prices a 30-calendar-day
at-the-money European Black-76 option at that realized volatility and at the
VXN close. It prints the comparison and, when Matplotlib is installed, writes
`examples/output/nq_realized_vs_vxn.png`.

Two realized-volatility figures are reported.

- **Five-minute.** Sample standard deviation of log returns between bars that
  are exactly five minutes apart, multiplied by `sqrt(252 * 276)`. The 276
  intervals are a 23-hour electronic session. The daily halt and the weekend
  are longer than five minutes, so those jumps are left out.
- **Thirty-day close-to-close.** Weekday cash-close proxy: the 15:55
  America/New_York bar, which ends at 16:00 if the stamp is the start of the
  interval. Log returns over the last 30 calendar days, multiplied by
  `sqrt(252)`. This is the headline realized input because VXN is a 30-day
  implied volatility. The chart uses a rolling 21-return window of the same
  weekday series so the path can be compared with VXN.

The discount rate in the example is 4 percent, continuously compounded, and
is an illustration rather than a market quote. The strike equals the futures
proxy. That is model at-the-money, not a listed strike. USD premiums multiply
index points by 20, the E-mini Nasdaq-100 multiplier.

VXN measures 30-day implied volatility of the Nasdaq-100 cash index. It is not
an implied volatility from NQ futures options, and the futures basis means the
two underlyings are not the same. The script is a comparison of inputs.

On the bundled sample the script reports, for the 15:55 America/New_York close
on 2 November 2026 (futures 28137.25):

| Input | Annualized vol | ATM call and put (index points) |
| --- | ---: | ---: |
| 5-minute realized | 16.92% | 542.45 |
| 30-day close-to-close | 22.45% | 719.89 |
| VXN close | 30.89% | 990.16 |

Call and put premiums match because the strike is set equal to the futures
price. The USD column in the script multiplies by 20.

## Data

The files in `data/` are included so the example can be rerun.

| File | Series | Range | Source |
| --- | --- | --- | --- |
| `data/NQ_F_5m.csv` | Nasdaq-100 futures, 5-minute bars (Yahoo Finance `NQ=F`). Columns: Datetime, Adj Close, Close, High, Low, Open, Volume. The code uses Close. | 24 Aug 2026 00:00 through 2 Nov 2026 23:20, exchange-local time in the file | Yahoo Finance |
| `data/VXN_daily.csv` | CBOE Nasdaq-100 Volatility Index, daily close (Yahoo Finance `^VXN`) | 3 Nov 2025 through 2 Nov 2026 | Yahoo Finance. The index is published by CBOE and is quoted in annualized percentage points |

Yahoo Finance data is used here for research and education. It is not an
official settlement file and it can differ from an exchange print. VXN's
methodology is CBOE's, not this library's.

## Tests

```bash
pytest
```

The suite checks put-call parity, European tree convergence to
Black-Scholes-Merton, American early exercise where it is expected, Monte
Carlo error against the closed form, implied-volatility round trips, analytic
Greeks against finite differences, the gamma-exposure sign convention, and the
bundled sample files.

## Limitations

- Dividend yield is continuous. There is no discrete cash dividend.
- The tree is Cox-Ross-Rubinstein only. It has the usual even/odd oscillation.
- Monte Carlo prices European vanillas. It does not price early exercise.
- Implied volatility is the European Black-Scholes-Merton (or Black-76) root.
- Gamma exposure uses Black-Scholes gamma and a fixed dealer-sign assumption.
- There is no rate curve, no calendar spread, and no stochastic volatility.
- The Nasdaq-100 study is a teaching comparison. It is not a tradable signal
  and it does not reproduce the VXN calculation or the listed NQ option specs.

## License

MIT. See [LICENSE](LICENSE).
