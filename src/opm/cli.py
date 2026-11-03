"""Command line for the pricing routines."""

from __future__ import annotations

import argparse
import sys

from opm import __version__
from opm.binomial import binomial_price
from opm.black_scholes import black76_price, black_scholes_price
from opm.gex import net_gex_by_strike, read_chain_csv
from opm.greeks import analytic_greeks, black76_greeks, numeric_greeks
from opm.implied_vol import implied_volatility
from opm.monte_carlo import monte_carlo_price


def _add_contract(parser: argparse.ArgumentParser, *, with_volatility: bool) -> None:
    parser.add_argument("--spot", type=float, required=True, help="spot or futures price")
    parser.add_argument("--strike", type=float, required=True)
    parser.add_argument(
        "--time",
        type=float,
        required=True,
        dest="time_to_expiry",
        help="year fraction to expiry",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help="continuously compounded risk-free rate (default: 0)",
    )
    parser.add_argument(
        "--dividend-yield",
        type=float,
        default=0.0,
        dest="dividend_yield",
        help="continuous dividend yield (default: 0). Ignored for Black-76.",
    )
    parser.add_argument(
        "--type",
        default="call",
        dest="option_type",
        choices=["call", "put"],
        help="call or put (default: call)",
    )
    parser.add_argument(
        "--model",
        default="bsm",
        choices=["bsm", "black76"],
        help="bsm, or black76 where --spot is the futures price (default: bsm)",
    )
    if with_volatility:
        parser.add_argument(
            "--vol",
            type=float,
            required=True,
            dest="volatility",
            help="annualized volatility as a decimal, for example 0.20",
        )


def _contract(args: argparse.Namespace, *, volatility: float) -> dict[str, float | str]:
    dividend = args.rate if args.model == "black76" else args.dividend_yield
    return {
        "spot": args.spot,
        "strike": args.strike,
        "time_to_expiry": args.time_to_expiry,
        "rate": args.rate,
        "volatility": volatility,
        "dividend_yield": dividend,
        "option_type": args.option_type,
    }


def _print_price(model: str, option_type: str, price: float) -> None:
    print(f"model  {model}")
    print(f"type   {option_type}")
    print(f"price  {price:.8f}")


def _print_greeks(greeks: object) -> None:
    print("units  vega per 1.00 vol, theta per year, rho per 1.00 rate")
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        print(f"{name:<6} {getattr(greeks, name):.8f}")


def _command_price(args: argparse.Namespace) -> None:
    if args.model == "black76":
        price = black76_price(
            args.spot,
            args.strike,
            args.time_to_expiry,
            args.rate,
            args.volatility,
            args.option_type,
        )
        label = "black-76"
    else:
        price = black_scholes_price(**_contract(args, volatility=args.volatility))
        label = "black-scholes-merton"
    _print_price(label, args.option_type, price)


def _black76_pricer(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    option_type: str,
) -> float:
    """Black-76 adapter. The dividend argument is unused; carry stays in the futures."""

    del dividend_yield
    return black76_price(spot, strike, time_to_expiry, rate, volatility, option_type)


def _command_greeks(args: argparse.Namespace) -> None:
    if args.model == "black76" and args.method == "analytic":
        greeks = black76_greeks(
            args.spot,
            args.strike,
            args.time_to_expiry,
            args.rate,
            args.volatility,
            args.option_type,
        )
    elif args.model == "black76":
        greeks = numeric_greeks(
            **_contract(args, volatility=args.volatility),
            pricer=_black76_pricer,
        )
    elif args.method == "analytic":
        greeks = analytic_greeks(**_contract(args, volatility=args.volatility))
    else:
        greeks = numeric_greeks(**_contract(args, volatility=args.volatility))
    _print_greeks(greeks)


def _command_iv(args: argparse.Namespace) -> None:
    dividend = args.rate if args.model == "black76" else args.dividend_yield
    volatility = implied_volatility(
        args.price,
        args.spot,
        args.strike,
        args.time_to_expiry,
        args.rate,
        dividend,
        args.option_type,
    )
    print(f"implied_vol  {volatility:.8f}")
    print(f"percent      {100.0 * volatility:.4f}")


def _command_binomial(args: argparse.Namespace) -> None:
    price = binomial_price(
        **_contract(args, volatility=args.volatility),
        steps=args.steps,
        exercise=args.exercise,
    )
    _print_price(f"crr-{args.exercise}", args.option_type, price)


def _command_monte_carlo(args: argparse.Namespace) -> None:
    result = monte_carlo_price(
        **_contract(args, volatility=args.volatility),
        n_paths=args.paths,
        antithetic=not args.no_antithetic,
        seed=args.seed,
    )
    _print_price("monte-carlo", args.option_type, result.price)
    print(f"stderr {result.standard_error:.8f}")
    print(f"paths  {result.n_paths}")
    print(f"antithetic  {str(result.antithetic).lower()}")


def _command_gex(args: argparse.Namespace) -> None:
    chain = read_chain_csv(args.chain)
    result = net_gex_by_strike(
        chain,
        spot=args.spot,
        time_to_expiry=args.time_to_expiry,
        rate=args.rate,
        dividend_yield=args.dividend_yield,
        multiplier=args.multiplier,
    )
    print(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opm",
        description="Price European and American vanillas, implied volatility, and dealer gamma exposure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"opm {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    price = commands.add_parser("price", help="Black-Scholes-Merton or Black-76 price")
    _add_contract(price, with_volatility=True)
    price.set_defaults(func=_command_price)

    greeks = commands.add_parser("greeks", help="analytic or finite-difference Greeks")
    _add_contract(greeks, with_volatility=True)
    greeks.add_argument("--method", choices=["analytic", "numeric"], default="analytic")
    greeks.set_defaults(func=_command_greeks)

    iv = commands.add_parser("iv", help="implied volatility from a European premium")
    _add_contract(iv, with_volatility=False)
    iv.add_argument("--price", type=float, required=True, help="option premium to match")
    iv.set_defaults(func=_command_iv)

    tree = commands.add_parser("binomial", help="Cox-Ross-Rubinstein tree")
    _add_contract(tree, with_volatility=True)
    tree.add_argument("--steps", type=int, default=200)
    tree.add_argument("--exercise", choices=["european", "american"], default="european")
    tree.set_defaults(func=_command_binomial)

    monte = commands.add_parser(
        "monte-carlo",
        aliases=["mc"],
        help="European Monte Carlo with optional antithetic variates",
    )
    _add_contract(monte, with_volatility=True)
    monte.add_argument("--paths", type=int, default=100_000)
    monte.add_argument("--seed", type=int, default=None)
    monte.add_argument(
        "--no-antithetic",
        action="store_true",
        help="use independent normals instead of antithetic pairs",
    )
    monte.set_defaults(func=_command_monte_carlo)

    gex = commands.add_parser("gex", help="net dealer gamma exposure by strike")
    gex.add_argument("--chain", required=True, help="CSV with strike, oi, iv, type")
    gex.add_argument("--spot", type=float, required=True)
    gex.add_argument("--time", type=float, required=True, dest="time_to_expiry")
    gex.add_argument("--rate", type=float, default=0.0)
    gex.add_argument("--dividend-yield", type=float, default=0.0, dest="dividend_yield")
    gex.add_argument(
        "--multiplier",
        type=float,
        default=100.0,
        help="underlying units per contract (100 equity, 20 E-mini Nasdaq-100)",
    )
    gex.set_defaults(func=_command_gex)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
