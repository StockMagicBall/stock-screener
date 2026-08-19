"""
Directional strategy layer on top of the screener's movement score.

The screener score (from backtest.py) predicts MOVEMENT SIZE, not direction
(the backtest showed ~53-58% positive days on high-score days -- barely
above a coin flip). This module adds a trend/direction filter and combines
it with the movement score into an actual long-only trading rule, then
simulates real trades (entry next-day open, hold N days, exit at close,
minus round-trip costs) to see whether the combination has a genuine edge.

IMPORTANT HONESTY NOTE: this simulates trades assuming perfect fills at
open/close prices, with a simple flat cost estimate. No system can promise
winning trades -- treat all output here as a hypothesis test, not a
guarantee, and paper-trade before risking real money.

Run with:
    python strategy.py --tickers AAPL MSFT NVDA TSLA AMD --period 3y
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

from swing_screener import fetch_history, macd
from backtest import compute_indicator_frame, compute_score_series

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Direction filter
# ---------------------------------------------------------------------------

def compute_direction(df: pd.DataFrame) -> pd.Series:
    """
    Simple trend filter: price above its 50-day average AND MACD line above
    its signal line = 'bullish'. Price below 50-day average AND MACD below
    signal = 'bearish'. Otherwise 'neutral'.
    """
    close = df["close"]
    sma50 = close.rolling(50).mean()
    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    direction = pd.Series("neutral", index=df.index)
    bullish = (close > sma50) & (macd_line > signal_line)
    bearish = (close < sma50) & (macd_line < signal_line)
    direction[bullish] = "bullish"
    direction[bearish] = "bearish"
    direction[sma50.isna()] = np.nan
    return direction


# ---------------------------------------------------------------------------
# Trade simulation (long-only: movement score flags an unusual setup,
# direction filter requires the trend to already be up)
# ---------------------------------------------------------------------------

def simulate_trades(
    df: pd.DataFrame,
    ticker: str,
    score_quantile: float = 0.8,
    holding_days: int = 3,
    cost_bps: float = 10.0,  # round-trip cost estimate in basis points (0.10%)
) -> pd.DataFrame:
    ind = compute_indicator_frame(df)
    score = compute_score_series(ind)
    direction = compute_direction(df)

    valid = score.notna() & direction.notna()
    threshold = score[valid].quantile(score_quantile)

    entry_signal = valid & (score >= threshold) & (direction == "bullish")
    entry_dates = df.index[entry_signal]

    trades = []
    for entry_date in entry_dates:
        pos = df.index.get_loc(entry_date)
        entry_idx = pos + 1  # enter next trading day's open -- no lookahead
        exit_idx = entry_idx + holding_days - 1
        if exit_idx >= len(df):
            continue  # not enough future data to complete this trade

        entry_price = df["open"].iloc[entry_idx]
        exit_price = df["close"].iloc[exit_idx]
        gross_return = exit_price / entry_price - 1
        net_return = gross_return - (cost_bps / 10000)

        trades.append(
            {
                "ticker": ticker,
                "signal_date": entry_date.date(),
                "entry_date": df.index[entry_idx].date(),
                "exit_date": df.index[exit_idx].date(),
                "entry_price": round(float(entry_price), 2),
                "exit_price": round(float(exit_price), 2),
                "score": round(float(score.loc[entry_date]), 1),
                "gross_return_pct": round(float(gross_return) * 100, 2),
                "net_return_pct": round(float(net_return) * 100, 2),
                "win": bool(net_return > 0),
            }
        )

    return pd.DataFrame(trades)


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}

    returns = trades["net_return_pct"] / 100
    wins = trades["win"]
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1).min()

    gross_profit = returns[returns > 0].sum()
    gross_loss = -returns[returns < 0].sum()

    return {
        "total_trades": len(trades),
        "win_rate_pct": round(wins.mean() * 100, 1),
        "avg_return_per_trade_pct": round(returns.mean() * 100, 2),
        "avg_win_pct": round(returns[wins].mean() * 100, 2) if wins.any() else 0.0,
        "avg_loss_pct": round(returns[~wins].mean() * 100, 2) if (~wins).any() else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "cumulative_return_pct": round((equity.iloc[-1] - 1) * 100, 1),
        "max_drawdown_pct": round(drawdown * 100, 1),
    }


def get_todays_signals(tickers: list, period: str = "1y", score_quantile: float = 0.8) -> pd.DataFrame:
    """What the strategy would flag RIGHT NOW, for each ticker's most recent day."""
    rows = []
    for t in tickers:
        t = t.strip().upper()
        df = fetch_history(t, period=period)
        if df is None or len(df) < 60:
            continue

        ind = compute_indicator_frame(df)
        score = compute_score_series(ind)
        direction = compute_direction(df)

        valid = score.notna() & direction.notna()
        threshold = score[valid].quantile(score_quantile)

        last = -1
        s = score.iloc[last]
        d = direction.iloc[last]
        signal = "LONG SETUP" if (pd.notna(s) and s >= threshold and d == "bullish") else "no signal"

        rows.append(
            {
                "ticker": t,
                "close": round(float(df["close"].iloc[last]), 2),
                "score": round(float(s), 1) if pd.notna(s) else None,
                "score_threshold": round(float(threshold), 1),
                "direction": d,
                "signal": signal,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backtest the directional trading strategy")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--period", default="3y")
    parser.add_argument("--holding-days", type=int, default=3)
    parser.add_argument("--score-quantile", type=float, default=0.8)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--out", default="strategy_trades.csv")
    args = parser.parse_args()

    all_trades = []
    print(f"Simulating trades across {len(args.tickers)} tickers over {args.period}...", file=sys.stderr)
    for t in args.tickers:
        df = fetch_history(t, period=args.period)
        if df is None or len(df) < 100:
            print(f"  skip {t}: insufficient data", file=sys.stderr)
            continue
        trades = simulate_trades(
            df, t, score_quantile=args.score_quantile,
            holding_days=args.holding_days, cost_bps=args.cost_bps,
        )
        if not trades.empty:
            all_trades.append(trades)

    if not all_trades:
        print("No trades generated -- try a longer period or lower --score-quantile.")
        return

    trades = pd.concat(all_trades, ignore_index=True).sort_values("signal_date")
    trades.to_csv(args.out, index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(f"\n=== {len(trades)} simulated trades (holding {args.holding_days} days, "
          f"{args.cost_bps}bps round-trip cost) ===")
    print(trades.tail(20).to_string(index=False))

    print("\n=== Strategy summary ===")
    summary = summarize_trades(trades)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(
        "\nHow to read this:\n"
        "  - win_rate_pct: needs to clear ~50% and, more importantly, avg_win should\n"
        "    outweigh avg_loss -- a 45% win rate can still be profitable if wins run bigger.\n"
        "  - profit_factor: gross profit / gross loss. Below 1.0 = losing strategy.\n"
        "    Above 1.5 is respectable for a simple rules-based approach; treat anything\n"
        "    above ~2.5 with suspicion (likely overfit to this specific sample).\n"
        "  - max_drawdown_pct: worst peak-to-trough dip in simulated equity. This is the\n"
        "    number that tells you if you could actually stomach running this for real.\n"
        "  - This backtest assumes perfect fills and a flat cost estimate -- real slippage,\n"
        "    spread, and execution timing will make live results worse than this, not better.\n"
        f"\nFull trade log saved to {args.out}"
    )


if __name__ == "__main__":
    main()
