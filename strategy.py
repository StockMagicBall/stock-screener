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

def conviction_tier(score: float) -> str:
    """
    Bucket a signal-day score into a rough conviction tier, based on what the
    scoring system's components mean (see backtest.py's compute_score_series):
    a score in the 30s+ usually means several signals stacked together
    (volume spike, breakout, RSI extreme, etc), while a score under ~18 is
    typically just one weak factor.

    IMPORTANT: this is NOT a predicted win probability. The backtests showed
    a real but modest correlation (roughly 0.12-0.24) between score and
    MOVEMENT SIZE -- not a score that predicts a winning trade. Treat this
    as "how many independent things lined up," not "how likely to profit."
    """
    if score >= 40:
        return "HIGH"
    if score >= 25:
        return "MODERATE"
    return "LOW"


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


def simulate_portfolio(
    tickers: list,
    period: str = "3y",
    start: str = None,
    end: str = None,
    score_quantile: float = 0.8,
    max_holding_days: int = 20,
    cost_bps: float = 10.0,
    stop_loss_pct: float = 5.0,
    trailing_stop_pct: float = 8.0,
    trend_exit: bool = True,
    starting_capital: float = 10000.0,
    max_concurrent: int = 5,
    require_confirmation: bool = True,
    confirm_window_days: int = 3,
) -> dict:
    """
    Swing-trade portfolio backtest. A fixed number of position 'slots' share
    real capital (so simultaneous signals can't each claim 100%). Each
    position is held as long as the trend holds, exiting on whichever
    comes first:
      - initial stop-loss (stop_loss_pct below entry)
      - trailing stop (trailing_stop_pct below the highest price reached
        since entry -- this is what lets winners run instead of forcing
        an exit after a fixed number of days)
      - trend reversal (price/MACD trend filter turns off, if trend_exit=True)
      - max_holding_days safety cap, in case none of the above ever fires

    ENTRY-TIMING CONFIRMATION (if require_confirmation=True): a raw signal
    does not enter immediately. It waits up to confirm_window_days for price
    to close above the signal day's high -- real follow-through, not just a
    one-day flicker. If that never happens, the signal is dropped entirely
    (no trade). This delays entries but filters out breakouts that
    immediately fail. Entry then happens at the open of the day AFTER
    confirmation, so this never looks ahead.

    Pass start (and optionally end) as "YYYY-MM-DD" to backtest a specific
    historical window (e.g. start="2022-01-01", end="2022-12-31") instead
    of the rolling `period` window.
    """
    ticker_data = {}
    ticker_direction = {}
    candidates = []

    for t in tickers:
        t = t.strip().upper()
        df = fetch_history(t, period=period, start=start, end=end)
        if df is None or len(df) < 100:
            continue
        ticker_data[t] = df

        ind = compute_indicator_frame(df)
        score = compute_score_series(ind)
        direction = compute_direction(df)
        ticker_direction[t] = direction

        valid = score.notna() & direction.notna()
        if not valid.any():
            continue
        threshold = score[valid].quantile(score_quantile)
        entry_signal = valid & (score >= threshold) & (direction == "bullish")

        for signal_date in df.index[entry_signal]:
            sig_pos = df.index.get_loc(signal_date)

            if require_confirmation:
                signal_high = df["high"].iloc[sig_pos]
                confirmed_idx = None
                for offset in range(1, confirm_window_days + 1):
                    check_idx = sig_pos + offset
                    if check_idx >= len(df):
                        break
                    if df["close"].iloc[check_idx] > signal_high:
                        confirmed_idx = check_idx
                        break
                if confirmed_idx is None:
                    continue  # never confirmed -- drop this signal, no trade
                entry_idx = confirmed_idx + 1
                days_to_confirm = confirmed_idx - sig_pos
            else:
                entry_idx = sig_pos + 1
                days_to_confirm = 0

            if entry_idx >= len(df):
                continue
            candidates.append(
                {
                    "ticker": t,
                    "signal_date": signal_date,
                    "entry_date": df.index[entry_idx],
                    "entry_idx": entry_idx,
                    "score": round(float(score.loc[signal_date]), 1),
                    "days_to_confirm": days_to_confirm,
                    "used": False,
                }
            )

    if not candidates:
        return {"trades": pd.DataFrame(), "equity_curve": pd.DataFrame(), "summary": {}, "skipped": 0}

    candidates.sort(key=lambda c: c["entry_date"])

    all_dates = sorted(set().union(*[set(df.index) for df in ticker_data.values()]))

    position_size = starting_capital / max_concurrent
    cash = starting_capital
    open_positions = []
    closed_trades = []
    equity_curve = []
    skipped = 0

    def current_value(pos, date):
        df = ticker_data[pos["ticker"]]
        if date in df.index:
            close = df.loc[date, "close"]
            return position_size * (close / pos["entry_price"])
        return position_size

    def close_position(pos, exit_price, date, reason, days_held):
        net_return = (exit_price / pos["entry_price"] - 1) - (cost_bps / 10000)
        pnl = position_size * net_return
        nonlocal cash
        cash += position_size + pnl
        closed_trades.append(
            {
                "ticker": pos["ticker"],
                "signal_date": pos["signal_date"].date(),
                "entry_date": pos["entry_date"].date(),
                "exit_date": date.date(),
                "days_to_confirm": pos["days_to_confirm"],
                "days_held": days_held,
                "entry_price": round(pos["entry_price"], 2),
                "exit_price": round(float(exit_price), 2),
                "score": pos["score"],
                "net_return_pct": round(net_return * 100, 2),
                "exit_reason": reason,
                "win": bool(net_return > 0),
            }
        )

    for date in all_dates:
        # 1. process exits
        still_open = []
        for pos in open_positions:
            df = ticker_data[pos["ticker"]]
            direction = ticker_direction[pos["ticker"]]
            if date not in df.index:
                still_open.append(pos)
                continue
            day_idx = df.index.get_loc(date)
            if day_idx <= pos["entry_idx"]:
                still_open.append(pos)
                continue

            row = df.loc[date]
            days_held = day_idx - pos["entry_idx"]
            is_last_bar = day_idx == len(df) - 1

            # update trailing peak using the day's high before checking stops
            pos["peak_price"] = max(pos["peak_price"], row["high"])
            trailing_stop_price = pos["peak_price"] * (1 - trailing_stop_pct / 100) if trailing_stop_pct else -1
            effective_stop = max(pos["initial_stop_price"], trailing_stop_price)

            exit_price, reason = None, None
            if row["low"] <= effective_stop:
                exit_price = effective_stop
                reason = "trailing_stop" if effective_stop > pos["initial_stop_price"] else "stop_loss"
            elif trend_exit and direction.loc[date] != "bullish":
                exit_price, reason = row["close"], "trend_exit"
            elif days_held >= max_holding_days:
                exit_price, reason = row["close"], "time_cap"
            elif is_last_bar:
                exit_price, reason = row["close"], "data_end"

            if exit_price is not None:
                close_position(pos, exit_price, date, reason, days_held)
            else:
                still_open.append(pos)
        open_positions = still_open

        # 2. process new entries (only if a slot is free)
        for c in candidates:
            if c["entry_date"] != date or c["used"]:
                continue
            c["used"] = True
            if len(open_positions) >= max_concurrent:
                skipped += 1
                continue
            df = ticker_data[c["ticker"]]
            entry_price = float(df["open"].iloc[c["entry_idx"]])
            cash -= position_size
            initial_stop_price = entry_price * (1 - stop_loss_pct / 100) if stop_loss_pct else -1
            open_positions.append(
                {
                    "ticker": c["ticker"],
                    "signal_date": c["signal_date"],
                    "entry_date": c["entry_date"],
                    "entry_idx": c["entry_idx"],
                    "entry_price": entry_price,
                    "peak_price": entry_price,
                    "initial_stop_price": initial_stop_price,
                    "score": c["score"],
                    "days_to_confirm": c["days_to_confirm"],
                }
            )

        # 3. mark equity for today
        unrealized = sum(current_value(p, date) for p in open_positions)
        equity_curve.append({"date": date, "equity": cash + unrealized})

    trades = pd.DataFrame(closed_trades)
    equity_df = pd.DataFrame(equity_curve)

    summary = {}
    if not trades.empty:
        returns = trades["net_return_pct"] / 100
        wins = trades["win"]
        gross_profit = returns[returns > 0].sum()
        gross_loss = -returns[returns < 0].sum()
        running_max = equity_df["equity"].cummax()
        drawdown = (equity_df["equity"] / running_max - 1).min()

        summary = {
            "total_trades": len(trades),
            "trades_skipped_capacity": skipped,
            "win_rate_pct": round(wins.mean() * 100, 1),
            "avg_return_per_trade_pct": round(returns.mean() * 100, 2),
            "avg_win_pct": round(returns[wins].mean() * 100, 2) if wins.any() else 0.0,
            "avg_loss_pct": round(returns[~wins].mean() * 100, 2) if (~wins).any() else 0.0,
            "avg_days_held": round(trades["days_held"].mean(), 1),
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
            "stopped_out_pct": round((trades["exit_reason"].isin(["stop_loss", "trailing_stop"])).mean() * 100, 1),
            "final_equity": round(float(equity_df["equity"].iloc[-1]), 2),
            "total_return_pct": round((equity_df["equity"].iloc[-1] / starting_capital - 1) * 100, 1),
            "max_drawdown_pct": round(float(drawdown) * 100, 1),
        }

    return {"trades": trades, "equity_curve": equity_df, "summary": summary, "skipped": skipped}


def simulate_buy_and_hold(
    tickers: list, period: str = "3y", start: str = None, end: str = None,
    starting_capital: float = 10000.0,
) -> dict:
    """
    Equal-weight buy-and-hold benchmark: split starting_capital evenly across
    tickers on day 1, hold to the end, no trading. This is the baseline any
    active strategy needs to beat to justify its extra complexity and risk.
    """
    tickers = [t.strip().upper() for t in tickers]
    per_ticker_capital = starting_capital / len(tickers)

    value_series = {}
    for t in tickers:
        df = fetch_history(t, period=period, start=start, end=end)
        if df is None or len(df) < 2:
            continue
        shares = per_ticker_capital / df["close"].iloc[0]
        value_series[t] = df["close"] * shares

    if not value_series:
        return {"equity_curve": pd.DataFrame(), "summary": {}}

    combined = pd.concat(value_series.values(), axis=1).sort_index()
    combined = combined.ffill().dropna()
    equity = combined.sum(axis=1)

    running_max = equity.cummax()
    drawdown = (equity / running_max - 1).min()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    summary = {
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round(total_return * 100, 1),
        "annualized_return_pct": round(cagr * 100, 1),
        "max_drawdown_pct": round(float(drawdown) * 100, 1),
    }
    equity_df = equity.reset_index()
    equity_df.columns = ["date", "equity"]
    return {"equity_curve": equity_df, "summary": summary}


def get_todays_signals(
    tickers: list, period: str = "1y", score_quantile: float = 0.8,
    include_context: bool = False,
    require_confirmation: bool = True, confirm_window_days: int = 3,
    watch_near_pct: float = 15.0,
) -> pd.DataFrame:
    """
    What the strategy would flag RIGHT NOW, for each ticker's most recent day.

    With require_confirmation=True (matches the backtest logic), a raw
    score+trend signal must show real follow-through -- a close above the
    signal day's high, within confirm_window_days -- before it counts as an
    actionable setup. Signal states:
      - "LONG SETUP (confirmed)": raw signal fired recently AND price has
        already closed above that signal day's high -- actionable now.
      - "AWAITING CONFIRMATION": raw signal fired recently but hasn't shown
        follow-through yet -- still inside its confirmation window, don't
        act on it yet.
      - "WATCH (near threshold)": no raw signal yet, but today's score is
        within watch_near_pct of the entry threshold AND the trend is
        already bullish -- getting close, worth keeping an eye on.
      - "no signal": nothing meeting the criteria currently.

    If include_context=True, also pulls news sentiment and institutional
    ownership for confirmed setups -- these are DISPLAY CONTEXT ONLY, not
    part of the entry criteria, since sentiment is noisy/lagging and
    institutional data is quarterly and stale.
    """
    from swing_screener import get_news_sentiment, get_institutional_context

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
        raw_signal = valid & (score >= threshold) & (direction == "bullish")

        last = len(df) - 1
        s = score.iloc[last]
        d = direction.iloc[last]

        signal = "no signal"
        days_to_confirm = None
        signal_day_score = None
        signal_date = None

        if require_confirmation:
            # scan all raw signals within the lookback window -- if ANY of them
            # has since confirmed, that's an actionable setup; otherwise if any
            # raw signal exists unconfirmed, it's still awaiting follow-through.
            # Track the score/date FROM THE SIGNAL DAY ITSELF, not today --
            # today's score can drift well below what actually triggered this.
            lookback_start = max(0, last - confirm_window_days)
            found_confirmed, found_awaiting = False, False
            for sig_pos in range(lookback_start, last + 1):
                if not raw_signal.iloc[sig_pos]:
                    continue
                signal_high = df["high"].iloc[sig_pos]
                confirmed_this = False
                for offset in range(1, min(confirm_window_days, last - sig_pos) + 1):
                    if df["close"].iloc[sig_pos + offset] > signal_high:
                        confirmed_this = True
                        days_to_confirm = offset
                        break
                if confirmed_this:
                    found_confirmed = True
                    signal_day_score = float(score.iloc[sig_pos])
                    signal_date = df.index[sig_pos]
                else:
                    found_awaiting = True
                    if not found_confirmed:  # don't let an awaiting hit overwrite a confirmed one
                        signal_day_score = float(score.iloc[sig_pos])
                        signal_date = df.index[sig_pos]
            signal = (
                "LONG SETUP (confirmed)" if found_confirmed
                else "AWAITING CONFIRMATION" if found_awaiting
                else "no signal"
            )
        else:
            signal = "LONG SETUP" if (pd.notna(s) and s >= threshold and d == "bullish") else "no signal"
            if signal == "LONG SETUP":
                signal_day_score = float(s)
                signal_date = df.index[last]

        # Near-threshold watch: only applies when nothing above already fired --
        # i.e. no raw signal today or in the recent confirmation window.
        if signal == "no signal" and pd.notna(s) and d == "bullish" and threshold > 0:
            watch_floor = threshold * (1 - watch_near_pct / 100)
            if watch_floor <= s < threshold:
                signal = "WATCH (near threshold)"

        display_score = signal_day_score if signal_day_score is not None else (float(s) if pd.notna(s) else None)

        conviction = None
        if signal in ("LONG SETUP", "LONG SETUP (confirmed)") and display_score is not None:
            conviction = conviction_tier(display_score)

        row = {
            "ticker": t,
            "close": round(float(df["close"].iloc[last]), 2),
            "score": round(display_score, 1) if display_score is not None else None,
            "conviction": conviction,
            "today_score": round(float(s), 1) if pd.notna(s) else None,
            "score_threshold": round(float(threshold), 1),
            "direction": d,
            "signal": signal,
        }
        if signal_date is not None:
            row["signal_date"] = signal_date.date()
        if require_confirmation:
            row["days_to_confirm"] = days_to_confirm

        if include_context and signal in ("LONG SETUP", "LONG SETUP (confirmed)"):
            news = get_news_sentiment(t)
            inst = get_institutional_context(t)
            row["news_sentiment"] = news.get("avg_sentiment")
            row["news_headlines_checked"] = news.get("headline_count")
            row["pct_institutional_ownership"] = inst.get("pct_institutions")

        rows.append(row)
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
