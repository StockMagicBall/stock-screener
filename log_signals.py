"""
Live signal track record.

Every time a confirmed LONG SETUP fires, it gets logged with its entry price
and the date. A fixed number of trading days later, this script checks what
actually happened and records the real, out-of-sample result.

This is the single most important honesty check in the whole system: a
backtest already knows the answer. A live, forward-looking log does not --
it's the closest thing to a genuine track record this tool can produce.

The log lives in signal_log.csv and is committed back to the repo by the
GitHub Actions workflow, so it persists and accumulates across every
scheduled run.

Run with:
    python log_signals.py
"""

import os
import sys
from datetime import datetime

import pandas as pd

from strategy import get_todays_signals, fetch_history

LOG_PATH = "signal_log.csv"
EVAL_HORIZON_DAYS = 10  # trading days after logging before we score the outcome

LOG_COLUMNS = [
    "ticker", "logged_date", "entry_price", "score", "conviction",
    "eval_date", "outcome", "realized_return_pct", "evaluated_on",
]


def load_log() -> pd.DataFrame:
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        for col in LOG_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[LOG_COLUMNS]
    return pd.DataFrame(columns=LOG_COLUMNS)


def add_new_signals(log: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Log any newly-confirmed signal that isn't already logged for today."""
    confirmed = signals[signals["signal"] == "LONG SETUP (confirmed)"]
    today = datetime.utcnow().date().isoformat()

    already_logged_today = set(
        log.loc[log["logged_date"] == today, "ticker"]
    ) if not log.empty else set()

    new_rows = []
    for _, row in confirmed.iterrows():
        if row["ticker"] in already_logged_today:
            continue  # already logged this exact signal today, don't duplicate
        new_rows.append(
            {
                "ticker": row["ticker"],
                "logged_date": today,
                "entry_price": row["close"],  # observed price at logging time, not a real fill
                "score": row["score"],
                "conviction": row.get("conviction"),
                "eval_date": None,  # filled in once EVAL_HORIZON_DAYS trading days have passed
                "outcome": "pending",
                "realized_return_pct": None,
                "evaluated_on": None,
            }
        )

    if new_rows:
        print(f"Logging {len(new_rows)} new confirmed signal(s): "
              f"{[r['ticker'] for r in new_rows]}", file=sys.stderr)
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        print("No new confirmed signals to log today.", file=sys.stderr)

    return log


def evaluate_pending(log: pd.DataFrame) -> pd.DataFrame:
    """Score any pending entry that's old enough to evaluate now."""
    if log.empty:
        return log

    today = datetime.utcnow().date()
    pending = log[log["outcome"] == "pending"]
    evaluated_count = 0

    for idx, row in pending.iterrows():
        logged_date = pd.to_datetime(row["logged_date"]).date()
        calendar_days_elapsed = (today - logged_date).days
        # Rough proxy: trading days ~ 5/7 of calendar days. Good enough for a
        # "has enough time passed" gate -- not used for the return calc itself.
        if calendar_days_elapsed < EVAL_HORIZON_DAYS * 1.5:
            continue  # not old enough yet

        df = fetch_history(row["ticker"], period="3mo")
        if df is None or df.empty:
            continue  # couldn't fetch -- leave pending, try again next run

        current_price = float(df["close"].iloc[-1])
        entry_price = float(row["entry_price"])
        realized_return = (current_price / entry_price - 1) * 100

        log.loc[idx, "eval_date"] = today.isoformat()
        log.loc[idx, "outcome"] = "win" if realized_return > 0 else "loss"
        log.loc[idx, "realized_return_pct"] = round(realized_return, 2)
        log.loc[idx, "evaluated_on"] = today.isoformat()
        evaluated_count += 1

    print(f"Evaluated {evaluated_count} pending signal(s) that reached the "
          f"{EVAL_HORIZON_DAYS}-trading-day mark.", file=sys.stderr)
    return log


def print_summary(log: pd.DataFrame) -> None:
    resolved = log[log["outcome"].isin(["win", "loss"])]
    pending = log[log["outcome"] == "pending"]
    print(f"\nTrack record so far: {len(resolved)} resolved, {len(pending)} pending", file=sys.stderr)
    if not resolved.empty:
        win_rate = (resolved["outcome"] == "win").mean() * 100
        avg_return = resolved["realized_return_pct"].mean()
        print(f"  Win rate: {win_rate:.1f}%  |  Avg return: {avg_return:.2f}%", file=sys.stderr)


def main():
    tickers = [
        t.strip().upper() for t in os.environ.get(
            "TICKERS",
            "AAPL,MSFT,NVDA,TSLA,AMD,AMZN,GOOGL,META,NFLX,AVGO,AMC,BYND,GME,GPRO,"
            "^HSI,SPY,IWM,PDD,JD,TSLL,BULL,RKT,ENPH,^VIX,INTC,DOGE-USD,BABA,PYPL,"
            "BTC-USD,DJT,HOOD,ROBN,ETSY,GOOG,NKE,SOFI,COIN,BIDU,UBER,FUBO,SHOP,"
            "ARKG,KOSS,NIO,SMCI,BB,MU,DIS,DELL,PLTR,BRK-A,LULU,ROKU,ABNB,UVXY,AI",
        ).split(",") if t.strip()
    ]
    score_quantile = float(os.environ.get("SCORE_QUANTILE", "0.8"))
    confirm_window_days = int(os.environ.get("CONFIRM_WINDOW_DAYS", "3"))

    log = load_log()
    log = evaluate_pending(log)

    print(f"Checking {len(tickers)} tickers for new confirmed signals to log...", file=sys.stderr)
    signals = get_todays_signals(
        tickers, score_quantile=score_quantile,
        require_confirmation=True, confirm_window_days=confirm_window_days,
    )
    if not signals.empty:
        log = add_new_signals(log, signals)

    log.to_csv(LOG_PATH, index=False)
    print_summary(log)


if __name__ == "__main__":
    main()
