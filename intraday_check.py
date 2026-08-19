"""
Intraday movement alert -- a separate, simpler, EXPERIMENTAL tool from the
main daily strategy.

IMPORTANT HONESTY NOTE: the daily screener/strategy/confirmation logic was
backtested and validated entirely on DAILY CLOSING PRICES. Running that same
scoring against intraday, still-forming bars would answer a different
question than what was actually tested, and would silently borrow
credibility it hasn't earned. This module does NOT do that. Instead it's a
much simpler, honestly-scoped heuristic: flag a ticker if it has moved more
than a threshold % from yesterday's close, with a volume sanity check,
during the current trading session. This has NOT been backtested. Treat it
as a "something's moving, go look" nudge, not a validated signal.

Runs every ~15-30 minutes during market hours via GitHub Actions. Checks
whether the market is actually open (using SPY as a proxy) before doing
anything, so it's a no-op outside trading hours, on weekends, and on
holidays -- no need for the cron schedule itself to know the exact calendar.

Required environment variables: same EMAIL_FROM / EMAIL_PASSWORD / EMAIL_TO
as daily_alert.py.

Optional environment variables:
    INTRADAY_TICKERS    -- comma-separated, defaults to a short priority list
    INTRADAY_MOVE_PCT   -- default 3.0 (% move from prior close to trigger)
    INTRADAY_MIN_RVOL   -- default 1.5 (current volume vs typical, sanity check)

Run manually with:
    python intraday_check.py
"""

import os
import sys
from datetime import datetime, timezone

import pandas as pd

from swing_screener import get_current_price, fetch_history
from daily_alert import send_email

DEFAULT_INTRADAY_TICKERS = "AAPL,MSFT,NVDA,TSLA,AMD,AMZN,GOOGL,META,NFLX,AVGO"
DEDUP_LOG_PATH = "intraday_alerts_today.csv"
DEDUP_COLUMNS = ["date", "ticker", "alerted_at_utc", "move_pct"]


def is_market_open() -> bool:
    """Uses SPY's reported market state as a proxy for 'is the US market open right now'."""
    try:
        import yfinance as yf
        info = yf.Ticker("SPY").info or {}
        return info.get("marketState") == "REGULAR"
    except Exception as e:
        print(f"Could not determine market state ({e}) -- skipping this run to be safe.", file=sys.stderr)
        return False


def load_dedup_log() -> pd.DataFrame:
    if os.path.exists(DEDUP_LOG_PATH):
        df = pd.read_csv(DEDUP_LOG_PATH)
        for col in DEDUP_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[DEDUP_COLUMNS]
    return pd.DataFrame(columns=DEDUP_COLUMNS)


def get_previous_close(ticker: str) -> float | None:
    try:
        import yfinance as yf
        fi = yf.Ticker(ticker).fast_info
        prev = fi.get("previousClose") or fi.get("previous_close")
        if prev is not None:
            return float(prev)
    except Exception:
        pass
    try:
        df = fetch_history(ticker, period="5d")
        if df is not None and len(df) >= 2:
            return float(df["close"].iloc[-2])
    except Exception:
        pass
    return None


def get_relative_volume(ticker: str) -> float | None:
    """Today's volume-so-far vs the 20-day average full-session volume, as a rough sanity check."""
    try:
        df = fetch_history(ticker, period="1mo")
        if df is None or len(df) < 10:
            return None
        avg_vol = df["volume"].iloc[:-1].mean()  # exclude today's partial bar
        today_vol = df["volume"].iloc[-1]
        if avg_vol and avg_vol > 0:
            return float(today_vol / avg_vol)
    except Exception:
        pass
    return None


def main():
    if not is_market_open():
        print("Market is not currently open (per SPY) -- nothing to do.", file=sys.stderr)
        return

    tickers = [
        t.strip().upper() for t in os.environ.get("INTRADAY_TICKERS", DEFAULT_INTRADAY_TICKERS).split(",")
        if t.strip()
    ]
    move_threshold = float(os.environ.get("INTRADAY_MOVE_PCT", "3.0"))
    min_rvol = float(os.environ.get("INTRADAY_MIN_RVOL", "1.5"))

    today = datetime.now(timezone.utc).date().isoformat()
    dedup = load_dedup_log()
    already_alerted_today = set(dedup.loc[dedup["date"] == today, "ticker"]) if not dedup.empty else set()

    new_movers = []
    print(f"Checking {len(tickers)} tickers intraday (market is open)...", file=sys.stderr)
    for ticker in tickers:
        if ticker in already_alerted_today:
            continue

        price_info = get_current_price(ticker)
        current_price = price_info.get("price")
        prev_close = get_previous_close(ticker)
        if current_price is None or prev_close is None or prev_close == 0:
            continue

        move_pct = (current_price / prev_close - 1) * 100
        if abs(move_pct) < move_threshold:
            continue

        rvol = get_relative_volume(ticker)
        if rvol is not None and rvol < min_rvol:
            continue

        new_movers.append({
            "ticker": ticker, "current_price": round(current_price, 2),
            "prev_close": round(prev_close, 2), "move_pct": round(move_pct, 2),
            "rvol": round(rvol, 2) if rvol is not None else None,
        })

    print(f"New movers this run: {len(new_movers)}", file=sys.stderr)

    if new_movers:
        subject = "[Swing Screener] Intraday movement: " + ", ".join(
            f"{m['ticker']} ({m['move_pct']:+.1f}%)" for m in new_movers
        )
        lines = [
            "EXPERIMENTAL intraday alert -- NOT the backtested daily strategy, and has not been "
            "validated the way the daily signals have. This is a simple 'something's moving, go "
            "look' nudge based on % move from yesterday's close plus a volume sanity check.\n",
        ]
        for m in new_movers:
            rvol_str = f", {m['rvol']}x avg volume" if m["rvol"] is not None else ""
            lines.append(
                f"  {m['ticker']}: ${m['current_price']} ({m['move_pct']:+.2f}% from prior close "
                f"${m['prev_close']}{rvol_str})"
            )
        lines.append(
            "\n---\nNot financial advice. This intraday tool is much simpler and less rigorously "
            "tested than the daily strategy -- treat it as a prompt to go look, not a signal to act on."
        )
        body = "\n".join(lines)

        try:
            send_email(subject, body)
            print("Intraday alert email sent.", file=sys.stderr)
        except Exception as e:
            print(f"Failed to send intraday alert email: {e}", file=sys.stderr)

        new_rows = pd.DataFrame([
            {
                "date": today, "ticker": m["ticker"],
                "alerted_at_utc": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "move_pct": m["move_pct"],
            }
            for m in new_movers
        ])
        dedup = pd.concat([dedup, new_rows], ignore_index=True)
        dedup["date"] = dedup["date"].astype(str)
        cutoff = (pd.Timestamp.utcnow() - pd.Timedelta(days=30)).date().isoformat()
        dedup = dedup[dedup["date"] >= cutoff]
        dedup.to_csv(DEDUP_LOG_PATH, index=False)
    else:
        print("No new intraday movers to alert on.", file=sys.stderr)


if __name__ == "__main__":
    main()
