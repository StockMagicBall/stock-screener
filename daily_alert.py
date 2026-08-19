"""
Daily signal alert -- checks tickers for confirmed LONG SETUP signals and
emails you if any fire. Designed to run on a schedule via GitHub Actions
(see .github/workflows/daily_signal_check.yml), since Streamlit Cloud only
runs your app when someone is actively viewing the page.

Required environment variables (set as GitHub Actions secrets, never commit
these to the repo):
    EMAIL_FROM      -- the Gmail address sending the alert
    EMAIL_PASSWORD  -- a Gmail App Password (NOT your regular password)
    EMAIL_TO        -- where to send the alert (can be the same address)

Optional environment variables (have sensible defaults if not set):
    TICKERS             -- comma-separated, defaults to the app's default list
    SCORE_QUANTILE      -- default 0.8
    CONFIRM_WINDOW_DAYS -- default 3
    WATCH_NEAR_PCT      -- default 15 (how close to threshold counts as "watch")
    MIN_CONVICTION      -- default HIGH (only email confirmed signals at or above this tier:
                            HIGH, MODERATE, or LOW)
    ACCOUNT_EQUITY      -- default 10000 (used for the position sizing shown in each email)
    RISK_PCT            -- default 1.0 (% of account risked per trade for sizing)
    STOP_LOSS_PCT       -- default 5.0 (must match the stop-loss % used elsewhere, so the
                            sizing reflects the actual stop the strategy would use)
    SMTP_SERVER         -- default smtp.gmail.com
    SMTP_PORT           -- default 587

Run manually with:
    python daily_alert.py
"""

import os
import smtplib
import sys
from email.mime.text import MIMEText

from strategy import get_todays_signals, calculate_position_size, CONVICTION_RANK

DEFAULT_TICKERS = "AAPL,MSFT,NVDA,TSLA,AMD,AMZN,GOOGL,META,NFLX,AVGO,AMC,BYND,GME,GPRO,^HSI,SPY,IWM,PDD,JD,TSLL,BULL,RKT,ENPH,^VIX,INTC,DOGE-USD,BABA,PYPL,BTC-USD,DJT,HOOD,ROBN,ETSY,GOOG,NKE,SOFI,COIN,BIDU,UBER,FUBO,SHOP,ARKG,KOSS,NIO,SMCI,BB,MU,DIS,DELL,PLTR,BRK-A,LULU,ROKU,ABNB,UVXY,AI"


def build_email_body(confirmed_df, awaiting_df, account_equity, risk_pct, stop_loss_pct) -> str:
    lines = []
    lines.append(f"Confirmed HIGH-conviction LONG SETUP signals (sized for ${account_equity:,.0f} account, {risk_pct}% risk/trade):\n")
    for _, row in confirmed_df.iterrows():
        entry_price = row["close"]
        stop_price = entry_price * (1 - stop_loss_pct / 100)
        sizing = calculate_position_size(account_equity, risk_pct, entry_price, stop_price)

        lines.append(
            f"  {row['ticker']}  |  close: ${row['close']}  |  score: {row['score']} "
            f"({row.get('conviction', '?')} conviction)  "
            f"|  confirmed {row.get('days_to_confirm', '?')} day(s) after signal"
        )
        if sizing:
            lines.append(
                f"      -> Suggested size: {sizing['shares']} shares (${sizing['position_value']:,.2f}, "
                f"{sizing['pct_of_account']}% of account) | stop: ${sizing['stop_price']} "
                f"| risking: ${sizing['risk_amount']:,.2f}"
            )
        else:
            lines.append("      -> Could not compute sizing for this price.")

    if not awaiting_df.empty:
        lines.append("\nAlso awaiting confirmation (not yet actionable):\n")
        for _, row in awaiting_df.iterrows():
            lines.append(f"  {row['ticker']}  |  close: ${row['close']}  |  score: {row['score']}")

    lines.append(
        "\n---\nReminder: this is a hypothesis-testing tool, not financial advice. Position "
        "sizing shown is a standard risk-based calculation, not a recommendation for your "
        "specific situation -- verify the numbers yourself before acting. Conviction tiers "
        "reflect how many signal components stacked together, not a predicted win probability. "
        "No system can guarantee winning trades -- confirm with your own research "
        "before acting on anything here."
    )
    return "\n".join(lines)


def build_watch_email_body(awaiting_df, near_threshold_df) -> str:
    lines = []
    lines.append(
        "Nothing is actionable yet -- this is an early heads-up, not a buy signal.\n"
    )

    if not awaiting_df.empty:
        lines.append("Awaiting confirmation (a raw signal fired, watching for follow-through):\n")
        for _, row in awaiting_df.iterrows():
            lines.append(f"  {row['ticker']}  |  close: ${row['close']}  |  score: {row['score']}")
        lines.append("")

    if not near_threshold_df.empty:
        lines.append("Near threshold (approaching a raw signal, trend already bullish):\n")
        for _, row in near_threshold_df.iterrows():
            lines.append(
                f"  {row['ticker']}  |  close: ${row['close']}  |  score: {row['score']} "
                f"(threshold: {row['score_threshold']})"
            )
        lines.append("")

    lines.append(
        "---\nThis is a watch-list alert, not a trade signal. Nothing here has confirmed "
        "yet -- treat it as 'worth checking on,' not 'time to act.'"
    )
    return "\n".join(lines)


def send_email(subject: str, body: str) -> None:
    email_from = os.environ["EMAIL_FROM"]
    email_password = os.environ["EMAIL_PASSWORD"]
    email_to = os.environ["EMAIL_TO"]
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(email_from, email_password)
        server.send_message(msg)


def main():
    if os.environ.get("TEST_MODE", "").lower() in ("1", "true", "yes"):
        print("TEST_MODE enabled -- sending a delivery-confirmation email regardless of signals.", file=sys.stderr)
        try:
            send_email(
                "[Swing Screener] Test alert -- delivery working",
                "This is a test email to confirm the Daily Signal Check workflow can "
                "successfully send you email. If you're reading this, delivery works.\n\n"
                "Real alerts will look like this but list actual confirmed LONG SETUP tickers.",
            )
            print("Test email sent successfully.", file=sys.stderr)
        except KeyError as e:
            print(f"Missing required environment variable: {e}. Set EMAIL_FROM, EMAIL_PASSWORD, EMAIL_TO.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Failed to send test email: {e}", file=sys.stderr)
            sys.exit(1)
        return

    tickers = [t.strip().upper() for t in os.environ.get("TICKERS", DEFAULT_TICKERS).split(",") if t.strip()]
    score_quantile = float(os.environ.get("SCORE_QUANTILE", "0.8"))
    confirm_window_days = int(os.environ.get("CONFIRM_WINDOW_DAYS", "3"))
    watch_near_pct = float(os.environ.get("WATCH_NEAR_PCT", "15"))
    min_conviction = os.environ.get("MIN_CONVICTION", "HIGH").upper()
    account_equity = float(os.environ.get("ACCOUNT_EQUITY", "10000"))
    risk_pct = float(os.environ.get("RISK_PCT", "1.0"))
    stop_loss_pct = float(os.environ.get("STOP_LOSS_PCT", "5.0"))

    print(f"Checking {len(tickers)} tickers for confirmed signals...", file=sys.stderr)
    signals = get_todays_signals(
        tickers, score_quantile=score_quantile,
        require_confirmation=True, confirm_window_days=confirm_window_days,
        watch_near_pct=watch_near_pct,
    )

    if signals.empty:
        print("No data returned -- check tickers and network access.", file=sys.stderr)
        return

    all_confirmed = signals[signals["signal"] == "LONG SETUP (confirmed)"]
    min_rank = CONVICTION_RANK.get(min_conviction, 2)
    confirmed = all_confirmed[
        all_confirmed["conviction"].map(lambda c: CONVICTION_RANK.get(c, -1)) >= min_rank
    ]
    awaiting = signals[signals["signal"] == "AWAITING CONFIRMATION"]
    near_threshold = signals[signals["signal"] == "WATCH (near threshold)"]

    print(
        f"Confirmed (all): {len(all_confirmed)}, Confirmed (>={min_conviction}): {len(confirmed)}, "
        f"Awaiting: {len(awaiting)}, Near threshold: {len(near_threshold)}",
        file=sys.stderr,
    )

    # Confirmed buy signals at or above the conviction floor -- the main alert, highest priority
    if not confirmed.empty:
        subject = f"[Swing Screener] {len(confirmed)} {min_conviction}-conviction signal(s): " + ", ".join(confirmed["ticker"])
        body = build_email_body(confirmed, awaiting, account_equity, risk_pct, stop_loss_pct)
        try:
            send_email(subject, body)
            print("Confirmed-signal email sent successfully.", file=sys.stderr)
        except KeyError as e:
            print(f"Missing required environment variable: {e}. Set EMAIL_FROM, EMAIL_PASSWORD, EMAIL_TO.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Failed to send confirmed-signal email: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"No confirmed signals at or above {min_conviction} conviction -- no buy-alert email sent.", file=sys.stderr)

    # High-watch alert -- separate, lower-urgency trigger for setups getting close
    watch_count = len(awaiting) + len(near_threshold)
    if watch_count > 0:
        watch_tickers = list(awaiting["ticker"]) + list(near_threshold["ticker"])
        subject = f"[Swing Screener] High Watch: {watch_count} ticker(s) getting close: " + ", ".join(watch_tickers)
        body = build_watch_email_body(awaiting, near_threshold)
        try:
            send_email(subject, body)
            print("High-watch email sent successfully.", file=sys.stderr)
        except KeyError as e:
            print(f"Missing required environment variable: {e}. Set EMAIL_FROM, EMAIL_PASSWORD, EMAIL_TO.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Failed to send high-watch email: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Nothing near threshold -- no watch email sent.", file=sys.stderr)


if __name__ == "__main__":
    main()
