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
    SMTP_SERVER         -- default smtp.gmail.com
    SMTP_PORT           -- default 587

Run manually with:
    python daily_alert.py
"""

import os
import smtplib
import sys
from email.mime.text import MIMEText

from strategy import get_todays_signals

DEFAULT_TICKERS = "AAPL,MSFT,NVDA,TSLA,AMD,AMZN,GOOGL,META,NFLX,AVGO"


def build_email_body(confirmed_df, awaiting_df) -> str:
    lines = []
    lines.append("Confirmed LONG SETUP signals:\n")
    for _, row in confirmed_df.iterrows():
        lines.append(
            f"  {row['ticker']}  |  close: ${row['close']}  |  score: {row['score']}  "
            f"|  confirmed {row.get('days_to_confirm', '?')} day(s) after signal"
        )

    if not awaiting_df.empty:
        lines.append("\nAlso awaiting confirmation (not yet actionable):\n")
        for _, row in awaiting_df.iterrows():
            lines.append(f"  {row['ticker']}  |  close: ${row['close']}  |  score: {row['score']}")

    lines.append(
        "\n---\nReminder: this is a hypothesis-testing tool, not financial advice. "
        "No system can guarantee winning trades -- confirm with your own research "
        "before acting on anything here."
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
    tickers = [t.strip().upper() for t in os.environ.get("TICKERS", DEFAULT_TICKERS).split(",") if t.strip()]
    score_quantile = float(os.environ.get("SCORE_QUANTILE", "0.8"))
    confirm_window_days = int(os.environ.get("CONFIRM_WINDOW_DAYS", "3"))

    print(f"Checking {len(tickers)} tickers for confirmed signals...", file=sys.stderr)
    signals = get_todays_signals(
        tickers, score_quantile=score_quantile,
        require_confirmation=True, confirm_window_days=confirm_window_days,
    )

    if signals.empty:
        print("No data returned -- check tickers and network access.", file=sys.stderr)
        return

    confirmed = signals[signals["signal"] == "LONG SETUP (confirmed)"]
    awaiting = signals[signals["signal"] == "AWAITING CONFIRMATION"]

    print(f"Confirmed: {len(confirmed)}, Awaiting: {len(awaiting)}", file=sys.stderr)

    if confirmed.empty:
        print("No confirmed signals -- no email sent.", file=sys.stderr)
        return

    subject = f"[Swing Screener] {len(confirmed)} confirmed signal(s): " + ", ".join(confirmed["ticker"])
    body = build_email_body(confirmed, awaiting)

    try:
        send_email(subject, body)
        print("Email sent successfully.", file=sys.stderr)
    except KeyError as e:
        print(f"Missing required environment variable: {e}. Set EMAIL_FROM, EMAIL_PASSWORD, EMAIL_TO.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Failed to send email: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
