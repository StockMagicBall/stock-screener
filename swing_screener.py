"""
Swing/Day-Trade Screener
=========================
Pulls recent daily price/volume data for a list of US tickers, computes a set
of short-term technical signals, and produces a ranked "probable movement"
watchlist.

This is a SCREENING tool, not a prediction engine. It surfaces stocks showing
statistically unusual momentum, volatility, or volume behavior relative to
their own recent history. Treat the output as a shortlist to investigate
further, not a buy/sell signal.

Requirements:
    pip install yfinance pandas numpy

Usage:
    python swing_screener.py --tickers AAPL MSFT NVDA TSLA AMD
    python swing_screener.py --tickers-file tickers.txt --top 15
"""

import argparse
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_history(ticker: str, period: str = "6mo") -> pd.DataFrame | None:
    """Fetch daily OHLCV data for a single ticker via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        sys.exit(
            "yfinance is not installed. Run: pip install yfinance pandas numpy"
        )

    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty or len(df) < 30:
        return None

    # yfinance sometimes returns MultiIndex columns for single tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None
    return df[list(required)].dropna()


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = (upper - lower) / mid
    return upper, mid, lower, bandwidth


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    avg_vol = volume.rolling(period).mean().shift(1)  # avoid using today's own vol in its own baseline
    return volume / avg_vol


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    ticker: str
    close: float
    pct_change_1d: float
    rsi_14: float
    macd_hist: float
    bb_bandwidth: float
    bb_position: float  # 0 = at lower band, 1 = at upper band
    atr_pct: float  # ATR as % of price -- volatility magnitude
    rel_volume: float
    range_break: str  # "20d_high", "20d_low", or "none"
    score: float = 0.0
    notes: list = field(default_factory=list)


def score_ticker(df: pd.DataFrame, ticker: str) -> Signal | None:
    if df is None or len(df) < 30:
        return None

    close = df["close"]
    high20 = df["high"].rolling(20).max()
    low20 = df["low"].rolling(20).min()

    r = rsi(close)
    macd_line, signal_line, hist = macd(close)
    upper, mid, lower, bw = bollinger_bands(close)
    a = atr(df)
    rv = relative_volume(df["volume"])

    last = -1
    last_close = close.iloc[last]
    prev_close = close.iloc[last - 1]
    pct_change_1d = (last_close / prev_close - 1) * 100

    bb_range = (upper.iloc[last] - lower.iloc[last])
    bb_position = (
        (last_close - lower.iloc[last]) / bb_range if bb_range and not np.isnan(bb_range) else 0.5
    )

    range_break = "none"
    if last_close >= high20.iloc[last - 1]:
        range_break = "20d_high"
    elif last_close <= low20.iloc[last - 1]:
        range_break = "20d_low"

    sig = Signal(
        ticker=ticker,
        close=round(float(last_close), 2),
        pct_change_1d=round(float(pct_change_1d), 2),
        rsi_14=round(float(r.iloc[last]), 1) if not np.isnan(r.iloc[last]) else 50.0,
        macd_hist=round(float(hist.iloc[last]), 4),
        bb_bandwidth=round(float(bw.iloc[last]), 4) if not np.isnan(bw.iloc[last]) else 0.0,
        bb_position=round(float(bb_position), 2),
        atr_pct=round(float(a.iloc[last] / last_close * 100), 2) if not np.isnan(a.iloc[last]) else 0.0,
        rel_volume=round(float(rv.iloc[last]), 2) if not np.isnan(rv.iloc[last]) else 1.0,
        range_break=range_break,
    )

    # --- Weighted scoring: higher = more "unusual" short-term setup ---
    score = 0.0
    notes = []

    # Volume anomaly is the strongest short-term tell
    if sig.rel_volume >= 3:
        score += 30; notes.append(f"volume {sig.rel_volume}x avg")
    elif sig.rel_volume >= 1.5:
        score += 15; notes.append(f"volume {sig.rel_volume}x avg")

    # Range breakout
    if sig.range_break == "20d_high":
        score += 20; notes.append("broke 20d high")
    elif sig.range_break == "20d_low":
        score += 20; notes.append("broke 20d low")

    # Momentum extremes (overbought/oversold, prone to short-term continuation or reversal)
    if sig.rsi_14 >= 70:
        score += 12; notes.append(f"RSI overbought ({sig.rsi_14})")
    elif sig.rsi_14 <= 30:
        score += 12; notes.append(f"RSI oversold ({sig.rsi_14})")

    # MACD histogram sign flip magnitude (using raw value as proxy for momentum strength)
    if abs(sig.macd_hist) > 0:
        macd_strength = min(abs(sig.macd_hist) / (sig.close * 0.005), 1.0) * 10
        score += macd_strength
        if macd_strength > 5:
            notes.append("strong MACD momentum")

    # Bollinger squeeze = volatility contraction, often precedes a move
    if sig.bb_bandwidth and sig.bb_bandwidth < 0.05:
        score += 15; notes.append("BB squeeze (low volatility, watch for breakout)")

    # Already at/beyond bands = active expansion
    if sig.bb_position >= 1.0:
        score += 10; notes.append("price at/above upper BB")
    elif sig.bb_position <= 0.0:
        score += 10; notes.append("price at/below lower BB")

    # Elevated ATR relative to typical single-digit % suggests an active mover
    if sig.atr_pct >= 5:
        score += 8; notes.append(f"high volatility (ATR {sig.atr_pct}% of price)")

    sig.score = round(score, 1)
    sig.notes = notes
    return sig


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_screen(tickers: list, period: str = "6mo") -> pd.DataFrame:
    rows = []
    for t in tickers:
        t = t.strip().upper()
        if not t:
            continue
        df = fetch_history(t, period=period)
        sig = score_ticker(df, t)
        if sig is None:
            print(f"  skip {t}: insufficient data", file=sys.stderr)
            continue
        rows.append(sig)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame([r.__dict__ for r in rows])
    out["notes"] = out["notes"].apply(lambda n: "; ".join(n) if n else "")
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Short-term swing/day-trade screener")
    parser.add_argument("--tickers", nargs="+", help="List of tickers, e.g. AAPL MSFT NVDA")
    parser.add_argument("--tickers-file", help="Path to a text file, one ticker per line")
    parser.add_argument("--period", default="6mo", help="History window (e.g. 3mo, 6mo, 1y)")
    parser.add_argument("--top", type=int, default=20, help="Number of results to show")
    parser.add_argument("--out", default="watchlist.csv", help="CSV output path")
    args = parser.parse_args()

    tickers = list(args.tickers) if args.tickers else []
    if args.tickers_file:
        with open(args.tickers_file) as f:
            tickers += [line.strip() for line in f if line.strip()]

    if not tickers:
        parser.error("Provide --tickers or --tickers-file")

    print(f"Screening {len(tickers)} tickers...", file=sys.stderr)
    result = run_screen(tickers, period=args.period)

    if result.empty:
        print("No results — check tickers and data availability.")
        return

    result.head(args.top).to_csv(args.out, index=False)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(result.head(args.top).to_string(index=False))
    print(f"\nSaved full ranked list to {args.out}")


if __name__ == "__main__":
    main()
