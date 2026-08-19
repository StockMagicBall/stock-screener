"""
Backtester for the swing/day-trade screener.

Walks through each ticker's full price history, computes what the screener's
score WOULD HAVE BEEN on every past day (using only data available up to
that day -- no lookahead), then checks what actually happened in the
following 1/3/5 trading days. This tells you whether high scores actually
precede bigger-than-usual moves, or whether the tool is just noise dressed
up as insight.

Run with:
    python backtest.py --tickers AAPL MSFT NVDA TSLA AMD --period 2y

Requirements: same as swing_screener.py (yfinance, pandas, numpy)
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

from swing_screener import fetch_history, rsi, macd, bollinger_bands, atr, relative_volume

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Vectorized indicator + score computation (mirrors swing_screener's
# per-row scoring logic, but applied to every historical day at once)
# ---------------------------------------------------------------------------

def compute_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high20 = df["high"].rolling(20).max()
    low20 = df["low"].rolling(20).min()

    r = rsi(close)
    _, _, hist = macd(close)
    upper, mid, lower, bw = bollinger_bands(close)
    a = atr(df)
    rv = relative_volume(df["volume"])

    bb_range = (upper - lower).replace(0, np.nan)
    bb_position = (close - lower) / bb_range

    range_break = pd.Series("none", index=df.index)
    range_break[close >= high20.shift(1)] = "20d_high"
    range_break[close <= low20.shift(1)] = "20d_low"

    ind = pd.DataFrame(
        {
            "close": close,
            "rsi_14": r,
            "macd_hist": hist,
            "bb_bandwidth": bw,
            "bb_position": bb_position,
            "atr_pct": a / close * 100,
            "rel_volume": rv,
            "range_break": range_break,
        }
    )
    return ind


def compute_score_series(ind: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=ind.index)

    rv = ind["rel_volume"]
    score += np.where(rv >= 3, 30, np.where(rv >= 1.5, 15, 0))

    score += np.where(ind["range_break"] != "none", 20, 0)

    r14 = ind["rsi_14"]
    score += np.where((r14 >= 70) | (r14 <= 30), 12, 0)

    macd_strength = (ind["macd_hist"].abs() / (ind["close"] * 0.005)).clip(upper=1) * 10
    score += macd_strength.fillna(0)

    score += np.where(ind["bb_bandwidth"] < 0.05, 15, 0)

    score += np.where((ind["bb_position"] >= 1.0) | (ind["bb_position"] <= 0.0), 10, 0)

    score += np.where(ind["atr_pct"] >= 5, 8, 0)

    # Rows without enough history for indicators to be valid -> NaN out
    invalid = ind[["rsi_14", "bb_bandwidth", "atr_pct", "rel_volume"]].isna().any(axis=1)
    score[invalid] = np.nan

    return score


# ---------------------------------------------------------------------------
# Backtest evaluation
# ---------------------------------------------------------------------------

def backtest_ticker(df: pd.DataFrame, horizons=(1, 3, 5)) -> pd.DataFrame:
    ind = compute_indicator_frame(df)
    score = compute_score_series(ind)
    close = df["close"]

    rows = []
    for h in horizons:
        fwd_return = close.shift(-h) / close - 1
        valid = score.notna() & fwd_return.notna()
        s, f = score[valid], fwd_return[valid]
        if len(s) < 20:
            continue

        abs_f = f.abs()
        top_thresh = s.quantile(0.8)
        top_mask = s >= top_thresh

        rows.append(
            {
                "horizon_days": h,
                "n_days": len(s),
                "corr_score_vs_abs_move": round(s.corr(abs_f), 3),
                "avg_abs_move_top20pct_score": round(abs_f[top_mask].mean() * 100, 2),
                "avg_abs_move_rest": round(abs_f[~top_mask].mean() * 100, 2),
                "avg_signed_return_top20pct_score": round(f[top_mask].mean() * 100, 2),
                "pct_positive_top20pct_score": round((f[top_mask] > 0).mean() * 100, 1),
            }
        )
    return pd.DataFrame(rows)


def run_backtest(tickers: list, period: str = "2y", horizons=(1, 3, 5)) -> pd.DataFrame:
    all_results = []
    for t in tickers:
        t = t.strip().upper()
        if not t:
            continue
        df = fetch_history(t, period=period)
        if df is None or len(df) < 60:
            print(f"  skip {t}: insufficient data", file=sys.stderr)
            continue
        res = backtest_ticker(df, horizons=horizons)
        if res.empty:
            continue
        res.insert(0, "ticker", t)
        all_results.append(res)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Pool all tickers together per horizon for an overall verdict."""
    summary = (
        results.groupby("horizon_days")
        .apply(
            lambda g: pd.Series(
                {
                    "total_days": g["n_days"].sum(),
                    "avg_corr_score_vs_abs_move": round(g["corr_score_vs_abs_move"].mean(), 3),
                    "avg_move_top20pct_score(%)": round(
                        g["avg_abs_move_top20pct_score"].mean(), 2
                    ),
                    "avg_move_rest(%)": round(g["avg_abs_move_rest"].mean(), 2),
                    "avg_pct_positive_top20pct": round(
                        g["pct_positive_top20pct_score"].mean(), 1
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Backtest the swing screener's scoring logic")
    parser.add_argument("--tickers", nargs="+", help="List of tickers")
    parser.add_argument("--tickers-file", help="Path to a text file, one ticker per line")
    parser.add_argument("--period", default="2y", help="History window (e.g. 1y, 2y, 5y)")
    parser.add_argument("--out", default="backtest_results.csv", help="Per-ticker CSV output path")
    args = parser.parse_args()

    tickers = list(args.tickers) if args.tickers else []
    if args.tickers_file:
        with open(args.tickers_file) as f:
            tickers += [line.strip() for line in f if line.strip()]

    if not tickers:
        parser.error("Provide --tickers or --tickers-file")

    print(f"Backtesting {len(tickers)} tickers over {args.period}...", file=sys.stderr)
    results = run_backtest(tickers, period=args.period)

    if results.empty:
        print("No results — check tickers and data availability.")
        return

    results.to_csv(args.out, index=False)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)

    print("\n=== Per-ticker results ===")
    print(results.to_string(index=False))

    print("\n=== Pooled summary (this is the headline verdict) ===")
    summary = summarize(results)
    print(summary.to_string(index=False))

    print(
        "\nHow to read this:\n"
        "  - 'avg_move_top20pct_score' vs 'avg_move_rest': if the top-score days show a\n"
        "     noticeably BIGGER average move than the rest, the score is catching real signal.\n"
        "     If they're close, the score isn't adding much beyond random days.\n"
        "  - 'avg_corr_score_vs_abs_move': correlation between score and move size.\n"
        "     Near 0 = no relationship. 0.15+ is a meaningfully useful edge for this kind\n"
        "     of tool; 0.3+ would be unusually strong for short-term technical signals.\n"
        "  - 'avg_pct_positive_top20pct': for the top-score days, what % moved UP. Near 50%\n"
        "     means the score flags volatility/movement but not direction (expected --\n"
        "     this score wasn't designed to predict direction, only 'something's happening').\n"
        f"\nFull per-ticker breakdown saved to {args.out}"
    )


if __name__ == "__main__":
    main()
