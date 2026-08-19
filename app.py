"""
Streamlit web UI for the swing/day-trade screener + directional strategy.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

from swing_screener import run_screen
from strategy import simulate_portfolio, simulate_buy_and_hold, get_todays_signals

st.set_page_config(page_title="Swing Screener", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Theme: trading-terminal aesthetic -- dark navy, monospace data, amber signature
# ---------------------------------------------------------------------------
TICKER_TAPE = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "GOOGL", "META", "NFLX", "AVGO"]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background-color: #0A0E14; }

    /* Headers */
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }
    h1 {
        background: linear-gradient(90deg, #F0A93A 0%, #FFD98A 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }

    /* Numbers everywhere: monospace, like a real terminal */
    [data-testid="stMetricValue"], .stDataFrame, code {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(155deg, #121822 0%, #0D1219 100%);
        border: 1px solid #212B38;
        border-radius: 10px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] { color: #6B7889 !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #212B38; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif; font-weight: 500; color: #6B7889;
        padding: 10px 18px;
    }
    .stTabs [aria-selected="true"] {
        color: #F0A93A !important; border-bottom: 2px solid #F0A93A !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #F0A93A, #E0922B);
        color: #0A0E14; font-weight: 700; border: none; border-radius: 8px;
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 16px rgba(240, 169, 58, 0.45); transform: translateY(-1px);
    }

    /* Ticker tape -- signature element */
    .ticker-tape-wrap {
        overflow: hidden; white-space: nowrap; border-top: 1px solid #212B38;
        border-bottom: 1px solid #212B38; background: #0D1219; padding: 8px 0; margin-bottom: 1.2rem;
    }
    .ticker-tape {
        display: inline-block; padding-left: 100%;
        animation: ticker-scroll 32s linear infinite;
        font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #6B7889;
    }
    .ticker-tape span { margin-right: 2.5rem; }
    .ticker-tape .dot { color: #2FD9A8; margin-right: 4px; }
    @keyframes ticker-scroll {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-100%); }
    }
    @media (prefers-reduced-motion: reduce) {
        .ticker-tape { animation: none; padding-left: 0; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_tape_html = "".join(f'<span><span class="dot">●</span>{t}</span>' for t in TICKER_TAPE * 3)
st.markdown(f'<div class="ticker-tape-wrap"><div class="ticker-tape">{_tape_html}</div></div>', unsafe_allow_html=True)

st.title("📈 Swing / Day-Trade Screener")
st.caption(
    "Educational tool, not financial advice. The movement score flags unusual "
    "short-term setups; the strategy tab adds a trend filter on top and backtests "
    "it as real trades. No system can guarantee winning trades — treat this as a "
    "hypothesis-testing tool, and paper-trade before risking real money."
)


def style_pnl(df: pd.DataFrame, pct_col: str = "net_return_pct"):
    """Color-code a returns column green/coral, terminal-style."""
    def _color(val):
        if pd.isna(val):
            return ""
        color = "#2FD9A8" if val > 0 else ("#FF5C6C" if val < 0 else "#6B7889")
        return f"color: {color}; font-weight: 600;"
    return df.style.map(_color, subset=[pct_col]) if pct_col in df.columns else df


def style_signal(df: pd.DataFrame, col: str = "signal"):
    """Highlight rows that meet the entry criteria, terminal-style."""
    def _row(row):
        if col in row and row[col] == "LONG SETUP":
            return ["background-color: rgba(240, 169, 58, 0.12); font-weight: 600;"] * len(row)
        return [""] * len(row)
    return df.style.apply(_row, axis=1) if col in df.columns else df


tab_screen, tab_strategy, tab_today = st.tabs(
    ["🔍 Screener", "🧪 Strategy Backtest", "🎯 Today's Signals"]
)

# ---------------------------------------------------------------------------
# Tab 1: original movement screener
# ---------------------------------------------------------------------------
with tab_screen:
    with st.sidebar:
        st.header("Screener Settings")
        tickers_input = st.text_area(
            "Tickers (one per line, or comma-separated)",
            value="AAPL\nMSFT\nNVDA\nTSLA\nAMD\nAMZN\nGOOGL\nMETA\nNFLX\nAVGO",
            height=220,
            key="screener_tickers",
        )
        period = st.selectbox("History window", ["3mo", "6mo", "1y"], index=1, key="screener_period")
        top_n = st.slider("Show top N results", 5, 50, 15, key="screener_topn")
        run_button = st.button("Run Screen", type="primary", key="run_screen")

    if run_button:
        raw = tickers_input.replace(",", "\n")
        tickers = [t.strip().upper() for t in raw.splitlines() if t.strip()]
        if not tickers:
            st.warning("Enter at least one ticker.")
        else:
            with st.spinner(f"Pulling data and scoring {len(tickers)} tickers..."):
                result = run_screen(tickers, period=period)
            if result.empty:
                st.error("No results — check your tickers and try again.")
            else:
                top = result.head(top_n)
                st.subheader(f"Top {len(top)} results")
                st.dataframe(
                    top, use_container_width=True, hide_index=True,
                    column_config={
                        "score": st.column_config.ProgressColumn(
                            "score", min_value=0, max_value=100, format="%.1f"
                        ),
                    },
                )
                csv = top.to_csv(index=False).encode("utf-8")
                st.download_button("Download as CSV", data=csv, file_name="watchlist.csv", mime="text/csv")
    else:
        st.info("Set your tickers in the sidebar and click **Run Screen** to begin.")

# ---------------------------------------------------------------------------
# Tab 2: strategy backtest (movement score + trend direction -> simulated trades)
# ---------------------------------------------------------------------------
with tab_strategy:
    st.subheader("Backtest the swing-trading strategy")
    st.caption(
        "Portfolio-level simulation: a fixed number of position slots share a real "
        "capital pool. Positions ride the trend using a trailing stop — instead of "
        "exiting on a fixed day count — and close out when the trend actually reverses."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        strat_tickers_input = st.text_area(
            "Tickers to backtest",
            value="AAPL\nMSFT\nNVDA\nTSLA\nAMD\nAMZN\nGOOGL\nMETA\nNFLX\nAVGO",
            height=180,
            key="strategy_tickers",
        )
        date_mode = st.radio(
            "Time window", ["Recent (rolling)", "Custom date range"], key="date_mode", horizontal=True
        )
        if date_mode == "Recent (rolling)":
            strat_period = st.selectbox("History window", ["1y", "2y", "3y", "5y"], index=2, key="strategy_period")
            strat_start, strat_end = None, None
        else:
            strat_period = "3y"
            dcol1, dcol2 = st.columns(2)
            strat_start = dcol1.date_input("Start date", value=pd.Timestamp("2022-01-01"), key="strat_start")
            strat_end = dcol2.date_input("End date", value=pd.Timestamp("2022-12-31"), key="strat_end")
            st.caption("Tip: 2022 was a rough year for tech stocks — a good stress test for whether this strategy earns its keep when buy-and-hold doesn't.")
        score_quantile = st.slider("Score threshold (percentile)", 0.5, 0.95, 0.8, step=0.05, key="score_q")
    with col2:
        stop_loss_pct = st.slider("Initial stop-loss (%)", 1.0, 15.0, 5.0, step=0.5, key="stop_loss")
        trailing_stop_pct = st.slider("Trailing stop (%)", 2.0, 20.0, 8.0, step=0.5, key="trailing_stop")
        max_holding_days = st.slider("Max holding days (safety cap)", 5, 60, 20, key="max_holding")
        trend_exit = st.checkbox("Exit early if trend reverses", value=True, key="trend_exit")
    with col3:
        max_concurrent = st.slider("Max concurrent positions", 1, 10, 5, key="max_concurrent")
        starting_capital = st.number_input("Starting capital ($)", min_value=1000, value=10000, step=1000, key="capital")
        cost_bps = st.number_input("Round-trip cost (bps)", min_value=0.0, value=10.0, step=1.0, key="cost_bps")

    if st.button("Run Strategy Backtest", type="primary", key="run_strategy"):
        raw = strat_tickers_input.replace(",", "\n")
        tickers = [t.strip().upper() for t in raw.splitlines() if t.strip()]

        with st.spinner("Simulating portfolio..."):
            result = simulate_portfolio(
                tickers, period=strat_period,
                start=str(strat_start) if strat_start else None,
                end=str(strat_end) if strat_end else None,
                score_quantile=score_quantile,
                max_holding_days=max_holding_days, cost_bps=cost_bps,
                stop_loss_pct=stop_loss_pct, trailing_stop_pct=trailing_stop_pct,
                trend_exit=trend_exit, starting_capital=starting_capital,
                max_concurrent=max_concurrent,
            )

        trades, equity_df, summary = result["trades"], result["equity_curve"], result["summary"]

        if trades.empty:
            st.error("No trades generated — try a longer period or lower score threshold.")
        else:
            st.subheader("Results")
            cols = st.columns(4)
            cols[0].metric("Total trades", summary["total_trades"])
            cols[1].metric("Win rate", f"{summary['win_rate_pct']}%")
            cols[2].metric("Profit factor", summary["profit_factor"])
            cols[3].metric("Max drawdown", f"{summary['max_drawdown_pct']}%")

            cols2 = st.columns(4)
            cols2[0].metric("Total return", f"{summary['total_return_pct']}%")
            cols2[1].metric("Final equity", f"${summary['final_equity']:,.0f}")
            cols2[2].metric("Avg days held", summary["avg_days_held"])
            cols2[3].metric("Skipped (no free slot)", summary["trades_skipped_capacity"])

            if summary["profit_factor"] > 2.5:
                st.warning(
                    "Profit factor above 2.5 on a simple rules-based strategy is unusually "
                    "high — treat this as a sign of overfitting rather than a strategy to trust outright."
                )

            st.line_chart(equity_df.set_index("date")["equity"], height=250)
            st.caption("Real portfolio equity curve — capital shared across concurrent positions.")

            exit_counts = trades["exit_reason"].value_counts()
            st.caption(
                "Exit reasons: " + ", ".join(f"{v} {k}" for k, v in exit_counts.items())
            )

            st.dataframe(style_pnl(trades, "net_return_pct"), use_container_width=True, hide_index=True)
            csv = trades.to_csv(index=False).encode("utf-8")
            st.download_button("Download trade log as CSV", data=csv, file_name="strategy_trades.csv", mime="text/csv")

            st.caption(
                "Reminder: perfect fills assumed at the stop/exit price, flat cost estimate. "
                "Real slippage and execution timing will make live results worse than this, not better."
            )

            # --- Buy-and-hold benchmark comparison ---
            st.divider()
            st.subheader("vs. Buy-and-Hold Benchmark")
            st.caption(
                "The question that matters most: does the strategy's extra complexity and "
                "risk actually beat just buying and holding the same tickers?"
            )
            with st.spinner("Computing buy-and-hold benchmark..."):
                bh = simulate_buy_and_hold(
                    tickers, period=strat_period,
                    start=str(strat_start) if strat_start else None,
                    end=str(strat_end) if strat_end else None,
                    starting_capital=starting_capital,
                )

            if bh["summary"]:
                bh_summary = bh["summary"]
                bcol = st.columns(3)
                bcol[0].metric(
                    "Strategy total return", f"{summary['total_return_pct']}%",
                    delta=f"{round(summary['total_return_pct'] - bh_summary['total_return_pct'], 1)}pp vs buy-and-hold",
                )
                bcol[1].metric("Buy-and-hold total return", f"{bh_summary['total_return_pct']}%")
                bcol[2].metric(
                    "Strategy drawdown vs buy-and-hold",
                    f"{summary['max_drawdown_pct']}%",
                    delta=f"{round(summary['max_drawdown_pct'] - bh_summary['max_drawdown_pct'], 1)}pp",
                    delta_color="inverse",
                )

                strat_eq = equity_df.set_index("date")["equity"].rename("Strategy")
                bh_eq = bh["equity_curve"].set_index("date")["equity"].rename("Buy & Hold")
                combined = pd.concat([strat_eq, bh_eq], axis=1).dropna()
                st.line_chart(combined, height=300)

                if summary["total_return_pct"] > bh_summary["total_return_pct"]:
                    st.success(
                        f"The strategy outperformed buy-and-hold by "
                        f"{round(summary['total_return_pct'] - bh_summary['total_return_pct'], 1)} percentage points "
                        f"over this period — though check whether the lower/higher drawdown justifies the added complexity."
                    )
                else:
                    st.warning(
                        f"Buy-and-hold actually outperformed the strategy by "
                        f"{round(bh_summary['total_return_pct'] - summary['total_return_pct'], 1)} percentage points "
                        f"over this period. Unless the strategy's drawdown is meaningfully better, simply holding "
                        f"these tickers would have been the better — and far simpler — choice."
                    )
            else:
                st.info("Could not compute benchmark for these tickers.")

# ---------------------------------------------------------------------------
# Tab 3: today's live signals
# ---------------------------------------------------------------------------
with tab_today:
    st.subheader("What the strategy flags right now")
    today_tickers_input = st.text_area(
        "Tickers to check",
        value="AAPL\nMSFT\nNVDA\nTSLA\nAMD\nAMZN\nGOOGL\nMETA\nNFLX\nAVGO",
        height=180,
        key="today_tickers",
    )
    today_quantile = st.slider("Score threshold (percentile)", 0.5, 0.95, 0.8, step=0.05, key="today_q")
    show_context = st.checkbox(
        "Also pull news sentiment + institutional ownership for flagged tickers",
        value=False, key="show_context",
    )
    if show_context:
        st.caption(
            "News sentiment is scored from recent headlines and is noisy/backward-looking — "
            "it often reflects a move that already happened rather than predicting the next one. "
            "Institutional ownership comes from 13F filings, which can be up to ~4 months stale and "
            "shows position size, not recent buying or selling. Both are context, not entry signals."
        )

    if st.button("Check Today's Signals", type="primary", key="run_today"):
        raw = today_tickers_input.replace(",", "\n")
        tickers = [t.strip().upper() for t in raw.splitlines() if t.strip()]
        with st.spinner("Checking latest data..."):
            signals = get_todays_signals(
                tickers, score_quantile=today_quantile, include_context=show_context,
            )

        if signals.empty:
            st.error("No data returned — check your tickers.")
        else:
            flagged = signals[signals["signal"] == "LONG SETUP"]
            if not flagged.empty:
                st.success(f"{len(flagged)} ticker(s) currently meet the strategy's entry criteria")
            else:
                st.info("No tickers currently meet the strategy's entry criteria — that's normal, and better than false positives.")

            st.dataframe(style_signal(signals, "signal"), use_container_width=True, hide_index=True)
            st.caption(
                "A 'LONG SETUP' here means: movement score above your threshold AND price "
                "trending up (above 50-day average, MACD bullish) as of the most recent close. "
                "It is not a recommendation to buy — confirm with your own research."
            )
