"""
Streamlit web UI for the swing/day-trade screener + directional strategy.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

from swing_screener import run_screen
from strategy import simulate_trades, summarize_trades, get_todays_signals
from swing_screener import fetch_history

st.set_page_config(page_title="Swing Screener", layout="wide")
st.title("📈 Swing / Day-Trade Screener")
st.caption(
    "Educational tool, not financial advice. The movement score flags unusual "
    "short-term setups; the strategy tab adds a trend filter on top and backtests "
    "it as real trades. No system can guarantee winning trades — treat this as a "
    "hypothesis-testing tool, and paper-trade before risking real money."
)

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
    st.subheader("Backtest the directional strategy")
    st.caption(
        "Combines the movement score with a trend filter (price vs 50-day average + "
        "MACD) into long-only entries, then simulates trades: enter next day's open, "
        "hold N days, exit at close, minus an estimated round-trip cost."
    )

    col1, col2 = st.columns(2)
    with col1:
        strat_tickers_input = st.text_area(
            "Tickers to backtest",
            value="AAPL\nMSFT\nNVDA\nTSLA\nAMD\nAMZN\nGOOGL\nMETA\nNFLX\nAVGO",
            height=180,
            key="strategy_tickers",
        )
        strat_period = st.selectbox("History window", ["1y", "2y", "3y", "5y"], index=2, key="strategy_period")
    with col2:
        holding_days = st.slider("Holding period (trading days)", 1, 10, 3, key="holding_days")
        score_quantile = st.slider("Score threshold (percentile)", 0.5, 0.95, 0.8, step=0.05, key="score_q")
        cost_bps = st.number_input("Round-trip cost (bps)", min_value=0.0, value=10.0, step=1.0, key="cost_bps")

    if st.button("Run Strategy Backtest", type="primary", key="run_strategy"):
        raw = strat_tickers_input.replace(",", "\n")
        tickers = [t.strip().upper() for t in raw.splitlines() if t.strip()]

        all_trades = []
        progress = st.progress(0.0, text="Backtesting...")
        for i, t in enumerate(tickers):
            df = fetch_history(t, period=strat_period)
            if df is not None and len(df) >= 100:
                trades = simulate_trades(
                    df, t, score_quantile=score_quantile,
                    holding_days=holding_days, cost_bps=cost_bps,
                )
                if not trades.empty:
                    all_trades.append(trades)
            progress.progress((i + 1) / len(tickers))
        progress.empty()

        if not all_trades:
            st.error("No trades generated — try a longer period or lower score threshold.")
        else:
            trades = pd.concat(all_trades, ignore_index=True).sort_values("signal_date")
            summary = summarize_trades(trades)

            st.subheader("Results")
            cols = st.columns(4)
            cols[0].metric("Total trades", summary["total_trades"])
            cols[1].metric("Win rate", f"{summary['win_rate_pct']}%")
            cols[2].metric("Profit factor", summary["profit_factor"])
            cols[3].metric("Max drawdown", f"{summary['max_drawdown_pct']}%")

            cols2 = st.columns(4)
            cols2[0].metric("Avg return/trade", f"{summary['avg_return_per_trade_pct']}%")
            cols2[1].metric("Avg win", f"{summary['avg_win_pct']}%")
            cols2[2].metric("Avg loss", f"{summary['avg_loss_pct']}%")
            cols2[3].metric("Cumulative return", f"{summary['cumulative_return_pct']}%")

            if summary["profit_factor"] > 2.5:
                st.warning(
                    "Profit factor above 2.5 on a simple rules-based strategy is unusually "
                    "high — treat this as a sign of overfitting to this specific sample "
                    "rather than a strategy you should trust at face value."
                )

            equity = (1 + trades["net_return_pct"] / 100).cumprod()
            st.line_chart(equity.reset_index(drop=True), height=250)
            st.caption("Simulated equity curve (sequential trades, not compounded across overlapping positions)")

            st.dataframe(trades, use_container_width=True, hide_index=True)
            csv = trades.to_csv(index=False).encode("utf-8")
            st.download_button("Download trade log as CSV", data=csv, file_name="strategy_trades.csv", mime="text/csv")

            st.caption(
                "Reminder: perfect fills assumed, flat cost estimate. Real slippage and "
                "execution timing will make live results worse than this, not better."
            )

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

    if st.button("Check Today's Signals", type="primary", key="run_today"):
        raw = today_tickers_input.replace(",", "\n")
        tickers = [t.strip().upper() for t in raw.splitlines() if t.strip()]
        with st.spinner("Checking latest data..."):
            signals = get_todays_signals(tickers, score_quantile=today_quantile)

        if signals.empty:
            st.error("No data returned — check your tickers.")
        else:
            flagged = signals[signals["signal"] == "LONG SETUP"]
            if not flagged.empty:
                st.success(f"{len(flagged)} ticker(s) currently meet the strategy's entry criteria")
            else:
                st.info("No tickers currently meet the strategy's entry criteria — that's normal, and better than false positives.")

            st.dataframe(signals, use_container_width=True, hide_index=True)
            st.caption(
                "A 'LONG SETUP' here means: movement score above your threshold AND price "
                "trending up (above 50-day average, MACD bullish) as of the most recent close. "
                "It is not a recommendation to buy — confirm with your own research."
            )
