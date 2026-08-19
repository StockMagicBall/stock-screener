"""
Streamlit web UI for the swing/day-trade screener + directional strategy.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

from swing_screener import run_screen, get_current_price
from strategy import simulate_portfolio, simulate_buy_and_hold, get_todays_signals
import journal_store

st.set_page_config(page_title="Swing Screener", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Theme: trading-terminal aesthetic -- dark navy, monospace data, amber signature
# ---------------------------------------------------------------------------
TICKER_TAPE = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "GOOGL", "META", "NFLX", "AVGO", "AMC", "BYND", "GME", "GPRO", "^HSI", "SPY", "IWM", "PDD", "JD", "TSLL", "BULL", "RKT", "ENPH", "^VIX", "INTC", "DOGE-USD", "BABA", "PYPL", "BTC-USD", "DJT", "HOOD", "ROBN", "ETSY", "GOOG", "NKE", "SOFI", "COIN", "BIDU", "UBER", "FUBO", "SHOP", "ARKG", "KOSS", "NIO", "SMCI", "BB", "MU", "DIS", "DELL", "PLTR", "BRK-A", "LULU", "ROKU", "ABNB", "UVXY", "AI"]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@500;700;900&family=Manrope:wght@400;500;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }

    .stApp {
        background:
            radial-gradient(60% 50% at 15% 10%, rgba(139,92,246,0.18), transparent 60%),
            radial-gradient(50% 45% at 85% 0%, rgba(255,62,165,0.14), transparent 60%),
            radial-gradient(55% 50% at 50% 100%, rgba(34,211,238,0.10), transparent 60%),
            #0B0710;
    }

    h1, h2, h3 { font-family: 'Unbounded', sans-serif !important; letter-spacing: -0.01em; }
    h1 {
        background: linear-gradient(90deg, #8B5CF6 0%, #FF3EA5 55%, #22D3EE 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 900 !important; text-transform: uppercase; font-size: 2.2rem !important;
    }
    h2, h3 { color: #F5F3FF !important; font-weight: 700 !important; }

    /* Body text + captions -- default Streamlit gray is unreadable on this background */
    p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] {
        color: #D8D3E8 !important;
    }
    [data-testid="stCaptionContainer"] p {
        color: #A79FC4 !important; /* still muted, but readable */
    }

    [data-testid="stMetricValue"], .stDataFrame, code {
        font-family: 'JetBrains Mono', monospace !important;
    }

    [data-testid="stMetric"] {
        background: #16101F;
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 16px;
        padding: 16px 18px;
    }
    [data-testid="stMetricLabel"] { color: #9891A8 !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
    [data-testid="stMetricValue"] { color: #F5F3FF !important; }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: none; padding-bottom: 8px; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Unbounded', sans-serif; font-weight: 500; font-size: 0.82rem;
        color: #9891A8; background: #16101F;
        border-radius: 999px; padding: 10px 20px; border: 1px solid rgba(255,255,255,0.07);
    }
    .stTabs [aria-selected="true"] {
        color: #0B0710 !important;
        background: linear-gradient(90deg, #8B5CF6, #FF3EA5) !important;
        border: none !important; font-weight: 700;
    }

    .stButton>button {
        background: linear-gradient(90deg, #8B5CF6, #FF3EA5 60%, #22D3EE);
        color: #0B0710; font-weight: 700; font-family: 'Unbounded', sans-serif; font-size: 0.85rem;
        border: none; border-radius: 999px; padding: 0.6rem 1.4rem;
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(255,62,165,0.4);
    }

    .stTextArea textarea, .stSelectbox [data-baseweb="select"], .stNumberInput input {
        background: #16101F !important;
        border: 1px solid rgba(255,255,255,0.09) !important;
        color: #F5F3FF !important; border-radius: 10px !important;
    }

    .ticker-tape-wrap {
        overflow: hidden; white-space: nowrap;
        border-radius: 999px; background: #16101F;
        border: 1px solid rgba(255,255,255,0.07);
        padding: 10px 0; margin-bottom: 1.4rem;
    }
    .ticker-tape {
        display: inline-block; padding-left: 100%;
        animation: ticker-scroll var(--ticker-duration, 30s) linear infinite;
        font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #C9C4DA;
    }
    .ticker-tape span { margin-right: 2.5rem; }
    .ticker-tape .dot { margin-right: 5px; font-size: 0.7rem; }
    @keyframes ticker-scroll {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-100%); }
    }
    @media (prefers-reduced-motion: reduce) { .ticker-tape { animation: none; padding-left: 0; } }
    </style>
    """,
    unsafe_allow_html=True,
)

_dot_colors = ["#8B5CF6", "#FF3EA5", "#22D3EE", "#B4FF39"]
_tape_html = "".join(
    f'<span><span class="dot" style="color:{_dot_colors[i % 4]}">●</span>{t}</span>'
    for i, t in enumerate(TICKER_TAPE * 3)
)
# Scroll speed scales with ticker count so it stays calm and readable no matter
# how many tickers are added later -- roughly 3.5 seconds of travel per ticker.
_tape_duration = max(30, len(TICKER_TAPE) * 3.5)
st.markdown(
    f'<div class="ticker-tape-wrap"><div class="ticker-tape" '
    f'style="--ticker-duration: {_tape_duration}s;">{_tape_html}</div></div>',
    unsafe_allow_html=True,
)

st.title("📈 Swing / Day-Trade Screener")
st.caption(
    "Educational tool, not financial advice. The movement score flags unusual "
    "short-term setups; the strategy tab adds a trend filter on top and backtests "
    "it as real trades. No system can guarantee winning trades — treat this as a "
    "hypothesis-testing tool, and paper-trade before risking real money."
)


def style_pnl(df: pd.DataFrame, pct_col: str = "net_return_pct"):
    """Color-code a returns column neon-lime/hot-pink, gen-z terminal style."""
    def _color(val):
        if pd.isna(val):
            return ""
        color = "#B4FF39" if val > 0 else ("#FF4D6D" if val < 0 else "#9891A8")
        return f"color: {color}; font-weight: 700;"
    return df.style.map(_color, subset=[pct_col]) if pct_col in df.columns else df


def style_signal(df: pd.DataFrame, col: str = "signal"):
    """Highlight rows that meet the entry criteria with a gradient glow tint."""
    def _row(row):
        if col in row and row[col] in ("LONG SETUP", "LONG SETUP (confirmed)"):
            return ["background: linear-gradient(90deg, rgba(139,92,246,0.16), rgba(255,62,165,0.10)); font-weight: 700;"] * len(row)
        if col in row and row[col] == "AWAITING CONFIRMATION":
            return ["background: rgba(240, 200, 60, 0.08);"] * len(row)
        return [""] * len(row)
    return df.style.apply(_row, axis=1) if col in df.columns else df


tab_screen, tab_strategy, tab_today, tab_track, tab_size, tab_journal = st.tabs(
    ["🔍 Screener", "🧪 Strategy Backtest", "🎯 Today's Signals", "📊 Track Record", "💰 Position Sizing", "📒 Trade Journal"]
)

# ---------------------------------------------------------------------------
# Tab 1: original movement screener
# ---------------------------------------------------------------------------
with tab_screen:
    with st.sidebar:
        st.header("Screener Settings")
        tickers_input = st.text_area(
            "Tickers (one per line, or comma-separated)",
            value="AAPL\nMSFT\nNVDA\nTSLA\nAMD\nAMZN\nGOOGL\nMETA\nNFLX\nAVGO\nAMC\nBYND\nGME\nGPRO\n^HSI\nSPY\nIWM\nPDD\nJD\nTSLL\nBULL\nRKT\nENPH\n^VIX\nINTC\nDOGE-USD\nBABA\nPYPL\nBTC-USD\nDJT\nHOOD\nROBN\nETSY\nGOOG\nNKE\nSOFI\nCOIN\nBIDU\nUBER\nFUBO\nSHOP\nARKG\nKOSS\nNIO\nSMCI\nBB\nMU\nDIS\nDELL\nPLTR\nBRK-A\nLULU\nROKU\nABNB\nUVXY\nAI",
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
            value="AAPL\nMSFT\nNVDA\nTSLA\nAMD\nAMZN\nGOOGL\nMETA\nNFLX\nAVGO\nAMC\nBYND\nGME\nGPRO\n^HSI\nSPY\nIWM\nPDD\nJD\nTSLL\nBULL\nRKT\nENPH\n^VIX\nINTC\nDOGE-USD\nBABA\nPYPL\nBTC-USD\nDJT\nHOOD\nROBN\nETSY\nGOOG\nNKE\nSOFI\nCOIN\nBIDU\nUBER\nFUBO\nSHOP\nARKG\nKOSS\nNIO\nSMCI\nBB\nMU\nDIS\nDELL\nPLTR\nBRK-A\nLULU\nROKU\nABNB\nUVXY\nAI",
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
        require_confirmation = st.checkbox(
            "Require breakout confirmation before entry", value=True, key="require_confirmation",
        )
        confirm_window_days = st.slider(
            "Confirmation window (days)", 1, 10, 3, key="confirm_window",
            disabled=not require_confirmation,
        )
        if require_confirmation:
            st.caption(
                "A signal only trades once price closes above the signal day's high within "
                "this window — filters out breakouts that immediately fail, at the cost of "
                "entering a bit later and missing a few real ones too."
            )

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
                require_confirmation=require_confirmation, confirm_window_days=confirm_window_days,
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
        value="AAPL\nMSFT\nNVDA\nTSLA\nAMD\nAMZN\nGOOGL\nMETA\nNFLX\nAVGO\nAMC\nBYND\nGME\nGPRO\n^HSI\nSPY\nIWM\nPDD\nJD\nTSLL\nBULL\nRKT\nENPH\n^VIX\nINTC\nDOGE-USD\nBABA\nPYPL\nBTC-USD\nDJT\nHOOD\nROBN\nETSY\nGOOG\nNKE\nSOFI\nCOIN\nBIDU\nUBER\nFUBO\nSHOP\nARKG\nKOSS\nNIO\nSMCI\nBB\nMU\nDIS\nDELL\nPLTR\nBRK-A\nLULU\nROKU\nABNB\nUVXY\nAI",
        height=180,
        key="today_tickers",
    )
    today_quantile = st.slider("Score threshold (percentile)", 0.5, 0.95, 0.8, step=0.05, key="today_q")
    today_require_confirmation = st.checkbox(
        "Require breakout confirmation", value=True, key="today_require_confirmation",
    )
    today_confirm_window = st.slider(
        "Confirmation window (days)", 1, 10, 3, key="today_confirm_window",
        disabled=not today_require_confirmation,
    )
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
                require_confirmation=today_require_confirmation, confirm_window_days=today_confirm_window,
            )

        if signals.empty:
            st.error("No data returned — check your tickers.")
        else:
            confirmed = signals[signals["signal"].isin(["LONG SETUP", "LONG SETUP (confirmed)"])]
            awaiting = signals[signals["signal"] == "AWAITING CONFIRMATION"]
            if not confirmed.empty:
                st.success(f"{len(confirmed)} ticker(s) confirmed and meet the strategy's entry criteria")
            elif not awaiting.empty:
                st.info(f"{len(awaiting)} ticker(s) have a raw signal but haven't confirmed yet — not actionable until they do.")
            else:
                st.info("No tickers currently meet the strategy's entry criteria — that's normal, and better than false positives.")

            st.dataframe(style_signal(signals, "signal"), use_container_width=True, hide_index=True)
            st.caption(
                "'score' is the score FROM THE DAY THE SIGNAL FIRED (this is what actually triggered "
                "it) -- 'today_score' is today's score, which can drift well below that by the time a "
                "setup confirms. 'LONG SETUP (confirmed)' means: movement score above threshold, price "
                "trending up, AND price has already closed above the signal day's high -- real "
                "follow-through, not just a one-day flicker. 'AWAITING CONFIRMATION' means the setup "
                "fired but hasn't proven itself yet. Neither is a recommendation to buy -- confirm with "
                "your own research."
            )

# ---------------------------------------------------------------------------
# Tab 4: live, out-of-sample track record
# ---------------------------------------------------------------------------
with tab_track:
    st.subheader("Live track record")
    st.caption(
        "Every confirmed LONG SETUP gets logged automatically by the daily scheduled check, "
        "then scored ~10 trading days later against what actually happened. Unlike every other "
        "tab here, this data is genuinely out-of-sample -- it didn't exist when the signal fired. "
        "This is the real test of whether the system works, not a backtest."
    )

    import os
    log_path = "signal_log.csv"
    if not os.path.exists(log_path):
        st.info(
            "No track record yet -- this file is created automatically the first time the "
            "scheduled GitHub Action logs a confirmed signal. Check back after it's run a few times."
        )
    else:
        log_df = pd.read_csv(log_path)
        if log_df.empty:
            st.info("Log file exists but is empty -- no confirmed signals logged yet.")
        else:
            resolved = log_df[log_df["outcome"].isin(["win", "loss"])]
            pending = log_df[log_df["outcome"] == "pending"]

            cols = st.columns(4)
            cols[0].metric("Total logged", len(log_df))
            cols[1].metric("Resolved", len(resolved))
            cols[2].metric("Pending", len(pending))
            if not resolved.empty:
                win_rate = (resolved["outcome"] == "win").mean() * 100
                cols[3].metric("Live win rate", f"{win_rate:.1f}%")
            else:
                cols[3].metric("Live win rate", "—")

            if not resolved.empty:
                avg_return = resolved["realized_return_pct"].mean()
                st.metric("Avg realized return (resolved trades)", f"{avg_return:.2f}%")
                if len(resolved) < 20:
                    st.warning(
                        f"Only {len(resolved)} resolved signal(s) so far -- far too small a sample "
                        "to draw real conclusions. Treat these numbers as provisional until this "
                        "grows to at least 30-50 resolved trades."
                    )

            st.dataframe(
                style_pnl(log_df, "realized_return_pct") if "realized_return_pct" in log_df.columns else log_df,
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "'entry_price' is the price observed when the signal was logged, not a guaranteed "
                "fill. Results are evaluated a fixed ~10 trading days later, regardless of what the "
                "strategy's own trailing-stop/trend-exit rules would have done -- this is a simpler, "
                "more transparent test of the raw signal, not the full trading strategy."
            )

# ---------------------------------------------------------------------------
# Tab 5: risk-based position sizing calculator
# ---------------------------------------------------------------------------
with tab_size:
    st.subheader("Position sizing calculator")
    st.caption(
        "Not financial advice -- this is a standard, widely-used risk-management calculation, "
        "not a recommendation for your specific situation. Sizes each trade by how much you're "
        "willing to LOSE if the stop hits, not by an equal dollar split across positions -- this "
        "differs from the backtest, which used equal-weight slots for simplicity."
    )

    col1, col2 = st.columns(2)
    with col1:
        account_equity = st.number_input(
            "Total account equity ($)", min_value=100.0, value=10000.0, step=500.0, key="size_equity",
        )
        risk_pct = st.slider(
            "Risk per trade (% of account)", 0.25, 5.0, 1.0, step=0.25, key="size_risk_pct",
        )
        if risk_pct > 2.0:
            st.warning(
                "Risking more than ~2% per trade on a single position is aggressive by common "
                "convention -- a short losing streak can do real damage to an account at this size."
            )
    with col2:
        entry_price = st.number_input("Planned entry price ($)", min_value=0.01, value=100.0, step=0.5, key="size_entry")
        stop_price = st.number_input("Stop-loss price ($)", min_value=0.01, value=95.0, step=0.5, key="size_stop")

    if stop_price >= entry_price:
        st.error("Stop-loss price must be below your entry price for a long position.")
    else:
        risk_amount = account_equity * (risk_pct / 100)
        risk_per_share = entry_price - stop_price
        shares = int(risk_amount // risk_per_share)
        position_value = shares * entry_price
        pct_of_account = (position_value / account_equity * 100) if account_equity > 0 else 0
        stop_distance_pct = (risk_per_share / entry_price) * 100

        st.divider()
        cols = st.columns(4)
        cols[0].metric("Dollar risk (if stopped out)", f"${risk_amount:,.2f}")
        cols[1].metric("Shares to buy", f"{shares:,}")
        cols[2].metric("Position value", f"${position_value:,.2f}")
        cols[3].metric("% of account in this trade", f"{pct_of_account:.1f}%")

        if shares == 0:
            st.warning(
                "Stop distance is too wide relative to your risk budget -- 0 shares fit within your "
                "risk limit at this entry/stop combination. Either widen your risk %, tighten the "
                "stop, or skip this trade."
            )
        elif pct_of_account > 50:
            st.warning(
                f"This position would use {pct_of_account:.0f}% of your total account. A tight stop "
                "distance can produce a large position value even at low % risk -- worth double-"
                "checking this isn't over-concentrating your account in one stock."
            )

        st.caption(f"Stop distance: {stop_distance_pct:.1f}% below entry.")

    st.divider()
    st.markdown("**Portfolio-level check (optional)**")
    max_open = st.slider("Planned max concurrent positions", 1, 10, 5, key="size_max_open")
    total_heat = risk_pct * max_open
    st.metric("Total portfolio risk if ALL positions stopped out same day", f"{total_heat:.1f}%")
    if total_heat > 10:
        st.warning(
            "Total portfolio heat above ~10% is high -- a genuinely bad, correlated day (several "
            "positions moving against you at once, which happens more often than independent "
            "probability suggests) could produce a serious drawdown. Consider fewer concurrent "
            "positions or a lower risk % per trade."
        )
    else:
        st.success("Total portfolio heat is within a commonly-used conservative range.")

    st.caption(
        "Reminder: this calculator only sizes the position -- it doesn't predict whether the trade "
        "wins. Pick your risk % and stop level BEFORE entering, and don't widen a stop mid-trade "
        "to avoid taking a loss -- that's one of the most common ways disciplined plans break down."
    )

# ---------------------------------------------------------------------------
# Tab 6: personal trade journal (your real trades, not signal outcomes)
# ---------------------------------------------------------------------------
with tab_journal:
    st.subheader("Trade journal")
    st.caption(
        "Your actual trades -- distinct from the Track Record tab, which scores what the "
        "SIGNALS did automatically. This is what YOU actually did with them."
    )

    try:
        journal_df, journal_sha = journal_store.load_journal()
        journal_available = True
    except Exception as e:
        journal_available = False
        st.error(
            f"Couldn't connect to the journal storage ({e}). This needs GITHUB_TOKEN and "
            "GITHUB_REPO set in this app's Streamlit Cloud secrets (Settings -> Secrets), "
            "separate from your GitHub Actions secrets."
        )

    if journal_available:
        with st.expander("➕ Log a new trade", expanded=journal_df.empty):
            c1, c2, c3 = st.columns(3)
            with c1:
                j_ticker = st.text_input("Ticker", key="j_ticker").strip().upper()
                j_entry_date = st.date_input("Entry date", value=pd.Timestamp.today(), key="j_entry_date")
            with c2:
                j_entry_price = st.number_input("Entry price ($)", min_value=0.01, value=100.0, step=0.5, key="j_entry_price")
                j_units = st.number_input("Units (shares)", min_value=1, value=1, step=1, key="j_units")
            with c3:
                j_stop_loss = st.number_input("Stop-loss price ($)", min_value=0.0, value=0.0, step=0.5, key="j_stop_loss")
                j_sell_trigger = st.number_input("Sell trigger / target ($)", min_value=0.0, value=0.0, step=0.5, key="j_sell_trigger")
            j_notes = st.text_input("Notes (optional)", key="j_notes")

            if st.button("Add to journal", type="primary", key="j_add"):
                if not j_ticker:
                    st.warning("Enter a ticker first.")
                else:
                    new_row = {
                        "ticker": j_ticker,
                        "entry_date": str(j_entry_date),
                        "entry_price": j_entry_price,
                        "units": j_units,
                        "total_cost": round(j_entry_price * j_units, 2),
                        "stop_loss": j_stop_loss if j_stop_loss > 0 else None,
                        "sell_trigger": j_sell_trigger if j_sell_trigger > 0 else None,
                        "status": "open",
                        "exit_date": None, "exit_price": None,
                        "realized_pnl": None, "realized_pnl_pct": None,
                        "notes": j_notes,
                    }
                    updated = pd.concat([journal_df, pd.DataFrame([new_row])], ignore_index=True)
                    try:
                        journal_store.save_journal(updated, journal_sha, f"Log trade: {j_ticker}")
                        st.success(f"Logged {j_ticker}.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save: {e}")

        open_trades = journal_df[journal_df["status"] == "open"]
        closed_trades_j = journal_df[journal_df["status"] == "closed"]

        if not open_trades.empty:
            st.markdown("**Open positions**")

            refresh_col, ts_col = st.columns([1, 3])
            do_refresh = refresh_col.button("🔄 Refresh live prices", key="j_refresh_prices")
            if do_refresh or "j_price_cache" not in st.session_state:
                cache = {}
                with st.spinner("Fetching current prices..."):
                    for tkr in open_trades["ticker"].unique():
                        cache[tkr] = get_current_price(tkr)
                st.session_state["j_price_cache"] = cache
                st.session_state["j_price_fetched_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

            price_cache = st.session_state.get("j_price_cache", {})
            if "j_price_fetched_at" in st.session_state:
                ts_col.caption(f"Live prices as of {st.session_state['j_price_fetched_at']} (local time)")

            display_df = open_trades.drop(columns=["exit_date", "exit_price", "realized_pnl", "realized_pnl_pct"]).copy()
            regular_prices, extended_prices, extended_labels = [], [], []
            market_values, unrealized_pnls, unrealized_pcts = [], [], []
            fetch_errors = {}
            for _, row in display_df.iterrows():
                info = price_cache.get(row["ticker"], {})
                price = info.get("price")  # most-current available, used for P&L
                regular_prices.append(info.get("regular_price"))
                extended_prices.append(info.get("extended_price"))
                extended_labels.append(info.get("extended_label"))
                if price is not None:
                    mv = price * float(row["units"])
                    pnl = mv - float(row["total_cost"])
                    pnl_pct = (price / float(row["entry_price"]) - 1) * 100
                    market_values.append(round(mv, 2))
                    unrealized_pnls.append(round(pnl, 2))
                    unrealized_pcts.append(round(pnl_pct, 2))
                else:
                    market_values.append(None)
                    unrealized_pnls.append(None)
                    unrealized_pcts.append(None)
                    if info.get("error"):
                        fetch_errors[row["ticker"]] = info["error"]

            display_df["regular_price"] = pd.to_numeric(pd.Series(regular_prices), errors="coerce")
            display_df["extended_price"] = pd.to_numeric(pd.Series(extended_prices), errors="coerce")
            display_df["extended_session"] = pd.Series(extended_labels).fillna("")
            display_df["market_value"] = pd.to_numeric(pd.Series(market_values), errors="coerce")
            display_df["unrealized_pnl"] = pd.to_numeric(pd.Series(unrealized_pnls), errors="coerce")
            display_df["unrealized_pnl_pct"] = pd.to_numeric(pd.Series(unrealized_pcts), errors="coerce")

            if fetch_errors:
                st.warning(
                    "Couldn't fetch a live price for: " +
                    ", ".join(f"{t} ({e})" for t, e in fetch_errors.items())
                )

            st.dataframe(
                style_pnl(display_df, "unrealized_pnl_pct"), use_container_width=True, hide_index=True,
            )
            st.caption(
                "'regular_price' is the standard trading-session price (or last close if markets "
                "are shut, or the live 24h price for crypto). 'extended_price' only fills in during "
                "actual pre-market/after-hours windows -- blank the rest of the time, which is "
                "correct, not a bug. Market value and P&L use whichever price is most current. "
                "Click Refresh to update -- prices don't auto-update on their own."
            )

            with st.expander("✅ Close a position"):
                open_labels = [f"{r['ticker']} ({r['entry_date']}, {r['units']} units)" for _, r in open_trades.iterrows()]
                selected = st.selectbox("Select position to close", open_labels, key="j_close_select")
                sel_idx = open_trades.index[open_labels.index(selected)]

                cc1, cc2 = st.columns(2)
                j_exit_date = cc1.date_input("Exit date", value=pd.Timestamp.today(), key="j_exit_date")
                j_exit_price = cc2.number_input("Exit price ($)", min_value=0.01, value=100.0, step=0.5, key="j_exit_price")

                if st.button("Close this trade", type="primary", key="j_close_btn"):
                    row = journal_df.loc[sel_idx]
                    entry_price = float(row["entry_price"])
                    units = float(row["units"])
                    pnl = (j_exit_price - entry_price) * units
                    pnl_pct = (j_exit_price / entry_price - 1) * 100

                    journal_df.loc[sel_idx, "status"] = "closed"
                    journal_df.loc[sel_idx, "exit_date"] = str(j_exit_date)
                    journal_df.loc[sel_idx, "exit_price"] = j_exit_price
                    journal_df.loc[sel_idx, "realized_pnl"] = round(pnl, 2)
                    journal_df.loc[sel_idx, "realized_pnl_pct"] = round(pnl_pct, 2)

                    try:
                        journal_store.save_journal(journal_df, journal_sha, f"Close trade: {row['ticker']}")
                        st.success(f"Closed {row['ticker']} at ${j_exit_price} ({pnl_pct:+.2f}%).")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
        else:
            st.info("No open positions logged yet.")

        if not closed_trades_j.empty:
            st.markdown("**Closed positions**")
            cols = st.columns(3)
            total_pnl = closed_trades_j["realized_pnl"].sum()
            win_rate = (closed_trades_j["realized_pnl"] > 0).mean() * 100
            cols[0].metric("Total realized P&L", f"${total_pnl:,.2f}")
            cols[1].metric("Win rate", f"{win_rate:.1f}%")
            cols[2].metric("Trades closed", len(closed_trades_j))
            st.dataframe(style_pnl(closed_trades_j, "realized_pnl_pct"), use_container_width=True, hide_index=True)
