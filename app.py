"""
Streamlit web UI for the swing/day-trade screener.

Run with:
    streamlit run app.py

This opens a local website in your browser (usually http://localhost:8501)
where you can type tickers, click a button, and see the ranked results
as a table -- no command-line typing needed after this point.
"""

import streamlit as st
import pandas as pd

from swing_screener import run_screen

st.set_page_config(page_title="Swing Screener", layout="wide")

st.title("📈 Swing / Day-Trade Screener")
st.caption(
    "Screens tickers for unusual short-term momentum, volume, and volatility. "
    "This is a shortlist tool, not a prediction engine — always investigate "
    "flagged names further before acting."
)

with st.sidebar:
    st.header("Settings")
    tickers_input = st.text_area(
        "Tickers (one per line, or comma-separated)",
        value="AAPL\nMSFT\nNVDA\nTSLA\nAMD\nAMZN\nGOOGL\nMETA\nNFLX\nAVGO",
        height=220,
    )
    period = st.selectbox("History window", ["3mo", "6mo", "1y"], index=1)
    top_n = st.slider("Show top N results", min_value=5, max_value=50, value=15)
    run_button = st.button("Run Screen", type="primary")

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
                top,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "score": st.column_config.ProgressColumn(
                        "score", min_value=0, max_value=100, format="%.1f"
                    ),
                },
            )

            csv = top.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download as CSV", data=csv, file_name="watchlist.csv", mime="text/csv"
            )

            with st.expander("What do these columns mean?"):
                st.markdown(
                    """
- **pct_change_1d** — today's % price move
- **rsi_14** — momentum (>70 overbought, <30 oversold)
- **macd_hist** — momentum trend strength/direction
- **bb_bandwidth** — Bollinger Band width (low = volatility squeeze)
- **bb_position** — where price sits within its bands (0=lower band, 1=upper band)
- **atr_pct** — volatility as a % of price
- **rel_volume** — today's volume vs. 20-day average (>1.5 is notable)
- **range_break** — did price break its 20-day high/low today
- **score** — weighted combination of the above; higher = more unusual setup
                    """
                )
else:
    st.info("Set your tickers in the sidebar and click **Run Screen** to begin.")
