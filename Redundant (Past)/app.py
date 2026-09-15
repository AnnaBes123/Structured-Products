"""
FCN Probability Editor

A Fixed Coupon Note (FCN) can be decomposed into a zero-coupon bond plus a
short put option on the underlying. The investor is exposed to downside risk
if the stock closes below the strike at maturity. This tool estimates that
probability from historical data:

  1. Pick a ticker and pull its price history from Yahoo Finance.
  2. Set a strike (as a price or as % of current spot).
  3. Set the note's tenor (how many trading days until maturity).
  4. The tool looks at every historical window of that length, applies the
     historical return to today's spot price to get a simulated terminal
     price, and reports what fraction of those simulated outcomes fall
     below the strike. This is a standard historical-simulation estimate.

Run with:
    source venv/bin/activate
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="FCN Probability Editor", layout="wide")

PERIOD_OPTIONS = {
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "10 Years": "10y",
    "Max": "max",
}


@st.cache_data(ttl=3600)
def fetch_prices(ticker, period):
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        return None
    close = data["Close"].dropna()
    close.name = "Close"
    return close


def simulated_terminal_prices(close, spot, horizon_days):
    returns = close.pct_change(horizon_days).dropna()
    return spot * (1 + returns.values)


st.title("FCN Probability Editor")
st.caption(
    "Estimate the historical probability that a stock closes below a given "
    "strike at the maturity of a Fixed Coupon Note."
)

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    period_label = st.selectbox("Historical data", list(PERIOD_OPTIONS.keys()), index=2)
    horizon_days = st.number_input(
        "Note tenor (trading days)", min_value=1, max_value=2520, value=252, step=1,
        help="252 trading days ≈ 1 year, 63 ≈ 1 quarter, 21 ≈ 1 month.",
    )
    strike_mode = st.radio("Set strike by", ["% of current spot", "Price"], horizontal=True)

if not ticker:
    st.info("Enter a ticker to begin.")
    st.stop()

close = fetch_prices(ticker, PERIOD_OPTIONS[period_label])

if close is None or len(close) < 2:
    st.error(f"No data found for '{ticker}'. Check the ticker symbol.")
    st.stop()

spot = float(close.iloc[-1])
last_date = close.index[-1].date()

with st.sidebar:
    if strike_mode == "% of current spot":
        strike_pct = st.slider("Strike (% of spot)", min_value=30, max_value=100, value=70, step=1)
        strike = spot * strike_pct / 100
    else:
        strike = st.number_input("Strike price", min_value=0.0, value=round(spot * 0.7, 2), step=0.5)
        strike_pct = strike / spot * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Spot", f"${spot:,.2f}", help=f"Last close, {last_date}")
col2.metric("Strike", f"${strike:,.2f}")
col3.metric("Strike as % of spot", f"{strike_pct:.1f}%")
col4.metric("Tenor", f"{horizon_days} trading days")

if len(close) <= horizon_days:
    st.warning(
        f"Only {len(close)} trading days of history are available, which is not "
        f"enough to build a full {horizon_days}-day rolling window. Results below "
        "use whatever overlapping windows are available and may be unreliable."
    )

sim_prices = simulated_terminal_prices(close, spot, horizon_days)

if len(sim_prices) == 0:
    st.error("Not enough historical data to compute a probability for this tenor.")
    st.stop()

prob_below = float(np.mean(sim_prices < strike))
naive_prob_below = float((close < strike).mean())

st.subheader("Results")
r1, r2 = st.columns(2)
r1.metric(
    f"P(close < strike at +{horizon_days}d)",
    f"{prob_below:.1%}",
    help="Fraction of historical return windows of this length that, applied to today's "
    "spot, would land below the strike.",
)
r2.metric(
    "Naive: % of historical days already below strike",
    f"{naive_prob_below:.1%}",
    help="Simple frequency of historical closes below the strike level, no horizon applied. "
    "Shown for reference only.",
)

st.subheader("Price history")
fig = go.Figure()
fig.add_trace(go.Scatter(x=close.index, y=close.values, name="Close", line=dict(color="#1f77b4")))
fig.add_hline(y=strike, line_dash="dash", line_color="red", annotation_text="Strike")
fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Price ($)")
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"Distribution of simulated {horizon_days}-day terminal prices")
hist_fig = go.Figure()
hist_fig.add_trace(go.Histogram(x=sim_prices, nbinsx=60, marker_color="#1f77b4", name="Simulated terminal price"))
hist_fig.add_vline(x=strike, line_dash="dash", line_color="red", annotation_text="Strike")
hist_fig.add_vline(x=spot, line_dash="dot", line_color="green", annotation_text="Spot")
hist_fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Simulated price ($)")
st.plotly_chart(hist_fig, use_container_width=True)

st.caption(
    f"Based on {len(sim_prices):,} overlapping {horizon_days}-day historical windows "
    f"from {close.index[0].date()} to {last_date}."
)
