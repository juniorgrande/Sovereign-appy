import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

# --- Function to fetch historical data ---
def fetch_data(ticker, interval="1d"):
    """
    Fetches data from Yahoo Finance with longer periods depending on interval.
    """
    try:
        if interval == "1d":
            df = yf.download(ticker, period="1y", interval=interval, progress=False)
        else:
            df = yf.download(ticker, period="60d", interval=interval, progress=False)
        
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

# --- Chart Display ---
def display_chart(ticker, interval="1d"):
    df = fetch_data(ticker, interval)
    if df.empty:
        st.warning("No data available for this asset/timeframe.")
        return

    # Full chart
    st.subheader(f"{ticker} Chart ({interval})")
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']
    )])
    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=0, r=0, b=0, t=0),
        dragmode=False  # locks scrolling/zooming by default
    )
    st.plotly_chart(fig, use_container_width=True)

    # Mini chart (last 100 bars)
    st.subheader("Mini Trend Overview")
    mini_df = df.tail(100)
    fig_mini = go.Figure(data=[go.Candlestick(
        x=mini_df.index,
        open=mini_df['Open'], high=mini_df['High'], low=mini_df['Low'], close=mini_df['Close']
    )])
    fig_mini.update_layout(
        template="plotly_dark",
        height=150,
        margin=dict(l=0, r=0, b=0, t=0),
        dragmode=False
    )
    st.plotly_chart(fig_mini, use_container_width=True)

# --- Sidebar for user selection ---
with st.sidebar:
    st.header("Chart Settings")
    asset_list = ["BTC-USD", "ETH-USD", "EURUSD=X", "GC=F", "AAPL", "TSLA"]
    selected_asset = st.selectbox("Select Asset", asset_list)
    interval_option = st.selectbox("Select Timeframe", ["1d", "1h", "30m", "15m"])

# --- Display charts ---
display_chart(selected_asset, interval_option)
