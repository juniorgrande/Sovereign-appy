import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import t

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Sovereign Apex v5", layout="wide")

# ---------------- STATE ----------------
if 'shove_history' not in st.session_state:
    st.session_state.shove_history = []
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0
if 'chart_unlocked' not in st.session_state:
    st.session_state.chart_unlocked = False

# ---------------- ASSETS ----------------
all_assets = [
    "GC=F", "CL=F", "BTC-USD", "ETH-USD", "EURUSD=X", "GBPUSD=X",
    "AAPL", "TSLA", "MSFT", "NVDA", "USDJPY=X", "SI=F"
]

# ---------------- SIDEBAR ----------------
view = st.sidebar.radio("Select View", ["Dashboard", "Chart", "Watchlist", "News"])
sel_asset = st.sidebar.selectbox("Select Asset", all_assets)
timeframe = st.sidebar.selectbox("Timeframe", ["15m", "30m", "1h", "4h", "1d"])
unlock_chart = st.sidebar.checkbox("🔓 Unlock Chart Interaction", value=False)
st.session_state.chart_unlocked = unlock_chart

# ---------------- UTILITIES ----------------
def fetch_data(ticker, period="5d", interval="1h"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

def analyze_displacement(df):
    if df.empty or len(df) < 3:
        return False, 0.0
    c1, c3 = df.iloc[-3], df.iloc[-1]
    is_fvg = float(c3['Low']) > float(c1['High'])
    body = abs(float(c3['Close']) - float(c3['Open']))
    total = float(c3['High']) - float(c3['Low'])
    conviction = (body / total * 100) if total > 0 else 0
    return is_fvg, conviction

def monte_carlo_prob(df, sims=1000):
    if df.empty or 'Close' not in df.columns or len(df) < 20:
        return 0.5
    returns = df['Close'].pct_change().dropna()
    try:
        params = t.fit(returns)
        paths = [t.rvs(*params, size=10).sum() for _ in range(sims)]
        return np.mean([1 if p > 0.01 else 0 for p in paths])
    except:
        return 0.5

# ---------------- VIEWS ----------------
if view == "Dashboard":
    st.title("🔱 SOVEREIGN APEX: DASHBOARD")
    win_rate = len(st.session_state.shove_history) / max(st.session_state.total_scans, 1) * 100
    col1, col2, col3 = st.columns(3)
    col1.metric("Win Rate", f"{win_rate:.1f}%")
    col2.metric("Total Gains", f"${len(st.session_state.shove_history)*10}")
    col3.metric("Total Scans", st.session_state.total_scans)

    st.subheader("🛰 High-Rank Trade Notifications")
    results = []
    for a in all_assets:
        df = fetch_data(a, interval=timeframe)
        fvg, conv = analyze_displacement(df)
        prob = monte_carlo_prob(df)
        if fvg and conv > 60:
            color = "🟡" if conv > 75 else "🔴"
            results.append({"Asset": a, "Conviction": f"{conv:.1f}%", "MC Prob": f"{prob*100:.1f}%", "Rank": color})
    if results:
        st.table(pd.DataFrame(results))
    else:
        st.info("No high-quality trades detected.")

elif view == "Chart":
    st.title(f"📈 Chart View: {sel_asset} ({timeframe})")
    df_chart = fetch_data(sel_asset, interval=timeframe)
    if not df_chart.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=df_chart.index,
            open=df_chart['Open'],
            high=df_chart['High'],
            low=df_chart['Low'],
            close=df_chart['Close']
        )])
        fig.update_layout(
            template="plotly_dark",
            height=600,
            margin=dict(l=0, r=0, b=0, t=0),
            dragmode='zoom' if st.session_state.chart_unlocked else False,
            xaxis=dict(fixedrange=not st.session_state.chart_unlocked),
            yaxis=dict(fixedrange=not st.session_state.chart_unlocked),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data to display.")

elif view == "Watchlist":
    st.title("📋 Watchlist")
    selected_assets = st.multiselect("Select assets to track", all_assets, default=["GC=F", "BTC-USD"])
    for a in selected_assets:
        st.subheader(a)
        df_w = fetch_data(a, interval=timeframe)
        if not df_w.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=df_w.index,
                open=df_w['Open'],
                high=df_w['High'],
                low=df_w['Low'],
                close=df_w['Close']
            )])
            fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,b=0,t=0),
                              dragmode='zoom' if st.session_state.chart_unlocked else False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No data for {a}")

elif view == "News":
    st.title("📰 Market News")
    ticker = yf.Ticker(sel_asset)
    news_items = ticker.news[:10] if hasattr(ticker, "news") else []
    if news_items:
        for n in news_items:
            st.write(f"**{n['title']}**")
            st.caption(f"Source: {n.get('publisher','N/A')}")
    else:
        st.info("No news available.")

# ---------------- LOG $10 SHOVE ----------------
with st.sidebar:
    if st.button("✅ LOG $10 SUCCESS"):
        st.session_state.shove_history.append({"Time": pd.Timestamp.now(), "Asset": sel_asset})
        st.toast("Victory Logged.")
