import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t

# --- APP CONFIG ---
st.set_page_config(page_title="Sovereign Apex Cockpit", layout="wide")

# --- STATE MANAGEMENT ---
if 'shove_history' not in st.session_state:
    st.session_state.shove_history = []
if 'notifications' not in st.session_state:
    st.session_state.notifications = []

# --- ASSETS AND TIMEFRAMES ---
assets = ["GC=F", "CL=F", "BTC-USD", "ETH-USD", "EURUSD=X", "GBPUSD=X", "AAPL", "TSLA", "MSFT"]
timeframes = ["15m", "30m", "1h", "4h", "1d"]

# --- LAYOUT SELECTION ---
tab = st.sidebar.radio("Go to Section", ["Chart", "Watchlist", "News"])

# --- HELPER FUNCTIONS ---
def fetch_data(ticker, interval="1h", period="5d"):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
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

def monte_carlo_prob(df, sims=500):
    if df.empty or 'Close' not in df.columns or len(df) < 20:
        return 0.5
    returns = df['Close'].pct_change().dropna()
    try:
        params = t.fit(returns)
        paths = [t.rvs(*params, size=10).sum() for _ in range(sims)]
        return np.mean([1 if p > 0.01 else 0 for p in paths])
    except:
        return 0.5

def generate_author_notification(df, asset, tf):
    # Placeholder: Combine logic from 40 authors
    # For demo, we score based on displacement conviction + Monte Carlo
    fvg, conviction = analyze_displacement(df)
    mc_prob = monte_carlo_prob(df)
    rank = "Low"
    if fvg and conviction > 60 and mc_prob > 0.8:
        rank = "High"
    return {
        "Asset": asset,
        "Timeframe": tf,
        "Conviction": conviction,
        "MC Prob": mc_prob,
        "Rank": rank
    }

# --- SECTION: CHART ---
if tab == "Chart":
    st.title("📈 Chart Viewer")
    selected_asset = st.selectbox("Select Asset", assets)
    selected_tf = st.selectbox("Select Timeframe", timeframes)
    df_chart = fetch_data(selected_asset, interval=selected_tf)
    
    if not df_chart.empty:
        fig = go.Figure(data=[go.Candlestick(x=df_chart.index,
                                             open=df_chart['Open'],
                                             high=df_chart['High'],
                                             low=df_chart['Low'],
                                             close=df_chart['Close'])])
        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Market data not available.")

# --- SECTION: WATCHLIST ---
elif tab == "Watchlist":
    st.title("📋 Watchlist")
    watchlist_data = []
    for a in assets:
        df = fetch_data(a, interval="1h")
        fvg, conv = analyze_displacement(df)
        mc = monte_carlo_prob(df)
        watchlist_data.append({
            "Asset": a,
            "Displacement": fvg,
            "Conviction %": f"{conv:.1f}",
            "Monte Carlo Prob": f"{mc*100:.1f}%"
        })
    st.dataframe(pd.DataFrame(watchlist_data))

# --- SECTION: NEWS ---
elif tab == "News":
    st.title("📰 News & Market Intelligence")
    selected_asset = st.selectbox("Select Asset", assets)
    try:
        ticker = yf.Ticker(selected_asset)
        news_items = ticker.news[:5] if hasattr(ticker, 'news') else []
        if news_items:
            for n in news_items:
                st.write(f"**{n['title']}**")
                st.caption(f"Source: {n['publisher']}")
        else:
            st.info("No recent news found for this asset.")
    except:
        st.error("Unable to fetch news.")

# --- BOTTOM DASHBOARD PANEL ---
st.divider()
st.subheader("📊 Sovereign Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("Total $10 Gains", f"${len(st.session_state.shove_history)*10}")
col2.metric("Total Scans", len(st.session_state.notifications))
col3.button("Log $10 Successful Shove", key="log_gain", on_click=lambda: st.session_state.shove_history.append({"Asset": selected_asset, "Time": pd.Timestamp.now()}))

# Author Notifications
st.markdown("### 🔔 Notifications")
if st.button("Show High-Rank Trade Notifications"):
    st.session_state.notifications = []
    for a in assets:
        for tf in ["15m", "30m", "1h", "4h"]:
            df = fetch_data(a, interval=tf)
            note = generate_author_notification(df, a, tf)
            st.session_state.notifications.append(note)
    # Display notifications
    if st.session_state.notifications:
        notif_df = pd.DataFrame(st.session_state.notifications)
        notif_df['Rank Color'] = notif_df['Rank'].apply(lambda x: '🟡' if x=="High" else '🔴')
        st.dataframe(notif_df)
    else:
        st.info("No high-rank trade setups detected.")
