import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import t

# --- APP CONFIG ---
st.set_page_config(page_title="Sovereign Apex v5", layout="wide")

# --- STATE MANAGEMENT ---
if 'shove_history' not in st.session_state: st.session_state.shove_history = []
if 'total_scans' not in st.session_state: st.session_state.total_scans = 0

# --- APP SECTIONS NAVIGATION ---
section = st.sidebar.radio("Select Section", ["Dashboard", "Charts", "Watchlist"])

# --- TIMEFRAME SELECTION ---
timeframes = ["15m", "30m", "1h", "4h", "1d"]
selected_tf = st.sidebar.selectbox("Select Timeframe", timeframes)

# --- ASSETS LIST ---
assets = ["GC=F", "CL=F", "BTC-USD", "ETH-USD", "EURUSD=X", "GBPUSD=X", "AAPL", "TSLA"]

# --- UTILITY FUNCTIONS ---
def fetch_data(ticker, period="30d", interval="1h"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

def analyze_displacement(df):
    if df.empty or len(df) < 3: return False, 0.0
    c1, c3 = df.iloc[-3], df.iloc[-1]
    is_fvg = float(c3['Low']) > float(c1['High'])
    body = abs(float(c3['Close']) - float(c3['Open']))
    total = float(c3['High']) - float(c3['Low'])
    conviction = (body / total * 100) if total > 0 else 0
    return is_fvg, conviction

def monte_carlo_prob(df, sims=1000):
    if df.empty or 'Close' not in df.columns or len(df) < 20: return 0.5
    returns = df['Close'].pct_change().dropna()
    try:
        params = t.fit(returns)
        paths = [t.rvs(*params, size=10).sum() for _ in range(sims)]
        return np.mean([1 if p > 0.01 else 0 for p in paths])
    except:
        return 0.5

# =========================
# --- DASHBOARD SECTION ---
# =========================
if section == "Dashboard":
    st.title("🔱 SOVEREIGN APEX: DASHBOARD")
    
    st.subheader("Leaderboard & Notifications")
    win_rate = (len(st.session_state.shove_history) / max(st.session_state.total_scans,1)) * 100
    st.metric("Win Rate", f"{win_rate:.1f}%")
    st.metric("Total Gains", f"${len(st.session_state.shove_history)*10}")
    st.metric("Total High Rank Scans", st.session_state.total_scans)
    
    st.divider()
    st.subheader("Shove History")
    if st.session_state.shove_history:
        st.table(pd.DataFrame(st.session_state.shove_history))
    else:
        st.caption("No high-rank shoves logged yet.")

# =========================
# --- CHARTS SECTION ---
# =========================
elif section == "Charts":
    st.title("📊 SOVEREIGN APEX: CHARTS")
    sel_asset = st.selectbox("Select Asset", assets)
    
    # Fetch data according to selected timeframe
    interval = selected_tf
    df = fetch_data(sel_asset, period="30d", interval=interval)
    
    if df.empty:
        st.warning("No data available for this asset/timeframe.")
    else:
        # Candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']
        )])
        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # Displacement & Monte Carlo notifications
        fvg, conv = analyze_displacement(df)
        prob = monte_carlo_prob(df)
        if fvg and conv > 60:
            st.success(f"High-rank displacement detected! Conviction: {conv:.1f}%, Monte Carlo: {prob*100:.1f}%")
        else:
            st.info("No high-quality setups detected.")

        # Log shove button
        if st.button("✅ LOG SUCCESSFUL $10 SHOVE"):
            st.session_state.shove_history.append({"Time": pd.Timestamp.now(), "Asset": sel_asset, "TF": selected_tf})
            st.toast("Victory Recorded.")

# =========================
# --- WATCHLIST SECTION ---
# =========================
elif section == "Watchlist":
    st.title("📋 SOVEREIGN APEX: WATCHLIST")
    
    selected_assets = st.multiselect("Select assets to monitor", assets, default=assets[:4])
    
    watchlist_data = []
    for a in selected_assets:
        df = fetch_data(a, period="7d", interval=selected_tf)
        fvg, conv = analyze_displacement(df)
        prob = monte_carlo_prob(df)
        watchlist_data.append({
            "Asset": a,
            "Conviction": f"{conv:.1f}%",
            "MonteCarlo": f"{prob*100:.1f}%",
            "High Rank": "✅" if fvg and conv>60 else "❌"
        })
    st.table(pd.DataFrame(watchlist_data))
