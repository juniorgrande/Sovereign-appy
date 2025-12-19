import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Initialize NLTK ---
@st.cache_resource
def load_sia():
    try:
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception as e:
        st.error(f"NLTK Load Error: {e}")
        return None

sia = load_sia()

# --- App Config ---
st.set_page_config(page_title="Sovereign Apex v5.0", layout="wide")

# --- Session State ---
if 'shove_history' not in st.session_state: st.session_state.shove_history = []
if 'total_scans' not in st.session_state: st.session_state.total_scans = 0

# --- Utility Functions ---
def fetch_safe_data(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        st.warning(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

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

def analyze_displacement(df):
    if df.empty or len(df) < 3: return False, 0.0
    try:
        c1, c3 = df.iloc[-3], df.iloc[-1]
        is_fvg = float(c3['Low']) > float(c1['High'])
        body = abs(float(c3['Close']) - float(c3['Open']))
        total = float(c3['High']) - float(c3['Low'])
        conviction = (body / total * 100) if total > 0 else 0
        return is_fvg, conviction
    except:
        return False, 0.0

def get_sentiment(ticker):
    if not sia: return 0, "Neutral"
    try:
        news = yf.Ticker(ticker).news[:10]
        if not news: return 0, "Neutral"
        scores = [sia.polarity_scores(n.get('title',''))['compound'] for n in news]
        avg = np.mean(scores)
        label = "BULLISH" if avg > 0.05 else "BEARISH" if avg < -0.05 else "NEUTRAL"
        return avg, label
    except:
        return 0, "Neutral"

# --- Layout Sections ---
menu = ["Chart", "Watchlist", "Dashboard"]
selection = st.sidebar.radio("Navigate Sections", menu)

# --- Assets & Timeframes ---
assets = ["GC=F","CL=F","BTC-USD","EURUSD=X","AAPL","TSLA"]
timeframes = {
    "1min":"1m", "5min":"5m", "15min":"15m", "1h":"1h",
    "1D":"1d", "1M":"1mo", "3M":"3mo", "6M":"6mo", "1Y":"1y"
}
sel_asset = st.sidebar.selectbox("Select Asset", assets)
sel_tf = st.sidebar.selectbox("Select Timeframe", list(timeframes.keys()))

# --- Chart Section ---
if selection == "Chart":
    st.subheader(f"📈 {sel_asset} Chart - {sel_tf}")
    df = fetch_safe_data(sel_asset, period=timeframes[sel_tf], interval=timeframes[sel_tf])
    if not df.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']
        )])
        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,b=0,t=30))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No chart data available.")

    # Collapsible Panels
    with st.expander("🔔 Notifications"):
        fvg, conv = analyze_displacement(df)
        prob = monte_carlo_prob(df)
        if fvg:
            rank_color = "gold" if conv>70 else "red"
            st.markdown(f"**Displacement Detected** - Conviction: {conv:.1f}%, Monte Carlo Prob: {prob*100:.1f}%")
        else:
            st.write("No high-rank displacement detected.")
    
    with st.expander("📚 Authors' Views (Combined Logic)"):
        # Placeholder for combined logic of 40 masters
        st.info("High-rank setup according to combined 40 masters' logic will appear here.")

# --- Watchlist Section ---
elif selection == "Watchlist":
    st.subheader("📋 Watchlist")
    for a in assets:
        df = fetch_safe_data(a, period="5d", interval="1h")
        price = df['Close'].iloc[-1] if not df.empty else "N/A"
        st.metric(label=a, value=f"{price}")

# --- Dashboard Section ---
elif selection == "Dashboard":
    st.subheader("🏆 Dashboard & Leaderboard")
    win_count = len(st.session_state.shove_history)
    total_scans = st.session_state.total_scans
    st.metric("Win Count", win_count)
    st.metric("Total Scans", total_scans)
    st.metric("Estimated Gains", f"${win_count*10}")
    
    st.table(pd.DataFrame(st.session_state.shove_history))
    
    with st.sidebar:
        if st.button("✅ Log $10 Shove"):
            st.session_state.shove_history.append({"Time": pd.Timestamp.now(), "Asset": sel_asset})
            st.toast("Victory recorded!")
