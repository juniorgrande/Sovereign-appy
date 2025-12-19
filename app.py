import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
import asyncio
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# --- NLTK INIT ---
@st.cache_resource
def load_nltk():
    try:
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except:
        return None

sia = load_nltk()

# --- APP CONFIG ---
st.set_page_config(page_title="Sovereign Apex PRO", layout="wide")

# --- STATE MANAGEMENT ---
if 'shove_history' not in st.session_state:
    st.session_state.shove_history = []

if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0

if 'asset_data_cache' not in st.session_state:
    st.session_state.asset_data_cache = {}

# --- ASSETS ---
ALL_ASSETS = ["GC=F", "CL=F", "BTC-USD", "ETH-USD", "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AAPL", "TSLA", "AMZN"]

# --- TIMEFRAMES ---
TIMEFRAMES = {
    "1m": "1m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1D": "1d",
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y"
}

# --- UTILITY FUNCTIONS ---
@st.cache_data
def fetch_yf_data(asset, period="1y", interval="1h"):
    try:
        df = yf.download(asset, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

def monte_carlo_prob(df, sims=1000):
    if df.empty or len(df) < 20:
        return 0.5
    returns = df['Close'].pct_change().dropna()
    try:
        params = t.fit(returns)
        paths = [t.rvs(*params, size=10).sum() for _ in range(sims)]
        return np.mean([1 if p > 0.01 else 0 for p in paths])
    except:
        return 0.5

def analyze_fvg(df):
    if df.empty or len(df) < 3: return False, 0.0
    c1, c3 = df.iloc[-3], df.iloc[-1]
    body = abs(c3['Close'] - c3['Open'])
    total = c3['High'] - c3['Low']
    conviction = (body / total * 100) if total > 0 else 0
    is_fvg = c3['Low'] > c1['High'] or c3['High'] < c1['Low']
    return is_fvg, conviction

def get_sentiment(asset):
    try:
        ticker = yf.Ticker(asset)
        news = ticker.news[:10]
        if not news: return 0, "Neutral"
        scores = [sia.polarity_scores(n['title'])['compound'] for n in news if 'title' in n]
        avg = np.mean(scores) if scores else 0
        label = "BULLISH" if avg > 0.05 else "BEARISH" if avg < -0.05 else "NEUTRAL"
        return avg, label
    except:
        return 0, "Neutral"

# --- AUTHOR LOGIC PLACEHOLDER ---
def merged_author_logic(df):
    # Here you could integrate Nisson, Murph, Homa, Chan, etc.
    # Currently we calculate a score based on FVG + Monte Carlo
    fvg, conv = analyze_fvg(df)
    prob = monte_carlo_prob(df)
    score = conv * prob  # weighted score
    rank = "Gold" if score > 70 else "Red"
    return {"score": score, "rank": rank, "fvg": fvg}

# --- UI LAYOUT ---
tabs = st.tabs(["Dashboard", "Charts", "Watchlist", "Notifications", "News"])

# --- DASHBOARD ---
with tabs[0]:
    st.header("🔱 Leaderboard & Mission Status")
    total_gains = len(st.session_state.shove_history) * 10
    st.metric("Total Gains", f"${total_gains}")
    st.metric("Total Scans", st.session_state.total_scans)
    if st.session_state.shove_history:
        st.subheader("🏆 Shove History")
        st.table(pd.DataFrame(st.session_state.shove_history))

# --- CHARTS ---
with tabs[1]:
    st.header("📈 Chart View")
    sel_asset_chart = st.selectbox("Select Asset", ALL_ASSETS, key="chart_asset")
    sel_timeframe = st.selectbox("Select Timeframe", list(TIMEFRAMES.keys()), key="chart_tf")
    period = "1y" if sel_timeframe in ["1D","1M","3M","6M","1Y"] else "6mo"
    interval = TIMEFRAMES[sel_timeframe]
    
    df_chart = fetch_yf_data(sel_asset_chart, period=period, interval=interval)
    if not df_chart.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=df_chart.index,
            open=df_chart['Open'], high=df_chart['High'],
            low=df_chart['Low'], close=df_chart['Close']
        )])
        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for this asset/timeframe.")

# --- WATCHLIST ---
with tabs[2]:
    st.header("📋 Watchlist")
    watchlist_assets = st.multiselect("Select Assets to Watch", ALL_ASSETS, default=ALL_ASSETS[:5])
    for asset in watchlist_assets:
        df_w = fetch_yf_data(asset, period="1mo", interval="1d")
        if not df_w.empty:
            last_close = df_w['Close'].iloc[-1]
            st.metric(asset, f"${last_close:.2f}")

# --- NOTIFICATIONS ---
with tabs[3]:
    st.header("🔔 Notifications & Author Insights")
    for asset in ALL_ASSETS[:5]:
        df_n = fetch_yf_data(asset, period="3mo", interval="1d")
        logic = merged_author_logic(df_n)
        st.write(f"{asset}: Rank: {logic['rank']}, Score: {logic['score']:.1f}, FVG Detected: {logic['fvg']}")

# --- NEWS ---
with tabs[4]:
    st.header("📰 News & Sentiment")
    sel_asset_news = st.selectbox("Asset for News", ALL_ASSETS, key="news_asset")
    news_list = yf.Ticker(sel_asset_news).news[:10]
    for n in news_list:
        if 'title' not in n: continue
        score = sia.polarity_scores(n['title'])['compound'] if sia else 0
        label = "🟢" if score>0 else "🔴" if score<0 else "⚪"
        st.write(f"{label} {n['title']}")
        st.caption(f"{n.get('publisher','Unknown')} | Score: {score:.2f}")

# --- LOGGING SUCCESSFUL $10 SHOVE ---
with st.sidebar:
    st.header("🛠️ Controls")
    if st.button("✅ LOG $10 SUCCESS"):
        st.session_state.shove_history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Asset": sel_asset_chart
        })
        st.toast("Victory Recorded.")
        st.experimental_rerun()
