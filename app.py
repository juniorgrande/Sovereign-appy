import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time

# ================== NLTK INITIALIZATION ==================
@st.cache_resource
def load_nltk():
    try:
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception as e:
        st.error(f"NLTK Load Error: {e}")
        return None

sia = load_nltk()

# ================== APP CONFIG ==================
st.set_page_config(page_title="Sovereign Apex Fortified v5.0", layout="wide")

# ================== SESSION STATE ==================
if 'shove_history' not in st.session_state:
    st.session_state.shove_history = []

if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0

MAX_HISTORY = 100  # Limit history length

# ================== SAFE DATA FETCH ==================
def fetch_safe_data(ticker, period="5d", interval="1h", retries=3, delay=1):
    for _ in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df.empty or len(df) < 1:
                time.sleep(delay)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception:
            time.sleep(delay)
            continue
    st.warning(f"Unable to fetch data for {ticker}.")
    return pd.DataFrame()

# ================== MONTE CARLO ==================
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

# ================== DISPLACEMENT ANALYSIS ==================
def analyze_displacement(df):
    if df.empty or len(df) < 3:
        return False, 0.0
    try:
        c1, c3 = df.iloc[-3], df.iloc[-1]
        is_fvg = float(c3['Low']) > float(c1['High'])
        body = abs(float(c3['Close']) - float(c3['Open']))
        total = float(c3['High']) - float(c3['Low'])
        conviction = (body / total * 100) if total > 0 else 0
        return is_fvg, conviction
    except:
        return False, 0.0

# ================== SENTIMENT ANALYSIS ==================
def get_sentiment_score(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = getattr(ticker, "news", [])
        if not news: return 0, "Neutral"
        scores = [sia.polarity_scores(n['title'])['compound'] for n in news]
        avg_score = np.mean(scores)
        if avg_score > 0.05:
            sentiment = "BULLISH"
        elif avg_score < -0.05:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"
        return avg_score, sentiment
    except:
        return 0, "No Data"

# ================== DASHBOARD ==================
st.title("🔱 SOVEREIGN APEX: FORTIFIED v5.0")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Intelligence Control")
    assets = ["GC=F", "CL=F", "BTC-USD", "ETH-USD", "EURUSD=X", "GBPUSD=X",
              "AAPL", "TSLA", "MSFT", "NVDA"]
    sel_asset = st.selectbox("Active Asset", assets)

    # Log $10 Shove
    if st.button("✅ LOG $10 SUCCESS"):
        st.session_state.shove_history.append({"Time": pd.Timestamp.now(), "Asset": sel_asset})
        st.session_state.shove_history = st.session_state.shove_history[-MAX_HISTORY:]
        st.toast("Victory Recorded.")
        st.rerun()

# ---------- MISSION STATUS ----------
st.subheader("📊 Mission Status")
col1, col2, col3 = st.columns(3)
total_gains = len(st.session_state.shove_history) * 10
win_rate = (len(st.session_state.shove_history) / max(st.session_state.total_scans, 1)) * 100
col1.metric("Win Rate", f"{win_rate:.1f}%")
col2.metric("Total Gains", f"${total_gains}")
col3.metric("Total High Rank Scans", st.session_state.total_scans)

st.divider()

# ---------- MARKET RADAR ----------
st.subheader("📡 High-Quality Displacement Radar")
if st.button("🛰 RUN MARKET SCAN"):
    tfs = ["15m", "30m", "1h", "4h"]
    radar_results = []
    for tf in tfs:
        df = fetch_safe_data(sel_asset, interval=tf)
        fvg, conv = analyze_displacement(df)
        prob = monte_carlo_prob(df)
        if fvg and conv > 60:
            radar_results.append({"Timeframe": tf, "Conviction": f"{conv:.1f}%", "Monte Carlo Prob": f"{prob*100:.1f}%"})
            st.session_state.total_scans += 1
    if radar_results:
        st.table(pd.DataFrame(radar_results))
    else:
        st.info("No high-quality shoves detected in this scan.")

# ---------- CHART VISUALIZATION ----------
st.subheader("📈 Live Candlestick Chart")
df_chart = fetch_safe_data(sel_asset, interval="1h")
if not df_chart.empty:
    fig = go.Figure(data=[go.Candlestick(x=df_chart.index,
                                         open=df_chart['Open'],
                                         high=df_chart['High'],
                                         low=df_chart['Low'],
                                         close=df_chart['Close'])])
    fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Market data unavailable.")

# ---------- NEWS FEED ----------
st.subheader("📰 Intelligence Feed")
news_items = getattr(yf.Ticker(sel_asset), "news", [])[:5]
if news_items:
    for n in news_items:
        score = sia.polarity_scores(n['title'])['compound'] if sia else 0
        label = "🟢" if score > 0 else "🔴" if score < 0 else "⚪"
        st.write(f"{label} **{n['title']}**")
        st.caption(f"Source: {n.get('publisher','Unknown')} | Score: {score:.2f}")
else:
    st.info("No recent news available.")

# ---------- SHOVE HISTORY ----------
st.subheader("📓 Shove History")
if st.session_state.shove_history:
    st.table(pd.DataFrame(st.session_state.shove_history))
else:
    st.caption("No recorded shoves yet. Scan the market and log victories!")
