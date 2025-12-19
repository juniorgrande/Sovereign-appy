import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- NLTK INITIALIZATION ---
@st.cache_resource
def load_nltk():
    try:
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception as e:
        st.error(f"NLTK Load Error: {e}")
        return None

sia = load_nltk()

# --- APP CONFIG ---
st.set_page_config(page_title="Sovereign Apex Fortified", layout="wide")

# --- STATE MANAGEMENT ---
if 'shove_history' not in st.session_state:
    st.session_state.shove_history = []
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0

# --- FETCH DATA SAFELY ---
def fetch_safe_data(ticker, period="5d", interval="1h"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty or len(df) < 1:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        st.sidebar.error(f"Fetch Error ({ticker}): {e}")
        return pd.DataFrame()

# --- MONTE CARLO PROBABILITY ---
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

# --- DISPLACEMENT ANALYSIS ---
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

# --- DASHBOARD ---
st.title("🔱 SOVEREIGN APEX: FORTIFIED")

with st.sidebar:
    st.header("Intelligence Control")
    assets = ["GC=F", "CL=F", "BTC-USD", "ETH-USD", "EURUSD=X", "GBPUSD=X", "AAPL", "TSLA", "MSFT"]
    sel_asset = st.selectbox("Active Asset", assets)
    
    if st.button("✅ LOG $10 SUCCESS"):
        st.session_state.shove_history.append({"Time": pd.Timestamp.now(), "Asset": sel_asset})
        st.toast("Victory Recorded.")
        st.rerun()

# --- RADAR SCAN ---
st.subheader("📡 Displacement Radar")
if st.button("🛰 RUN MARKET SCAN"):
    tfs = ["15m", "30m", "1h", "4h"]
    results = []
    for tf in tfs:
        data = fetch_safe_data(sel_asset, interval=tf)
        fvg, conv = analyze_displacement(data)
        prob = monte_carlo_prob(data)
        if fvg and conv > 60:
            st.session_state.total_scans += 1
            results.append({"Timeframe": tf, "Conviction": f"{conv:.1f}%", "MC Prob": f"{prob*100:.1f}%"})
    if results:
        st.table(pd.DataFrame(results))
    else:
        st.info("No high-quality shoves detected. Discipline is key.")

# --- CHART VISUALIZATION ---
df_chart = fetch_safe_data(sel_asset)
if not df_chart.empty:
    fig = go.Figure(data=[go.Candlestick(
        x=df_chart.index,
        open=df_chart['Open'],
        high=df_chart['High'],
        low=df_chart['Low'],
        close=df_chart['Close']
    )])
    fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Market is currently unresponsive or closed.")

# --- NEWS FEED & SENTIMENT ---
st.subheader("📰 Intelligence Feed")
news_items = getattr(yf.Ticker(sel_asset), "news", [])[:10]
if news_items:
    for n in news_items:
        title = n.get('title', 'No Title')
        score = sia.polarity_scores(title)['compound'] if sia else 0
        label = "🟢" if score > 0 else "🔴" if score < 0 else "⚪"
        st.write(f"{label} **{title}**")
        st.caption(f"Source: {n.get('publisher','Unknown')} | Score: {score:.2f}")
else:
    st.info("No recent news available.")

# --- SHOVE HISTORY ---
st.subheader("📓 Shove History")
if st.session_state.shove_history:
    st.table(pd.DataFrame(st.session_state.shove_history))
else:
    st.caption("No shoves logged yet.")
