import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- NLTK Setup ---
@st.cache_resource
def load_nltk():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_nltk()

# --- CONFIG ---
st.set_page_config(page_title="Sovereign Apex v5 Optimized", layout="wide")

# --- STATE ---
if 'shove_history' not in st.session_state: st.session_state.shove_history = []
if 'total_scans' not in st.session_state: st.session_state.total_scans = 0

# --- CACHED DATA FETCH ---
@st.cache_data(ttl=3600)
def fetch_data(ticker, period="1y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# --- CACHED MONTE CARLO ---
@st.cache_data(ttl=3600)
def monte_carlo_prob(df, sims=500):
    if df.empty or 'Close' not in df.columns or len(df) < 20: 
        return 0.5
    returns = df['Close'].pct_change().dropna()
    params = t.fit(returns)
    paths = [t.rvs(*params, size=10).sum() for _ in range(sims)]
    return np.mean([1 if p > 0.01 else 0 for p in paths])

# --- CACHED NEWS ---
@st.cache_data(ttl=1800)
def fetch_news(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        news_list = ticker_obj.news[:10] if ticker_obj.news else []
        return news_list
    except:
        return []

# --- ANALYSIS FUNCTIONS ---
def analyze_displacement(df):
    if df.empty or len(df) < 3: return False, 0.0
    c1, c3 = df.iloc[-3], df.iloc[-1]
    is_fvg = float(c3['Low']) > float(c1['High'])
    body = abs(float(c3['Close']) - float(c3['Open']))
    total = float(c3['High']) - float(c3['Low'])
    conviction = (body / total * 100) if total > 0 else 0
    return is_fvg, conviction

# --- DASHBOARD ---
st.title("🔱 SOVEREIGN APEX: FORTIFIED V5")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Intelligence Control")
    assets = ["GC=F", "CL=F", "BTC-USD", "EURUSD=X", "AAPL", "TSLA", "ETH-USD", "MSFT"]
    sel_asset = st.selectbox("Active Asset", assets)
    timeframe = st.selectbox("Time Frame", ["1d","5d","1mo","3mo","6mo","1y","5y","max"])
    
    if st.button("✅ LOG $10 SUCCESS"):
        st.session_state.shove_history.append({"Time": pd.Timestamp.now(), "Asset": sel_asset})
        st.toast("Victory Recorded.")
        st.rerun()

# --- RADAR SCAN ---
st.subheader("📡 Displacement Radar")
if st.button("🛰 RUN MARKET SCAN"):
    tfs = ["15m","30m","1h","4h","1d"]
    results = []
    for tf in tfs:
        df_tf = fetch_data(sel_asset, period=timeframe, interval=tf)
        fvg, conv = analyze_displacement(df_tf)
        prob = monte_carlo_prob(df_tf)
        if fvg and conv > 60:
            st.session_state.total_scans += 1
            results.append({"TimeFrame": tf, "Conviction": f"{conv:.1f}%", "MonteCarloProb": f"{prob*100:.1f}%"})
    if results: st.table(pd.DataFrame(results))
    else: st.info("No high-quality shoves detected.")

# --- CHART VIEW (FULL PAGE ON DEMAND) ---
with st.expander("📊 Chart View", expanded=True):
    df_chart = fetch_data(sel_asset, period=timeframe, interval="1h")
    if not df_chart.empty:
        fig = go.Figure(data=[go.Candlestick(x=df_chart.index,
                                             open=df_chart['Open'], high=df_chart['High'],
                                             low=df_chart['Low'], close=df_chart['Close'])])
        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Chart data unavailable.")

# --- NEWS VIEW ---
with st.expander("📰 Intelligence Feed", expanded=False):
    news_items = fetch_news(sel_asset)
    if news_items:
        for n in news_items:
            score = sia.polarity_scores(n['title'])['compound'] if sia else 0
            label = "🟢" if score>0.05 else "🔴" if score<-0.05 else "⚪"
            st.write(f"{label} **{n['title']}**")
            st.caption(f"Source: {n['publisher']} | Score: {score:.2f}")
    else:
        st.info("No news available.")

# --- SHOVE HISTORY ---
st.subheader("📓 Shove History")
if st.session_state.shove_history:
    st.table(pd.DataFrame(st.session_state.shove_history))
else:
    st.caption("No trades logged yet.")
