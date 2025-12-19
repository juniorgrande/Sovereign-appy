import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# ---------------------- NLTK SENTIMENT ----------------------
@st.cache_resource
def load_nltk():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()
sia = load_nltk()

# ---------------------- APP CONFIG ----------------------
st.set_page_config(page_title="Sovereign Apex v5.0", layout="wide")

# ---------------------- STATE MANAGEMENT ----------------------
if 'shove_history' not in st.session_state: st.session_state.shove_history = []
if 'total_scans' not in st.session_state: st.session_state.total_scans = 0
if 'notifications' not in st.session_state: st.session_state.notifications = []

# ---------------------- ASSETS ----------------------
assets = [
    "GC=F","CL=F","BTC-USD","ETH-USD","EURUSD=X","GBPUSD=X",
    "AAPL","TSLA","MSFT","NFLX","AMZN","USDJPY=X","SI=F"
]

# ---------------------- TIMEFRAMES ----------------------
timeframes = {"15min":"15m","30min":"30m","1h":"1h","4h":"4h","1d":"1d"}

# ---------------------- DATA FETCH ----------------------
def fetch_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

# ---------------------- MONTE CARLO ----------------------
def monte_carlo_prob(df, sims=1000):
    if df.empty or 'Close' not in df.columns or len(df) < 20: return 0.5
    returns = df['Close'].pct_change().dropna()
    try:
        params = t.fit(returns)
        paths = [t.rvs(*params, size=10).sum() for _ in range(sims)]
        return np.mean([1 if p > 0.01 else 0 for p in paths])
    except:
        return 0.5

# ---------------------- FVG DISPLACEMENT ----------------------
def analyze_displacement(df):
    if df.empty or len(df) < 3: return False, 0.0
    c1, c3 = df.iloc[-3], df.iloc[-1]
    try:
        is_fvg = float(c3['Low']) > float(c1['High'])
        body = abs(float(c3['Close']) - float(c3['Open']))
        total = float(c3['High']) - float(c3['Low'])
        conviction = (body / total * 100) if total > 0 else 0
        return is_fvg, conviction
    except:
        return False, 0.0

# ---------------------- NEWS & SENTIMENT ----------------------
def get_news_sentiment(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        news = ticker_obj.news[:10]
        news_list = []
        for n in news:
            score = sia.polarity_scores(n['title'])['compound'] if sia else 0
            label = "🟢 BULLISH" if score>0.05 else "🔴 BEARISH" if score<-0.05 else "⚪ NEUTRAL"
            news_list.append({"title":n['title'], "source":n['publisher'], "score":score, "label":label})
        return news_list
    except:
        return []

# ---------------------- SIDEBAR ----------------------
st.sidebar.header("Sovereign Controls")
selected_asset = st.sidebar.selectbox("Select Asset", assets)
selected_tf = st.sidebar.selectbox("Timeframe", list(timeframes.keys()))
if st.sidebar.button("✅ LOG $10 SUCCESS"):
    st.session_state.shove_history.append({"Time": datetime.now().strftime("%Y-%m-%d %H:%M"), "Asset": selected_asset})
    st.toast("Victory Recorded!")
    st.rerun()

# ---------------------- TABS ----------------------
tab_chart, tab_watchlist, tab_dashboard, tab_news = st.tabs(["📊 Chart View", "📋 Watchlist", "🏆 Dashboard", "📰 News"])

# ---------------------- CHART TAB ----------------------
with tab_chart:
    st.subheader(f"{selected_asset} Chart ({selected_tf})")
    period = "1y" if selected_tf=="1d" else "60d"
    interval = timeframes[selected_tf]
    df_chart = fetch_data(selected_asset, period, interval)
    if not df_chart.empty:
        fig = go.Figure(data=[go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], 
                                             low=df_chart['Low'], close=df_chart['Close'])])
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)
        # Mini chart (last 100 bars)
        st.markdown("**Mini Chart (Last 100 Bars)**")
        mini_df = df_chart.tail(100)
        mini_fig = go.Figure(data=[go.Candlestick(x=mini_df.index, open=mini_df['Open'], high=mini_df['High'], 
                                                  low=mini_df['Low'], close=mini_df['Close'])])
        mini_fig.update_layout(template="plotly_dark", height=200, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(mini_fig, use_container_width=True)
    else:
        st.warning("No chart data available for this asset/timeframe.")

# ---------------------- WATCHLIST TAB ----------------------
with tab_watchlist:
    st.subheader("Watchlist Signals")
    watch_results = []
    for asset in assets:
        df_w = fetch_data(asset, "60d", "1d")
        fvg, conv = analyze_displacement(df_w)
        prob = monte_carlo_prob(df_w)
        rank_color = "🟡" if conv>70 else "🔴"
        watch_results.append({"Asset":asset, "Conviction":f"{conv:.1f}%", "MC Prob":f"{prob*100:.1f}%", "Rank":rank_color})
    st.table(pd.DataFrame(watch_results))

# ---------------------- DASHBOARD TAB ----------------------
with tab_dashboard:
    st.subheader("Top-Ranked Trades & Shove History")
    if st.session_state.shove_history:
        df_hist = pd.DataFrame(st.session_state.shove_history)
        st.table(df_hist)
    else:
        st.caption("No shoves logged yet.")

# ---------------------- NEWS TAB ----------------------
with tab_news:
    st.subheader(f"{selected_asset} News & Sentiment")
    news_items = get_news_sentiment(selected_asset)
    if news_items:
        for n in news_items:
            st.write(f"{n['label']} **{n['title']}**")
            st.caption(f"Source: {n['source']} | Score: {n['score']:.2f}")
    else:
        st.info("No recent news available.")

# ---------------------- NOTIFICATIONS ----------------------
if st.session_state.notifications:
    st.sidebar.subheader("🔔 Notifications")
    for note in st.session_state.notifications:
        st.sidebar.write(note)
